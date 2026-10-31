"""
Train and evaluate bearing-fault models on CWRU vibration features.

Pipeline: load .mat -> 12 kHz windows -> vibration features -> leak-aware split ->
compare candidates -> pick the best DEPLOYABLE model on the VALIDATION set ->
tune the alarm threshold on validation -> report once on the untouched TEST set.

Deployable models are 4-class sklearn pipelines (Normal / Inner Race / Ball / Outer Race).
The failure probability served to the twin is  1 - P(Normal).
Isolation Forest (trained on normal data only) and a rule baseline are compared for context.
"""

import json
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:  # plots are optional
    HAS_PLOTTING = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from src.data import (
    CLASS_LABELS, CLASS_NAMES, assign_splits, build_window_table, describe_split_protocol,
    load_all_recordings, recording_summary,
)
from src.features import FEATURE_NAMES

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SEED = 42


# ──────────────────────────── comparison-only models ────────────────────────────
class ThresholdBaseline:
    """Rule baseline: share of {rms, kurtosis, crest, impulse} above the normal 99th percentile."""

    cols = ["rms", "kurtosis", "crest_factor", "impulse_factor"]

    def fit(self, X: pd.DataFrame, y_binary):
        normal = X[np.asarray(y_binary) == 0]
        self.limits_ = normal[self.cols].quantile(0.99)
        return self

    def fault_proba(self, X: pd.DataFrame) -> np.ndarray:
        hits = np.column_stack([(X[c] > self.limits_[c]).to_numpy() for c in self.cols])
        return hits.mean(axis=1)


class IsolationForestDetector:
    """Unsupervised: fitted on NORMAL training windows only; score scaled with train stats."""

    def fit(self, X: pd.DataFrame, y_binary):
        normal = X[np.asarray(y_binary) == 0]
        self.scaler_ = StandardScaler().fit(normal)
        self.model_ = IsolationForest(n_estimators=200, random_state=SEED).fit(self.scaler_.transform(normal))
        s = -self.model_.score_samples(self.scaler_.transform(normal))
        self.lo_, self.hi_ = float(np.percentile(s, 50)), float(np.percentile(s, 99.9))
        return self

    def fault_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = -self.model_.score_samples(self.scaler_.transform(X))
        return np.clip((s - self.lo_) / (self.hi_ - self.lo_ + 1e-9), 0.0, 1.0)


# ──────────────────────────── metrics helpers ────────────────────────────
def binary_metrics(y_true, p_fault, threshold: float) -> Dict[str, float]:
    y_pred = (p_fault >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two = len(np.unique(y_true)) > 1
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p_fault)), 4) if two else 0.0,
        "pr_auc": round(float(average_precision_score(y_true, p_fault)), 4) if two else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": round(float(threshold), 4),
    }


def find_optimal_threshold(y_val, p_val) -> Tuple[float, float]:
    """Centre of the plateau of thresholds that reach the best validation F1 (robust when
    classes separate perfectly, where 'first best' would pick an extreme threshold)."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    f1s = np.array([f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid])
    best = f1s.max()
    plateau = grid[f1s >= best - 1e-9]
    return float(np.median(plateau)), float(best)


def multiclass_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def candidate_models() -> Dict[str, Pipeline]:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                             min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
        ]),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                         class_weight="balanced", random_state=SEED, verbosity=-1)),
        ])
    return models


def train_and_evaluate(data_dir: str = os.path.join(DATA_DIR, "cwru")) -> Dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    recs = load_all_recordings(data_dir)
    rec_table = recording_summary(recs)
    print(rec_table.to_string(index=False))

    table = assign_splits(build_window_table(recs))
    protocol = describe_split_protocol(table)
    table.to_csv(os.path.join(DATA_DIR, "cwru_windows.csv"), index=False)

    split = {s: table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, d in split.items():
        print(f"{s:5s}: {len(d):4d} windows  {d['class_name'].value_counts().to_dict()}")
    for s in ("train", "val", "test"):
        if split[s]["class_id"].nunique() < 4:
            raise RuntimeError(f"'{s}' split is missing a class; check the CWRU files in {data_dir}")

    X = {s: d[FEATURE_NAMES] for s, d in split.items()}
    y = {s: d["class_id"].to_numpy() for s, d in split.items()}
    yb = {s: (y[s] != 0).astype(int) for s in y}
    normal_idx = CLASS_NAMES.index("normal")

    rows, fitted, thresholds = [], {}, {}

    def record(name, p_val, p_test, deployable, mc_val=None, mc_test=None, ll=np.nan):
        th, val_f1 = find_optimal_threshold(yb["val"], p_val)
        thresholds[name] = th
        tm = binary_metrics(yb["test"], p_test, th)
        rows.append({
            "model": name, "deployable": deployable, "threshold": th,
            "val_f1": round(val_f1, 4),
            "val_macro_f1": mc_val["macro_f1"] if mc_val else np.nan,
            "val_logloss": round(ll, 5) if ll == ll else np.nan,
            "test_precision": tm["precision"], "test_recall": tm["recall"], "test_f1": tm["f1"],
            "test_pr_auc": tm["pr_auc"], "test_roc_auc": tm["roc_auc"],
            "test_accuracy_4class": mc_test["accuracy"] if mc_test else np.nan,
            "test_macro_f1_4class": mc_test["macro_f1"] if mc_test else np.nan,
            "test_tp": tm["tp"], "test_fp": tm["fp"], "test_tn": tm["tn"], "test_fn": tm["fn"],
        })
        print(f"[{name:20s}] thr={th:.2f}  test P={tm['precision']:.3f} R={tm['recall']:.3f} "
              f"F1={tm['f1']:.3f}  ROC-AUC={tm['roc_auc']:.3f}")

    # comparison-only models
    for name, det in (("Rule Baseline", ThresholdBaseline()), ("Isolation Forest", IsolationForestDetector())):
        det.fit(X["train"], yb["train"])
        record(name, det.fault_proba(X["val"]), det.fault_proba(X["test"]), deployable=False)

    # deployable 4-class models
    for name, pipe in candidate_models().items():
        pipe.fit(X["train"], y["train"])
        pv, pt = pipe.predict_proba(X["val"]), pipe.predict_proba(X["test"])
        record(
            name, 1 - pv[:, normal_idx], 1 - pt[:, normal_idx], deployable=True,
            mc_val=multiclass_metrics(y["val"], pv.argmax(1)),
            mc_test=multiclass_metrics(y["test"], pt.argmax(1)),
            ll=float(log_loss(y["val"], pv, labels=list(range(4)))),
        )
        fitted[name] = pipe

    results = pd.DataFrame(rows)
    deploy = results[results["deployable"]].sort_values(["val_macro_f1", "val_logloss"], ascending=[False, True])
    best_name = deploy.iloc[0]["model"]  # chosen on VALIDATION, never on test
    results = results.sort_values(["deployable", "test_f1"], ascending=[False, False])
    results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    best = fitted[best_name]
    thr = thresholds[best_name]
    p_test = best.predict_proba(X["test"])
    p_fault = 1 - p_test[:, normal_idx]
    pred_cls = p_test.argmax(1)
    final_bin = binary_metrics(yb["test"], p_fault, thr)
    final_mc = multiclass_metrics(y["test"], pred_cls)
    cm = confusion_matrix(y["test"], pred_cls, labels=list(range(4)))
    print(f"\nBEST (by validation): {best_name}  | test 4-class acc={final_mc['accuracy']}  "
          f"macro-F1={final_mc['macro_f1']}  | fault detection F1={final_bin['f1']}")

    # final deployable model: refit on train+val (hyper-parameters / threshold already fixed)
    final_pipe = candidate_models()[best_name]
    final_pipe.fit(pd.concat([X["train"], X["val"]]), np.concatenate([y["train"], y["val"]]))
    bundle = {
        "pipeline": final_pipe,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "model_name": best_name,
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, "best_model.joblib"))

    # sanity: refit model must still do well on the held-out test split
    p_ref = final_pipe.predict_proba(X["test"])
    refit_bin = binary_metrics(yb["test"], 1 - p_ref[:, normal_idx], thr)

    metadata = {
        "best_model_name": best_name,
        "dataset": "CWRU Bearing Data Center, drive-end accelerometer, resampled to 12 kHz",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "class_labels": CLASS_LABELS,
        "window_size": 2048,
        "window_stride": 512,
        "sample_rate_hz": 12000,
        "optimal_threshold": thr,
        "metrics_test": final_bin,
        "metrics_test_refit_all_train_val": refit_bin,
        "metrics_test_4class": final_mc,
        "confusion_matrix_4class": cm.tolist(),
        "status_thresholds": {
            "normal_max": round(thr * 0.6, 4),
            "warning_max": round(thr, 4),
            "critical_min": round(thr, 4),
        },
        "split_protocol": protocol,
        "split_sizes": {s: {CLASS_NAMES[k]: int(v) for k, v in d["class_id"].value_counts().items()}
                        for s, d in split.items()},
        "recordings": rec_table.to_dict(orient="records"),
        "caveat": ("Only one recording per fault class (0.007in, 0 HP). Fault test windows come from "
                   "later time blocks of the SAME recordings (purged gap, no shared samples); the normal "
                   "class is tested on an entire unseen recording/load. Add more CWRU fault files "
                   "(other loads / sizes) and re-run training for a recording-level fault evaluation."),
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__},
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "binary": final_bin, "four_class": final_mc,
                   "comparison": results.to_dict(orient="records")}, f, indent=2, default=float)

    if HAS_PLOTTING:
        make_plots(best, best_name, X["test"], y["test"], yb["test"], p_fault, pred_cls, cm, final_bin, table)
    print("Training complete. Artifacts written to models/ and reports/.")
    return metadata


def make_plots(best, best_name, X_test, y_test, yb_test, p_fault, pred_cls, cm, final_bin, table):
    # confusion matrix
    labels = [CLASS_LABELS[c].replace(" Fault", "") for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Test confusion matrix ({best_name})", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    # ROC / PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fpr, tpr, _ = roc_curve(yb_test, p_fault)
    a1.plot(fpr, tpr, lw=2, label=f"AUC = {final_bin['roc_auc']:.3f}"); a1.plot([0, 1], [0, 1], "--", c="gray")
    a1.set(title="ROC - fault detection", xlabel="False positive rate", ylabel="True positive rate"); a1.legend()
    pr, rc, _ = precision_recall_curve(yb_test, p_fault)
    a2.plot(rc, pr, lw=2, c="#e74c3c", label=f"PR-AUC = {final_bin['pr_auc']:.3f}")
    a2.set(title="Precision-Recall - fault detection", xlabel="Recall", ylabel="Precision"); a2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, "roc_pr_curves.png"), dpi=150); plt.close()

    # feature importance (model-native, else permutation)
    model = best.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        title = f"Feature importance ({best_name})"
    else:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=SEED).importances_mean
        title = f"Permutation importance ({best_name})"
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(FEATURE_NAMES)[order], imp[order], color="#3498db"); ax.set_title(title, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150); plt.close()

    # class balance per split
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = table[table["split"] != "drop"].groupby(["split", "class_name"]).size().unstack(fill_value=0)
    ct.loc[["train", "val", "test"]].plot(kind="bar", ax=ax, rot=0); ax.set_ylabel("windows")
    ax.set_title("Windows per class and split", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS_DIR, "class_balance.png"), dpi=150); plt.close()


if __name__ == "__main__":
    train_and_evaluate()
