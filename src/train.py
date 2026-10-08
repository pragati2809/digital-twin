"""
Model training, evaluation, threshold tuning, and artifact serialization module.

Implements stratified Train/Validation/Test splitting, model comparison
(Heuristic Baseline, Isolation Forest, Random Forest, LightGBM/XGBoost),
decision threshold optimization for imbalanced failure detection, and saves
best model pipeline and evaluation metrics.
"""

import json
import os
import sys
from typing import Dict, Tuple, Any

# Add project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data import load_dataset, get_standardized_dataframe
from src.features import create_features_df, ENGINEERED_FEATURE_NAMES

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))


class HeuristicThresholdModel:
    """
    Domain rule-based baseline model for industrial failure detection.
    Failure rules:
    1. Tool Wear > 200 min AND Torque > 40 Nm
    2. Heat Dissipation failure: Temp Diff < 8.6 K AND RPM < 1380
    3. Power Failure: Mechanical Power < 3500 W OR Mechanical Power > 9000 W
    """
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=ENGINEERED_FEATURE_NAMES)

        # Rule scores
        score = np.zeros(len(df))
        rule1 = (df["tool_wear"] > 200) & (df["torque"] > 40)
        rule2 = (df["temp_diff"] < 8.6) & (df["rpm"] < 1380)
        rule3 = (df["power_w"] < 3500) | (df["power_w"] > 9000)

        score[rule1] += 0.4
        score[rule2] += 0.4
        score[rule3] += 0.4

        proba_1 = np.clip(score, 0.0, 1.0)
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


class IsolationForestWrapper:
    """
    Unsupervised Anomaly Detection wrapper around IsolationForest.
    Maps anomaly scores [-1, 1] to pseudo-probabilities [0, 1].
    """
    def __init__(self, contamination=0.035, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict_proba(self, X):
        # decision_function returns negative values for anomalies
        scores = self.model.decision_function(X)
        # Normalize score into range [0, 1] where 1 is anomalous
        # Higher score = more anomalous = higher failure prob
        scaled_scores = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
        return np.column_stack([1.0 - scaled_scores, scaled_scores])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


def evaluate_predictions(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(
    y_val: np.ndarray, y_proba_val: np.ndarray, target_recall: float = 0.80
) -> Tuple[float, float]:
    """
    Sweeps thresholds on validation set to maximize F1 score while maintaining target recall.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.05, 0.95, 91)

    for th in thresholds:
        y_pred = (y_proba_val >= th).astype(int)
        rec = recall_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        # Prioritize F1 subject to reasonable recall (> target_recall * 0.9)
        if rec >= (target_recall * 0.85) and f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    # Fallback if no threshold met target recall condition
    if best_f1 < 0:
        for th in thresholds:
            y_pred = (y_proba_val >= th).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th

    return float(best_thresh), float(best_f1)


def train_and_evaluate(data_path: str = "data/ai4i2020.csv") -> Dict[str, Any]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df_raw = load_dataset(data_path)
    df_std = get_standardized_dataframe(df_raw)

    X = create_features_df(df_raw)
    y = df_std["machine_failure"].values

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val
    )  # 0.17647 * 0.85 = 0.15

    print(f"Data Split Sizes: Train={len(X_train)} (Failures={y_train.sum()}), Val={len(X_val)} (Failures={y_val.sum()}), Test={len(X_test)} (Failures={y_test.sum()})")

    # Define candidate models
    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    candidates = {
        "Heuristic Baseline": HeuristicThresholdModel(),
        "Isolation Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForestWrapper(contamination=0.035, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }

    if HAS_LIGHTGBM:
        candidates["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgb.LGBMClassifier(
                n_estimators=100, scale_pos_weight=pos_weight, random_state=42, verbosity=-1
            )),
        ])
    elif HAS_XGBOOST:
        candidates["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBClassifier(
                n_estimators=100, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss"
            )),
        ])

    results = []
    trained_pipelines = {}
    optimal_thresholds = {}

    for name, pipeline in candidates.items():
        print(f"\n--- Training {name} ---")
        if name == "Heuristic Baseline":
            pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(X_train, y_train)

        # Validation prediction & threshold tuning
        if hasattr(pipeline, "predict_proba"):
            val_proba = pipeline.predict_proba(X_val)[:, 1]
            test_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            val_proba = pipeline.predict(X_val)
            test_proba = pipeline.predict(X_test)

        opt_th, val_f1 = find_optimal_threshold(y_val, val_proba, target_recall=0.80)
        optimal_thresholds[name] = opt_th

        val_metrics = evaluate_predictions(y_val, val_proba, opt_th)
        test_metrics = evaluate_predictions(y_test, test_proba, opt_th)

        print(f"[{name}] Optimal Threshold = {opt_th:.4f}")
        print(f"[{name}] Test Metrics: Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}")

        trained_pipelines[name] = pipeline

        record = {
            "model": name,
            "threshold": opt_th,
            "val_f1": val_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_tp": test_metrics["tp"],
            "test_fp": test_metrics["fp"],
            "test_tn": test_metrics["tn"],
            "test_fn": test_metrics["fn"],
        }
        results.append(record)

    results_df = pd.DataFrame(results).sort_values(by="test_f1", ascending=False)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_threshold = optimal_thresholds[best_model_name]

    print(f"\n==========================================")
    print(f" BEST MODEL: {best_model_name} (F1: {results_df.iloc[0]['test_f1']:.4f})")
    print(f"==========================================")

    # Save best model pipeline
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved best model pipeline to {best_model_path}")

    # Best model predictions on test set
    best_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    final_test_metrics = evaluate_predictions(y_test, best_test_proba, best_threshold)

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_names": ENGINEERED_FEATURE_NAMES,
        "optimal_threshold": best_threshold,
        "metrics_test": final_test_metrics,
        "status_thresholds": {
            "normal_max": round(best_threshold * 0.6, 4),
            "warning_max": round(best_threshold, 4),
            "critical_min": round(best_threshold, 4),
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_model_name, "metrics": final_test_metrics, "comparison": results}, f, indent=2)

    if not HAS_PLOTTING:
        print("Matplotlib/Seaborn not installed yet; skipping PNG plots.")
        return metadata

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if hasattr(best_pipeline, "named_steps") and hasattr(best_pipeline.named_steps.get("model"), "feature_importances_"):
        importances = best_pipeline.named_steps["model"].feature_importances_
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(ENGINEERED_FEATURE_NAMES)[sorted_idx], importances[sorted_idx], color="#3498db")
        ax.set_title(f"Feature Importances ({best_model_name})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Gini / Gain Importance")
    else:
        # Fallback to Permutation Importance
        perm_imp = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = np.argsort(perm_imp.importances_mean)
        ax.barh(np.array(ENGINEERED_FEATURE_NAMES)[sorted_idx], perm_imp.importances_mean[sorted_idx], color="#3498db")
        ax.set_title(f"Permutation Feature Importance ({best_model_name})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Accuracy/F1 Decrease")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()

    # ROC and PR Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, best_test_proba)
    ax1.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC = {final_test_metrics['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    # PR Curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, best_test_proba)
    ax2.plot(rec_arr, prec_arr, color="#e74c3c", lw=2, label=f"PR (AUC = {final_test_metrics['pr_auc']:.3f})")
    ax2.axvline(final_test_metrics["recall"], color="black", linestyle=":", label=f"Optimal Rec = {final_test_metrics['recall']:.2f}")
    ax2.set_title("Precision-Recall Curve (PR)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_pr_curves.png"), dpi=300)
    plt.close()

    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, (best_test_proba >= best_threshold).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Predicted Normal", "Predicted Failure"],
                yticklabels=["Actual Normal", "Actual Failure"])
    ax.set_title(f"Test Confusion Matrix ({best_model_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    print("Successfully completed model training, evaluation, and plot generation.")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
