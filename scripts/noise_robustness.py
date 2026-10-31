"""
Stress test: add Gaussian noise to held-out test windows and measure 4-class accuracy per class.
Writes reports/noise_robustness.csv. Shows where the model breaks (ball faults are the weak class).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

from src.data import CLASS_NAMES, assign_splits, build_window_table, get_window, load_all_recordings
from src.features import TARGET_FS, extract_features, features_to_frame

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(per_file: int = 40, seed: int = 0):
    recs = load_all_recordings()
    pipe = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))["pipeline"]
    test = assign_splits(build_window_table(recs))
    test = test[test["split"] == "test"]
    rng = np.random.default_rng(seed)
    rows = []
    for snr in [None, 30, 20, 10, 5, 0]:
        for fid, g in test.groupby("source_file"):
            sample = g.sample(min(per_file, len(g)), random_state=seed)
            hits = []
            for _, r in sample.iterrows():
                w = get_window(recs[fid], int(r["win_idx"])).copy()
                if snr is not None:
                    w += rng.normal(0, np.sqrt(np.mean((w - w.mean()) ** 2)) / 10 ** (snr / 20), w.size)
                p = pipe.predict_proba(features_to_frame(extract_features(w, TARGET_FS, recs[fid].rpm)))[0]
                hits.append(int(p.argmax()) == recs[fid].class_id)
            rows.append({"snr_db": "clean" if snr is None else snr, "class": recs[fid].class_name,
                         "file": fid, "accuracy": float(np.mean(hits)), "windows": len(hits)})
    df = pd.DataFrame(rows)
    out = df.groupby(["snr_db", "class"], sort=False)["accuracy"].mean().unstack().round(3)[CLASS_NAMES]
    print(out)
    df.to_csv(os.path.join(ROOT, "reports", "noise_robustness.csv"), index=False)


if __name__ == "__main__":
    main()
