"""EDA for the CWRU bearing recordings: summary table + waveform/spectrum and feature plots."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import (CLASS_LABELS, CLASS_NAMES, WINDOW_SIZE, assign_splits, build_window_table, get_window,
                      load_all_recordings, recording_summary)
from src.features import FEATURE_NAMES, TARGET_FS

REPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
COLORS = {"normal": "#10b981", "inner_race": "#ef4444", "ball": "#f59e0b", "outer_race": "#8b5cf6"}


def main():
    os.makedirs(REPORTS, exist_ok=True)
    recs = load_all_recordings()
    summary = recording_summary(recs)
    print(summary.to_string(index=False))

    # one representative recording per class: waveform + spectrum
    rep = {}
    for r in recs.values():
        rep.setdefault(r.class_name, r)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    for row, cname in enumerate(CLASS_NAMES):
        r = rep[cname]
        w = get_window(r, 100)
        w = w - w.mean()
        t = np.arange(w.size) / TARGET_FS * 1000
        axes[row, 0].plot(t, w, color=COLORS[cname], lw=0.8)
        axes[row, 0].set_title(f"{CLASS_LABELS[cname]} ({r.file_id}.mat) - waveform", fontsize=10)
        axes[row, 0].set_ylabel("g")
        f = np.fft.rfftfreq(w.size, 1 / TARGET_FS)
        s = np.abs(np.fft.rfft(w * np.hanning(w.size))) / w.size
        axes[row, 1].plot(f, s, color=COLORS[cname], lw=0.8)
        axes[row, 1].set_title("amplitude spectrum", fontsize=10)
        axes[row, 1].set_xlim(0, 6000)
    axes[-1, 0].set_xlabel("ms")
    axes[-1, 1].set_xlabel("Hz")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "signals_by_class.png"), dpi=130)
    plt.close()

    # feature distributions
    table = assign_splits(build_window_table(recs))
    show = ["rms", "kurtosis", "crest_factor", "env_bpfo", "env_bpfi", "env_bsf", "band_2000_3000", "spectral_centroid"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, feat in zip(axes.ravel(), show):
        data = [table.loc[table["class_id"] == i, feat] for i in range(4)]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=["Norm", "Inner", "Ball", "Outer"])
        for patch, c in zip(bp["boxes"], CLASS_NAMES):
            patch.set_facecolor(COLORS[c])
        ax.set_title(feat, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "feature_distributions.png"), dpi=130)
    plt.close()

    out = {
        "recordings": summary.to_dict(orient="records"),
        "windows_total": int(len(table)),
        "windows_per_class": {CLASS_NAMES[k]: int(v) for k, v in table["class_id"].value_counts().items()},
        "window_samples": WINDOW_SIZE,
        "n_features": len(FEATURE_NAMES),
        "feature_class_means": table.groupby("class_name")[FEATURE_NAMES].mean().round(4).to_dict(orient="index"),
    }
    with open(os.path.join(REPORTS, "eda_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("EDA written to reports/")


if __name__ == "__main__":
    main()
