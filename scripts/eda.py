"""
Exploratory Data Analysis (EDA) script for AI4I 2020 Predictive Maintenance Dataset.

Generates data summary statistics, class imbalance metrics, failure-mode breakdown,
and saves visualization plots to the reports/ directory.
"""

import json
import os
import sys

# Add project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/script execution
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from src.data import load_dataset, get_standardized_dataframe
from src.features import create_features_df

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))


def run_eda(data_path: str = "data/ai4i2020.csv") -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_raw = load_dataset(data_path)
    df_std = get_standardized_dataframe(df_raw)

    total_count = len(df_std)
    failure_count = int(df_std["machine_failure"].sum())
    normal_count = total_count - failure_count
    failure_rate_pct = (failure_count / total_count) * 100.0

    # Breakdown by machine type
    type_counts = df_std["type"].value_counts().to_dict()
    type_failures = (
        df_std.groupby("type")["machine_failure"]
        .agg(total="count", failures="sum")
        .reset_index()
    )
    type_failures["failure_rate_pct"] = (
        type_failures["failures"] / type_failures["total"]
    ) * 100.0
    type_summary = type_failures.to_dict(orient="records")

    # Failure modes breakdown
    failure_modes = ["twf", "hdf", "pwf", "osf", "rnf"]
    mode_counts = {mode.upper(): int(df_std[mode].sum()) for mode in failure_modes}

    # Summary statistics for numeric features
    numeric_cols = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    stats_overall = df_std[numeric_cols].describe().to_dict()
    stats_failure = df_std[df_std["machine_failure"] == 1][numeric_cols].describe().to_dict()
    stats_normal = df_std[df_std["machine_failure"] == 0][numeric_cols].describe().to_dict()

    summary_dict = {
        "total_records": total_count,
        "normal_records": normal_count,
        "failure_records": failure_count,
        "failure_rate_pct": round(failure_rate_pct, 4),
        "type_summary": type_summary,
        "failure_mode_counts": mode_counts,
        "overall_stats": stats_overall,
        "normal_stats": stats_normal,
        "failure_stats": stats_failure,
    }

    # Write summary JSON
    json_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved EDA summary JSON to {json_path}")

    if not HAS_PLOTTING:
        print("Matplotlib/Seaborn not installed yet; skipping PNG plots.")
        return summary_dict

    # Set aesthetic plotting style
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Plot 1: Class Imbalance
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Normal (0)", "Failure (1)"],
        [normal_count, failure_count],
        color=["#2ecc71", "#e74c3c"],
        width=0.5,
    )
    ax.set_title("Machine Failure Class Distribution (Target Imbalance)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Record Count", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total_count) * 100.0
        ax.annotate(
            f"{height:,}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "class_imbalance.png"), dpi=300)
    plt.close()

    # Plot 2: Failure Modes Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = list(mode_counts.keys())
    counts = list(mode_counts.values())
    bars = ax.bar(modes, counts, color="#e67e22", width=0.5)
    ax.set_title("Specific Failure Mode Occurrences", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("Failure Mode (TWF: Tool Wear, HDF: Heat Dissipation, PWF: Power, OSF: Overstrain, RNF: Random)", fontsize=9)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "failure_modes.png"), dpi=300)
    plt.close()

    # Plot 3: Feature Distributions by Class
    features_df = create_features_df(df_raw)
    features_df["machine_failure"] = df_std["machine_failure"]

    plot_cols = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "temp_diff", "power_w", "wear_torque"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, col in enumerate(plot_cols):
        sns.boxplot(
            data=features_df,
            x="machine_failure",
            y=col,
            hue="machine_failure",
            ax=axes[idx],
            palette={0: "#2ecc71", 1: "#e74c3c"},
            legend=False,
        )
        axes[idx].set_title(f"{col} by Class", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Machine Failure (0 vs 1)", fontsize=9)

    plt.suptitle("Feature Distributions Split by Machine Failure Status", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_distributions.png"), dpi=300)
    plt.close()

    # Plot 4: Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = features_df.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Feature & Target Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "correlation_matrix.png"), dpi=300)
    plt.close()

    print("EDA plots successfully generated and saved to reports/")
    return summary_dict


if __name__ == "__main__":
    run_eda()
