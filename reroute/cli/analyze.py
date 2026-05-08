"""Analysis figures from simulation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from reroute.core.logging import get_logger
from reroute.model.risk import (
    features_for_scenario,
    synthesize_misconnect_labels,
    train_from_scenarios,
)
from reroute.sim.generator import generate_dataset

logger = get_logger(__name__)

PLOT_STYLE = {
    "figure.figsize": (8, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlepad": 12,
    "font.family": "sans-serif",
}

ACCENT_TEAL = "#1D9E75"
ACCENT_RED = "#E24B4A"
ACCENT_BLUE = "#1A5276"
ACCENT_GRAY = "#5F5E5A"


def fig_improvement_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots()
    bins = np.arange(-5, 50, 2.5)
    ax.hist(df["delta_pct"], bins=bins, color=ACCENT_TEAL, alpha=0.85, edgecolor="white")
    ax.axvline(df["delta_pct"].median(), color=ACCENT_BLUE, linestyle="--",
               linewidth=2, label=f"Median: {df['delta_pct'].median():.1f}%")
    ax.set_xlabel("Cost reduction vs manual triage (%)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"Reroute LP improvement across {len(df)} scenarios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "improvement_distribution.png", dpi=160)
    plt.close()


def fig_scarcity_vs_savings(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots()
    bins = pd.cut(
        df["seat_demand_ratio"], bins=[0, 0.7, 0.9, 1.1, 1.3, 1.5],
        labels=["very tight\n(<0.70)", "tight\n(0.70-0.90)",
                "balanced\n(0.90-1.10)", "loose\n(1.10-1.30)",
                "ample\n(1.30+)"]
    )
    grouped = df.groupby(bins, observed=True)["delta_pct"].agg(["mean", "count"])
    ax.bar(range(len(grouped)), grouped["mean"], color=ACCENT_TEAL,
           alpha=0.85, edgecolor="white")
    for i, (mean, count) in enumerate(zip(grouped["mean"], grouped["count"])):
        ax.text(i, mean + 0.5, f"n={count}", ha="center", fontsize=9, color=ACCENT_GRAY)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index, fontsize=9)
    ax.set_ylabel("Mean cost reduction (%)")
    ax.set_xlabel("Seat supply / demand ratio")
    ax.set_title("Cost reduction by supply/demand regime")
    plt.tight_layout()
    plt.savefig(out_dir / "scarcity_vs_savings.png", dpi=160)
    plt.close()


def fig_feature_importance(model, out_dir: Path) -> None:
    fig, ax = plt.subplots()
    fi = {k: v for k, v in model.train_results.feature_importance.items() if v > 0.001}
    items = sorted(fi.items(), key=lambda kv: kv[1])
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    ax.barh(range(len(items)), vals, color=ACCENT_BLUE, alpha=0.85)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Permutation importance (normalized)")
    ax.set_title("Risk model feature importance")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=160)
    plt.close()


def fig_calibration_curve(model, out_dir: Path) -> None:
    test_scns = generate_dataset(n_scenarios=100, seed=999)
    feature_dfs = [features_for_scenario(s) for s in test_scns]
    df = pd.concat(feature_dfs, ignore_index=True)
    rng = np.random.default_rng(123)
    labels = synthesize_misconnect_labels(df, rng)
    probs = model.predict_proba(df)
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=ACCENT_GRAY, linestyle="--",
            label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color=ACCENT_TEAL,
            linewidth=2, markersize=7, label="Trained model (Platt-calibrated)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of misconnect")
    ax.set_title("Risk model calibration on held-out data")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=160)
    plt.close()


def fig_solve_time_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots()
    ax.hist(df["lp_solve_ms"], bins=20, color=ACCENT_BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(df["lp_solve_ms"].median(), color=ACCENT_RED, linestyle="--",
               linewidth=2, label=f"Median: {df['lp_solve_ms'].median():.1f} ms")
    ax.axvline(df["lp_solve_ms"].quantile(0.95), color=ACCENT_GRAY, linestyle=":",
               linewidth=2, label=f"P95: {df['lp_solve_ms'].quantile(0.95):.1f} ms")
    ax.set_xlabel("Solve time (ms)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("LP solve time distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "solve_time_distribution.png", dpi=160)
    plt.close()


def run_analysis(input_dir: Path) -> None:
    """Generate all analysis figures."""
    plt.rcParams.update(PLOT_STYLE)

    csv_path = input_dir / "comparison.csv"
    if not csv_path.exists():
        logger.error(f"No simulation results at {csv_path}. Run `reroute simulate` first.")
        return

    df = pd.read_csv(csv_path)
    fig_dir = input_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    logger.info("Retraining model for figures...")
    train_scns = generate_dataset(n_scenarios=200, seed=42)
    model, _, _ = train_from_scenarios(train_scns)

    logger.info("Generating figures...")
    fig_improvement_distribution(df, fig_dir)
    logger.info("  ✓ improvement_distribution.png")
    fig_scarcity_vs_savings(df, fig_dir)
    logger.info("  ✓ scarcity_vs_savings.png")
    fig_feature_importance(model, fig_dir)
    logger.info("  ✓ feature_importance.png")
    fig_calibration_curve(model, fig_dir)
    logger.info("  ✓ calibration_curve.png")
    fig_solve_time_distribution(df, fig_dir)
    logger.info("  ✓ solve_time_distribution.png")
    logger.info(f"All figures saved to {fig_dir}")
