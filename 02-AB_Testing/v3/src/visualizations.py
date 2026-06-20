from __future__ import annotations
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_theme(style="whitegrid")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_conversion_rates(stats: dict, save_path: Path | None = None) -> plt.Figure:
    """Plot conversion rates as a grouped bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [stats["control"]["label"].capitalize(), stats["treatment"]["label"].capitalize()]
    rates = [stats["control"]["rate_pct"], stats["treatment"]["rate_pct"]]
    bars = ax.bar(groups, rates, color=COLORS[:2], alpha=0.85, edgecolor="black", linewidth=1.2)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rate:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Conversion Rate (%)", fontsize=12)
    ax.set_title("A/B Test: Conversion Rates by Group", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.25)

    diff = stats["difference"]["absolute_pp"]
    ax.annotate(f"Difference: {diff:+.2f} pp", xy=(0.5, 0.92),
                xycoords="axes fraction", ha="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confidence_intervals(stats: dict, save_path: Path | None = None) -> plt.Figure:
    """Plot conversion rates with 95% Wilson confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [stats["control"]["label"].capitalize(), stats["treatment"]["label"].capitalize()]
    rates = [stats["control"]["rate"], stats["treatment"]["rate"]]
    cis = [stats["control"]["ci_95"], stats["treatment"]["ci_95"]]
    y_pos = [1, 0]

    for i, (group, rate, ci) in enumerate(zip(groups, rates, cis)):
        ax.errorbar(rate, y_pos[i], xerr=[[rate - ci[0]], [ci[1] - rate]],
                    fmt="o", color=COLORS[i], capsize=6, capthick=2, markersize=10, label=group)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Conversion Rate", fontsize=12)
    ax.set_title("95% Confidence Intervals for Conversion Rates", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_bayesian_posteriors(bayes_results: dict, labels: tuple = None, save_path: Path = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    p_better = bayes_results["p_treatment_better"]
    control_label = (labels[0] if labels else "Control").capitalize()
    treatment_label = (labels[1] if labels else "Treatment").capitalize()

    axes[0].bar([control_label, treatment_label],
                [1 - p_better, p_better],
                color=COLORS[:2], alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Posterior Probability", fontsize=12)
    axes[0].set_title("Posterior Means", fontsize=14, fontweight="bold")

    axes[1].pie([p_better, 1 - p_better],
                labels=[f"{treatment_label} Better\n{p_better:.1%}",
                        f"{control_label} Better\n{1-p_better:.1%}"],
                colors=COLORS[:2], autopct="", startangle=90, explode=(0.05, 0))
    axes[1].set_title("Probability Treatment is Better", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_segmentation(seg_results: dict, save_path: Path | None = None) -> plt.Figure:
    """Plot segment-level conversion rates with difference annotations."""
    rows = []
    for col_name, segments in seg_results.items():
        for seg_name, data in segments.items():
            if isinstance(data, dict) and "control_rate" in data:
                rows.append({
                    "Segment": f"{col_name}: {seg_name}"[:20],
                    "Control": data["control_rate"] * 100,
                    "Treatment": data["treatment_rate"] * 100,
                    "Diff": (data["treatment_rate"] - data["control_rate"]) * 100,
                })

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No categorical segments found", ha="center", va="center", fontsize=12)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    df_seg = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(8, len(df_seg) * 1.5), 5))
    x = np.arange(len(df_seg))
    w = 0.35
    ax.bar(x - w / 2, df_seg["Control"], w, label="Control", color=COLORS[0], alpha=0.85)
    ax.bar(x + w / 2, df_seg["Treatment"], w, label="Treatment", color=COLORS[1], alpha=0.85)

    for i, row in df_seg.iterrows():
        color = "green" if row["Diff"] > 0 else "red"
        ax.annotate(f"{row['Diff']:+.2f}pp", (i, max(row["Control"], row["Treatment"]) + 0.2),
                    ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df_seg["Segment"], rotation=30, ha="right")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Segmentation Analysis")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_power_analysis(mde: float, power_obs: float, save_path: Path | None = None) -> plt.Figure:
    """Plot observed power vs 80% target as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Observed Power", "80% Target"]
    values = [power_obs * 100, 80]
    colors = [COLORS[1], COLORS[2]]
    bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold", fontsize=11)
    ax.set_xlabel("Statistical Power (%)")
    ax.set_title(f"Post-hoc Power Analysis\nMDE = {mde:.4f} (Cohen's h)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_daily_trends(daily: pd.DataFrame | None, save_path: Path | None = None) -> plt.Figure:
    """Plot daily conversion rate trends per group."""
    if daily is None or len(daily) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No temporal data available", ha="center", va="center", fontsize=12)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, grp in enumerate(daily["group"].unique()):
        data = daily[daily["group"] == grp]
        ax.plot(data["date"].astype(str), data["rate"] * 100, marker=".", linewidth=1.5,
                label=str(grp).capitalize(), color=COLORS[i], alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Daily Conversion Rates Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
