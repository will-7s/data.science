import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.store import dataset, schema, reset, col_meta, all_cols, target_candidates, group_candidates, id_candidates, datetime_cols, binary_cols, num_cols, cat_cols
from src.data_loader import load_data, resolve_column_names, prepare_data, get_group_stats
from src.data_quality import report_data_quality
from src.descriptive_stats import compute_descriptive_stats
from src.hypothesis_testing import run_all_tests
from src.effect_size import compute_all_effect_sizes
from src.logistic_regression import (
    simple_logistic_regression,
    enriched_logistic_regression,
    likelihood_ratio_test,
)
from src.bayesian_analysis import beta_binomial_analysis
from src.segmentation import run_segmentation_analysis
from src.temporal_analysis import run_temporal_analysis
from src.robustness import run_robustness_checks
from src.report_generator import generate_text_report, save_text_report
from src.visualizations import (
    plot_conversion_rates,
    plot_confidence_intervals,
    plot_bayesian_posteriors,
    plot_segmentation,
    plot_power_analysis,
    plot_daily_trends,
)
from config import OUTPUT_DIR, REPORT_DIR
from statsmodels.stats.power import NormalIndPower


def run_analysis(
    data_path: str = None,
    output_dir: str = None,
    generate_plots: bool = True,
    target_col: str = "",
    group_col: str = "",
    covariate_cols: list[str] = None,
    time_col: str = "",
    id_col: str = "",
):
    print("=" * 72)
    print("  FLEXIBLE A/B TESTING ANALYSIS PIPELINE")
    print("=" * 72)
    print()

    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading data...")
    df = load_data(Path(data_path or "ab_data.csv"))
    print(f"    Loaded {len(df):,} rows x {len(df.columns)} columns")

    print("[2] Initializing store & inferring column types...")
    reset(df)
    print(f"    {len(all_cols)} columns detected")
    print(f"    Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")
    print(f"    Target candidates (binary): {target_candidates}")
    print(f"    Group candidates: {group_candidates}")

    print("[3] Resolving column mapping...")
    resolve_column_names(
        target_col=target_col,
        group_col=group_col,
        covariate_cols=covariate_cols or [],
        time_col=time_col,
        id_col=id_col,
    )
    print(f"    Schema: {schema.description()}")

    print("[4] Preparing data...")
    try:
        prepared = prepare_data()
    except ValueError as e:
        print(f"    ERROR: {e}")
        return
    gs = get_group_stats(prepared)
    print(f"    Groups: {gs['n_control']:,} vs {gs['n_treatment']:,}")
    print(f"    Target mean: control={gs['rate_control']*100:.2f}%, treatment={gs['rate_treatment']*100:.2f}%")

    print("[5] Running hypothesis tests...")
    tests = run_all_tests(prepared)
    print(f"    Z-test p-value: {tests['z_test']['p_value_two_sided']:.4f}")

    print("[6] Computing effect sizes...")
    effect_sizes = compute_all_effect_sizes(prepared)
    print(f"    Cohen's h: {effect_sizes['cohens_h']:.4f}")

    print("[7] Power analysis...")
    power_analysis = NormalIndPower()
    es = abs(effect_sizes["cohens_h"])
    if es < 1e-9:
        power = {
            "power_observed": 0.0, "n_needed_80pct": float("inf"),
            "mde_cohens_h": float("nan"), "mde_pp": float("nan"),
        }
        print("    Effect size ~0, power analysis skipped")
    else:
        try:
            power_obs = power_analysis.solve_power(
                effect_size=es, nobs1=gs["n_control"],
                ratio=gs["n_treatment"] / max(gs["n_control"], 1),
                alpha=0.05, alternative="two-sided",
            )
            n_needed = power_analysis.solve_power(
                effect_size=es, power=0.80, alpha=0.05, ratio=1.0,
                alternative="two-sided",
            )
            mde_h = power_analysis.solve_power(
                nobs1=gs["n_control"], ratio=gs["n_treatment"] / max(gs["n_control"], 1),
                power=0.80, alpha=0.05, alternative="two-sided",
            )
            mde_pp = 2 * np.sin(mde_h / 2) * np.sqrt(
                gs["rate_control"] * (1 - gs["rate_control"])
            )
            power = {
                "power_observed": float(power_obs),
                "n_needed_80pct": float(n_needed),
                "mde_cohens_h": float(mde_h),
                "mde_pp": float(mde_pp),
            }
            print(f"    Power: {power_obs*100:.1f}%")
        except Exception:
            power = {
                "power_observed": 0.0, "n_needed_80pct": float("inf"),
                "mde_cohens_h": float("nan"), "mde_pp": float("nan"),
            }
            print("    Power analysis skipped (error)")

    print("[8] Logistic regression...")
    log_reg_simple = simple_logistic_regression(prepared)
    log_reg_enriched = enriched_logistic_regression(prepared)
    lr_test = likelihood_ratio_test(log_reg_simple["model"], log_reg_enriched["model"])
    print(f"    Simple OR: {log_reg_simple['odds_ratio']:.4f} (p={log_reg_simple['p_value']:.4f})")

    print("[9] Bayesian analysis...")
    bayesian = beta_binomial_analysis(prepared)
    print(f"    P(treatment > control): {bayesian['p_treatment_better_pct']:.1f}%")

    print("[10] Segmentation, temporal, robustness...")
    segmentation = run_segmentation_analysis(prepared)
    temporal = run_temporal_analysis(prepared)
    robustness = run_robustness_checks(prepared)

    quality = report_data_quality()
    stats = compute_descriptive_stats(prepared)

    print("\nGenerating report...")
    report = generate_text_report(quality, stats, tests, effect_sizes, power,
                                   bayesian, segmentation, temporal, robustness,
                                   log_reg_simple, log_reg_enriched, lr_test)
    save_text_report(report)
    print(f"  Report saved to {REPORT_DIR / 'ab_testing_report.txt'}")

    if generate_plots:
        print("Generating plots...")
        plot_conversion_rates(stats, plot_dir / "conversion_rates.png")
        plot_confidence_intervals(stats, plot_dir / "confidence_intervals.png")
        plot_bayesian_posteriors(bayesian,
                                 (stats["control"]["label"], stats["treatment"]["label"]),
                                 plot_dir / "bayesian_posteriors.png")
        plot_segmentation(segmentation, plot_dir / "segmentation.png")
        plot_power_analysis(power["mde_cohens_h"], power["power_observed"],
                           plot_dir / "power_analysis.png")
        plot_daily_trends(temporal["daily_data"], plot_dir / "daily_trends.png")
        print(f"  Plots saved to {plot_dir}")

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)
    print()
    print(report)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible A/B Testing Analysis")
    parser.add_argument("--data", type=str, default="ab_data.csv", help="Path to data file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--target", type=str, default="", help="Target/conversion column name")
    parser.add_argument("--group", type=str, default="", help="Group/treatment column name")
    parser.add_argument("--covariates", type=str, nargs="*", default=[], help="Covariate column names")
    parser.add_argument("--time", type=str, default="", help="Time column name")
    parser.add_argument("--id", type=str, default="", help="ID column name")
    args = parser.parse_args()

    run_analysis(
        data_path=args.data,
        output_dir=args.output,
        generate_plots=not args.no_plots,
        target_col=args.target,
        group_col=args.group,
        covariate_cols=args.covariates,
        time_col=args.time,
        id_col=args.id,
    )
