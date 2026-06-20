import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

from src.store import schema, reset, all_cols, target_candidates, group_candidates, num_cols, cat_cols
from src.data_loader import load_data, resolve_column_names, prepare_data, get_group_stats
from src.data_quality import report_data_quality
from src.descriptive_stats import compute_descriptive_stats
from src.hypothesis_testing import run_all_tests
from src.effect_size import compute_all_effect_sizes
from src.power_analysis import compute_power          # FIX: from central module
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
    plot_conversion_rates, plot_confidence_intervals,
    plot_bayesian_posteriors, plot_segmentation,
    plot_power_analysis, plot_daily_trends,
)
from config import OUTPUT_DIR, REPORT_DIR, ALPHA


def run_analysis(
    data_path: str = None,
    output_dir: str = None,
    generate_plots: bool = True,
    target_col: str = "",
    group_col: str = "",
    covariate_cols: list[str] = None,
    time_col: str = "",
    id_col: str = "",
    control_value: str = "",
    page_col: str = "",        # for mismatch-cleaning (e.g. "landing_page")
):
    print("=" * 72)
    print("  FLEXIBLE A/B TESTING ANALYSIS PIPELINE")
    print("=" * 72)

    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading data...")
    df = load_data(Path(data_path or "ab_data.csv"))
    print(f"    Loaded {len(df):,} rows × {len(df.columns)} columns")

    print("[2] Initialising store & inferring column types...")
    reset(df)
    print(f"    {len(all_cols)} columns | numeric: {len(num_cols)} | categorical: {len(cat_cols)}")
    print(f"    Target candidates: {target_candidates}")
    print(f"    Group  candidates: {group_candidates}")

    print("[3] Resolving column mapping...")
    resolve_column_names(
        target_col=target_col,
        group_col=group_col,
        covariate_cols=covariate_cols or [],
        time_col=time_col,
        id_col=id_col,
        control_value=control_value,
    )
    print(f"    Schema: {schema.description()}")

    print("[4] Cleaning & preparing data...")
    try:
        prepared = prepare_data(page_col=page_col or None)
    except ValueError as e:
        print(f"    ERROR: {e}")
        return None

    cr = prepared.get("cleaning_report", {})
    if cr:
        print(f"    Mismatches removed: {cr['n_mismatch_removed']:,}")
        print(f"    Duplicates removed: {cr['n_dupes_removed']:,}")
        print(f"    Clean rows:         {cr['n_clean']:,}")

    gs = get_group_stats(prepared)
    print(f"    Control: {gs['n_control']:,} ({gs['rate_control']*100:.2f}%)  |  "
          f"Treatment: {gs['n_treatment']:,} ({gs['rate_treatment']*100:.2f}%)")

    print("[5] Hypothesis tests...")
    tests = run_all_tests(prepared)
    print(f"    Z-test (2-sided) p = {tests['z_test']['p_value_two_sided']:.4f}")

    print("[6] Effect sizes...")
    effect_sizes = compute_all_effect_sizes(prepared)
    print(f"    Cohen's h = {effect_sizes['cohens_h']:.4f} ({effect_sizes['cohens_h_interpretation']})")
    print(f"    OR = {effect_sizes['odds_ratio']:.4f}  |  {effect_sizes['nnt_label']} = {effect_sizes['nnt']:.0f}")

    print("[7] Power analysis...")
    power = compute_power(effect_sizes["cohens_h"], gs["n_control"], gs["n_treatment"], alpha=ALPHA)
    if power["skipped"]:
        print("    Skipped (effect size ~0)")
    else:
        print(f"    Observed power: {power['power_observed']*100:.1f}%  |  "
              f"N/group for 80%: {power['n_needed_80pct']:,.0f}")

    print("[8] Logistic regression...")
    log_reg_simple   = simple_logistic_regression(prepared)
    log_reg_enriched = enriched_logistic_regression(prepared)
    lr_test          = likelihood_ratio_test(log_reg_simple["model"], log_reg_enriched["model"])
    print(f"    Simple  OR = {log_reg_simple['odds_ratio']:.4f}  p = {log_reg_simple['p_value']:.4f}")
    print(f"    Enriched OR = {log_reg_enriched['odds_ratio']:.4f}  p = {log_reg_enriched['p_value']:.4f}")

    print("[9] Bayesian analysis...")
    bayesian = beta_binomial_analysis(prepared)
    print(f"    P(T > C) = {bayesian['p_treatment_better_pct']:.1f}%  |  decision: {bayesian['decision']}")

    print("[10] Segmentation, temporal, robustness...")
    segmentation = run_segmentation_analysis(prepared)
    temporal     = run_temporal_analysis(prepared)
    robustness   = run_robustness_checks(prepared)

    quality = report_data_quality(cleaning_report=cr)
    stats   = compute_descriptive_stats(prepared)

    print("\n[11] Generating report...")
    report = generate_text_report(
        quality, stats, tests, effect_sizes, power, bayesian,
        segmentation, temporal, robustness,
        log_reg_simple, log_reg_enriched, lr_test,
    )
    save_text_report(report)
    print(f"     Saved → {REPORT_DIR / 'ab_testing_report.txt'}")

    if generate_plots:
        print("[12] Generating plots...")
        plot_conversion_rates(stats, plot_dir / "conversion_rates.png")
        plot_confidence_intervals(stats, plot_dir / "confidence_intervals.png")
        plot_bayesian_posteriors(bayesian, (stats["control"]["label"], stats["treatment"]["label"]),
                                 plot_dir / "bayesian_posteriors.png")
        plot_segmentation(segmentation, plot_dir / "segmentation.png")
        plot_power_analysis(power["mde_cohens_h"], power["power_observed"],
                            plot_dir / "power_analysis.png")
        plot_daily_trends(temporal["daily_data"], plot_dir / "daily_trends.png")
        print(f"     Saved → {plot_dir}")

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)
    print()
    print(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible A/B Testing Analysis Pipeline")
    parser.add_argument("--data",       type=str, default="ab_data.csv")
    parser.add_argument("--output",     type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots",   action="store_true")
    parser.add_argument("--target",     type=str, default="")
    parser.add_argument("--group",      type=str, default="")
    parser.add_argument("--control",    type=str, default="",  help="Control group label")
    parser.add_argument("--page-col",   type=str, default="",  help="Landing-page column for mismatch cleaning")
    parser.add_argument("--covariates", type=str, nargs="*", default=[])
    parser.add_argument("--time",       type=str, default="")
    parser.add_argument("--id",         type=str, default="")
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
        control_value=args.control,
        page_col=args.page_col,
    )
