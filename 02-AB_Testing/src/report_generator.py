from pathlib import Path
from config import REPORT_DIR


def generate_text_report(
    quality: dict,
    stats: dict,
    tests: dict,
    effect_sizes: dict,
    power: dict,
    bayesian: dict,
    segmentation: dict,
    temporal: dict,
    robustness: dict,
    log_reg_simple: dict,
    log_reg_enriched: dict,
    lr_test: dict,
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  A/B TESTING ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append("")

    lines.append("-" * 72)
    lines.append("1. DATA QUALITY")
    lines.append("-" * 72)
    lines.append(f"  Total rows:           {quality['total_rows']:>10,}")
    lines.append(f"  Total columns:        {quality['total_columns']:>10}")
    if quality.get("schema"):
        lines.append(f"  Schema:               {quality['schema']}")
    if quality.get("group_distribution"):
        for k, v in quality["group_distribution"].items():
            lines.append(f"  Group '{k}':          {v:>10,}")
    if quality.get("missing_values"):
        lines.append(f"  Missing values:       {quality['missing_values']}")
    else:
        lines.append("  Missing values:       None")
    lines.append("")

    lines.append("-" * 72)
    lines.append("2. DESCRIPTIVE STATISTICS")
    lines.append("-" * 72)
    for grp_key in ["control", "treatment"]:
        s = stats[grp_key]
        lines.append(f"  {s['label'].capitalize():15s}: n={s['n']:>10,}  "
                     f"conversions={s['conversions']:>6,}  "
                     f"rate={s['rate_pct']:.2f}%  "
                     f"CI95=[{s['ci_95'][0]*100:.2f}%, {s['ci_95'][1]*100:.2f}%]")
    lines.append(f"  {'Difference':15s}: {stats['difference']['absolute_pp']:+.3f} pp  "
                 f"({stats['difference']['relative_pct']:+.2f}% relative)")
    lines.append("")

    lines.append("-" * 72)
    lines.append("3. FREQUENTIST HYPOTHESIS TESTS")
    lines.append("-" * 72)
    for name, res in [
        ("Chi-squared", tests["chi_squared"]),
        ("Z-test (2-sided)", tests["z_test"]),
        ("T-test (Welch)", tests["t_test"]),
        ("Mann-Whitney U", tests["mann_whitney"]),
    ]:
        p = res.get("p_value_two_sided", res.get("p_value", 0))
        stat = res.get("z_statistic_two_sided", res.get("statistic", res.get("t_statistic", res.get("u_statistic", 0))))
        sig = "SIG" if p < 0.05 else "NS"
        lines.append(f"  {name:20s}: stat={stat:.2f},  p={p:.6f}  {sig}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("4. EFFECT SIZES")
    lines.append("-" * 72)
    lines.append(f"  Cohen's h:         {effect_sizes['cohens_h']:.4f} ({effect_sizes['cohens_h_interpretation']})")
    lines.append(f"  Odds Ratio:        {effect_sizes['odds_ratio']:.4f}  "
                 f"CI95=[{effect_sizes['odds_ratio_ci_95'][0]:.4f}, {effect_sizes['odds_ratio_ci_95'][1]:.4f}]")
    lines.append(f"  Risk Ratio (RR):   {effect_sizes['risk_ratio']:.4f}")
    lines.append(f"  NNT:               {effect_sizes['nnt']:.1f}")
    lines.append(f"  Phi coefficient:   {effect_sizes['phi_coefficient']:.4f}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("5. STATISTICAL POWER")
    lines.append("-" * 72)
    lines.append(f"  Observed power:        {power['power_observed']*100:.2f}%")
    lines.append(f"  Needed per group (80%): {power['n_needed_80pct']:>8,.0f}")
    lines.append(f"  MDE (Cohen's h):       {power['mde_cohens_h']:.4f}")
    lines.append(f"  MDE (approx pp):       +/-{power['mde_pp']*100:.2f} pp")
    lines.append("")

    lines.append("-" * 72)
    lines.append("6. LOGISTIC REGRESSION")
    lines.append("-" * 72)
    lines.append(f"  Simple model:  OR={log_reg_simple['odds_ratio']:.4f}  "
                 f"CI95=[{log_reg_simple['or_ci_95'][0]:.4f}, {log_reg_simple['or_ci_95'][1]:.4f}]  "
                 f"p={log_reg_simple['p_value']:.6f}")
    lines.append(f"  Enriched model: OR={log_reg_enriched['odds_ratio']:.4f}  "
                 f"CI95=[{log_reg_enriched['or_ci_95'][0]:.4f}, {log_reg_enriched['or_ci_95'][1]:.4f}]  "
                 f"p={log_reg_enriched['p_value']:.6f}")
    lines.append(f"  LR test (simple vs enriched): LR={lr_test['lr_statistic']:.2f}, "
                 f"df={lr_test['df_difference']}, p={lr_test['p_value']:.6f}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("7. BAYESIAN ANALYSIS (Beta-Binomial)")
    lines.append("-" * 72)
    lines.append(f"  P(treatment > control): {bayesian['p_treatment_better_pct']:.2f}%")
    lines.append(f"  Expected loss (choose T): {bayesian['expected_loss']:.6f}")
    lines.append(f"  Median difference: {bayesian['median_difference']:.4f}")
    lines.append(f"  CI 95% difference: [{bayesian['ci_95'][0]:.4f}, {bayesian['ci_95'][1]:.4f}]")
    lines.append(f"  Decision (ROPE):   {bayesian['decision']}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("8. SEGMENTATION ANALYSIS")
    lines.append("-" * 72)
    if segmentation:
        for col_name, segments in segmentation.items():
            for seg_name, data in segments.items():
                if isinstance(data, dict) and "control_rate" in data:
                    lines.append(f"  {col_name}:{seg_name:15s}: Control={data['control_rate']*100:.2f}% (n={data['control_n']:>6,}), "
                                 f"Treatment={data['treatment_rate']*100:.2f}% (n={data['treatment_n']:>6,}), "
                                 f"p={data['p_value']:.4f}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("9. ROBUSTNESS CHECKS")
    lines.append("-" * 72)
    lines.append(f"  Bootstrap CI 95%:  [{robustness['bootstrap']['ci_95'][0]:.4f}, {robustness['bootstrap']['ci_95'][1]:.4f}]")
    lines.append(f"  Bootstrap pct > 0: {robustness['bootstrap']['pct_positive']:.2f}%")
    lines.append(f"  Permutation p (2-sided): {robustness['permutation']['p_value_two_sided']:.4f}")
    lines.append(f"  Permutation p (1-sided): {robustness['permutation']['p_value_one_sided']:.4f}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("10. TEMPORAL ANALYSIS")
    lines.append("-" * 72)
    if temporal.get("trends"):
        for grp, trend in temporal["trends"].items():
            lines.append(f"  {grp:15s}: r={trend['pearson_r']:.4f}, "
                         f"p={trend['p_value']:.4f} {'(significant trend)' if trend['significant_trend'] else '(no trend)'}")
    else:
        lines.append("  No time column provided.")
    lines.append("")

    lines.append("=" * 72)
    lines.append("  CONCLUSION & RECOMMENDATION")
    lines.append("=" * 72)
    lines.append("")

    p_z = tests["z_test"]["p_value_two_sided"]
    bayes_pct = bayesian["p_treatment_better_pct"]

    if p_z < 0.01:
        evidence = "STRONG: Statistically very significant difference (p < 0.01)."
    elif p_z < 0.05:
        evidence = "MODERATE: Statistically significant difference (p < 0.05)."
    elif p_z < 0.10:
        evidence = "WEAK: Trend without reaching significance (p < 0.10)."
    else:
        evidence = "NONE: No statistically significant difference detected (p >= 0.05)."

    lines.append(f"  Evidence strength: {evidence}")
    lines.append("")
    lines.append("  Key findings:")
    lines.append(f"    - Control rate:  {stats['control']['rate_pct']:.2f}%")
    lines.append(f"    - Treatment rate: {stats['treatment']['rate_pct']:.2f}%")
    lines.append(f"    - Difference: {stats['difference']['absolute_pp']:+.3f} pp "
                 f"({stats['difference']['relative_pct']:+.2f}% relative)")
    lines.append(f"    - Bayesian P(T > C): {bayes_pct:.1f}%")
    lines.append(f"    - Effect size: {effect_sizes['cohens_h_interpretation']} (Cohen's h = {effect_sizes['cohens_h']:.4f})")
    lines.append("")

    if bayes_pct > 95 or p_z < 0.05:
        rec = "Deploy the treatment -- the improvement is statistically justified."
    elif bayes_pct > 80 or p_z < 0.10:
        rec = "Favorable trend but insufficient evidence. Consider extending the test."
    else:
        rec = "Do NOT deploy the treatment. Keep the current version (control)."

    lines.append(f"  Recommendation: {rec}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)

    return "\n".join(lines)


def save_text_report(report_text: str, path: Path = None):
    if path is None:
        path = REPORT_DIR / "ab_testing_report.txt"
    path.write_text(report_text, encoding="utf-8")
    return path
