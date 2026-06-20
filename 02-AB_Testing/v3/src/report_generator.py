from __future__ import annotations
from pathlib import Path
from config import ALPHA, REPORT_DIR


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
    lines: list[str] = []
    lines += ["=" * 72, "  A/B TESTING ANALYSIS REPORT", "=" * 72, ""]

    # ── 1. DATA QUALITY ────────────────────────────────────────────────────
    lines += ["-" * 72, "1. DATA QUALITY", "-" * 72]
    lines.append(f"  Total rows (raw):     {quality['total_rows']:>10,}")
    lines.append(f"  Total columns:        {quality['total_columns']:>10}")
    if quality.get("schema"):
        lines.append(f"  Schema:               {quality['schema']}")
    if cl := quality.get("cleaning"):
        lines.append(f"  Mismatches removed:   {cl['n_mismatch_removed']:>10,}")
        lines.append(f"  Duplicates removed:   {cl['n_dupes_removed']:>10,}")
        lines.append(f"  Rows after cleaning:  {cl['n_clean']:>10,}  ({cl['pct_removed']:.2f}% removed)")
    if quality.get("group_distribution"):
        for k, v in quality["group_distribution"].items():
            lines.append(f"  Group '{k}':          {v:>10,}")
    lines.append(f"  Missing values:       {quality['missing_values'] or 'None'}")
    lines.append("")

    # ── 2. DESCRIPTIVE STATISTICS ──────────────────────────────────────────
    lines += ["-" * 72, "2. DESCRIPTIVE STATISTICS", "-" * 72]
    for key in ("control", "treatment"):
        s = stats[key]
        lines.append(
            f"  {s['label'].capitalize():15s}: n={s['n']:>10,}  "
            f"conversions={s['conversions']:>6,}  "
            f"rate={s['rate_pct']:.4f}%  "
            f"CI95=[{s['ci_95'][0]*100:.4f}%, {s['ci_95'][1]*100:.4f}%]"
        )
    d = stats["difference"]
    lines.append(
        f"  {'Difference':15s}: {d['absolute_pp']:+.4f} pp  "
        f"({d['relative_pct']:+.2f}% relative)  "
        f"CI95=[{d['ci_95'][0]*100:.4f}%, {d['ci_95'][1]*100:.4f}%]"
    )
    lines.append("")

    # ── 3. FREQUENTIST HYPOTHESIS TESTS ───────────────────────────────────
    lines += ["-" * 72, "3. FREQUENTIST HYPOTHESIS TESTS", "-" * 72]
    for name, key, stat_key, p_key in [
        ("Chi-squared",      "chi_squared",  "statistic",           "p_value"),
        ("Z-test (2-sided)", "z_test",       "z_statistic_two_sided","p_value_two_sided"),
        ("Z-test (T>C)",     "z_test",       "z_statistic_one_sided","p_value_one_sided"),
        ("T-test (Welch)",   "t_test",       "t_statistic",         "p_value"),
        ("Mann-Whitney U",   "mann_whitney", "u_statistic",         "p_value"),
    ]:
        res = tests[key]
        stat = res.get(stat_key, 0)
        p    = res.get(p_key, 1)
        sig  = "SIG" if p < ALPHA else "NS"
        lines.append(f"  {name:22s}: stat={stat:>10.2f},  p={p:.6f}  {sig}")
    lines.append("")

    # ── 4. EFFECT SIZES ────────────────────────────────────────────────────
    lines += ["-" * 72, "4. EFFECT SIZES", "-" * 72]
    lines.append(f"  Cohen's h:         {effect_sizes['cohens_h']:.4f} ({effect_sizes['cohens_h_interpretation']})")
    or_lo, or_hi = effect_sizes['odds_ratio_ci_95']
    lines.append(f"  Odds Ratio:        {effect_sizes['odds_ratio']:.4f}  CI95=[{or_lo:.4f}, {or_hi:.4f}]")
    lines.append(f"  Risk Ratio (RR):   {effect_sizes['risk_ratio']:.4f}")
    # FIX: display NNT or NNH correctly
    nnt_label = effect_sizes.get("nnt_label", "NNT")
    lines.append(f"  {nnt_label}:               {effect_sizes['nnt']:.1f}  ({effect_sizes.get('nnt_direction','')})")
    lines.append(f"  Phi coefficient:   {effect_sizes['phi_coefficient']:.4f}")
    lines.append("")

    # ── 5. STATISTICAL POWER ───────────────────────────────────────────────
    lines += ["-" * 72, "5. STATISTICAL POWER", "-" * 72]
    if power.get("skipped"):
        lines.append("  Power analysis skipped (effect size ~0)")
    else:
        lines.append(f"  Observed power:        {power['power_observed']*100:.2f}%")
        lines.append(f"  Needed per group (80%): {power['n_needed_80pct']:>8,.0f}")
        lines.append(f"  MDE (Cohen's h):       {power['mde_cohens_h']:.4f}")
        lines.append(f"  MDE (approx pp):       ±{power['mde_pp']*100:.2f} pp")
    lines.append("")

    # ── 6. LOGISTIC REGRESSION ────────────────────────────────────────────
    lines += ["-" * 72, "6. LOGISTIC REGRESSION", "-" * 72]
    for label, res in [("Simple", log_reg_simple), ("Enriched", log_reg_enriched)]:
        lo, hi = res['or_ci_95']
        lines.append(
            f"  {label:8s} model:  OR={res['odds_ratio']:.4f}  "
            f"CI95=[{lo:.4f}, {hi:.4f}]  p={res['p_value']:.6f}  "
            f"{'SIG' if res['significant'] else 'NS'}"
        )
    lines.append(
        f"  LR test (simple vs enriched): LR={lr_test['lr_statistic']:.2f}, "
        f"df={lr_test['df_difference']}, p={lr_test['p_value']:.6f}"
    )
    lines.append("")

    # ── 7. BAYESIAN ANALYSIS ──────────────────────────────────────────────
    lines += ["-" * 72, "7. BAYESIAN ANALYSIS (Beta-Binomial)", "-" * 72]
    rope_lo = bayesian.get("rope_lower", -0.002)
    rope_hi = bayesian.get("rope_upper",  0.002)
    lines.append(f"  ROPE bounds:           [{rope_lo:+.4f}, {rope_hi:+.4f}]")
    lines.append(f"  Simulations:           {bayesian.get('n_simulations', 50000):,}")
    lines.append(f"  P(treatment > control): {bayesian['p_treatment_better_pct']:.2f}%")
    lines.append(f"  P(diff in ROPE):        {bayesian['p_rope_region']*100:.2f}%")
    lines.append(f"  Expected loss (T→C):   {bayesian['expected_loss']:.6f}")
    lines.append(f"  Median difference:     {bayesian['median_difference']:+.4f}")
    c1, c2 = bayesian['ci_95']
    lines.append(f"  CI 95% difference:     [{c1:+.4f}, {c2:+.4f}]")
    lines.append(f"  Decision (ROPE):       {bayesian['decision']}")
    lines.append("")

    # ── 8. SEGMENTATION ───────────────────────────────────────────────────
    lines += ["-" * 72, "8. SEGMENTATION ANALYSIS (FDR-corrected p-values)", "-" * 72]
    if segmentation:
        for col_name, segs in segmentation.items():
            for seg_name, data in segs.items():
                if not isinstance(data, dict) or "control_rate" not in data:
                    continue
                p_raw = data.get("p_value_raw", data.get("p_value", 1.0))
                p_adj = data.get("p_value", 1.0)
                sig   = "SIG*" if data.get("significant") else "NS"
                lines.append(
                    f"  {col_name}:{seg_name:12s}  "
                    f"C={data['control_rate']*100:.2f}% (n={data['control_n']:>6,})  "
                    f"T={data['treatment_rate']*100:.2f}% (n={data['treatment_n']:>6,})  "
                    f"p_raw={p_raw:.4f}  p_adj={p_adj:.4f}  {sig}"
                )
    else:
        lines.append("  No suitable categorical segments found.")
    lines.append("  * significant after BH FDR correction")
    lines.append("")

    # ── 9. ROBUSTNESS ─────────────────────────────────────────────────────
    lines += ["-" * 72, "9. ROBUSTNESS CHECKS", "-" * 72]
    b = robustness["bootstrap"]
    lines.append(f"  Bootstrap CI 95%:        [{b['ci_95'][0]:+.4f}, {b['ci_95'][1]:+.4f}]")
    lines.append(f"  Bootstrap pct T > C:     {b['pct_positive']:.2f}%")
    p = robustness["permutation"]
    lines.append(f"  Permutation p (2-sided): {p['p_value_two_sided']:.4f}")
    lines.append(f"  Permutation p (1-sided): {p['p_value_one_sided']:.4f}")
    lines.append("")

    # ── 10. TEMPORAL ──────────────────────────────────────────────────────
    lines += ["-" * 72, "10. TEMPORAL ANALYSIS", "-" * 72]
    if temporal.get("trends"):
        for grp, trend in temporal["trends"].items():
            r, pv = trend["pearson_r"], trend["p_value"]
            sig = "(significant trend)" if trend["significant_trend"] else "(no trend)"
            lines.append(f"  {grp:15s}: r={r:.4f},  p={pv:.4f}  {sig}")
    else:
        lines.append(f"  {temporal.get('error', 'No temporal data.')}")
    lines.append("")

    # ── CONCLUSION ────────────────────────────────────────────────────────
    lines += ["=" * 72, "  CONCLUSION & RECOMMENDATION", "=" * 72, ""]
    p_z    = tests["z_test"]["p_value_two_sided"]
    b_pct  = bayesian["p_treatment_better_pct"]

    if p_z < 0.01:
        evidence = "STRONG — p < 0.01"
    elif p_z < 0.05:
        evidence = "MODERATE — p < 0.05"
    elif p_z < 0.10:
        evidence = "WEAK — trend, p < 0.10"
    else:
        evidence = "NONE — p ≥ 0.05"

    lines.append(f"  Evidence strength:  {evidence}")
    lines.append("")
    lines.append("  Key findings:")
    lines.append(f"    Control rate:    {stats['control']['rate_pct']:.4f}%")
    lines.append(f"    Treatment rate:  {stats['treatment']['rate_pct']:.4f}%")
    d = stats["difference"]
    lines.append(f"    Difference:      {d['absolute_pp']:+.4f} pp  ({d['relative_pct']:+.2f}% relative)")
    lines.append(f"    Bayesian P(T>C): {b_pct:.1f}%")
    lines.append(f"    Effect size:     {effect_sizes['cohens_h_interpretation']} (Cohen's h = {effect_sizes['cohens_h']:.4f})")
    lines.append("")

    if b_pct > 95 or p_z < 0.05:
        rec = "Deploy the treatment — the improvement is statistically justified."
    elif b_pct > 80 or p_z < 0.10:
        rec = "Favorable trend but insufficient evidence. Consider extending the test."
    else:
        rec = "Do NOT deploy the treatment. Keep the current version (control)."

    lines.append(f"  Recommendation: {rec}")
    lines += ["", "=" * 72, "  END OF REPORT", "=" * 72]
    return "\n".join(lines)


def save_text_report(report_text: str, path: Path | None = None) -> Path:
    if path is None:
        path = REPORT_DIR / "ab_testing_report.txt"
    path.write_text(report_text, encoding="utf-8")
    return path
