from __future__ import annotations

import math

import pandas as pd


def generate_text_report(R: dict) -> str:
    lines = []
    lines += ["=" * 72, "  A/B TESTING ANALYSIS REPORT", "=" * 72, ""]
    s = R.get("stats", {})
    lines += ["-" * 72, "1. DESCRIPTIVE STATISTICS", "-" * 72]
    lines.append(f"  {'Control':15s}: n={s.get('control_n', 0):>10,}  rate={s.get('control_rate_pct', 0):.4f}%")
    lines.append(f"  {'Treatment':15s}: n={s.get('treatment_n', 0):>10,}  rate={s.get('treatment_rate_pct', 0):.4f}%")
    lines.append(f"  {'Difference':15s}: {s.get('diff_pp', 0):+.4f} pp  ({s.get('rel_pct', 0):+.2f}% relative)")
    lines.append("")
    lines += ["-" * 72, "2. FREQUENTIST HYPOTHESIS TESTS", "-" * 72]
    tests = R.get("tests", {})
    for name, key in [("Chi-squared","chi_squared"),("Z-test (2-sided)","ztest"),("T-test (Welch)","ttest"),("Mann-Whitney U","mann_whitney")]:
        t = tests.get(key) or {}
        st, p = t.get("statistic"), t.get("p_value")
        if st is not None and p is not None:
            sig = "SIG" if p < s.get("alpha", 0.05) else "NS"
            lines.append(f"  {name:22s}: stat={st:>10.4f},  p={p:.6f}  {sig}")
    pv = (tests.get("ztest") or {}).get("p_value")
    if pv is not None:
        lines.append(f"  {'Z-test (1-sided T>C)':22s}: p={pv / 2:.6f}")
    lines.append("")
    lines += ["-" * 72, "3. EFFECT SIZES", "-" * 72]
    es = R.get("effect_sizes", {})
    lines.append(f"  Cohen's h:         {es.get('cohens_d', 0):.4f} ({es.get('cohens_d_interpretation', '')})")
    lines.append(f"  Risk Ratio (RR):   {es.get('risk_ratio', 0):.4f}")
    nn = es.get("nnt")
    lines.append(f"  NNT:               {'∞' if nn is None or not math.isfinite(nn) else f'{nn:.1f}'}")
    lines.append("")
    lines += ["-" * 72, "4. STATISTICAL POWER", "-" * 72]
    pwr = R.get("power") or {}
    if pwr.get("skipped"):
        lines.append("  Power analysis skipped (effect size ~0)")
    else:
        lines.append(f"  Observed power:        {pwr.get('power_observed', 0) * 100:.2f}%")
        lines.append(f"  Needed per group (80%): {pwr.get('n_needed_80pct', 0):>8,.0f}")
        lines.append(f"  MDE (Cohen's h):       {pwr.get('mde_cohens_h', 0):.4f}")
        lines.append(f"  MDE (approx pp):       ±{pwr.get('mde_pp', 0) * 100:.2f} pp")
    lines.append("")
    lines += ["-" * 72, "5. LOGISTIC REGRESSION", "-" * 72]
    for lb, k in [("Simple","log_simple"),("Enriched","log_enriched")]:
        lr = R.get(k) or {}
        lo, hi = lr.get("or_ci_95", (0, 0))
        lines.append(f"  {lb:8s} model:  OR={lr.get('odds_ratio', 0):.4f}  CI95=[{lo:.4f},{hi:.4f}]  p={lr.get('p_value',0):.6f}  {'SIG' if lr.get('significant') else 'NS'}")
    lrt = R.get("lr_test") or {}
    lines.append(f"  LR test: LR={lrt.get('lr_statistic',0):.2f}, p={lrt.get('p_value',0):.6f}")
    lines.append("")
    lines += ["-" * 72, "6. BAYESIAN ANALYSIS", "-" * 72]
    by = R.get("bayesian", {})
    rl, rh = by.get("rope_lower",-0.002), by.get("rope_upper",0.002)
    lines.append(f"  ROPE bounds:           [{rl:+.4f}, {rh:+.4f}]")
    lines.append(f"  P(treatment > control): {by.get('p_treatment_better_pct',by.get('p_treatment_better',0)*100):.2f}%")
    lines.append(f"  P(diff in ROPE):        {by.get('p_rope_region',0)*100:.2f}%")
    lines.append(f"  Expected loss (T->C):   {by.get('expected_loss',0):.6f}")
    c1,c2 = by.get("ci_95",(0,0))
    lines.append(f"  CI 95% difference:     [{c1:+.4f}, {c2:+.4f}]")
    lines.append(f"  Decision (ROPE):       {by.get('decision','N/A')}")
    lines.append("")
    lines += ["-" * 72, "7. SEGMENTATION", "-" * 72]
    sg = R.get("segmentation", {})
    if sg:
        for cn, ss in sg.items():
            for sn, d in ss.items():
                if not isinstance(d, dict) or "control_rate" not in d:
                    continue
                pr = d.get("p_value_raw", d.get("p_value", 1.0))
                pa = d.get("p_value", 1.0)
                si = "SIG*" if d.get("significant") else "NS"
                lines.append(f"  {cn}:{sn:12s}  C={d['control_rate']*100:.2f}% (n={d.get('control_n',0):>6,})  T={d['treatment_rate']*100:.2f}% (n={d.get('treatment_n',0):>6,})  p_raw={pr:.4f}  p_adj={pa:.4f}  {si}")
    lines.append("")
    lines += ["-" * 72, "8. ROBUSTNESS", "-" * 72]
    bt = R.get("bootstrap", {})
    c9 = bt.get("ci_95", (0, 0))
    lines.append(f"  Bootstrap CI 95%:        [{c9[0]:+.4f}, {c9[1]:+.4f}]")
    lines.append(f"  Bootstrap pct T > C:     {bt.get('pct_positive',0):.2f}%")
    pm = R.get("permutation", {})
    lines.append(f"  Permutation p (2-sided): {pm.get('p_value_two_sided',0):.4f}")
    lines.append(f"  Permutation p (1-sided): {pm.get('p_value_one_sided',0):.4f}")
    lines.append("")
    lines += ["=" * 72, "  CONCLUSION", "=" * 72, ""]
    lines.append(f"  Verdict: {R.get('verdict','N/A')}")
    lines += ["", "=" * 72, "  END OF REPORT", "=" * 72]
    return "\n".join(lines)


def build_summary_df(R: dict) -> pd.DataFrame:
    s = R.get("stats", {})
    es = R.get("effect_sizes", {})
    tests = R.get("tests", {})
    bayes = R.get("bayesian", {})
    power = R.get("power") or {}
    boot = R.get("bootstrap", {})
    perm = R.get("permutation", {})
    zt = tests.get("ztest") or {}
    c9 = boot.get("ci_95", (0, 0))
    rows = [
        ("Control n",str(s.get("control_n",0))),
        ("Treatment n",str(s.get("treatment_n",0))),
        ("Control rate (%)",f"{s.get('control_rate_pct',0):.4f}"),
        ("Treatment rate (%)",f"{s.get('treatment_rate_pct',0):.4f}"),
        ("Absolute diff (pp)",f"{s.get('diff_pp',0):.4f}"),
        ("Relative diff (%)",f"{s.get('rel_pct',0):.4f}"),
        ("Z-test p (two-sided)",f"{zt.get('p_value',0):.6f}"),
        ("Cohen's h",f"{es.get('cohens_d',0):.6f}"),
        ("Odds Ratio",f"{es.get('odds_ratio',0):.6f}" if es.get("odds_ratio") is not None else "—"),
        ("P(T>C) Bayesian (%)",f"{bayes.get('p_treatment_better_pct',bayes.get('p_treatment_better',0)*100):.2f}"),
        ("Bayesian decision",bayes.get("decision","—")),
        ("Observed power (%)",f"{power.get('power_observed',0)*100:.2f}"),
        ("Bootstrap CI 95% low",f"{c9[0]*100:.4f}" if c9 else "—"),
        ("Bootstrap CI 95% high",f"{c9[1]*100:.4f}" if c9 else "—"),
        ("Permutation p (two-sided)",f"{perm.get('p_value_two_sided',0):.6f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric","Value"])
