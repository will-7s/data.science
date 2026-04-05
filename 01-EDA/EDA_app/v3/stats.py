"""
stats.py  —  Pure statistical computations. No Dash, no Plotly.

All functions accept NumPy arrays and return plain Python values / dicts.
Lilliefors uses the pre-computed MC cache in store.py (< 1 ms per call).
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sp
from scipy.special import ndtr
from dataclasses import dataclass, field
from typing import Optional
from utils import drop_nan

ALPHA = 0.05


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class NormalityResult:
    name:       str
    statistic:  Optional[float]
    p_value:    Optional[float]
    is_normal:  Optional[bool]
    conclusion: str
    notes:      list[str] = field(default_factory=list)


# ── Normality battery ─────────────────────────────────────────────────────────

def run_normality_battery(arr: np.ndarray) -> list[NormalityResult]:
    """Run all 5 normality tests. NaN values are dropped silently."""
    clean = drop_nan(arr)
    return [
        _shapiro_wilk(clean),
        _kolmogorov_smirnov(clean),
        _anderson_darling(clean),
        _lilliefors(clean),
        _dagostino_pearson(clean),
    ]


def _shapiro_wilk(c: np.ndarray) -> NormalityResult:
    name, n = "Shapiro-Wilk", len(c)
    notes = [f"n = {n}", "Most powerful for n ≤ 5 000."]
    if n < 3:
        return NormalityResult(name, None, None, None, "Cannot run: n < 3.", [f"n = {n}"])
    if n > 5_000:
        notes.append(f"n = {n} > 5 000 — test may over-reject for trivial deviations.")
    try:
        w, p = sp.shapiro(c)
    except Exception as e:
        return NormalityResult(name, None, None, None, f"Failed: {e}", notes)
    ok = p > ALPHA
    return NormalityResult(name, float(w), float(p), ok,
                           f"{'Normal' if ok else 'Non-normal'}  (W = {w:.4f},  p = {p:.4f})", notes)


def _kolmogorov_smirnov(c: np.ndarray) -> NormalityResult:
    name, n = "Kolmogorov-Smirnov", len(c)
    if n < 3:
        return NormalityResult(name, None, None, None, "Cannot run: n < 3.", [f"n = {n}"])
    mu, sigma = float(c.mean()), float(c.std(ddof=1))
    if sigma == 0:
        return NormalityResult(name, None, None, None, "Cannot run: zero variance.", [])
    notes = [f"n = {n}",
             "Parameters estimated from sample → test is conservative (prefer Lilliefors)."]
    d, p = sp.kstest(c, 'norm', args=(mu, sigma))
    ok = p > ALPHA
    return NormalityResult(name, float(d), float(p), ok,
                           f"{'Normal' if ok else 'Non-normal'}  (D = {d:.4f},  p = {p:.4f})", notes)


def _anderson_darling(c: np.ndarray) -> NormalityResult:
    name, n = "Anderson-Darling", len(c)
    if n < 7:
        return NormalityResult(name, None, None, None, f"Cannot run: n = {n} < 7.", [])
    notes = [f"n = {n}", "Emphasises tail deviations."]
    try:
        res  = sp.anderson(c, dist='norm', method='interpolate')
        stat, p = float(res.statistic), float(res.pvalue)
        ok   = p > ALPHA
        notes.append("p-value from interpolation of tabulated critical values.")
        return NormalityResult(name, stat, p, ok,
                               f"{'Normal' if ok else 'Non-normal'}  (A² = {stat:.4f},  p = {p:.4f})", notes)
    except TypeError:
        # scipy < 1.17 fallback
        res   = sp.anderson(c, dist='norm')
        stat  = float(res.statistic)
        crit5 = float(res.critical_values[2])
        ok    = stat < crit5
        notes.append(f"p-value unavailable — decision from 5 % critical value ({crit5:.4f}).")
        return NormalityResult(name, stat, None, ok,
                               f"{'Normal' if ok else 'Non-normal'}  (A² = {stat:.4f} vs crit = {crit5:.4f})", notes)


def _lilliefors(c: np.ndarray) -> NormalityResult:
    """
    Lilliefors test.  Uses the pre-computed MC null distribution from
    store._lilliefors_cache when available (< 1 ms); falls back to
    on-the-fly computation otherwise.
    """
    name, n = "Lilliefors", len(c)
    if n < 4:
        return NormalityResult(name, None, None, None, f"Cannot run: n = {n} < 4.", [])
    sigma = float(c.std(ddof=1))
    if sigma == 0:
        return NormalityResult(name, None, None, None, "Cannot run: zero variance.", [])

    z_obs   = (c - c.mean()) / sigma
    d_obs, _ = sp.kstest(z_obs, 'norm')
    d_obs   = float(d_obs)

    # Try cache first (set by store.reset at load time)
    import store as _store
    mc_stats = _store.get_lilliefors_mc(n)

    if mc_stats is None:
        # Cold path: compute on the fly (only happens outside store.reset flow)
        mc_stats = _store._lilliefors_mc(n)
        note_src = "on-the-fly MC"
    else:
        note_src = "cached MC"

    p    = float(max((mc_stats >= d_obs).mean(), 1.0 / len(mc_stats)))
    ok   = p > ALPHA
    notes = [f"n = {n}",
             f"Corrects KS for estimated parameters ({note_src}, {len(mc_stats)} reps)."]
    return NormalityResult(name, d_obs, p, ok,
                           f"{'Normal' if ok else 'Non-normal'}  (D = {d_obs:.4f},  p ≈ {p:.4f})", notes)


def _dagostino_pearson(c: np.ndarray) -> NormalityResult:
    name, n = "D'Agostino-Pearson", len(c)
    if n < 8:
        return NormalityResult(name, None, None, None, f"Cannot run: n = {n} < 8.", [])
    k2, p   = sp.normaltest(c)
    ok      = p > ALPHA
    skew    = float(sp.skew(c))
    kurt    = float(sp.kurtosis(c))
    notes   = [f"n = {n}",
               f"Skewness = {skew:+.3f}  (0 = symmetric)",
               f"Excess kurtosis = {kurt:+.3f}  (0 = normal)"]
    return NormalityResult(name, float(k2), float(p), ok,
                           f"{'Normal' if ok else 'Non-normal'}  (K² = {k2:.4f},  p = {p:.4f})", notes)


# ── Descriptive statistics ────────────────────────────────────────────────────

def descriptive_stats(arr: np.ndarray) -> dict:
    """Mean, median, mode, std, min, max, n — single NumPy pass."""
    clean = drop_nan(arr)
    if clean.size == 0:
        return {'n': 0}
    uniq, counts  = np.unique(clean, return_counts=True)
    top           = np.argmax(counts)
    return {
        'n':          int(clean.size),
        'mean':       float(clean.mean()),
        'median':     float(np.median(clean)),
        'mode':       float(uniq[top]),
        'mode_freq':  int(counts[top]),
        'mode_ties':  int((counts == counts[top]).sum()),
        'std':        float(clean.std(ddof=1)),
        'min':        float(clean.min()),
        'max':        float(clean.max()),
    }


def outlier_percentage(arr: np.ndarray) -> float:
    """IQR × 1.5 outlier rate (0–100 %)."""
    clean = drop_nan(arr)
    if clean.size == 0:
        return 0.0
    q1, q3 = np.quantile(clean, [0.25, 0.75])
    iqr    = q3 - q1
    return float(((clean < q1 - 1.5*iqr) | (clean > q3 + 1.5*iqr)).mean() * 100)


# ── Compatibility shims ───────────────────────────────────────────────────────

def normality_test(arr: np.ndarray):
    """Legacy shim — Shapiro-Wilk only."""
    clean = drop_nan(arr)
    if len(clean) < 3:
        return None, None
    return sp.shapiro(clean)


def normality_label(p: Optional[float]) -> str:
    if p is None:
        return "Insufficient data"
    return f"Normal (p = {p:.3f})" if p > ALPHA else f"Non-normal (p = {p:.3f})"


# ── Group statistics ──────────────────────────────────────────────────────────

def group_means(num_arr: np.ndarray, cat_arr: np.ndarray) -> dict[str, float]:
    return {
        str(c): float(drop_nan(num_arr[cat_arr == c]).mean())
        for c in np.unique(cat_arr)
        if drop_nan(num_arr[cat_arr == c]).size
    }


# ── Correlation matrix ────────────────────────────────────────────────────────

def correlation_matrix(data: dict, num_cols: list[str]):
    if len(num_cols) < 2:
        return None, None
    n   = len(num_cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = data[num_cols[i]], data[num_cols[j]]
            mask = ~(np.isnan(a) | np.isnan(b))
            r    = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 1 else 0.0
            mat[i, j] = mat[j, i] = r
    return mat, num_cols


# ── Bivariate tests ───────────────────────────────────────────────────────────

def bivariate_test(var1: str, var2: str, data: dict, meta: dict) -> dict:
    t1, t2 = meta[var1], meta[var2]
    d1, d2 = data[var1], data[var2]
    if t1 == 'numeric' and t2 == 'numeric':
        return _test_num_num(d1, d2, var1, var2)
    if t1 != t2:
        num, cat = (d1, d2) if t1 == 'numeric' else (d2, d1)
        return _test_num_cat(num, cat)
    return _test_cat_cat(d1, d2)


def _test_num_num(a, b, v1, v2) -> dict:
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return {"test_name": "Error", "results": ["Need ≥ 3 paired observations."]}
    rp, pp = sp.pearsonr(a, b)
    rs, ps = sp.spearmanr(a, b)
    rk, pk = sp.kendalltau(a, b)
    _, p1  = normality_test(a)
    _, p2  = normality_test(b)
    both   = (p1 or 0) > ALPHA and (p2 or 0) > ALPHA
    rec    = ("Pearson (both normal)" if both
              else "Kendall (small sample)" if len(a) < 30
              else "Spearman (non-normal)")
    sig    = lambda p: "✓ sig." if p < ALPHA else "✗ n.s."
    return {"test_name": "Correlation Analysis", "results": [
        f"n = {len(a)}",
        "",
        "── Correlation Coefficients ──",
        f"Pearson   r = {rp:+.4f}   p = {pp:.4f}  {sig(pp)}",
        f"  Measures linear association. Assumes bivariate normality.",
        f"Spearman  ρ = {rs:+.4f}   p = {ps:.4f}  {sig(ps)}",
        f"  Rank-based; robust to outliers and non-normality.",
        f"Kendall   τ = {rk:+.4f}   p = {pk:.4f}  {sig(pk)}",
        f"  Preferred for small or heavily tied samples.",
        "",
        f"Normality — {v1}: {'✓' if (p1 or 0)>ALPHA else '✗'}   {v2}: {'✓' if (p2 or 0)>ALPHA else '✗'}",
        f"Recommended: {rec}",
    ]}


def _test_num_cat(num, cat) -> dict:
    cats   = np.unique(cat)
    groups = [drop_nan(num[cat == c]) for c in cats]
    valid  = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return {"test_name": "Error", "results": ["Need ≥ 2 groups with ≥ 2 observations."]}

    lev_w, lev_p  = sp.levene(*valid)
    homo          = lev_p > ALPHA
    f_stat, ap    = sp.f_oneway(*valid)
    h_stat, kp    = sp.kruskal(*valid)
    sig = lambda p: "✓ sig." if p < ALPHA else "✗ n.s."

    # Normality check per group
    norm_flags = []
    for g in valid:
        if 3 <= len(g) <= 5_000:
            _, pg = sp.shapiro(g)
            norm_flags.append(pg > ALPHA)
        else:
            norm_flags.append(None)
    all_norm = all(f for f in norm_flags if f is not None)

    if all_norm and homo:
        rec = "ANOVA (normality ✓, equal variances ✓)"
    elif all_norm:
        rec = "Kruskal-Wallis or Welch ANOVA (normality ✓, variances unequal)"
    else:
        rec = "Kruskal-Wallis (normality not met)"

    results = [
        f"k = {len(valid)} groups  |  sizes: {[len(g) for g in valid]}",
        "",
        "── Variance Homogeneity (Levene) ──",
        f"W = {lev_w:.4f},  p = {lev_p:.4f}  →  {'Equal ✓' if homo else 'Unequal ✗  (ANOVA may be unreliable)'}",
        "",
        "── One-Way ANOVA (parametric) ──",
        f"F = {f_stat:.4f},  p = {ap:.4f}  {sig(ap)}",
        f"  Assumes: normality per group, equal variances.",
        "" if homo else "  ⚠ Levene rejected — consider Kruskal-Wallis.",
        "",
        "── Kruskal-Wallis (non-parametric) ──",
        f"H = {h_stat:.4f},  p = {kp:.4f}  {sig(kp)}",
        "  No normality or equal-variance assumption.",
        "",
        f"Recommended: {rec}",
    ]

    if len(valid) == 2:
        g1, g2 = valid
        t_stat, tp = sp.ttest_ind(g1, g2, equal_var=homo)
        u_stat, up = sp.mannwhitneyu(g1, g2, alternative='two-sided')
        var_lbl    = "Student" if homo else "Welch"
        results += [
            "",
            "── Two-Group Tests ──",
            f"t-test ({var_lbl}):  t = {t_stat:.4f},  p = {tp:.4f}  {sig(tp)}",
            f"Mann-Whitney U:     U = {u_stat:.1f},   p = {up:.4f}  {sig(up)}",
        ]
        if len(g1) == len(g2):
            try:
                w_stat, wp = sp.wilcoxon(g1, g2)
                results.append(
                    f"Wilcoxon (paired):  W = {w_stat:.1f},  p = {wp:.4f}  {sig(wp)}"
                )
            except Exception:
                results.append("Wilcoxon: no non-zero differences.")
        else:
            results.append(
                f"Wilcoxon: skipped — unequal sizes ({len(g1)} vs {len(g2)})."
            )

    return {"test_name": "Group Comparison", "results": results}


def _test_cat_cat(a, b) -> dict:
    ua, ub  = np.unique(a), np.unique(b)
    k, r    = len(ua), len(ub)
    ct      = np.array([[np.sum((a==va)&(b==vb)) for vb in ub] for va in ua], dtype=float)
    n       = ct.sum()
    chi2, p, dof, exp = sp.chi2_contingency(ct)
    min_exp = exp.min()
    low_pct = (exp < 5).mean() * 100
    valid   = (min_exp >= 1) and (low_pct <= 20)
    sig     = "✓ sig." if p < ALPHA else "✗ n.s."
    min_dim = min(k, r)
    cv = float(np.sqrt(chi2 / (n*(min_dim-1)))) if min_dim>1 and n>0 else float('nan')
    tt = float(np.sqrt(chi2 / (n*np.sqrt((k-1)*(r-1))))) if (k-1)*(r-1)>0 and n>0 else float('nan')

    def _strength(v):
        thresholds = {1:(0.10,0.30,0.50), 2:(0.07,0.21,0.35), 3:(0.06,0.17,0.29)}
        s, m, l = thresholds.get(min_dim-1, (0.05,0.15,0.25))
        return "negligible" if v<s else "small" if v<m else "medium" if v<l else "large"

    results = [
        f"Table: {k}×{r}  |  n = {n}",
        "",
        "── Chi-square Test of Independence ──",
        f"χ²({dof}) = {chi2:.4f},  p = {p:.4f}  {sig}",
        f"  Min expected frequency = {min_exp:.2f}  {'✓ ≥ 1' if min_exp>=1 else '✗ < 1 — unreliable'}",
        f"  Cells with expected < 5: {low_pct:.1f}%  {'✓ ≤ 20%' if low_pct<=20 else '✗ > 20% — unreliable'}",
    ] + (["  ⚠ Assumptions violated — interpret χ² cautiously."] if not valid else []) + [
        "",
        "── Effect Sizes ──",
        f"Cramér's V    = {cv:.4f}  ({_strength(cv)})" if not np.isnan(cv) else "Cramér's V: n/a",
        f"Tschuprow's T = {tt:.4f}" if not np.isnan(tt) else "Tschuprow's T: n/a",
        "  V ∈ [0,1]: 0 = no assoc, 1 = perfect. T ≤ V for non-square tables.",
    ]

    if k == 2 and r == 2:
        oddsratio, fp = sp.fisher_exact(ct.astype(int))
        results += [
            "",
            "── Fisher's Exact Test (2×2) ──",
            f"Odds ratio = {oddsratio:.4f},  p = {fp:.4f}  {'✓ sig.' if fp<ALPHA else '✗ n.s.'}",
            "  Exact p-value; preferred when expected frequencies are low.",
        ]
    else:
        results.append(f"Fisher's exact: not applicable ({k}×{r} table — requires 2×2).")

    return {"test_name": "Association Tests", "results": results}
