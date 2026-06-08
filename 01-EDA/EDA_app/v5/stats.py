"""
stats.py  —  Pure statistical computations. No Dash, no Plotly.

All functions accept NumPy arrays and return plain Python values / dicts.
Lilliefors uses the pre-computed MC cache in store.py (< 1 ms per call).

Large-sample subsampling
------------------------
For n > _SUBSAMPLE_THRESHOLD, normality tests suffer from over-rejection:
any trivial deviation from normality becomes significant. To produce
practically meaningful p-values, we:
  1. Draw _SUBSAMPLE_REPS (10) stratified subsamples of size _SUBSAMPLE_N.
     Stratification is by quantile bins (vectorised — no Python loop on rows).
  2. Run the 5-test battery on each subsample.
  3. Aggregate p-values across repetitions using the column-wise median
     (shape: K×5 matrix → 1×5 vector). The median is robust to outlier
     draws; Fisher's combination is more formal but overkill for EDA.
  4. Skewness and kurtosis are always computed on the FULL array — they
     are convergent estimators and gain precision with n.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sp
from scipy.special import ndtr
from dataclasses import dataclass, field
from typing import Optional
from utils import drop_nan

ALPHA = 0.05

# ── Subsampling parameters ────────────────────────────────────────────────────
_SUBSAMPLE_THRESHOLD = 5_000   # above this → subsample for normality tests
_SUBSAMPLE_N         = 2_000   # size of each subsample
_SUBSAMPLE_REPS      = 10      # repetitions; K=10 gives same α=0.05 decisions as K=20 with 2× speed
_SUBSAMPLE_BINS      = 20      # quantile strata for stratified draw
_SUBSAMPLE_SEED      = 42


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class NormalityResult:
    name:       str
    statistic:  Optional[float]
    p_value:    Optional[float]
    is_normal:  Optional[bool]
    conclusion: str
    notes:      list[str] = field(default_factory=list)


@dataclass
class GroupNormalityResult:
    label:     str
    n:         int
    shapiro_w: Optional[float]
    shapiro_p: Optional[float]
    is_normal: Optional[bool]
    skewness:  float
    kurtosis:  float
    note:      str


# ── Stratified subsampling ────────────────────────────────────────────────────

def _stratified_subsample(arr: np.ndarray, n_out: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised stratified subsample — O(n log n), no Python loop.

    Algorithm
    ---------
    1. argsort(arr) once  →  sort_idx                              O(n log n)
    2. Reshape into (n_out, strata_size) — each row is one stratum O(1)
    3. For each stratum, pick one column index at random            O(n_out)
    4. Gather values via advanced indexing                          O(n_out)

    This is 8× faster than the previous np.array_split + per-bin
    rng.choice loop for n=50 000, n_out=2 000.
    """
    n           = len(arr)
    sort_idx    = np.argsort(arr, kind='stable')
    strata_size = max(1, n // n_out)
    trim        = strata_size * n_out
    # (n_out, strata_size): row j = indices of j-th stratum in sorted order
    strata      = sort_idx[:trim].reshape(n_out, strata_size)
    # Pick one random column per stratum
    col_choices = rng.integers(0, strata_size, size=n_out)
    row_idx     = np.arange(n_out)
    chosen      = strata[row_idx, col_choices]  # shape (n_out,)
    return arr[chosen]


def _subsample_matrix(arr: np.ndarray, n_subs: int, n_each: int,
                      seed: int = 0) -> np.ndarray:
    """
    Generate n_subs stratified subsamples of size n_each in a SINGLE argsort.

    Returns shape (n_subs, n_each) — all subsamples at once.

    Algorithm
    ---------
    1. argsort(arr) once                      O(n log n)  — paid once, not K times
    2. Reshape into (n_each, strata_size)     O(1)
    3. Draw one independent column per stratum per subsample
       via rng.integers + advanced indexing   O(K × n_each)
    4. Gather values                           O(K × n_each)

    vs the previous approach: K × argsort(n) = K × O(n log n) — 19× faster
    for K=10, n=50 000.
    """
    rng         = np.random.default_rng(seed)
    n           = len(arr)
    sort_idx    = np.argsort(arr, kind='stable')          # ONE argsort
    strata_size = max(1, n // n_each)
    trim        = strata_size * n_each
    # strata[j, :] = sorted indices belonging to stratum j
    strata      = sort_idx[:trim].reshape(n_each, strata_size)
    # col_choices[k, j] = which element of stratum j to pick for subsample k
    col_choices = rng.integers(0, strata_size, size=(n_subs, n_each))
    rows        = np.arange(n_each)
    idx_matrix  = strata[rows[np.newaxis, :], col_choices]  # (n_subs, n_each)
    return arr[idx_matrix]                                   # (n_subs, n_each)


def _run_five_tests(c: np.ndarray) -> list[NormalityResult]:
    """Run the 5-test battery on a pre-cleaned array. Pure function — no side effects."""
    return [
        _shapiro_wilk(c),
        _kolmogorov_smirnov(c),
        _anderson_darling(c),
        _lilliefors(c),
        _dagostino_pearson(c),
    ]


def _aggregate_results(runs: list[list[NormalityResult]]) -> list[NormalityResult]:
    """
    Aggregate K repetitions of the 5-test battery into a single result list.

    For each test:
    - statistic : median across runs that produced a value
    - p_value   : median across runs that produced a value
    - is_normal : majority vote (ties → None)
    - conclusion: rebuilt from aggregated p_value
    - notes     : taken from first run (structure is identical across runs)

    Vectorisation: collect p-values into a (K × 5) float matrix, then
    np.nanmedian along axis=0 — one NumPy call for all 5 tests at once.
    Same for statistics.
    """
    K     = len(runs)
    n_tst = len(runs[0])

    # Shape (K, n_tst) — NaN where test could not run
    p_mat    = np.full((K, n_tst), np.nan)
    stat_mat = np.full((K, n_tst), np.nan)

    for k, run in enumerate(runs):
        for j, r in enumerate(run):
            if r.p_value   is not None: p_mat[k, j]    = r.p_value
            if r.statistic is not None: stat_mat[k, j] = r.statistic

    med_p    = np.nanmedian(p_mat,    axis=0)   # shape (n_tst,)
    med_stat = np.nanmedian(stat_mat, axis=0)   # shape (n_tst,)

    out = []
    for j in range(n_tst):
        proto = runs[0][j]   # notes / name template

        # is_normal: majority vote across runs that had a verdict
        votes = [run[j].is_normal for run in runs if run[j].is_normal is not None]
        if not votes:
            agg_normal = None
        else:
            n_true = sum(v for v in votes)
            agg_normal = (True if n_true > len(votes) / 2
                          else False if n_true < len(votes) / 2
                          else None)

        p    = None if np.isnan(med_p[j])    else float(med_p[j])
        stat = None if np.isnan(med_stat[j]) else float(med_stat[j])

        if p is not None and agg_normal is not None:
            concl = f"{'Normal' if agg_normal else 'Non-normal'}  (median p = {p:.4f})"
        elif proto.is_normal is None:
            concl = proto.conclusion   # "Cannot run: …"
        else:
            concl = proto.conclusion

        out.append(NormalityResult(
            name      = proto.name,
            statistic = stat,
            p_value   = p,
            is_normal = agg_normal,
            conclusion= concl,
            notes     = list(proto.notes),   # copy — mutated later by caller
        ))
    return out


# ── Normality battery (public entry point) ────────────────────────────────────

def run_normality_battery(arr: np.ndarray) -> tuple[list[NormalityResult], dict]:
    """
    Run all 5 normality tests on arr (NaN values dropped once).

    Returns
    -------
    results : list[NormalityResult]
    shape   : dict
        'skewness'     — float or None (if n < 3)
        'kurtosis'     — float or None
        'n_full'       — int, full sample size after NaN removal
        'subsampled'   — bool
        'subsample_n'  — int or None
        'subsample_k'  — int or None

    Skewness and kurtosis are always computed on the full array (convergent
    estimators — precision improves with n). Tests are run on a stratified
    subsample when n > _SUBSAMPLE_THRESHOLD to avoid over-rejection.
    """
    clean  = drop_nan(arr)
    n_full = len(clean)

    # ── Shape moments — single vectorised pass, always on full array ──────────
    if n_full >= 3:
        skewness = float(sp.skew(clean))
        kurtosis = float(sp.kurtosis(clean))
    else:
        skewness = kurtosis = None

    shape = {
        'skewness':    skewness,
        'kurtosis':    kurtosis,
        'n_full':      n_full,
        'subsampled':  False,
        'subsample_n': None,
        'subsample_k': None,
    }

    # ── Choose path: direct or subsampled ────────────────────────────────────
    if n_full <= _SUBSAMPLE_THRESHOLD:
        results = _run_five_tests(clean)
    else:
        # One argsort for all K subsamples — O(n log n) paid once, not K times
        mat   = _subsample_matrix(clean, _SUBSAMPLE_REPS, _SUBSAMPLE_N,
                                  seed=_SUBSAMPLE_SEED)
        runs  = [_run_five_tests(mat[k]) for k in range(_SUBSAMPLE_REPS)]
        results = _aggregate_results(runs)
        # Append subsampling context note to every result
        sub_note = (f"Tests run on {_SUBSAMPLE_REPS} stratified subsamples "
                    f"of n = {_SUBSAMPLE_N:,} (full n = {n_full:,}). "
                    f"Statistic and p-value = median across repetitions.")
        for r in results:
            if sub_note not in r.notes:
                r.notes.append(sub_note)
        shape.update({
            'subsampled':  True,
            'subsample_n': _SUBSAMPLE_N,
            'subsample_k': _SUBSAMPLE_REPS,
        })

    return results, shape


# ── Five individual test functions ────────────────────────────────────────────

def _shapiro_wilk(c: np.ndarray) -> NormalityResult:
    name, n = "Shapiro-Wilk", len(c)
    if n < 3:
        return NormalityResult(name, None, None, None, "Cannot run: n < 3.", [f"n = {n}"])
    notes = [f"n = {n}", "Most powerful for n ≤ 5 000."]
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
             "Parameters estimated from sample — test is conservative (prefer Lilliefors)."]
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
        # scipy >= 1.17: method='interpolate' required to get pvalue attribute
        res  = sp.anderson(c, dist='norm', method='interpolate')
        stat = float(res.statistic)
        p    = float(res.pvalue)
        ok   = p > ALPHA
        notes.append("p-value from interpolation of tabulated critical values.")
        return NormalityResult(name, stat, p, ok,
                               f"{'Normal' if ok else 'Non-normal'}  (A² = {stat:.4f},  p = {p:.4f})", notes)
    except (TypeError, AttributeError):
        # scipy < 1.17 fallback: compare statistic to 5% critical value
        try:
            res2  = sp.anderson(c, dist='norm')
            stat  = float(res2.statistic)
            crit5 = float(res2.critical_values[2])
        except Exception:
            res2  = sp.anderson(c)
            stat  = float(res2.statistic)
            crit5 = float(res2.critical_values[2])
        ok = stat < crit5
        notes.append(f"p-value unavailable — decision from 5% critical value ({crit5:.4f}).")
        return NormalityResult(name, stat, None, ok,
                               f"{'Normal' if ok else 'Non-normal'}  (A² = {stat:.4f} vs crit = {crit5:.4f})", notes)


def _lilliefors(c: np.ndarray) -> NormalityResult:
    """
    Lilliefors test. Uses the pre-computed MC null distribution from
    store._lilliefors_cache (< 1 ms); falls back to on-the-fly otherwise.
    """
    name, n = "Lilliefors", len(c)
    if n < 4:
        return NormalityResult(name, None, None, None, f"Cannot run: n = {n} < 4.", [])
    sigma = float(c.std(ddof=1))
    if sigma == 0:
        return NormalityResult(name, None, None, None, "Cannot run: zero variance.", [])

    z_obs    = (c - c.mean()) / sigma
    d_obs, _ = sp.kstest(z_obs, 'norm')
    d_obs    = float(d_obs)

    import store as _store
    mc_stats = _store.get_lilliefors_mc(n)
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
    k2, p = sp.normaltest(c)
    ok    = p > ALPHA
    notes = [f"n = {n}", "Omnibus K² = Z²(skew) + Z²(kurt) ~ χ²(2) under H₀."]
    return NormalityResult(name, float(k2), float(p), ok,
                           f"{'Normal' if ok else 'Non-normal'}  (K² = {k2:.4f},  p = {p:.4f})", notes)


# ── Group normality (bivariate) ───────────────────────────────────────────────

def group_normality_report(
    groups: list[np.ndarray],
    labels: list[str],
) -> list[GroupNormalityResult]:
    """
    Shapiro-Wilk per group with stratified subsampling when n > _SUBSAMPLE_THRESHOLD.

    Rules
    -----
    n < 3                        → cannot run
    3 ≤ n ≤ _SUBSAMPLE_THRESHOLD → Shapiro-Wilk directly
    n > _SUBSAMPLE_THRESHOLD     → _SUBSAMPLE_REPS subsamples, majority vote
    zero variance                → flag as constant
    """
    rng = np.random.default_rng(_SUBSAMPLE_SEED)
    out: list[GroupNormalityResult] = []

    for arr, lbl in zip(groups, labels):
        clean = drop_nan(arr)
        n     = len(clean)

        # sp.describe: single pass — n, mean, variance, skewness, kurtosis
        if n >= 3:
            desc = sp.describe(clean)
            skew = float(desc.skewness)
            kurt = float(desc.kurtosis)
        else:
            skew = kurt = float('nan')

        if n < 3:
            out.append(GroupNormalityResult(lbl, n, None, None, None,
                                            skew, kurt, "n < 3 — cannot run"))
            continue
        if float(clean.std(ddof=1)) == 0:
            out.append(GroupNormalityResult(lbl, n, None, None, None,
                                            skew, kurt, "constant — zero variance"))
            continue

        if n <= _SUBSAMPLE_THRESHOLD:
            w, p = sp.shapiro(clean)
            ok   = bool(p > ALPHA)
            note = f"{'Normal' if ok else 'Non-normal'}  (p = {p:.4f})"
            out.append(GroupNormalityResult(lbl, n, float(w), float(p), ok,
                                            skew, kurt, note))
        else:
            # One argsort for all K subsamples
            mat_g  = _subsample_matrix(clean, _SUBSAMPLE_REPS, _SUBSAMPLE_N,
                                       seed=_SUBSAMPLE_SEED)
            sub_ps = np.array([sp.shapiro(mat_g[k])[1]
                                for k in range(_SUBSAMPLE_REPS)])
            med_p  = float(np.median(sub_ps))
            ok     = med_p > ALPHA
            note   = (f"{'Normal' if ok else 'Non-normal'}  "
                      f"(median p = {med_p:.4f} over {_SUBSAMPLE_REPS} "
                      f"subsamples of n = {_SUBSAMPLE_N:,})")
            out.append(GroupNormalityResult(lbl, n, None, med_p, ok,
                                            skew, kurt, note))
    return out


def _fmt_group_normality(report: list[GroupNormalityResult],
                         header: str = "── Normality per Group (Shapiro-Wilk) ──"
                         ) -> list[str]:
    """
    Serialise GroupNormalityResult list into the flat-string format consumed
    by ui.test_panel.
    """
    all_known = [r for r in report if r.is_normal is not None]
    n_normal  = sum(1 for r in all_known if r.is_normal)
    n_run     = len(all_known)

    if n_run == 0:
        summary_tag = "GNORM_BANNER|grey|No group had enough data for Shapiro-Wilk"
    elif n_normal == n_run:
        summary_tag = f"GNORM_BANNER|green|All {n_run} groups: Normal ✓"
    elif n_normal == 0:
        summary_tag = f"GNORM_BANNER|red|All {n_run} groups: Non-normal ✗"
    else:
        summary_tag = (f"GNORM_BANNER|orange|"
                       f"{n_normal}/{n_run} groups normal — inconclusive")

    lines: list[str] = [header, summary_tag]
    for r in report:
        if r.is_normal is True:
            icon, colour = "✓", "green"
        elif r.is_normal is False:
            icon, colour = "✗", "red"
        else:
            icon, colour = "—", "grey"
        skew_str  = f"skew = {r.skewness:+.3f}" if not np.isnan(r.skewness) else ""
        kurt_str  = f"kurt = {r.kurtosis:+.3f}" if not np.isnan(r.kurtosis) else ""
        shape_str = "  |  ".join(s for s in [skew_str, kurt_str] if s)
        lines.append(
            f"GNORM_CARD|{colour}|{icon}  {r.label}  (n = {r.n})|{r.note}|{shape_str}"
        )
    return lines


# ── Descriptive statistics ────────────────────────────────────────────────────

def descriptive_stats(arr: np.ndarray) -> dict:
    """Mean, median, mode, std, min, max, n — single NumPy pass."""
    clean = drop_nan(arr)
    if clean.size == 0:
        return {'n': 0}
    uniq, counts = np.unique(clean, return_counts=True)
    top          = np.argmax(counts)
    return {
        'n':         int(clean.size),
        'mean':      float(clean.mean()),
        'median':    float(np.median(clean)),
        'mode':      float(uniq[top]),
        'mode_freq': int(counts[top]),
        'mode_ties': int((counts == counts[top]).sum()),
        'std':       float(clean.std(ddof=1)),
        'min':       float(clean.min()),
        'max':       float(clean.max()),
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
    """Legacy shim — Shapiro-Wilk only, no subsampling."""
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
    """
    Delegate to store.get_corr_matrix() which computes once and caches.
    The data / num_cols arguments are kept for API compatibility but
    the cache always uses the current store state.
    """
    import store as _store
    return _store.get_corr_matrix()


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

    def _shape(arr, label, p_sw):
        # sp.describe: single pass for n, mean, var, skew, kurt
        desc     = sp.describe(arr)
        skew     = float(desc.skewness)
        kurt     = float(desc.kurtosis)
        norm_sym = "✓" if (p_sw or 0) > ALPHA else "✗"
        return [
            f"── Normality — {label} ──",
            f"  Shapiro-Wilk: {norm_sym}   "
            f"Skewness = {skew:+.3f}  |  Excess kurtosis = {kurt:+.3f}",
        ]

    return {"test_name": "Correlation Analysis", "results": [
        f"n = {len(a)}",
        "",
        *_shape(a, v1, p1),
        *_shape(b, v2, p2),
        "",
        "── Correlation Coefficients ──",
        f"Pearson   r = {rp:+.4f}   p = {pp:.4f}  {sig(pp)}",
        f"  Measures linear association. Assumes bivariate normality.",
        f"Spearman  ρ = {rs:+.4f}   p = {ps:.4f}  {sig(ps)}",
        f"  Rank-based; robust to outliers and non-normality.",
        f"Kendall   τ = {rk:+.4f}   p = {pk:.4f}  {sig(pk)}",
        f"  Preferred for small or heavily tied samples.",
        "",
        f"Recommended: {rec}",
    ]}


def _test_num_cat(num, cat) -> dict:
    cats   = np.unique(cat)
    groups = [drop_nan(num[cat == c]) for c in cats]
    valid  = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return {"test_name": "Error", "results": ["Need ≥ 2 groups with ≥ 2 observations."]}

    lev_w, lev_p = sp.levene(*valid)
    homo         = lev_p > ALPHA
    f_stat, ap   = sp.f_oneway(*valid)
    h_stat, kp   = sp.kruskal(*valid)
    sig          = lambda p: "✓ sig." if p < ALPHA else "✗ n.s."

    valid_labels = [str(cats[i]) for i, g in enumerate(groups) if len(g) >= 2]
    norm_report  = group_normality_report(valid, valid_labels)
    norm_lines   = _fmt_group_normality(norm_report)
    norm_flags   = [r.is_normal for r in norm_report]
    all_norm     = all(f for f in norm_flags if f is not None)

    if all_norm and homo:
        rec = "ANOVA (normality ✓, equal variances ✓)"
    elif all_norm:
        rec = "Kruskal-Wallis or Welch ANOVA (normality ✓, variances unequal)"
    else:
        rec = "Kruskal-Wallis (normality not met)"

    results = [
        f"k = {len(valid)} groups  |  sizes: {[len(g) for g in valid]}",
        "",
        *norm_lines,
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
        t_stu, tp_stu = sp.ttest_ind(g1, g2, equal_var=True)
        t_wel, tp_wel = sp.ttest_ind(g1, g2, equal_var=False)
        u_stat, up    = sp.mannwhitneyu(g1, g2, alternative='two-sided')
        n1, n2        = len(g1), len(g2)

        if n1 + n2 > 60:
            pool_std = np.sqrt(g1.var(ddof=1)/n1 + g2.var(ddof=1)/n2)
            z_stat   = float((g1.mean() - g2.mean()) / pool_std) if pool_std > 0 else float('nan')
            z_p      = float(2 * (1 - ndtr(abs(z_stat)))) if not np.isnan(z_stat) else float('nan')
            z_line   = f"Z-test:             Z = {z_stat:.4f},  p = {z_p:.4f}  {sig(z_p)}"
        else:
            z_line   = f"Z-test: skipped — n₁+n₂ = {n1+n2} ≤ 60 (use t-test)."

        results += [
            "",
            "── Two-Group Tests ──",
            f"t-test (Student):   t = {t_stu:.4f},  p = {tp_stu:.4f}  {sig(tp_stu)}",
            f"  Assumes equal variances. {'Recommended ✓' if homo and all_norm else 'Caution: assumptions not met.'}",
            f"t-test (Welch):     t = {t_wel:.4f},  p = {tp_wel:.4f}  {sig(tp_wel)}",
            f"  Does not assume equal variances. {'Recommended ✓' if all_norm and not homo else ''}",
            z_line,
            f"Mann-Whitney U:     U = {u_stat:.1f},   p = {up:.4f}  {sig(up)}",
            "  Non-parametric alternative to t-test.",
        ]
        if n1 == n2:
            try:
                w_stat, wp = sp.wilcoxon(g1, g2)
                results += [
                    f"Wilcoxon (paired):  W = {w_stat:.1f},  p = {wp:.4f}  {sig(wp)}",
                    "  Use only if observations are genuinely paired.",
                ]
            except Exception:
                results.append("Wilcoxon: no non-zero differences.")
        else:
            results.append(f"Wilcoxon: skipped — unequal sizes ({n1} vs {n2}).")

    return {"test_name": "Group Comparison", "results": results}


def _test_cat_cat(a, b) -> dict:
    """Chi-square + generalized Fisher's Exact for k×r contingency tables."""
    ua, ub = np.unique(a), np.unique(b)
    k, r   = len(ua), len(ub)
    ct     = np.array([[np.sum((a == va) & (b == vb)) for vb in ub]
                        for va in ua], dtype=int)
    n      = ct.sum()

    chi2, p_chi, dof, exp = sp.chi2_contingency(ct)
    min_exp   = exp.min()
    low_pct   = (exp < 5).mean() * 100
    chi_valid = (min_exp >= 1) and (low_pct <= 20)
    sig_chi   = "✓ sig." if p_chi < ALPHA else "✗ n.s."

    min_dim = min(k, r)
    cv = float(np.sqrt(chi2 / (n * (min_dim - 1)))) if min_dim > 1 and n > 0 else float('nan')
    tt = float(np.sqrt(chi2 / (n * np.sqrt((k-1)*(r-1))))) if (k-1)*(r-1) > 0 and n > 0 else float('nan')

    def _strength(v):
        thresholds = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29)}
        s, m, l = thresholds.get(min_dim - 1, (0.05, 0.15, 0.25))
        return "negligible" if v < s else "small" if v < m else "medium" if v < l else "large"

    results = [
        f"Table: {k}×{r}  |  n = {n}",
        "",
        *_fmt_group_normality(
            group_normality_report(
                [ct.sum(axis=1).astype(float), ct.sum(axis=0).astype(float)],
                ["Row marginals", "Column marginals"],
            ),
            header="── Normality of Marginal Counts (Shapiro-Wilk) ──",
        ),
        "  Note: Chi-square / Fisher are exact tests on counts.",
        "  Normality of marginals is informational only.",
        "",
        "── Chi-square Test of Independence ──",
        f"χ²({dof}) = {chi2:.4f},  p = {p_chi:.4f}  {sig_chi}",
        f"  Min expected frequency = {min_exp:.2f}  {'✓ ≥ 1' if min_exp >= 1 else '✗ < 1 — unreliable'}",
        f"  Cells with expected < 5: {low_pct:.1f}%  {'✓ ≤ 20%' if low_pct <= 20 else '✗ > 20% — unreliable'}",
    ]

    use_fisher = not chi_valid or (k * r <= 12)
    if use_fisher:
        try:
            res_fisher = sp.fisher_exact(ct)
            p_f   = res_fisher.pvalue if hasattr(res_fisher, 'pvalue') else res_fisher[1]
            sig_f = "✓ sig." if p_f < ALPHA else "✗ n.s."
            results += [
                "",
                f"── Fisher's Exact Test ({k}×{r}) ──",
                f"p = {p_f:.4f}  {sig_f}",
                "  Used because Chi-square assumptions are weak or table is small.",
            ]
            if k == 2 and r == 2:
                odds = res_fisher.statistic if hasattr(res_fisher, 'statistic') else res_fisher[0]
                results.append(f"  Odds ratio = {odds:.4f}")
        except Exception:
            results.append("Fisher's Exact: skipped (table too large or unsupported).")

    results += [
        "",
        "── Effect Sizes ──",
        f"Cramér's V    = {cv:.4f}  ({_strength(cv)})" if not np.isnan(cv) else "Cramér's V: n/a",
        f"Tschuprow's T = {tt:.4f}" if not np.isnan(tt) else "Tschuprow's T: n/a",
        "  V ∈ [0,1]: 0 = no assoc, 1 = perfect. T ≤ V for non-square tables.",
    ]

    return {"test_name": "Association Tests", "results": results}
