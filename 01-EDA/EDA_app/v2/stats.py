"""
stats.py
────────
Pure statistical computations — no Dash, no Plotly, no UI concern.

Every function takes plain NumPy arrays and returns plain Python values
(floats, strings, dicts).  This makes them trivially unit-testable.

Public API
----------
normality_test(arr)                        → (statistic, p_value)
normality_label(p_value)                   → human-readable string
descriptive_stats(arr)                     → dict of scalar stats
outlier_percentage(arr)                    → float (0–100)
group_means(num_arr, cat_arr)              → dict[category, mean]
correlation_matrix(col_data, num_cols)     → (matrix, col_names) | (None, None)
bivariate_test(var1, var2, col_data, meta) → dict with test_name + results list
"""

import numpy as np
from scipy import stats as sp


# ── normality ─────────────────────────────────────────────────────────────────

def normality_test(arr: np.ndarray) -> tuple[float | None, float | None]:
    """
    Shapiro-Wilk test.  Returns (statistic, p_value).
    Returns (None, None) when the test cannot be applied.
    """
    clean = _drop_nan(arr)
    if len(clean) < 3 or len(clean) > 5000:
        return None, None
    try:
        return sp.shapiro(clean)
    except Exception:
        return None, None


def normality_label(p_value: float | None) -> str:
    """Convert a p-value to a readable normality verdict."""
    if p_value is None:
        return "Insufficient data or Numerical instability"
    return f"Normal (p={p_value:.3f})" if p_value > 0.05 else f"Non-normal (p={p_value:.3f})"


# ── descriptive stats ─────────────────────────────────────────────────────────

def descriptive_stats(arr: np.ndarray) -> dict:
    """Return key descriptive statistics for a numeric column."""
    clean = _drop_nan(arr)
    return {
        'mean':   float(np.mean(clean)),
        'median': float(np.median(clean)),
        'std':    float(np.std(clean)),
        'min':    float(np.min(clean)),
        'max':    float(np.max(clean)),
        'n':      int(len(clean)),
    }


def outlier_percentage(arr: np.ndarray) -> float:
    """IQR-based outlier rate (×1.5 rule), returned as 0–100 %."""
    clean = _drop_nan(arr)
    if len(clean) == 0:
        return 0.0
    q1, q3 = np.quantile(clean, [0.25, 0.75])
    iqr = q3 - q1
    outliers = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
    return float(outliers.mean() * 100)


# ── group stats ───────────────────────────────────────────────────────────────

def group_means(num_arr: np.ndarray, cat_arr: np.ndarray) -> dict[str, float]:
    """Mean of *num_arr* for each category in *cat_arr*."""
    result = {}
    for cat in np.unique(cat_arr):
        values = _drop_nan(num_arr[cat_arr == cat])
        result[str(cat)] = float(np.mean(values)) if len(values) > 0 else float('nan')
    return result


# ── correlation matrix ────────────────────────────────────────────────────────

def correlation_matrix(
    col_data: dict,
    num_cols: list[str],
) -> tuple[np.ndarray | None, list[str] | None]:
    """
    Pairwise Pearson correlation for all numeric columns.
    Returns (matrix, column_names) or (None, None) if < 2 numeric columns.
    """
    if len(num_cols) < 2:
        return None, None

    n = len(num_cols)
    matrix = np.eye(n)    # diagonal = 1 by definition

    for i in range(n):
        for j in range(i + 1, n):
            a, b = col_data[num_cols[i]], col_data[num_cols[j]]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() > 1:
                r = np.corrcoef(a[mask], b[mask])[0, 1]
            else:
                r = 0.0
            matrix[i, j] = matrix[j, i] = r

    return matrix, num_cols


# ── bivariate tests ───────────────────────────────────────────────────────────

def bivariate_test(
    var1: str,
    var2: str,
    col_data: dict,
    col_meta: dict,
) -> dict:
    """
    Choose and run the most appropriate test for the variable pair.

    Returns
    -------
    dict with keys:
        test_name : str
        results   : list[str]   — one line per statistic
    """
    t1 = col_meta.get(var1, 'categorical')
    t2 = col_meta.get(var2, 'categorical')
    d1, d2 = col_data[var1], col_data[var2]

    # numeric × numeric  →  Pearson + Spearman
    if t1 == 'numeric' and t2 == 'numeric':
        mask = ~(np.isnan(d1) | np.isnan(d2))
        a, b = d1[mask], d2[mask]
        if len(a) < 2:
            return _error("Not enough paired observations.")

        pearson_r,  pearson_p  = sp.pearsonr(a, b)
        spearman_r, spearman_p = sp.spearmanr(a, b)
        _, norm_p = normality_test(a)
        recommended = "Pearson" if (norm_p or 0) > 0.05 else "Spearman"

        return {
            'test_name': 'Correlation Analysis',
            'results': [
                f"Pearson  r = {pearson_r:.3f}  (p = {pearson_p:.4f})",
                f"Spearman r = {spearman_r:.3f}  (p = {spearman_p:.4f})",
                f"Recommended: {recommended}",
            ],
        }

    # numeric × categorical  →  ANOVA + Kruskal-Wallis
    if t1 != t2:
        num_arr = d1 if t1 == 'numeric' else d2
        cat_arr = d2 if t1 == 'numeric' else d1

        groups = [
            _drop_nan(num_arr[cat_arr == cat])
            for cat in np.unique(cat_arr)
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return _error("Need at least 2 non-empty groups.")

        f_stat,  anova_p   = sp.f_oneway(*groups)
        h_stat,  kruskal_p = sp.kruskal(*groups)
        recommended = "ANOVA" if kruskal_p > 0.05 else "Kruskal-Wallis"

        return {
            'test_name': 'Group Comparison',
            'results': [
                f"ANOVA          F = {f_stat:.3f}  (p = {anova_p:.4f})",
                f"Kruskal-Wallis H = {h_stat:.3f}  (p = {kruskal_p:.4f})",
                f"Recommended: {recommended}",
            ],
        }

    # categorical × categorical  →  Chi-square + Cramér's V
    unique1, unique2 = np.unique(d1), np.unique(d2)
    contingency = np.array([
        [np.sum((d1 == v1) & (d2 == v2)) for v2 in unique2]
        for v1 in unique1
    ])

    chi2, p_value, dof, _ = sp.chi2_contingency(contingency)
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0.0

    return {
        'test_name': 'Chi-square Test',
        'results': [
            f"Chi-square = {chi2:.3f}  (p = {p_value:.4f},  dof = {dof})",
            f"Cramér's V = {cramers_v:.3f}",
            f"Association: {'Significant' if p_value < 0.05 else 'Not significant'}",
        ],
    }


# ── private helpers ───────────────────────────────────────────────────────────

def _drop_nan(arr: np.ndarray) -> np.ndarray:
    """Return arr with NaN removed (safe for non-float arrays too)."""
    if arr.dtype.kind == 'f':
        return arr[~np.isnan(arr)]
    return arr


def _error(msg: str) -> dict:
    return {'test_name': 'Error', 'results': [msg]}
