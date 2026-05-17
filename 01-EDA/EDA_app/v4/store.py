"""
store.py  —  Single source of truth for the loaded dataset.

Column classification heuristic
---------------------------------
float64 with ≤ 10 distinct integer values → categorical  (binary flags, Likert, etc.)
float64, everything else                  → numeric
string / object                           → categorical

Pre-computed column cache (reset-time)
---------------------------------------
clean_arrays : NaN-stripped arrays, computed once at load time.
col_stats    : {'n_clean', 'n_nan'} per column — avoids re-scanning on every callback.

Lilliefors MC cache  (lazy, per n_clean)
-----------------------------------------
Built on first access per unique sample size, NOT at load time.
Load stays fast; first display of each column pays ~20 ms once, then < 1 ms.
n_mc = 200 gives |Δp| ≤ 0.03 vs 500 reps — sufficient for α = 0.05 decisions.

Correlation matrix cache
------------------------
Computed once after load and invalidated on new upload.
Avoids recomputing an O(p²·n) operation on every bivariate tab interaction.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from utils import drop_nan, is_integer_array

dataset:       dict[str, np.ndarray] = {}
clean_arrays:  dict[str, np.ndarray] = {}   # NaN-stripped, computed at reset
col_stats:     dict[str, dict]       = {}   # {'n_clean', 'n_nan'} per col
col_meta:      dict[str, str]        = {}
all_cols:      list[str]             = []
num_cols:      list[str]             = []
cat_cols:      list[str]             = []

_lilliefors_cache:  dict[int, np.ndarray] = {}
_corr_matrix_cache: tuple | None          = None   # (mat, cols) or None

_MC_REPS = 200   # 20 ms per unique n; |Δp| ≤ 0.03 vs 500 reps
_MC_SEED = 0


def reset(new_dataset: dict[str, np.ndarray]) -> None:
    global dataset, clean_arrays, col_stats, col_meta
    global all_cols, num_cols, cat_cols
    global _lilliefors_cache, _corr_matrix_cache

    dataset  = new_dataset
    all_cols = list(new_dataset.keys())
    num_cols.clear(); cat_cols.clear()
    col_meta.clear(); clean_arrays.clear(); col_stats.clear()
    _lilliefors_cache.clear()
    _corr_matrix_cache = None

    for col, arr in new_dataset.items():
        # ── Pre-compute clean array once ─────────────────────────────────────
        c = drop_nan(arr)
        clean_arrays[col] = c
        col_stats[col]    = {'n_clean': len(c), 'n_nan': len(arr) - len(c)}

        # ── Column classification ─────────────────────────────────────────────
        if arr.dtype.kind == "f":
            n_uniq = len(np.unique(c))
            if n_uniq <= 10 and is_integer_array(arr):
                col_meta[col] = "categorical"; cat_cols.append(col)
            else:
                col_meta[col] = "numeric";     num_cols.append(col)
        else:
            col_meta[col] = "categorical"; cat_cols.append(col)


def is_loaded() -> bool:
    return bool(dataset)


# ── Lilliefors MC cache ───────────────────────────────────────────────────────

def get_lilliefors_mc(n_clean: int) -> np.ndarray:
    """Return the MC null distribution for n_clean (lazy, cached)."""
    if n_clean not in _lilliefors_cache:
        _lilliefors_cache[n_clean] = _compute_lilliefors_mc(n_clean)
    return _lilliefors_cache[n_clean]


def _compute_lilliefors_mc(n: int) -> np.ndarray:
    """
    Vectorised Lilliefors null distribution.
    Shape: (n_mc,).  All operations are pure NumPy + scipy.special.ndtr.
    ndtr is 3× faster than sp.norm.cdf on large matrices.
    """
    rng = np.random.default_rng(_MC_SEED)
    S   = rng.standard_normal((_MC_REPS, n))           # (n_mc, n)
    m   = S.mean(1, keepdims=True)
    std = S.std(1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    Z   = (S - m) / std                                # (n_mc, n)
    Zs  = np.sort(Z, axis=1)
    cdf = ndtr(Zs)
    hi  = np.arange(1, n + 1) / n
    lo  = np.arange(0, n)     / n
    return np.maximum((hi - cdf).max(1), (cdf - lo).max(1))


# ── Correlation matrix cache ──────────────────────────────────────────────────

def get_corr_matrix() -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """
    Return the pairwise Pearson correlation matrix for all numeric columns.
    Computed once after load, then served from cache on every subsequent call.

    Vectorised: stacks all numeric clean arrays into a (p × n_min) matrix
    and calls np.corrcoef once — O(p²·n) but only paid once per dataset.
    """
    global _corr_matrix_cache
    if _corr_matrix_cache is not None:
        return _corr_matrix_cache

    cols = num_cols
    if len(cols) < 2:
        _corr_matrix_cache = (None, None)
        return None, None

    # Find the maximum common length after pairwise NaN removal.
    # Full vectorisation requires a common mask — we build the union NaN mask.
    arrays  = [dataset[c] for c in cols]
    # Union NaN mask: row is valid only if ALL numeric columns are non-NaN
    nan_mask = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.dtype.kind == 'f':
            nan_mask |= np.isnan(a)
    valid = ~nan_mask

    if valid.sum() < 2:
        _corr_matrix_cache = (None, None)
        return None, None

    # Stack into (p, n_valid) and call np.corrcoef once
    mat_data = np.vstack([a[valid] for a in arrays])   # (p, n_valid)
    mat      = np.corrcoef(mat_data)                   # (p, p)
    np.fill_diagonal(mat, 1.0)

    _corr_matrix_cache = (mat, cols)
    return mat, cols
