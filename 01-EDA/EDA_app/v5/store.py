"""
store.py  —  Single source of truth for the loaded dataset.

Column classification heuristic  (v5 fix)
------------------------------------------
The previous version only checked dtype.kind == 'f' (float64).
parsers.py v5 (via pandas infer_objects) can produce int8/16/32/64
and uint8/16/32/64 columns.  These must also be treated as numeric.

Classification rules (in order):
  1. String / object dtype  → categorical  (always)
  2. Boolean dtype          → categorical  (True/False are labels)
  3. Any integer dtype      → numeric      (int8…int64, uint8…uint64)
     UNLESS ≤ 10 distinct values and looks like integer codes → categorical
  4. Float dtype            → numeric
     UNLESS ≤ 10 distinct integer values → categorical
     (binary flags, Likert 1-5, ratings)

Caches (all invalidated on new upload)
---------------------------------------
clean_arrays : NaN-stripped float arrays, pre-computed at reset time.
               For integer/bool columns, stored as float64 for uniform
               treatment in stats functions.
col_stats    : {'n_clean', 'n_nan'} per column.
_corr_matrix_cache : pairwise Pearson matrix, computed once.
_pca_cache   : result of pca.run_pca(), computed once.
_lilliefors_cache : MC null distributions, lazy per unique n.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from utils import drop_nan, is_integer_array

dataset:       dict[str, np.ndarray] = {}
clean_arrays:  dict[str, np.ndarray] = {}
col_stats:     dict[str, dict]       = {}
col_meta:      dict[str, str]        = {}
all_cols:      list[str]             = []
num_cols:      list[str]             = []
cat_cols:      list[str]             = []

_lilliefors_cache:  dict[int, np.ndarray] = {}
_corr_matrix_cache: tuple | None          = None
_pca_cache:         dict | None           = None

_MC_REPS = 200
_MC_SEED = 0

_INT_KINDS  = frozenset('iu')   # signed and unsigned integers
_FLOAT_KIND = 'f'
_STR_KINDS  = frozenset('USOb') # unicode, bytes, object, bool


def _classify_column(arr: np.ndarray) -> str:
    """
    Return 'numeric' or 'categorical' for a column array.

    Handles float64, int*, uint*, bool, str/unicode, object.
    """
    kind = arr.dtype.kind

    # String / object / bool → always categorical
    if kind in _STR_KINDS:
        return 'categorical'

    # Integer dtypes → numeric by default; categorical if few distinct int values
    # Additional guard: n_uniq < 0.5*n avoids treating a small all-different
    # integer column (e.g. IDs) as categorical.
    if kind in _INT_KINDS:
        n_uniq = len(np.unique(arr))
        if n_uniq <= 10 and n_uniq < max(0.5 * len(arr), 3):
            return 'categorical'
        return 'numeric'

    # Float → categorical if looks like integer codes with few values
    if kind == _FLOAT_KIND:
        clean  = drop_nan(arr)
        n_uniq = len(np.unique(clean))
        if n_uniq <= 10 and is_integer_array(arr):
            return 'categorical'
        return 'numeric'

    # Fallback
    return 'categorical'


def _to_float(arr: np.ndarray) -> np.ndarray:
    """
    Convert any numeric array to float64, preserving NaN for missing values.
    Integer arrays have no NaN concept — they are cast directly.
    """
    if arr.dtype.kind == _FLOAT_KIND:
        return arr.astype(np.float64, copy=False)
    if arr.dtype.kind in _INT_KINDS:
        return arr.astype(np.float64)
    return arr   # string/object: return as-is (drop_nan handles in stats)


def reset(new_dataset: dict[str, np.ndarray]) -> None:
    global dataset, clean_arrays, col_stats, col_meta
    global all_cols, num_cols, cat_cols
    global _lilliefors_cache, _corr_matrix_cache, _pca_cache

    dataset  = new_dataset
    all_cols = list(new_dataset.keys())
    num_cols.clear(); cat_cols.clear()
    col_meta.clear(); clean_arrays.clear(); col_stats.clear()
    _lilliefors_cache.clear()
    _corr_matrix_cache = None
    _pca_cache         = None

    for col, arr in new_dataset.items():
        meta = _classify_column(arr)
        col_meta[col] = meta

        # Pre-compute clean float array for numeric columns
        if meta == 'numeric':
            farr = _to_float(arr)
            c    = drop_nan(farr)
            clean_arrays[col] = c
            col_stats[col]    = {'n_clean': len(c), 'n_nan': len(arr) - len(c)}
            num_cols.append(col)
            # Ensure dataset stores float64 for consistent downstream use
            dataset[col] = farr
        else:
            # Categorical: store as string for uniform display
            if arr.dtype.kind in _INT_KINDS:
                # Integer-coded categorical (e.g. quality 1-10)
                str_arr = arr.astype(str)
                dataset[col] = str_arr
                clean_arrays[col] = str_arr
            else:
                clean_arrays[col] = arr
            col_stats[col] = {'n_clean': len(arr), 'n_nan': 0}
            cat_cols.append(col)


def is_loaded() -> bool:
    return bool(dataset)


# ── Lilliefors MC cache ───────────────────────────────────────────────────────

def get_lilliefors_mc(n_clean: int) -> np.ndarray:
    if n_clean not in _lilliefors_cache:
        _lilliefors_cache[n_clean] = _compute_lilliefors_mc(n_clean)
    return _lilliefors_cache[n_clean]


def _compute_lilliefors_mc(n: int) -> np.ndarray:
    rng = np.random.default_rng(_MC_SEED)
    S   = rng.standard_normal((_MC_REPS, n))
    m   = S.mean(1, keepdims=True)
    std = S.std(1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    Z   = (S - m) / std
    Zs  = np.sort(Z, axis=1)
    cdf = ndtr(Zs)
    hi  = np.arange(1, n + 1) / n
    lo  = np.arange(0, n)     / n
    return np.maximum((hi - cdf).max(1), (cdf - lo).max(1))


# ── Correlation matrix cache ──────────────────────────────────────────────────

def get_corr_matrix() -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    global _corr_matrix_cache
    if _corr_matrix_cache is not None:
        return _corr_matrix_cache

    cols = num_cols
    if len(cols) < 2:
        _corr_matrix_cache = (None, None)
        return None, None

    arrays   = [dataset[c] for c in cols]
    nan_mask = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.dtype.kind == 'f':
            nan_mask |= np.isnan(a)
    valid = ~nan_mask

    if valid.sum() < 2:
        _corr_matrix_cache = (None, None)
        return None, None

    mat_data = np.vstack([a[valid] for a in arrays])
    mat      = np.corrcoef(mat_data)
    np.fill_diagonal(mat, 1.0)
    _corr_matrix_cache = (mat, cols)
    return mat, cols
