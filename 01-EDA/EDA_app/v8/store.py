"""
store.py  —  Single source of truth for the loaded dataset.

Column classification heuristic
--------------------------------
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
col_stats    : {'n_clean', 'n_nan'} per column.
_corr_matrix_cache : pairwise Pearson matrix, computed once.
_lilliefors_cache : MC null distributions, lazy per unique n.
"""
from __future__ import annotations
import re
import threading
import numpy as np
from scipy.special import ndtr
import config
from utils import drop_nan, is_integer_array

dataset:       dict[str, np.ndarray] = {}
clean_arrays:  dict[str, np.ndarray] = {}
source_name:   str                    = ""
col_stats:     dict[str, dict]       = {}
col_meta:      dict[str, str]        = {}
all_cols:      list[str]             = []
num_cols:      list[str]             = []
cat_cols:      list[str]             = []

_metadata_cache:    dict | None            = None
_lilliefors_cache:  dict[int, np.ndarray] = {}
_corr_matrix_cache: tuple | None          = None
_plot_cache:        dict                   = {}
_PLOT_CACHE_MAX = 50

cancel_event = threading.Event()

lightweight_mode: bool = False
_slow_count: int = 0


def track_computation_speed(sampled: bool) -> bool:
    global _slow_count, lightweight_mode
    if sampled:
        _slow_count += 1
        if _slow_count >= config.LIGHTWEIGHT_SLOW_COUNT:
            lightweight_mode = True
    else:
        _slow_count = 0
        lightweight_mode = False
    return lightweight_mode


def should_sample(arr: np.ndarray) -> bool:
    """True if array exceeds the sampling threshold."""
    return len(arr) > config.SAMPLE_THRESHOLD


_INT_KINDS  = frozenset('iu')   # signed and unsigned integers
_FLOAT_KIND = 'f'
_STR_KINDS  = frozenset('USOb') # unicode, bytes, object, bool


def _is_date_string(s: str) -> bool:
    s = str(s).strip()
    if not s or len(s) < 6:
        return False
    return bool(re.match(
        r'^\d{2,4}[-/]\d{1,2}[-/]\d{2,4}', s
    )) or bool(re.match(
        r'^\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}', s
    ))


def _is_temporal(arr: np.ndarray) -> bool:
    if arr.dtype.kind == 'M':
        return True
    if arr.dtype.kind in _STR_KINDS:
        flat = arr.ravel()
        non_empty = flat[flat != '']
        sample = non_empty[:min(20, len(non_empty))]
        if len(sample) == 0:
            return False
        matches = sum(1 for v in sample if _is_date_string(v))
        return matches / len(sample) >= 0.5
    return False


def _classify_column(arr: np.ndarray) -> str:
    """
    Return 'numeric', 'categorical', or 'temporal' for a column array.

    Handles float64, int*, uint*, bool, str/unicode, object,
    datetime64, and date-like strings.
    """
    kind = arr.dtype.kind

    # Datetime64 → temporal
    if kind == 'M':
        return 'temporal'

    # String / object / bool → check temporal first, then categorical
    if kind in _STR_KINDS:
        if _is_temporal(arr):
            return 'temporal'
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
        if n_uniq <= 10 and is_integer_array(clean):
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


def reset(new_dataset: dict[str, np.ndarray], source: str = "") -> None:
    global dataset, clean_arrays, col_stats, col_meta
    global all_cols, num_cols, cat_cols, source_name
    global _metadata_cache, _lilliefors_cache, _corr_matrix_cache

    dataset  = new_dataset
    all_cols = list(new_dataset.keys())
    source_name = source
    num_cols = []; cat_cols = []
    col_meta.clear(); clean_arrays.clear(); col_stats.clear()
    _metadata_cache = None
    _lilliefors_cache.clear()
    _corr_matrix_cache = None
    _plot_cache.clear()
    cancel_event.clear()
    global lightweight_mode, _slow_count
    lightweight_mode = False
    _slow_count = 0

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
            # Count missing before any conversion
            if arr.dtype.kind == _FLOAT_KIND:
                n_missing = int(np.isnan(arr).sum())
            elif arr.dtype.kind in ('U', 'O', 'S'):
                n_missing = int(((arr == '') | (arr == 'nan')).sum()) if arr.size else 0
            else:
                n_missing = 0

            # Categorical: store as string for uniform display
            if arr.dtype.kind in _INT_KINDS:
                str_arr = arr.astype(str)
                dataset[col] = str_arr
                clean_arrays[col] = str_arr
            else:
                clean_arrays[col] = arr
            col_stats[col] = {'n_clean': len(arr) - n_missing, 'n_nan': n_missing}
            cat_cols.append(col)


def is_loaded() -> bool:
    return bool(dataset)


def clear_cancel() -> None:
    cancel_event.clear()


def trim_plot_cache() -> None:
    if len(_plot_cache) > _PLOT_CACHE_MAX:
        keys = list(_plot_cache.keys())
        for k in keys[:len(keys) - _PLOT_CACHE_MAX]:
            _plot_cache.pop(k, None)


# ── Metadata cache ─────────────────────────────────────────────────────────────

def get_metadata() -> dict:
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    n_cols = len(all_cols)
    if not n_cols:
        _metadata_cache = {
            "n_rows": 0, "n_cols": 0, "col_types": {},
            "memory_bytes": 0, "top_missing": [], "source": source_name,
        }
        return _metadata_cache
    n_rows = len(next(iter(dataset.values())))

    col_types: dict[str, int] = {}
    for t in col_meta.values():
        col_types[t] = col_types.get(t, 0) + 1

    memory_bytes = int(sum(
        arr.nbytes + (arr.size * 64 if arr.dtype.kind == 'O' else 0)
        for arr in dataset.values()
    ))

    missing: list[tuple[str, int]] = []
    for col in all_cols:
        arr = dataset[col]
        if arr.dtype.kind == 'f':
            n_missing = int(np.isnan(arr).sum())
        elif arr.dtype.kind in ('U', 'O', 'S'):
            n_missing = int(((arr == '') | (arr == 'nan')).sum()) if arr.size else 0
        else:
            n_missing = 0
        if n_missing > 0:
            missing.append((col, n_missing))
    missing.sort(key=lambda x: x[1], reverse=True)
    top_missing = [
        {"col": col, "n_missing": n, "pct": round(100.0 * n / n_rows, 1) if n_rows else 0.0}
        for col, n in missing[:5]
    ]

    _metadata_cache = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "col_types": col_types,
        "memory_bytes": memory_bytes,
        "top_missing": top_missing,
        "source": source_name,
    }
    return _metadata_cache


# ── Lilliefors MC cache ───────────────────────────────────────────────────────

def get_lilliefors_mc(n_clean: int) -> np.ndarray:
    if n_clean not in _lilliefors_cache:
        _lilliefors_cache[n_clean] = _compute_lilliefors_mc(n_clean)
    return _lilliefors_cache[n_clean]


def _compute_lilliefors_mc(n: int) -> np.ndarray:
    rng = np.random.default_rng(config.MC_SEED)
    S   = rng.standard_normal((config.MC_REPS, n))
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
