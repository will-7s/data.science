"""
store.py  —  Global application state (module-level singleton).

FIX (v6.1)
----------
1. datetime_cols list exposed so pca.py can exclude timestamps from
   row-label detection (was missing in v6.0).
2. _id_col heuristic: excludes datetime-like columns from ID candidates
   (same fix as AB project — timestamps are high-cardinality but not IDs).

Architecture note
-----------------
This is a single-user singleton.  For multi-user / Gunicorn deployment,
wrap state in Flask session or dcc.Store(storage_type="session").
"""
from __future__ import annotations
import numpy as np
from utils import drop_nan

# ── Public state ──────────────────────────────────────────────────────────────
dataset:      dict[str, np.ndarray] = {}
clean_arrays: dict[str, np.ndarray] = {}
col_stats:    dict[str, dict]       = {}
col_meta:     dict[str, str]        = {}
all_cols:     list[str]             = []
num_cols:     list[str]             = []
cat_cols:     list[str]             = []
datetime_cols: list[str]            = []   # FIX: exposed for pca.py row-label detection

_pca_cache:        dict | None = None
_clustering_cache: dict | None = None

_INT_KINDS  = frozenset('iu')
_FLOAT_KIND = 'f'
_STR_KINDS  = frozenset('USOb')


def _classify_column(arr: np.ndarray, int_tol: float = 1e-9) -> str:
    kind = arr.dtype.kind
    if kind in _STR_KINDS:
        return 'categorical'
    if kind in _INT_KINDS:
        n_uniq = len(np.unique(arr))
        if n_uniq <= 10 and n_uniq < max(0.5 * len(arr), 3):
            return 'categorical'
        return 'numeric'
    if kind == _FLOAT_KIND:
        clean = drop_nan(arr)
        uniq = np.unique(clean)
        n_uniq = len(uniq)
        if n_uniq <= 10 and n_uniq > 0 and bool(np.all(np.abs(uniq % 1.0) < int_tol)):
            return 'categorical'
        return 'numeric'
    return 'categorical'


def _is_datetime_col(arr: np.ndarray) -> bool:
    """Detect datetime-like columns (numpy M dtype or parseable strings)."""
    import warnings
    import pandas as pd
    if arr.dtype.kind == 'M':
        return True
    if arr.dtype.kind in ('U', 'O'):
        sample = [str(x) for x in arr[:50] if x is not None and str(x).strip()]
        if not sample:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample[:20])
            return True
        except Exception:
            return False
    return False


def _to_float(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == _FLOAT_KIND:
        return arr.astype(np.float64, copy=False)
    if arr.dtype.kind in _INT_KINDS:
        return arr.astype(np.float64)
    return arr


def reset(new_dataset: dict[str, np.ndarray]) -> None:
    global dataset, clean_arrays, col_stats, col_meta
    global all_cols, num_cols, cat_cols, datetime_cols
    global _pca_cache, _clustering_cache

    dataset  = new_dataset
    all_cols = list(new_dataset.keys())

    num_cols.clear()
    cat_cols.clear()
    col_meta.clear()
    clean_arrays.clear()
    col_stats.clear()
    datetime_cols.clear()   # FIX: clear on reset

    _pca_cache        = None
    _clustering_cache = None

    for col, arr in new_dataset.items():
        # FIX: detect datetime before classifying, so pca.py can skip them
        if _is_datetime_col(arr):
            datetime_cols.append(col)

        meta = _classify_column(arr)
        col_meta[col] = meta

        if meta == 'numeric':
            farr = _to_float(arr)
            c    = drop_nan(farr)
            clean_arrays[col] = c
            col_stats[col]    = {'n_clean': len(c), 'n_nan': len(arr) - len(c)}
            num_cols.append(col)
            dataset[col] = farr
        else:
            if arr.dtype.kind in _INT_KINDS:
                str_arr = arr.astype(str)
                dataset[col] = str_arr
                clean_arrays[col] = str_arr
            else:
                clean_arrays[col] = arr
            col_stats[col] = {'n_clean': len(arr), 'n_nan': 0}
            cat_cols.append(col)


def is_loaded() -> bool:
    return bool(dataset)


def get_pca_cache() -> dict | None:
    return _pca_cache

def set_pca_cache(result: dict | None) -> None:
    global _pca_cache
    _pca_cache = result


def get_clustering_cache() -> dict | None:
    return _clustering_cache

def set_clustering_cache(result: dict | None) -> None:
    global _clustering_cache
    _clustering_cache = result
