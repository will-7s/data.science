"""
store.py  —  Global application state (module-level singleton).

Architecture
------------
This module acts as a single source of truth for the loaded dataset.
All Dash callbacks read from / write to these module-level globals.
There is no class, no context manager, no dependency injection —
the simplicity is intentional for a single-user Dash app.

State lifecycle
---------------
  1. User uploads a file → loader.load() → store.reset(new_dataset)
  2. reset() classifies columns, pre-computes clean float arrays, clears caches
  3. PCA / clustering modules read from store.dataset / store.col_meta
  4. Results cached in _pca_cache / _clustering_cache → O(1) lookups

Column classification heuristic
-------------------------------
  - String / object / bool → categorical  (always)
  - Integer dtypes        → numeric
    UNLESS ≤ 10 distinct int values (< 50 % of n) → treat as categorical
  - Float                 → numeric
    UNLESS ≤ 10 distinct integer values → treat as categorical
    (binary flags, Likert scales, ratings)

This heuristic is conservative — it prefers "numeric" for ambiguous columns
(e.g. integer IDs with > 10 unique values stay numeric) and lets the user
decide whether they are meaningful for PCA/clustering.

Caches
------
  _pca_cache          → computed once on upload, cleared on re-upload
  _clustering_cache   → computed when user clicks "Run clustering"
                        cleared on re-upload (user must re-run)
"""

from __future__ import annotations
import numpy as np
from utils import drop_nan, is_integer_array

# ── Public state ──────────────────────────────────────────────────────────────
# Accessed directly by callbacks.py, pca.py, clustering.py via `import store`.

dataset:      dict[str, np.ndarray] = {}   # {col_name: np.ndarray}
clean_arrays: dict[str, np.ndarray] = {}   # NaN-stripped float64 for numeric
col_stats:    dict[str, dict]       = {}   # {'n_clean': …, 'n_nan': …}
col_meta:     dict[str, str]        = {}   # 'numeric' | 'categorical'
all_cols:     list[str]             = []
num_cols:     list[str]             = []   # subset of all_cols
cat_cols:     list[str]             = []   # subset of all_cols

# ── Private cache ─────────────────────────────────────────────────────────────

_pca_cache:        dict | None = None
_clustering_cache: dict | None = None

# ── Internal constants ────────────────────────────────────────────────────────

_INT_KINDS  = frozenset('iu')       # signed + unsigned integer dtypes
_FLOAT_KIND = 'f'                   # float16/32/64
_STR_KINDS  = frozenset('USOb')     # unicode, bytes, object, bool


# ── Column classification ─────────────────────────────────────────────────────

def _classify_column(arr: np.ndarray) -> str:
    """
    Return 'numeric' or 'categorical' based on dtype and cardinality.

    The logic is ordered by likelihood:
      1. String-like dtypes → categorical (fast path, no further checks).
      2. Integer dtypes → numeric, unless few distinct int values.
         Additional guard: n_uniq < 0.5 * n  prevents treating ID columns
         (e.g. 1000 unique integers in 1000 rows) as categorical.
      3. Float → categorical only if it looks like integer codes
         (all values are integers AND ≤ 10 distinct values).

    Improvement suggestion
    ----------------------
    Add a manual override mechanism so the user can re-classify a column
    from the UI if the heuristic gets it wrong (e.g. integer ID kept as numeric).
    """
    kind = arr.dtype.kind

    # ── String / object / bool → always categorical ─────────────────────────
    if kind in _STR_KINDS:
        return 'categorical'

    # ── Integer dtypes → numeric by default ─────────────────────────────────
    if kind in _INT_KINDS:
        n_uniq = len(np.unique(arr))
        if n_uniq <= 10 and n_uniq < max(0.5 * len(arr), 3):
            return 'categorical'
        return 'numeric'

    # ── Float → categorical if it looks like integer codes with few values ──
    if kind == _FLOAT_KIND:
        clean  = drop_nan(arr)
        n_uniq = len(np.unique(clean))
        if n_uniq <= 10 and is_integer_array(arr):
            return 'categorical'
        return 'numeric'

    return 'categorical'


# ── Type conversion helpers ───────────────────────────────────────────────────

def _to_float(arr: np.ndarray) -> np.ndarray:
    """
    Safe cast to float64.
    - Float dtype → no copy (.astype(copy=False) is a view if possible).
    - Integer dtype → float64 (NaN does not exist in integer arrays).
    - String / object → returned as-is (caller handles categoricals).
    """
    if arr.dtype.kind == _FLOAT_KIND:
        return arr.astype(np.float64, copy=False)
    if arr.dtype.kind in _INT_KINDS:
        return arr.astype(np.float64)
    return arr


# ── State reset ───────────────────────────────────────────────────────────────

def reset(new_dataset: dict[str, np.ndarray]) -> None:
    """
    Replace the entire application state with a new dataset.

    Called by loader.load() after parsing and deduplication.

    Operations
    ----------
    1. Store the raw dataset.
    2. For each column: classify → convert → pre-compute clean array.
    3. Clear all caches (PCA, clustering).

    Pre-computed clean arrays accelerate downstream stats/plots:
    the NaN removal happens once here, not in every callback.
    """
    global dataset, clean_arrays, col_stats, col_meta
    global all_cols, num_cols, cat_cols, _pca_cache, _clustering_cache

    # ── Store ───────────────────────────────────────────────────────────────
    dataset  = new_dataset
    all_cols = list(new_dataset.keys())

    # ── Clear mutable lists and dicts ──────────────────────────────────────
    num_cols.clear()
    cat_cols.clear()
    col_meta.clear()
    clean_arrays.clear()
    col_stats.clear()

    # ── Clear caches ───────────────────────────────────────────────────────
    _pca_cache         = None
    _clustering_cache  = None

    # ── Classify and pre-process each column ───────────────────────────────
    for col, arr in new_dataset.items():
        meta = _classify_column(arr)
        col_meta[col] = meta

        if meta == 'numeric':
            # Convert to float64 + pre-strip NaN for fast downstream access.
            farr = _to_float(arr)
            c    = drop_nan(farr)
            clean_arrays[col] = c
            col_stats[col]    = {'n_clean': len(c), 'n_nan': len(arr) - len(c)}
            num_cols.append(col)
            dataset[col] = farr           # replace with float64 version
        else:
            # Categorical: store as string for uniform display downstream.
            if arr.dtype.kind in _INT_KINDS:
                str_arr = arr.astype(str)
                dataset[col] = str_arr
                clean_arrays[col] = str_arr
            else:
                clean_arrays[col] = arr
            col_stats[col] = {'n_clean': len(arr), 'n_nan': 0}
            cat_cols.append(col)


# ── Guard ─────────────────────────────────────────────────────────────────────

def is_loaded() -> bool:
    """True if a dataset has been uploaded and parsed."""
    return bool(dataset)


# ── PCA cache accessors ───────────────────────────────────────────────────────

def get_pca_cache() -> dict | None:
    """O(1) read — PCA result or None."""
    return _pca_cache

def set_pca_cache(result: dict | None) -> None:
    """Store PCA result (called once in on_upload)."""
    global _pca_cache
    _pca_cache = result


# ── Clustering cache accessors ────────────────────────────────────────────────

def get_clustering_cache() -> dict | None:
    """O(1) read — clustering result or None."""
    return _clustering_cache

def set_clustering_cache(result: dict | None) -> None:
    """Store clustering result (called once per "Run clustering" click)."""
    global _clustering_cache
    _clustering_cache = result
