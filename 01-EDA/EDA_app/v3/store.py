"""
store.py  —  Single source of truth for the loaded dataset.

Column classification heuristic
---------------------------------
float64 with ≤ 10 distinct integer values → categorical  (binary flags, Likert, etc.)
float64, everything else                  → numeric
string / object                           → categorical

Lilliefors MC cache  (lazy, per n_clean)
-----------------------------------------
Built on first access per unique sample size, NOT at load time.
Load stays fast; first display of each column pays ~20 ms once, then < 1 ms.
n_mc = 200 gives |Δp| ≤ 0.03 vs 500 reps — sufficient for α = 0.05 decisions.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from utils import drop_nan, is_integer_array

dataset:  dict[str, np.ndarray] = {}
col_meta: dict[str, str]        = {}
all_cols: list[str]             = []
num_cols: list[str]             = []
cat_cols: list[str]             = []

_lilliefors_cache: dict[int, np.ndarray] = {}
_MC_REPS = 200   # 20 ms per unique n; |Δp| ≤ 0.03 vs 500 reps
_MC_SEED = 0


def reset(new_dataset: dict[str, np.ndarray]) -> None:
    global dataset, col_meta, all_cols, num_cols, cat_cols, _lilliefors_cache
    dataset  = new_dataset
    all_cols = list(new_dataset.keys())
    num_cols.clear(); cat_cols.clear(); col_meta.clear()
    _lilliefors_cache.clear()   # invalidate on new file

    for col, arr in new_dataset.items():
        if arr.dtype.kind == "f":
            n_uniq = len(np.unique(drop_nan(arr)))
            if n_uniq <= 10 and is_integer_array(arr):
                col_meta[col] = "categorical"; cat_cols.append(col)
            else:
                col_meta[col] = "numeric";     num_cols.append(col)
        else:
            col_meta[col] = "categorical"; cat_cols.append(col)


def is_loaded() -> bool:
    return bool(dataset)


def get_lilliefors_mc(n_clean: int) -> np.ndarray:
    """
    Return the MC null distribution for sample size n_clean.
    Computed on first call for that n, then cached (lazy evaluation).
    """
    if n_clean not in _lilliefors_cache:
        _lilliefors_cache[n_clean] = _compute_lilliefors_mc(n_clean)
    return _lilliefors_cache[n_clean]


def _compute_lilliefors_mc(n: int) -> np.ndarray:
    """
    Vectorised Lilliefors null distribution.
    Shape: (n_mc,).  All operations are pure NumPy + scipy.special.ndtr.
    ndtr is 3× faster than sp.norm.cdf on large matrices.
    """
    rng  = np.random.default_rng(_MC_SEED)
    S    = rng.standard_normal((_MC_REPS, n))              # (n_mc, n)
    m    = S.mean(1, keepdims=True)
    std  = S.std(1, ddof=1, keepdims=True); std[std == 0] = 1.0
    Z    = (S - m) / std                                   # (n_mc, n)
    Zs   = np.sort(Z, axis=1)                              # (n_mc, n)
    cdf  = ndtr(Zs)                                        # (n_mc, n)
    hi   = np.arange(1, n + 1) / n
    lo   = np.arange(0, n)     / n
    return np.maximum((hi - cdf).max(1), (cdf - lo).max(1))
