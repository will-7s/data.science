"""
store.py
────────
Single source of truth for the dataset loaded at runtime.

All other modules import from here so there is exactly one copy of the
data in memory and no risk of stale module-level caches.

Structure
---------
- `dataset`  : dict[str, np.ndarray]   column name → values
- `col_meta` : dict[str, Literal['numeric', 'categorical']]
- Convenience lists derived from col_meta: `all_cols`, `num_cols`, `cat_cols`

Usage
-----
    import store
    store.dataset['age']          # numpy array
    store.num_cols                # ['age', 'salary', ...]
    store.col_meta['gender']      # 'categorical'
"""

import numpy as np

# ── live state (mutated by parsers/loader on each upload) ─────────────────────

dataset: dict[str, np.ndarray] = {}
col_meta: dict[str, str] = {}        # 'numeric' | 'categorical'

all_cols: list[str] = []
num_cols:  list[str] = []
cat_cols:  list[str] = []


def reset(new_dataset: dict[str, np.ndarray]) -> None:
    """Replace the current dataset with *new_dataset* and rebuild all metadata."""
    global dataset, col_meta, all_cols, num_cols, cat_cols

    dataset  = new_dataset
    col_meta = {}
    all_cols = list(new_dataset.keys())
    num_cols.clear()
    cat_cols.clear()

    for col, arr in new_dataset.items():
        if arr.dtype.kind == 'f':          # float64 ⇒ numeric
            col_meta[col] = 'numeric'
            num_cols.append(col)
        else:                              # str/object ⇒ categorical
            col_meta[col] = 'categorical'
            cat_cols.append(col)


def is_loaded() -> bool:
    return bool(dataset)
