from __future__ import annotations
import numpy as np
import config


# ── Sampling helpers ──────────────────────────────────────────────────────────

def sample_random(arrays: dict[str, np.ndarray], n: int,
                  seed: int = config.SAMPLE_SEED) -> dict[str, np.ndarray]:
    """
    Random row subsample preserving dict structure.

    Returns a dict with the same keys, each array truncated to `n` rows.
    If `n >= len(next(iter(arrays.values())))`, returns arrays as-is.
    """
    if not arrays:
        return {}
    n_rows = len(next(iter(arrays.values())))
    if n >= n_rows:
        return {k: v.copy() for k, v in arrays.items()}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, n, replace=False)
    return {k: v[idx].copy() for k, v in arrays.items()}


def sample_stratified(num_arr: np.ndarray, cat_arr: np.ndarray,
                      n_per_group: int,
                      seed: int = config.SAMPLE_SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-group stratified sampling for numeric-categorical pairs.

    Preserves all category groups. Samples `n_per_group` rows from each group.
    Returns (sampled_num, sampled_cat) with rows in group order.
    """
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(cat_arr, return_inverse=True)
    num_out, cat_out = [], []

    for group_idx in range(len(uniq)):
        mask = inverse == group_idx
        g_num = num_arr[mask]
        g_cat = cat_arr[mask]
        if len(g_num) <= n_per_group:
            num_out.append(g_num)
            cat_out.append(g_cat)
        else:
            idx = rng.choice(len(g_num), n_per_group, replace=False)
            num_out.append(g_num[idx])
            cat_out.append(g_cat[idx])

    return np.concatenate(num_out), np.concatenate(cat_out)


def sample_rows(arrays: list[np.ndarray], n: int,
                seed: int = config.SAMPLE_SEED) -> list[np.ndarray]:
    """
    Row sampling preserving indices across all arrays.

    All arrays must have the same length. Returns a list of sampled arrays,
    each with `n` rows (or the original length if `n >= len`).
    """
    if not arrays:
        return []
    n_rows = len(arrays[0])
    if n >= n_rows:
        return [a.copy() for a in arrays]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, n, replace=False)
    return [a[idx].copy() for a in arrays]
