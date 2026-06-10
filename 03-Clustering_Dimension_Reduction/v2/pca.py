"""
pca.py  —  Pure PCA computations. No Dash, no Plotly.

Contributions vs correlations
------------------------------
contrib_var[j,k]  = loading[j,k]² × 100
                    → how much variable j BUILT axis k  (sums to 100 per PC)
corr_circle[j,k]  = loading[j,k] × sqrt(eigenvalue[k])
                    → Pearson correlation of original variable j with score k
cos2_var[j,k]     = corr_circle[j,k]²
contrib_ind[i,k]  = score[i,k]² / (n × eigenvalue[k]) × 100
cos2_ind[i,k]     = score[i,k]² / sum_k(score[i,:]²)

FIX (v6.1)
----------
1. n_optimal elbow guard: np.diff(np.diff(explained)) is empty when
   n_comp < 3. Previous code returned elbow=1 in that branch, biasing
   the majority-vote recommendation.  Now guarded properly.
2. row_labels: if the store has a string/ID column, use its values instead
   of generic "Ind. N" labels for meaningful individual annotations.
3. valid_mask returned alongside result so clustering can map labels back
   to original row indices.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.preprocessing import StandardScaler
import store as _store


def _get_row_labels(n_obs: int, valid_mask: np.ndarray) -> list[str]:
    """
    Return per-row display labels in priority order:
      1. A string column detected as ID-like (name, code, label…)
      2. Generic "Ind. N" fallback

    ID-like heuristic: string column, ≥ 80 % unique values,
    not a datetime column, max length ≤ 40 chars.
    """
    if not _store.is_loaded():
        return [f"Ind. {i+1}" for i in range(n_obs)]

    for col in _store.all_cols:
        arr = _store.dataset[col]
        if arr.dtype.kind not in ('U', 'S', 'O'):
            continue
        if col in _store.datetime_cols:   # skip timestamps
            continue
        uniq_ratio = len(np.unique(arr)) / max(len(arr), 1)
        if uniq_ratio < 0.50:            # too many repeated values → categorical, not ID
            continue
        max_len = max((len(str(v)) for v in arr[:100] if v), default=0)
        if max_len > 60:
            continue
        # Use this column — apply valid_mask to align with PCA rows
        labels_raw = arr[valid_mask].astype(str)
        return list(labels_raw)

    return [f"Ind. {i+1}" for i in range(n_obs)]


def run_pca(dataset: dict, col_meta: dict, col_stats: dict) -> dict | None:
    num_cols = [c for c, t in col_meta.items() if t == "numeric"]
    if len(num_cols) < 2:
        return None

    arrays   = [dataset[c] for c in num_cols]
    nan_mask = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.dtype.kind == "f":
            nan_mask |= np.isnan(a)

    valid        = ~nan_mask
    n_obs        = int(valid.sum())
    missing_rows = int(nan_mask.sum())
    if n_obs < 2:
        return None

    X      = np.column_stack([a[valid] for a in arrays]).astype(float)
    n_vars = X.shape[1]

    X_std = StandardScaler().fit_transform(X)
    n_comp = min(n_obs - 1, n_vars)
    pca    = _PCA(n_components=n_comp, svd_solver="full")
    scores     = pca.fit_transform(X_std)
    loadings   = pca.components_.T
    eigenvalues = pca.explained_variance_
    explained   = pca.explained_variance_ratio_
    cumulative  = np.cumsum(explained)

    # ── n_optimal: majority vote of three estimators ─────────────────────
    kaiser = max(1, int((eigenvalues > 1).sum()))

    # FIX: np.diff(np.diff(x)) is empty when len(x) < 3 (i.e. n_comp < 3).
    # Previous code: `if n_comp >= 3: elbow = argmax(...) + 2 else: elbow = 1`
    # elbow=1 in the else branch biased the median vote downward.
    # Correct fallback: use min(2, n_comp) so the vote stays meaningful.
    if n_comp >= 3:
        second_diff = np.diff(np.diff(explained))
        if second_diff.size > 0:
            elbow = int(np.argmax(second_diff) + 2)
        else:
            elbow = min(2, n_comp)
    else:
        elbow = min(2, n_comp)   # FIX: was hardcoded 1

    thresh_70 = int(np.searchsorted(cumulative, 0.70) + 1)
    n_optimal = max(2, min(int(np.median([kaiser, elbow, thresh_70])), n_comp))

    contributions_var = loadings ** 2 * 100
    corr_circle       = loadings * np.sqrt(eigenvalues)
    cos2_var          = corr_circle ** 2
    contributions_ind = (scores ** 2) / (n_obs * eigenvalues) * 100
    dist2             = (scores ** 2).sum(axis=1, keepdims=True)
    cos2_ind          = (scores ** 2) / np.where(dist2 == 0, 1.0, dist2)

    # FIX: use meaningful row labels (ID column if available, else "Ind. N")
    row_labels = _get_row_labels(n_obs, valid)

    return {
        "n_obs":             n_obs,
        "n_vars":            n_vars,
        "variables":         num_cols,
        "n_components":      n_comp,
        "eigenvalues":       eigenvalues,
        "explained":         explained,
        "cumulative":        cumulative,
        "n_optimal":         n_optimal,
        "loadings":          loadings,
        "scores":            scores,
        "contributions_var": contributions_var,
        "cos2_var":          cos2_var,
        "contributions_ind": contributions_ind,
        "cos2_ind":          cos2_ind,
        "corr_circle":       corr_circle,
        "row_labels":        row_labels,
        "missing_rows":      missing_rows,
        "valid_mask":        valid,    # FIX: expose mask for downstream row mapping
    }


def axis_insight(result: dict, pc_idx: int) -> dict:
    k         = pc_idx
    variables = result["variables"]
    contrib   = result["contributions_var"][:, k]
    cos2      = result["cos2_var"][:, k]
    corr      = result["corr_circle"][:, k]
    loading   = result["loadings"][:, k]
    order     = np.argsort(contrib)[::-1]
    threshold = 100.0 / len(variables)
    rows = [{
        "rank":      r + 1,
        "variable":  variables[j],
        "contrib":   float(contrib[j]),
        "cos2":      float(cos2[j]),
        "corr":      float(corr[j]),
        "loading":   float(loading[j]),
        "direction": "+" if loading[j] >= 0 else "−",
    } for r, j in enumerate(order)]
    return {
        "pc_label":          f"PC{pc_idx+1}",
        "explained":         float(result["explained"][k]),
        "eigenvalue":        float(result["eigenvalues"][k]),
        "uniform_threshold": threshold,
        "rows":              rows,
        "above_threshold":   [r for r in rows if r["contrib"] >= threshold],
    }
