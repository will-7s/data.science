"""
pca.py  —  Pure PCA computations. No Dash, no Plotly.

Contributions vs correlations
------------------------------
contrib_var[j,k]  = loading[j,k]² × 100
                    → how much variable j BUILT axis k  (sums to 100 per PC)
corr_circle[j,k]  = loading[j,k] × sqrt(eigenvalue[k])
                    → Pearson correlation of original variable j with score k
                    → NOT the same as contribution

cos2_var[j,k]     = corr_circle[j,k]²
                    → quality of representation of variable j on axis k

contrib_ind[i,k]  = score[i,k]² / (n × eigenvalue[k]) × 100
cos2_ind[i,k]     = score[i,k]² / sum_k(score[i,:]²)
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.preprocessing import StandardScaler


def run_pca(dataset: dict, col_meta: dict, col_stats: dict) -> dict | None:
    num_cols = [c for c, t in col_meta.items() if t == "numeric"]
    if len(num_cols) < 2:
        return None

    arrays = [dataset[c] for c in num_cols]
    nan_mask = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.dtype.kind == "f":
            nan_mask |= np.isnan(a)
    valid = ~nan_mask
    n_obs = int(valid.sum())
    missing_rows = int(nan_mask.sum())
    if n_obs < 2:
        return None

    X = np.column_stack([a[valid] for a in arrays]).astype(float)
    n_vars = X.shape[1]

    X_std = StandardScaler().fit_transform(X)
    n_comp = min(n_obs - 1, n_vars)
    pca = _PCA(n_components=n_comp, svd_solver="full")
    scores   = pca.fit_transform(X_std)
    loadings = pca.components_.T
    eigenvalues = pca.explained_variance_
    explained   = pca.explained_variance_ratio_
    cumulative  = np.cumsum(explained)

    # n_optimal: majority vote of Kaiser, scree elbow, 70% threshold
    kaiser = max(1, int((eigenvalues > 1).sum()))
    if n_comp >= 3:
        elbow = int(np.argmax(np.diff(np.diff(explained))) + 2)
    else:
        elbow = 1
    thresh_70 = int(np.searchsorted(cumulative, 0.70) + 1)
    n_optimal = max(2, min(int(np.median([kaiser, elbow, thresh_70])), n_comp))

    contributions_var = loadings**2 * 100
    corr_circle       = loadings * np.sqrt(eigenvalues)
    cos2_var          = corr_circle**2
    contributions_ind = (scores**2) / (n_obs * eigenvalues) * 100
    dist2             = (scores**2).sum(axis=1, keepdims=True)
    cos2_ind          = (scores**2) / np.where(dist2 == 0, 1.0, dist2)

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
        "row_labels":        [f"Ind. {i+1}" for i in range(n_obs)],
        "missing_rows":      missing_rows,
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
