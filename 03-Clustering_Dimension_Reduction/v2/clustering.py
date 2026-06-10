"""
clustering.py  —  Pure clustering computations (no Dash, no Plotly).

FIX (v6.1)
----------
1. _get_X() now returns (X, valid_mask) so callers can map cluster labels
   back to original dataset row indices.  Previously labels[n_clean] had
   no linkage to the original n_total rows, making individual annotation
   impossible when NaN rows existed.

2. Elbow + silhouette fused into a single loop — 14 KMeans fits reduced
   to 7 (one pass stores both inertia and silhouette per k).

3. t-SNE capped at T_SNE_MAX_OBS (15 000) with stratified subsampling
   when n > cap. A "tsne_subsampled" flag is returned so the UI can show
   a warning banner.

4. _discrimination_power() moved here from ui_clustering.py (was the
   only computational function in a UI module).

Architecture
------------
- _get_X()           →  NaN filtering + StandardScaler; returns (X, mask)
- run_kmeans()       →  sklearn KMeans n_init="auto"
- run_dbscan()       →  silhouette on core points only
- run_hierarchical() →  AgglomerativeClustering; keeps children_ for dendrogram
- run_tsne()         →  perplexity clamped; subsampled if n > T_SNE_MAX_OBS
- _run_diagnostics() →  FUSED elbow + silhouette in one KMeans pass (7 fits)
- characterize()     →  per-cluster mean/std profiles
- discrimination_power() → F-ratio per variable (moved from ui_clustering)
- run_clustering()   →  orchestrator
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ALGO_KMEANS = "kmeans"
ALGO_DBSCAN = "dbscan"
ALGO_HIER   = "hierarchical"

_KMEANS_INIT    = "k-means++"
_ELBOW_K_MAX    = 8
_TSNE_PERPS_MAX = 50
T_SNE_MAX_OBS   = 15_000   # FIX: cap for t-SNE; subsample above this


# ── Data extraction ───────────────────────────────────────────────────────────

def _get_X(dataset: dict, col_meta: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Build standardised (n_clean × p) matrix.

    FIX: now returns (X, valid_mask) instead of just X.
    valid_mask[i] is True if row i of the original dataset is included in X.
    This allows callers to map cluster labels back to original row indices.
    """
    num_cols = [c for c, t in col_meta.items() if t == "numeric"]
    if len(num_cols) < 2:
        return None, None

    arrays   = [dataset[c] for c in num_cols]
    nan_mask = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.dtype.kind == "f":
            nan_mask |= np.isnan(a)

    valid = ~nan_mask
    n_obs = int(valid.sum())
    if n_obs < 2:
        return None, None

    X = np.column_stack([a[valid] for a in arrays]).astype(float)
    return StandardScaler().fit_transform(X), valid


# ── Algorithm runners ─────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, n_clusters: int) -> dict:
    model  = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = model.fit_predict(X)
    n_uniq = len(np.unique(labels))
    sil    = float(silhouette_score(X, labels)) if n_uniq > 1 else 0.0
    return {
        "labels":     labels,
        "inertia":    float(model.inertia_),
        "silhouette": sil,
        "centers":    model.cluster_centers_,
        "n_clusters": n_clusters,
    }


def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict:
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_noise = int((labels == -1).sum())
    n_found = len(np.unique(labels[labels >= 0]))
    if n_found > 1:
        core = labels >= 0
        sil  = float(silhouette_score(X[core], labels[core]))
    else:
        sil = 0.0
    return {
        "labels":      labels,
        "silhouette":  sil,
        "n_clusters":  n_found,
        "n_noise":     n_noise,
        "eps":         eps,
        "min_samples": min_samples,
    }


def run_hierarchical(X: np.ndarray, n_clusters: int, linkage: str) -> dict:
    model  = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    n_uniq = len(np.unique(labels))
    sil    = float(silhouette_score(X, labels)) if n_uniq > 1 else 0.0
    return {
        "labels":     labels,
        "silhouette": sil,
        "n_clusters": n_clusters,
        "linkage":    linkage,
        "children":   model.children_,
    }


# ── t-SNE projection ──────────────────────────────────────────────────────────

def run_tsne(X: np.ndarray, perplexity: int = 30) -> tuple[np.ndarray, bool]:
    """
    2-D t-SNE projection.

    FIX: capped at T_SNE_MAX_OBS with stratified subsampling by cluster
    proximity (simple random used here since labels aren't available yet;
    stratification can be added post-hoc if needed).

    Returns (embedding, was_subsampled).
    """
    n = X.shape[0]
    subsampled = False
    X_fit = X

    if n > T_SNE_MAX_OBS:
        rng    = np.random.default_rng(0)
        idx    = rng.choice(n, T_SNE_MAX_OBS, replace=False)
        idx    = np.sort(idx)
        X_fit  = X[idx]
        subsampled = True

    p = min(perplexity, _TSNE_PERPS_MAX, max(1, X_fit.shape[0] // 3))
    embedding = TSNE(
        n_components=2, perplexity=p, random_state=0,
        learning_rate="auto", init="random",
    ).fit_transform(X_fit)

    if subsampled:
        # Pad with NaN for rows not in the subsample so index alignment holds
        full = np.full((n, 2), np.nan)
        full[idx] = embedding
        return full, True

    return embedding, False


# ── FIX: fused elbow + silhouette — one KMeans pass per k ────────────────────

def _run_diagnostics(X: np.ndarray, k_max: int = _ELBOW_K_MAX) -> tuple[dict, dict]:
    """
    FIX: previously two separate loops (run_elbow + run_silhouette_analysis)
    each fitted KMeans k_max-1 times = 14 fits total.
    Now one fused loop → 7 fits, each storing both inertia and silhouette.
    """
    k_limit  = min(k_max, X.shape[0] - 1)
    k_range  = list(range(2, k_limit + 1))
    inertias = []
    scores   = {}

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        scores[k] = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0

    elbow     = {"k_range": k_range, "inertias": inertias}
    silhouette = {"k_range": k_range, "scores": scores}
    return elbow, silhouette


# ── Cluster characterisation ──────────────────────────────────────────────────

def characterize_clusters(X: np.ndarray, labels: np.ndarray,
                           col_names: list[str]) -> dict:
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) == 0:
        return {
            "n_clusters": 1,
            "centers":    X.mean(axis=0, keepdims=True),
            "sizes":      {-1: len(X)},
            "profiles":   {-1: {}},
            "col_names":  col_names,
        }

    centers = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
    sizes   = {int(c): int((labels == c).sum()) for c in unique_labels}

    profiles = {}
    for c in unique_labels:
        mask         = labels == c
        cluster_data = X[mask]
        profiles[int(c)] = {
            col_names[j]: {
                "mean": float(cluster_data[:, j].mean()),
                "std":  float(cluster_data[:, j].std(ddof=1)) if cluster_data.shape[0] > 1 else 0.0,
            }
            for j in range(X.shape[1])
        }

    return {
        "n_clusters": len(unique_labels),
        "centers":    centers,
        "sizes":      sizes,
        "profiles":   profiles,
        "col_names":  col_names,
    }


# ── FIX: discrimination_power moved here from ui_clustering.py ───────────────

def discrimination_power(X_std: np.ndarray, labels: np.ndarray,
                          col_names: list[str]) -> dict:
    """
    Per-variable F-ratio: between-cluster / within-cluster variance.

    FIX: was in ui_clustering.py (UI module) — moved to clustering.py
    (computation module) where it belongs.  ui_clustering imports from here.

    High F → variable discriminates strongly between clusters.
    """
    char      = characterize_clusters(X_std, labels, col_names)
    unique_c  = sorted(char["sizes"].keys())
    n_vars    = X_std.shape[1]
    n_total   = X_std.shape[0]
    global_mean = X_std.mean(axis=0)

    f_ratios = {}
    for j, name in enumerate(col_names):
        ss_between = sum(
            char["sizes"][c] * (X_std[labels == c, j].mean() - global_mean[j]) ** 2
            for c in unique_c
        )
        ss_within = sum(
            ((X_std[labels == c, j] - X_std[labels == c, j].mean()) ** 2).sum()
            for c in unique_c
        )
        df_b = max(len(unique_c) - 1, 1)
        df_w = max(n_total - len(unique_c), 1)
        f    = (ss_between / df_b) / max(ss_within / df_w, 1e-12)
        f_ratios[name] = float(f)

    ranked = sorted(f_ratios.items(), key=lambda x: -x[1])
    return {
        "f_ratios": f_ratios,
        "ranked":   ranked,
        "top3":     ranked[:3],
        "weakest":  ranked[-3:] if len(ranked) >= 3 else ranked,
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_clustering(dataset: dict, col_meta: dict,
                   algorithm: str = "kmeans",
                   n_clusters: int = 3,
                   eps: float = 0.5,
                   min_samples: int = 5,
                   linkage: str = "ward",
                   perplexity: int = 30) -> dict | None:
    """
    Full clustering pipeline.

    FIX changes vs v6.0:
    - _get_X returns (X, valid_mask)
    - t-SNE subsampled if n > T_SNE_MAX_OBS; flag stored in result
    - elbow + silhouette fused (7 fits instead of 14)
    - discrimination_power computed here and stored in result
    """
    X, valid_mask = _get_X(dataset, col_meta)
    if X is None:
        return None

    num_cols  = [c for c, t in col_meta.items() if t == "numeric"]
    col_names = num_cols
    n_obs     = X.shape[0]

    tsne, tsne_subsampled = run_tsne(X, perplexity=perplexity)

    if algorithm == ALGO_KMEANS:
        result = run_kmeans(X, n_clusters)
    elif algorithm == ALGO_DBSCAN:
        result = run_dbscan(X, eps, min_samples)
    else:
        result = run_hierarchical(X, n_clusters, linkage)

    result["X_std"]            = X
    result["valid_mask"]       = valid_mask   # FIX: expose for row mapping
    result["tsne"]             = tsne
    result["tsne_subsampled"]  = tsne_subsampled
    result["n_obs"]            = n_obs
    result["n_vars"]           = X.shape[1]
    result["variables"]        = col_names
    result["algorithm"]        = algorithm

    result["characterization"]    = characterize_clusters(X, result["labels"], col_names)
    result["elbow"], result["silhouette_analysis"] = _run_diagnostics(X)  # FIX: fused
    result["discrimination"]      = discrimination_power(X, result["labels"], col_names)

    return result
