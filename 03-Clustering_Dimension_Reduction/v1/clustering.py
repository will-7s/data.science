"""
clustering.py  —  Pure clustering computations (no Dash, no Plotly).

Architecture
------------
- _get_X()          →  one-pass NaN filtering + StandardScaler
- run_kmeans()      →  sklearn KMeans with n_init="auto" (avoids deprecation)
- run_dbscan()      →  silhouette computed on core points only (excludes noise)
- run_hierarchical()→  AgglomerativeClustering, keeps children_ for dendrogram
- run_tsne()        →  perplexity clamped to [1, n//3] for stability
- run_elbow()       →  sequential KMeans over k ∈ [2, k_max]
- characterize()    →  per-cluster mean/std profiles, vectorised axis loop

All public functions accept/return plain dicts — no Dash, no Plotly.
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ── Algorithm constants ───────────────────────────────────────────────────────
# Used by callbacks.py and charts_clustering.py to dispatch correctly.
ALGO_KMEANS = "kmeans"
ALGO_DBSCAN = "dbscan"
ALGO_HIER   = "hierarchical"

# ── Global defaults ──────────────────────────────────────────────────────────
# Lowering k_max from 10→8 keeps the elbow/silhouette scan fast for small data
# while covering the range users typically explore.
_KMEANS_INIT      = "k-means++"            # default, explicit for clarity
_ELBOW_K_MAX      = 8                      # max k for elbow + silhouette scans
_TSNE_PERPS_MAX   = 50                     # upper bound, clamped below to n//3


def _get_X(dataset: dict, col_meta: dict) -> np.ndarray | None:
    """
    Build a standardised observation matrix from the store's dataset.
    -----------------------------------------------------------------
    Pipeline (one pass over columns):
      1. Collect numeric columns from col_meta.
      2. Build a single NaN mask via bitwise OR across columns.
      3. Column-stack the NaN-filtered arrays into (n_clean, n_vars).
      4. Standardise (mean=0, std=1) so Euclidean geometry is meaningful.

    Optimisation note
    -----------------
    The NaN mask loop is O(n × n_vars) in Python but each iteration is
    a fully vectorised NumPy operation (C-level).  For typical EDA
    datasets (n < 5e5, n_vars < 50) this completes in < 1 ms.
    If n_vars > 500 (wider than tall data), consider a transpose-first
    approach — but that is an edge case that would also break PCA.

    Improvement suggestion
    ----------------------
    Return the row mask along with X so callers can map cluster labels
    back to original row indices (needed for row-level annotation).
    """
    # ── Gather numeric columns ───────────────────────────────────────────────
    num_cols = [c for c, t in col_meta.items() if t == "numeric"]
    if len(num_cols) < 2:
        return None

    # ── Single-pass NaN mask ─────────────────────────────────────────────────
    arrays  = [dataset[c] for c in num_cols]
    nan_mask = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.dtype.kind == "f":
            nan_mask |= np.isnan(a)      # C-level OR — no Python loop per row

    valid  = ~nan_mask
    n_obs  = int(valid.sum())
    if n_obs < 2:
        return None

    # ── Column stack + standardisation ───────────────────────────────────────
    X = np.column_stack([a[valid] for a in arrays]).astype(float)
    return StandardScaler().fit_transform(X)


# ── Algorithm runners ─────────────────────────────────────────────────────────
# Each runner returns a dict that *must* contain at least {"labels", "silhouette"}.
# The calling orchestrator (run_clustering) merges cross-cutting fields.

def run_kmeans(X: np.ndarray, n_clusters: int) -> dict:
    """K-Means clustering with n_init="auto" (sklearn ≥ 1.2)."""
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = model.fit_predict(X)

    # Silhouette requires ≥ 2 clusters and ≥ 2 samples per cluster.
    n_unique = len(np.unique(labels))
    sil = float(silhouette_score(X, labels)) if n_unique > 1 else 0.0

    return {
        "labels":     labels,
        "inertia":    float(model.inertia_),
        "silhouette": sil,
        "centers":    model.cluster_centers_,
        "n_clusters": n_clusters,
    }


def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict:
    """
    DBSCAN density-based clustering.

    Silhouette is computed on **core points only** (labels ≥ 0) because
    sklearn's silhouette_score does not accept noise (-1) labels.
    This gives a cleaner quality metric of the actual cluster structure.

    Improvement suggestion
    ----------------------
    If n_noise is large (> 30 %), silhouette becomes misleading.
    Expose the Davies-Bouldin index (no ground-truth needed) as a
    secondary quality metric for DBSCAN in the UI.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_noise = int((labels == -1).sum())
    n_found = len(np.unique(labels[labels >= 0]))

    # ── Silhouette on core points only ──────────────────────────────────────
    if n_found > 1:
        core  = labels >= 0
        sil   = float(silhouette_score(X[core], labels[core]))
    else:
        sil   = 0.0

    return {
        "labels":      labels,
        "silhouette":  sil,
        "n_clusters":  n_found,
        "n_noise":     n_noise,
        "eps":         eps,
        "min_samples": min_samples,
    }


def run_hierarchical(X: np.ndarray, n_clusters: int, linkage: str) -> dict:
    """
    Agglomerative hierarchical clustering.

    Linkage choices:
      - 'ward'     : minimises within-cluster variance (default; Euclidean only)
      - 'complete' : max pairwise distance
      - 'average'  : average pairwise distance (UPGMA)
      - 'single'   : nearest-neighbour (chaining effect)

    The `children_` attribute is kept for potential dendrogram rendering
    (not wired in the current UI but available via charts_clustering).
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    n_unique = len(np.unique(labels))
    sil = float(silhouette_score(X, labels)) if n_unique > 1 else 0.0

    return {
        "labels":     labels,
        "silhouette": sil,
        "n_clusters": n_clusters,
        "linkage":    linkage,
        "children":   model.children_,         # shape (n-1, 2)
    }


# ── t-SNE projection ──────────────────────────────────────────────────────────

def run_tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """
    2-D t-SNE projection for cluster visualisation.

    Perplexity is clamped:
      - upper bound = min(user_requested, _TSNE_PERPS_MAX)
      - lower bound = min(perplexity,  n_samples // 3)
    This avoids the "perplexity > n" error while keeping the parameter
    meaningful for downstream interpretation.

    Optimisation note
    -----------------
    TSNE(learning_rate="auto", init="random") uses Barnes-Hut for n > 2500
    and exact for smaller n.  The random_state=0 ensures reproducibility.

    Improvement suggestion
    ----------------------
    Add UMAP as an alternative projection backend (separate optional import)
    — it is often faster and preserves global structure better than t-SNE.
    """
    n = X.shape[0]
    # Clamp perplexity so the algorithm doesn't crash on small data.
    p = min(perplexity, _TSNE_PERPS_MAX, max(1, n // 3))
    return TSNE(n_components=2, perplexity=p, random_state=0,
                learning_rate="auto", init="random").fit_transform(X)


# ── Diagnostic helpers ────────────────────────────────────────────────────────
# These run K-Means across a range of k to produce elbow curves and silhouette
# profiles.  They are called once per full clustering run, cached afterward.

def run_elbow(X: np.ndarray, k_max: int = _ELBOW_K_MAX) -> dict:
    """
    Within-cluster inertia (sum of squared distances) for k ∈ [2, k_max].
    The "elbow" in this curve suggests a reasonable k.
    """
    inertias = []
    k_limit  = min(k_max, X.shape[0] - 1)          # k must be < n_samples
    k_range  = range(2, k_limit + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        km.fit(X)
        inertias.append(float(km.inertia_))

    return {"k_range": list(k_range), "inertias": inertias}


def run_silhouette_analysis(X: np.ndarray, k_max: int = _ELBOW_K_MAX) -> dict:
    """
    Mean silhouette score for k ∈ [2, k_max].
    Higher = better-separated clusters (range: -1 to 1).

    Optimisation note
    -----------------
    This re-runs KMeans from scratch for each k (same as run_elbow).
    For very large datasets (n > 100k), consider caching the KMeans
    models or using a subsample for the silhouette scan.

    Improvement suggestion
    ----------------------
    Replace the sequential loop with:
        joblib.Parallel(n_jobs=-1)(
            delayed(lambda k: KMeans(k, random_state=0, n_init="auto").fit_predict(X))(k)
            for k in k_range
        )
    This parallelises across CPU cores with minimal code change.
    """
    scores  = {}
    k_limit = min(k_max, X.shape[0] - 1)
    k_range = range(2, k_limit + 1)

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0

    return {"k_range": list(k_range), "scores": scores}


# ── Post-hoc analysis ─────────────────────────────────────────────────────────

def characterize_clusters(X: np.ndarray, labels: np.ndarray,
                           col_names: list[str]) -> dict:
    """
    Per-cluster summary: sizes, centres (means), per-variable mean/std.

    For DBSCAN, noise (-1) is excluded from the cluster profile.
    If only noise is present, a single pseudo-cluster with zero profiles
    is returned so the UI never crashes on empty results.

    Vectorisation
    -------------
    The inner loop over variables (for j in range(X.shape[1])) is kept as
    a Python loop because n_vars is typically < 50.  For datasets with
    hundreds of variables, vectorise with:
        grouped = np.array([X[labels==c].mean(0) for c in unique_labels])
    which is already the case for `centers`.
    """
    unique_labels = np.unique(labels)

    # ── Exclude noise (-1) from cluster characterisation ────────────────────
    unique_labels = unique_labels[unique_labels >= 0]

    # ── Edge case: no real clusters found (all noise / too few points) ──────
    if len(unique_labels) == 0:
        return {
            "n_clusters": 1,
            "centers":    X.mean(axis=0, keepdims=True),
            "sizes":      {-1: len(X)},
            "profiles":   {-1: {}},              # empty profiles → UI skips
            "col_names":  col_names,
        }

    # ── Centres and sizes — vectorised across clusters ─────────────────────
    centers = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
    sizes   = {int(c): int((labels == c).sum()) for c in unique_labels}

    # ── Per-variable profiles (Python loop over vars, C-level stats) ───────
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


# ── Orchestrator — called by the Dash callback ────────────────────────────────

def run_clustering(dataset: dict, col_meta: dict,
                   algorithm: str = "kmeans",
                   n_clusters: int = 3,
                   eps: float = 0.5,
                   min_samples: int = 5,
                   linkage: str = "ward",
                   perplexity: int = 30) -> dict | None:
    """
    Full clustering pipeline: extract → scale → project → cluster → diagnose.

    Returns a single dict with all results cached in store._clustering_cache.
    Downstream callbacks read the cache — they never re-run clustering.

    Output dict structure
    ---------------------
      labels[n]              — cluster assignments (-1 for DBSCAN noise)
      X_std[n, p]            — standardised data matrix
      tsne[n, 2]             — t-SNE projection
      n_obs, n_vars          — integer dimensions
      variables[p]           — variable names
      algorithm              — one of ALGO_*
      silhouette             — float
      characterization       — dict from characterize_clusters()
      elbow                  — dict from run_elbow()
      silhouette_analysis    — dict from run_silhouette_analysis()

    Plus algorithm-specific keys:
      K-Means        → inertia, centers
      DBSCAN         → n_clusters, n_noise, eps, min_samples
      Hierarchical   → linkage, children
    """
    # ── Step 1: extract + scale ─────────────────────────────────────────────
    X = _get_X(dataset, col_meta)
    if X is None:
        return None

    # ── Step 2: metadata ────────────────────────────────────────────────────
    num_cols  = [c for c, t in col_meta.items() if t == "numeric"]
    col_names = num_cols
    n_obs     = X.shape[0]

    # ── Step 3: t-SNE projection (shared across all algorithms) ─────────────
    tsne = run_tsne(X, perplexity=perplexity)

    # ── Step 4: run the chosen algorithm ────────────────────────────────────
    if algorithm == ALGO_KMEANS:
        result = run_kmeans(X, n_clusters)
    elif algorithm == ALGO_DBSCAN:
        result = run_dbscan(X, eps, min_samples)
    else:
        result = run_hierarchical(X, n_clusters, linkage)

    # ── Step 5: cross-cutting enrichment ────────────────────────────────────
    result["X_std"]  = X
    result["tsne"]   = tsne
    result["n_obs"]  = n_obs
    result["n_vars"] = X.shape[1]
    result["variables"] = col_names
    result["algorithm"] = algorithm

    # ── Step 6: post-hoc diagnostics (cached alongside the main result) ─────
    result["characterization"]   = characterize_clusters(X, result["labels"], col_names)
    result["elbow"]              = run_elbow(X)
    result["silhouette_analysis"] = run_silhouette_analysis(X)

    return result
