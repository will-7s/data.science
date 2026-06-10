"""
charts_clustering.py  —  Plotly figure factories for the Clustering tab.

Design principles
-----------------
1. All figures share a common LAYOUT dict for visual consistency.
2. Subsampling is applied transparently when n_obs > SCATTER_SAMPLE_MAX
   to keep browser rendering fast.
3. Every public function returns a go.Figure — never None.
   Empty states are handled via empty_clustering().
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import clustering as cl
from charts_common import layout, BLUE, RED, GREY, CLUSTER_COLORS

# ── Performance thresholds ────────────────────────────────────────────────────
# When n_obs exceeds these limits, we randomly subsample for the scatter plot
# to keep the SVG/WebGL workload manageable in the browser.
_SCATTER_SAMPLE_MAX = 3_000


def cluster_scatter(result: dict) -> go.Figure:
    """
    2-D scatter of individual observations coloured by cluster label.

    Projection
    ----------
    Uses t-SNE coordinates from result["tsne"] (computed once per run).
    For DBSCAN, noise points (-1) are rendered as grey crosses with
    lower opacity so core clusters remain visually dominant.

    Subsample logic
    ---------------
    If n_obs > _SCATTER_SAMPLE_MAX, we draw a random subset of size
    _SCATTER_SAMPLE_MAX (seeded for reproducibility).  All clusters
    are proportionally represented via random sampling rather than
    stratified — the trade-off is acceptable for visualisation.

    Improvement suggestion
    ----------------------
    Add a "projection" dropdown (t-SNE / PCA first-2-PCs) so the user
    can compare the same clusters in different latent spaces.
    """
    labels    = result["labels"]
    n_clusters = result["characterization"]["n_clusters"]
    tsne      = result["tsne"]
    n_obs     = result["n_obs"]
    alg       = result["algorithm"]

    x, y = tsne[:, 0], tsne[:, 1]

    note = ""
    show_mask = np.ones(n_obs, dtype=bool)
    if n_obs > _SCATTER_SAMPLE_MAX:
        rng = np.random.default_rng(0)
        idx_show = rng.choice(n_obs, _SCATTER_SAMPLE_MAX, replace=False)
        show_mask[:] = False
        show_mask[idx_show] = True
        note = f" (sample {_SCATTER_SAMPLE_MAX:,} / {n_obs:,})"

    fig = go.Figure()

    if alg == cl.ALGO_DBSCAN:
        core_mask = (labels >= 0) & show_mask
        if core_mask.any():
            for c in np.unique(labels[core_mask]):
                mask = (labels == c) & show_mask
                fig.add_trace(go.Scatter(
                    x=x[mask], y=y[mask], mode="markers",
                    marker=dict(size=5, color=CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)],
                                opacity=0.7),
                    name=f"Cluster {c}",
                    hovertemplate=f"Cluster {c}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
                ))
        noise_mask = (labels == -1) & show_mask
        if noise_mask.any():
            fig.add_trace(go.Scatter(
                x=x[noise_mask], y=y[noise_mask], mode="markers",
                marker=dict(size=3, color=GREY, symbol="x", opacity=0.4),
                name="Noise",
                hovertemplate="Noise<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            ))
    else:
        for c in range(n_clusters):
            mask = (labels == c) & show_mask
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=5, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                            opacity=0.7),
                name=f"Cluster {c}",
                hovertemplate=f"Cluster {c}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
            ))

    fig.update_layout(
        **layout(),
        title=dict(text=f"Cluster visualisation (t-SNE projection){note}", font_size=13),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font_size=9),
    )
    return fig


def elbow_plot(result: dict) -> go.Figure:
    """
    Elbow curve: within-cluster inertia vs. number of clusters.
    The "elbow" (bend in the curve) suggests a reasonable k.

    Edge case
    ---------
    If k_range is empty (happens when n_obs ≤ 3), returns an
    explanatory empty chart instead of crashing.
    """
    elbow = result["elbow"]
    if not elbow["k_range"]:
        return empty_clustering("Not enough data for elbow analysis.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elbow["k_range"], y=elbow["inertias"],
        mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(size=8, color=BLUE),
        hovertemplate="k=%{x}<br>Inertia=%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **layout(),
        title=dict(text="Elbow method — within-cluster inertia", font_size=13),
        xaxis_title="Number of clusters (k)",
        yaxis_title="Inertia (within-cluster sum of squares)",
    )
    return fig


def silhouette_plot(result: dict) -> go.Figure:
    """
    Silhouette score for k ∈ [2, k_max].
    The highest bar suggests the best-separated clustering.

    Visual cue
    ----------
    The bar for the best k is highlighted in red; others are blue.
    This is a quick visual heuristic — always cross-check with elbow
    and domain knowledge.

    Score range: [-1, 1].
      > 0.5  → reasonable structure
      > 0.7  → strong structure
      < 0.25 → weak or artificial structure
    """
    sil = result["silhouette_analysis"]
    if not sil["k_range"]:
        return empty_clustering("Not enough data for silhouette analysis.")

    vals = list(sil["scores"].values())
    best = max(vals) if vals else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(sil["scores"].keys()),
        y=vals,
        marker_color=[RED if v == best else BLUE for v in vals],
        hovertemplate="k=%{x}<br>Silhouette=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **layout(),
        title=dict(text="Silhouette score by number of clusters", font_size=13),
        xaxis_title="Number of clusters (k)",
        yaxis_title="Silhouette score",
        yaxis_range=[-0.1, 1.1],
    )
    return fig


def cluster_profiles_heatmap(result: dict) -> go.Figure:
    """
    Heatmap of cluster means per variable (rows = clusters, columns = variables).

    Interpretation
    --------------
    Rows are standardised (mean=0, std=1) because the input was scaled
    before clustering.  A red cell means the cluster is above average
    on that variable; blue means below average.

    Edge case
    ---------
    If no clusters were found (all noise), returns an empty chart.
    """
    char = result["characterization"]
    n_clusters = char["n_clusters"]
    col_names  = char["col_names"]

    if n_clusters == 0 or not col_names:
        return empty_clustering("No clusters to profile.")

    # ── Build (n_clusters × n_vars) matrix of means ─────────────────────────
    data = np.zeros((n_clusters, len(col_names)))
    for c in range(n_clusters):
        for j, col in enumerate(col_names):
            data[c, j] = char["profiles"][c][col]["mean"]

    labels = [f"Cluster {c}" for c in sorted(char["sizes"].keys())]

    fig = px.imshow(
        data, x=col_names, y=labels,
        color_continuous_scale="RdBu_r",
        text_auto=".2f", aspect="auto",
        labels=dict(color="Mean (std)"),
    )
    h = max(250, n_clusters * 50 + 100)
    fig.update_layout(
        **layout(height=h, margin=dict(l=100, r=50, t=50, b=100)),
        title=dict(text="Cluster profiles — variable means per cluster", font_size=13),
    )
    return fig


def empty_clustering(msg: str = "Run clustering to see results.") -> go.Figure:
    """
    Placeholder chart for empty / pre-run state.
    -----------------------------------------------------------------
    Shows a centred message instead of a broken graph.
    Used as the default output before the user clicks "Run clustering".
    """
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#7f8c8d"),
    )
    fig.update_layout(height=300, template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, -apple-system, sans-serif", size=12))
    return fig
