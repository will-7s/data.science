"""
ui_clustering.py  —  Dash HTML factories for the Clustering tab.

Separation of concerns
----------------------
- LAYOUT is in app.py (dbc.Col with dcc.Dropdown, dcc.Slider, …).
- CALLBACKS are in callbacks.py (Dash @app.callback decorators).
- This file produces *only* html.Div trees for the clustering sidebar
  and interpretation panels.

All functions are pure: they take a dict (the cached clustering result)
and return an html.Div.  No side effects, no state.
"""

from __future__ import annotations
from dash import html
import numpy as np


# ── Private helpers ────────────────────────────────────────────────────────────

def _row(label: str, value, color: str = "var(--slate-700)") -> html.Div:
    """Single stat row: '<label>: <value>'."""
    return html.Div([
        html.Span(f"{label}: ", style={"fontWeight": 600, "color": color,
                                        "fontSize": "0.75rem"}),
        html.Span(value if hasattr(value, "to_plotly_json") else str(value),
                  style={"color": "var(--text-secondary)", "fontSize": "0.75rem"}),
    ], style={"marginBottom": 4})


def _section(title: str, children: list) -> html.Div:
    """Section header with ▶ icon."""
    return html.Div([
        html.Div([
            html.Span("\u25b6", style={"fontSize": "0.625rem",
                                       "color": "var(--blue)"}),
            html.Span(title, style={"fontSize": "0.75rem", "fontWeight": 700,
                                    "color": "var(--blue)"}),
        ], className="section-header"),
        *children,
    ])


def _interp_header(title: str, subtitle: str = "") -> html.Div:
    """Panel header shown at top of every interpretation panel."""
    return html.Div([
        html.Div(title, style={"fontWeight": 700, "fontSize": "0.8125rem",
                                "marginBottom": 4, "color": "var(--slate-800)"}),
        *([html.Div(subtitle, style={"fontSize": "0.6875rem",
                                      "color": "var(--text-muted)",
                                      "marginBottom": 12})] if subtitle else []),
    ])


def _tip_box(text: str, kind: str = "info") -> html.Div:
    """Coloured tip / warning / expert box."""
    return html.Div(text, className=f"tip-box {kind}")


def _table(headers: list, rows: list, highlight_col: int = None) -> html.Table:
    """Styled table with alternating row colours."""
    th_style = {"background": "var(--slate-800)", "color": "#fff",
                "padding": "6px 12px", "fontSize": "0.6875rem",
                "fontWeight": 600, "textAlign": "left", "whiteSpace": "nowrap"}
    td_base = {"padding": "5px 12px", "fontSize": "0.6875rem",
               "borderBottom": "1px solid var(--border-light)",
               "verticalAlign": "top"}

    header_row = html.Tr([html.Th(h, style=th_style) for h in headers])
    data_rows = []
    for i, row in enumerate(rows):
        tds = []
        for j, cell in enumerate(row):
            extra = {"fontWeight": 600, "color": "var(--blue)"} if j == highlight_col else {}
            tds.append(html.Td(
                cell,
                style={**td_base, **extra,
                       "background": "var(--slate-50)" if i % 2 == 0 else "#fff"},
            ))
        data_rows.append(html.Tr(tds))

    return html.Table(
        [html.Thead(header_row), html.Tbody(data_rows)],
        className="modern-table",
    )


def _reading_grid(items: list) -> html.Div:
    """Grid of icon-label-description cards."""
    cards = []
    for icon, label, desc in items:
        cards.append(html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "1rem", "marginRight": 6}),
                html.Span(label, style={"fontWeight": 600, "fontSize": "0.6875rem",
                                         "color": "var(--slate-700)"}),
            ]),
            html.Div(desc, style={"fontSize": "0.625rem",
                                   "color": "var(--text-muted)",
                                   "marginTop": 2}),
        ], style={"marginBottom": 8}))
    return html.Div(cards)


# ── Discrimination power ──────────────────────────────────────────────────────

def _discrimination_power(result: dict) -> dict:
    """
    Compute per-variable F-ratio: between-cluster / within-cluster variance.

    The "discrimination power" of a variable is:
        F = Var_between / Var_within

    A high F means the variable differs strongly across clusters:
    that variable is the most useful for naming / understanding each cluster.

    This is computed here (not in clustering.py) because it is purely
    a UI-level interpretation helper — not needed for the algorithm itself.
    """
    X_std    = result["X_std"]
    labels   = result["labels"]
    char     = result["characterization"]
    col_names = char["col_names"]
    n_vars   = X_std.shape[1]
    unique_c = sorted(char["sizes"].keys())

    global_mean = X_std.mean(axis=0)
    n_total     = X_std.shape[0]

    f_ratios = {}
    for j, name in enumerate(col_names):
        ss_between = 0.0
        ss_within  = 0.0
        for c in unique_c:
            mask  = labels == c
            n_c   = mask.sum()
            mu_c  = X_std[mask, j].mean()
            ss_between += n_c * (mu_c - global_mean[j]) ** 2
            ss_within  += ((X_std[mask, j] - mu_c) ** 2).sum()
        df_between = len(unique_c) - 1
        df_within  = n_total - len(unique_c)
        ms_between = ss_between / max(df_between, 1)
        ms_within  = ss_within / max(df_within, 1)
        f = ms_between / max(ms_within, 1e-12)
        f_ratios[name] = float(f)

    # Sort by descending F
    ranked = sorted(f_ratios.items(), key=lambda x: -x[1])
    return {
        "f_ratios": f_ratios,
        "ranked":   ranked,
        "top3":     ranked[:3],
        "weakest":  ranked[-3:] if len(ranked) >= 3 else ranked,
    }


# ── Algorithm display name map ────────────────────────────────────────────────

_ALG_LABELS = {
    "kmeans":       "K-Means",
    "dbscan":       "DBSCAN",
    "hierarchical": "Hierarchical (Agglomerative)",
}

_ALG_DESCRIPTIONS = {
    "kmeans": (
        "K-Means partitions the data into k spherical clusters by minimising "
        "within-cluster variance (inertia). It assumes clusters are isotropic "
        "and roughly equal-sized. Best used when you have a prior estimate of k."
    ),
    "dbscan": (
        "DBSCAN groups together points that are densely connected (eps-neighbourhood "
        "with at least min_samples points). Points in low-density regions are labelled "
        "as noise (-1). It does NOT require specifying k — the number of clusters "
        "is data-driven."
    ),
    "hierarchical": (
        "Hierarchical (agglomerative) clustering builds a tree of merges. "
        "You cut the tree at the desired k. Ward linkage minimises within-cluster "
        "variance (similar to K-Means) but can capture non-spherical shapes."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sidebar summary panel
# ═══════════════════════════════════════════════════════════════════════════════

def clustering_summary_panel(result: dict) -> html.Div:
    """
    Sidebar summary of the current clustering result.

    Sections
    --------
    1. Overview — algorithm, n, clusters found, silhouette score
    2. Algorithm-specific params (K-Means inertia / DBSCAN eps-noise / Hierarchical linkage)
    3. Cluster sizes table with percentages
    4. Interpretation note
    """
    alg       = result["algorithm"]
    sil       = result["silhouette"]
    n_obs     = result["n_obs"]
    n_vars    = result["n_vars"]
    char      = result["characterization"]

    # ── Silhouette colour threshold ──────────────────────────────────────────
    if sil >= 0.50:
        sil_color = "var(--green)"
        sil_label = f"{sil:.4f}  ✓ Good separation"
    elif sil >= 0.25:
        sil_color = "var(--amber)"
        sil_label = f"{sil:.4f}  ~ Moderate"
    else:
        sil_color = "var(--red)"
        sil_label = f"{sil:.4f}  ✗ Weak"

    children = [
        html.Div("Clustering Overview", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 10, "color": "var(--slate-800)",
        }),
        _row("Algorithm",  _ALG_LABELS.get(alg, alg)),
        _row("Observations", f"{n_obs:,}"),
        _row("Variables (numeric)", n_vars),
        _row("Clusters found", f"{char['n_clusters']}"),
        _row("Silhouette score",
             html.Span(sil_label, style={"color": sil_color, "fontWeight": 600}),
             color=sil_color),
        html.Hr(className="section-divider"),
    ]

    if alg == "dbscan":
        children += [
            _row("Noise points",    f"{result['n_noise']:,}"),
            _row("Noise ratio",     f"{result['n_noise'] / n_obs * 100:.1f}%"),
            _row("Epsilon (eps)",   f"{result['eps']:.2f}"),
            _row("Min samples",     result["min_samples"]),
        ]
    elif alg == "kmeans":
        children += [
            _row("Selected k",      result["n_clusters"]),
            _row("Inertia",         f"{result['inertia']:.1f}"),
        ]
    else:
        children += [
            _row("Selected k",      result["n_clusters"]),
            _row("Linkage",         result["linkage"]),
        ]

    children += [
        html.Hr(className="section-divider"),
        _section("Cluster sizes", [
            html.Div([
                html.Span(f"Cluster {c}: ", style={"fontWeight": 600,
                                                   "fontSize": "0.6875rem",
                                                   "color": "var(--slate-700)"}),
                html.Span(f"{s:,} obs. ({s / n_obs * 100:.1f}%)",
                          style={"fontSize": "0.6875rem",
                                 "color": "var(--text-secondary)"}),
            ], style={"marginBottom": 2})
            for c, s in sorted(char["sizes"].items())
        ]),
        html.Hr(className="section-divider"),
        html.Div([
            html.Span("Note: ", style={"fontWeight": 700, "fontSize": "0.625rem",
                                        "color": "var(--blue-dark)"}),
            html.Span(
                "All numeric variables are standardised (mean=0, std=1) "
                "before clustering to remove scale effects. "
                "t-SNE is used for 2-D projection.",
                style={"fontSize": "0.625rem", "color": "var(--blue-dark)"},
            ),
        ], className="tip-box info"),
    ]

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cluster scatter interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_scatter_interpretation(result: dict) -> html.Div:
    """
    Interpretation panel for the cluster t-SNE scatter plot.

    Explains:
      - What t-SNE shows and its limitations
      - How well clusters are separated
      - Which clusters overlap or are well-isolated
      - DBSCAN noise interpretation
      - The trade-off between local accuracy and global distortion
    """
    char     = result["characterization"]
    alg      = result["algorithm"]
    labels   = result["labels"]
    tsne     = result["tsne"]
    n_obs    = result["n_obs"]
    sil      = result["silhouette"]
    n_clust  = char["n_clusters"]
    sizes    = char["sizes"]

    # ── Separation analysis ──────────────────────────────────────────────────
    centers = np.array([
        tsne[labels == c].mean(axis=0) for c in sorted(sizes.keys())
    ])
    centroid_distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            centroid_distances.append((i, j, d))
    centroid_distances.sort(key=lambda x: -x[2])

    # Pairwise overlap: inter-cluster / intra-cluster spread ratio
    overlap_pairs = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pts_i = tsne[labels == sorted(sizes.keys())[i]]
            pts_j = tsne[labels == sorted(sizes.keys())[j]]
            if len(pts_i) < 2 or len(pts_j) < 2:
                continue
            dist_ij = float(np.linalg.norm(centers[i] - centers[j]))
            spread_i = float(pts_i.std(axis=0).mean())
            spread_j = float(pts_j.std(axis=0).mean())
            ratio = dist_ij / max(spread_i + spread_j, 1e-12)
            overlap_pairs.append((i, j, ratio))
    overlap_pairs.sort(key=lambda x: x[2])

    best_separated = overlap_pairs[-3:] if len(overlap_pairs) >= 3 else overlap_pairs
    most_overlap   = overlap_pairs[:3] if len(overlap_pairs) >= 3 else overlap_pairs

    # ── Noise info (DBSCAN) ─────────────────────────────────────────────────
    n_noise = int((labels == -1).sum()) if alg == "dbscan" else 0
    noise_pct = n_noise / n_obs * 100 if n_obs > 0 else 0

    # ── Build panel ─────────────────────────────────────────────────────────
    children = [
        _interp_header(
            "Cluster Scatter — Expert Reading Guide",
            subtitle=(
                f"t-SNE projection · {n_obs:,} observations · "
                f"{n_clust} clusters · Silhouette = {sil:.4f}"
            ),
        ),
        _reading_grid([
            ("\u25cf", "Each dot = one observation",
             "Position reflects the t-SNE 2-D embedding. "
             "Nearby dots have similar profiles in the original space."),
            ("\ud83c\udfa8", "Colour = cluster assignment",
             "Each colour is one cluster identified by the algorithm. "
             "Grey crosses (DBSCAN only) = noise points not assigned to any cluster."),
            ("\ud83d\udd0d", "t-SNE preserves LOCAL structure",
             "t-SNE focuses on keeping nearby points together. "
             "Global distances (cluster A vs cluster B) are NOT meaningful. "
             "Do NOT interpret absolute positions — only relative neighbourhoods."),
            ("\u26a0", "Cluster overlap in 2-D may be misleading",
             "If two clusters overlap in t-SNE, they may still be separable "
             "in the original high-dimensional space. "
             "Use the Profiles heatmap to understand true differences."),
        ]),
    ]

    # ── Separation report ──────────────────────────────────────────────────
    if len(overlap_pairs) > 0:
        sep_rows = []
        for i, j, ratio in best_separated:
            sep_rows.append([
                f"C{list(sizes.keys())[i]}",
                f"C{list(sizes.keys())[j]}",
                f"{ratio:.2f}",
                html.Span("Well separated",
                          style={"color": "var(--green)", "fontWeight": 600}),
            ])
        for i, j, ratio in most_overlap:
            if ratio < 1.5:
                sep_rows.append([
                    f"C{list(sizes.keys())[i]}",
                    f"C{list(sizes.keys())[j]}",
                    f"{ratio:.2f}",
                    html.Span("Overlapping",
                              style={"color": "var(--red)", "fontWeight": 600}),
                ])

        children += [
            html.Div("Cluster separation in t-SNE space:", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "var(--text-secondary)", "marginBottom": 8, "marginTop": 4,
            }),
            _table(
                ["Cluster A", "Cluster B", "Separation ratio", "Status"],
                sep_rows,
                highlight_col=3,
            ),
            _tip_box(
                "Separation ratio = centroid distance / (avg spread). "
                "Ratio > 2.0 = well separated. "
                "Ratio < 1.0 = substantial overlap. "
                "Remember: t-SNE can create false separation for random data — "
                "always cross-check with silhouette scores and domain knowledge.",
                kind="warn",
            ),
        ]

    # ── DBSCAN noise ──────────────────────────────────────────────────────
    if alg == "dbscan":
        children += [
            _tip_box(
                f"DBSCAN found {n_noise:,} noise points ({noise_pct:.1f}% of data). "
                f"{'High noise ratio — consider lowering eps or increasing min_samples.'
                  if noise_pct > 30 else
                  'Low noise ratio — the density parameters are well chosen.'
                  if noise_pct < 10 else
                  'Moderate noise ratio — fine-tune eps for cleaner clusters.'}",
                kind="warn" if noise_pct > 30 else "info",
            ),
        ]

    # ── Silhouette quality ─────────────────────────────────────────────────
    if sil >= 0.50:
        sil_msg = (
            f"Silhouette = {sil:.4f} (good separation). "
            "Clusters are internally cohesive and externally well separated. "
            "The clustering structure is reliable."
        )
        sil_kind = "success"
    elif sil >= 0.25:
        sil_msg = (
            f"Silhouette = {sil:.4f} (moderate). "
            "Clusters have reasonable structure but some overlap exists. "
            "Consider trying a different k or algorithm."
        )
        sil_kind = "warn"
    else:
        sil_msg = (
            f"Silhouette = {sil:.4f} (weak). "
            "Clusters are poorly separated. "
            "The data may not contain natural clusters, "
            "or the chosen parameters are inappropriate."
        )
        sil_kind = "warn"

    children += [
        _tip_box(sil_msg, kind=sil_kind),
        _tip_box(
            "How to improve clustering: (1) Standardise variables (already done). "
            "(2) Try the Elbow and Silhouette tabs to find a better k. "
            "(3) Try a different algorithm (DBSCAN may capture non-spherical clusters). "
            "(4) Remove noise variables using the Profiles heatmap.",
            kind="expert",
        ),
    ]

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Profiles heatmap interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def profiles_interpretation(result: dict) -> html.Div:
    """
    Interpretation panel for the cluster profiles heatmap.

    Explains:
      - How to read the heatmap (red = above avg, blue = below avg)
      - Which variables most discriminate the clusters (F-ratio ranking)
      - Per-cluster profile summary
      - Variables that do NOT differentiate (noise variables)
    """
    char    = result["characterization"]
    X_std   = result["X_std"]
    labels  = result["labels"]
    n_clust = char["n_clusters"]
    sizes   = char["sizes"]
    prows   = char["profiles"]

    if n_clust < 2 or not prows:
        return html.Div(
            "Run clustering with ≥ 2 clusters to see the profile analysis.",
            style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16},
        )

    # ── Discrimination power ──────────────────────────────────────────────
    dp = _discrimination_power(result)
    top3  = dp["top3"]
    weak3 = dp["weakest"]
    all_f = dp["ranked"]

    # ── Per-cluster profile summary ───────────────────────────────────────
    cluster_summaries = []
    for c in sorted(sizes.keys()):
        prof  = prows[c]
        n_c   = sizes[c]
        pct_c = n_c / result["n_obs"] * 100

        # Top 2 variables where this cluster is most above/below average
        above = sorted(
            [(name, info["mean"]) for name, info in prof.items()],
            key=lambda x: -x[1],
        )[:3]
        below = sorted(
            [(name, info["mean"]) for name, info in prof.items()],
            key=lambda x: x[1],
        )[:3]

        above_str = ", ".join(f"{name} ({val:+.2f})" for name, val in above if val > 0.3)
        below_str = ", ".join(f"{name} ({val:+.2f})" for name, val in below if val < -0.3)

        traits = []
        if above_str:
            traits.append(f"high on {above_str}")
        if below_str:
            traits.append(f"low on {below_str}")
        profile_desc = " · ".join(traits) if traits else "average on all variables"

        cluster_summaries.append({
            "cluster": c,
            "size": n_c,
            "pct": pct_c,
            "profile": profile_desc,
        })

    # ── Build panel ───────────────────────────────────────────────────────
    children = [
        _interp_header(
            "Cluster Profiles — Expert Reading Guide",
            subtitle=(
                f"{n_clust} clusters · {len(char['col_names'])} variables · "
                f"Values = standardised means (mean=0, std=1)"
            ),
        ),
        _reading_grid([
            ("\ud83d\udd10", "Red cell = above average",
             "A positive value means this cluster is higher on this variable "
             "than the global mean (after standardisation). "
             "Darker red = more distinctively high."),
            ("\ud83d\udd35", "Blue cell = below average",
             "A negative value means the cluster is lower on this variable. "
             "Darker blue = more distinctively low."),
            ("\u26aa", "White / pale = near average",
             "Values near zero mean this variable does NOT differentiate "
             "this cluster from the global profile."),
            ("\ud83c\udfaf", "Reading by ROW",
             "Read each row left to right to understand what defines "
             "that cluster. A cluster with many strong colours "
             "has a very distinctive profile."),
        ]),
    ]

    # ── Discriminating variables ─────────────────────────────────────────
    if top3:
        disc_rows = []
        for name, f_val in top3:
            # Determine direction: which cluster is highest/lowest?
            best_c = max(sizes.keys(), key=lambda c: prows[c][name]["mean"])
            worst_c = min(sizes.keys(), key=lambda c: prows[c][name]["mean"])
            best_val = prows[best_c][name]["mean"]
            worst_val = prows[worst_c][name]["mean"]
            disc_rows.append([
                name,
                f"{f_val:.1f}",
                f"C{best_c} ({best_val:+.2f})",
                f"C{worst_c} ({worst_val:+.2f})",
                "Strong discriminator" if f_val > 5 else "Moderate discriminator",
            ])

        children += [
            html.Div("Variables that most differentiate the clusters:", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "var(--text-secondary)", "marginBottom": 8,
            }),
            _table(
                ["Variable", "F-ratio", "Highest cluster", "Lowest cluster", "Role"],
                disc_rows,
                highlight_col=4,
            ),
            _tip_box(
                "The F-ratio measures how much a variable varies BETWEEN clusters "
                "vs. WITHIN clusters. High F → the variable is crucial for "
                "understanding cluster differences. Low F → it's similar across "
                "all clusters (noise variable for the clustering task).",
                kind="info",
            ),
        ]

    # ── Per-cluster profile ─────────────────────────────────────────────
    if cluster_summaries:
        prof_rows = [
            [f"Cluster {s['cluster']}", f"{s['size']:,} ({s['pct']:.1f}%)", s["profile"]]
            for s in cluster_summaries
        ]
        children += [
            html.Div("What defines each cluster:", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "var(--text-secondary)", "marginBottom": 8,
            }),
            _table(
                ["Cluster", "Size", "Distinctive profile (std. mean)"],
                prof_rows,
                highlight_col=2,
            ),
        ]

    # ── Weak variables ──────────────────────────────────────────────────
    if weak3:
        weak_str = ", ".join(f"{name} (F={f:.1f})" for name, f in weak3)
        children += [
            _tip_box(
                f"Variables that barely differentiate clusters: {weak_str}. "
                "These are 'noise variables' for clustering — they have similar "
                "values across all groups. Consider removing them in a future run "
                "to sharpen cluster separation.",
                kind="warn",
            ),
        ]

    children += [
        _tip_box(
            "Professional interpretation tip: name each cluster based on its "
            "top discriminating variables. For example: 'Cluster 0 = High income, "
            "high education → affluent group' vs 'Cluster 1 = Low income, "
            "high age → retired with modest means'. Use the t-SNE scatter "
            "to verify that named clusters form coherent regions.",
            kind="expert",
        ),
    ]

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cluster axis insight (per-cluster deep dive)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_axis_insight_panel(result: dict) -> html.Div:
    """
    Per-cluster detailed analysis panel (right sidebar).
    """
    char     = result["characterization"]
    n_clust  = char["n_clusters"]
    sizes    = char["sizes"]
    prows    = char["profiles"]
    n_obs    = result["n_obs"]
    sil      = result["silhouette"]
    algorithm = result["algorithm"]

    if n_clust < 1 or not prows:
        return html.Div(
            "Run clustering to see per-cluster analysis.",
            style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16},
        )

    dp      = _discrimination_power(result)
    all_f   = dp["ranked"]
    top_f_names = {name for name, _ in all_f[:3]}

    # ── Per-cluster cards ───────────────────────────────────────────────
    cluster_cards = []
    for c in sorted(sizes.keys()):
        n_c   = sizes[c]
        pct_c = n_c / n_obs * 100
        prof  = prows[c]

        # Distinctive traits
        traits = sorted(
            [(name, info["mean"]) for name, info in prof.items()],
            key=lambda x: -abs(x[1]),
        )[:4]

        trait_lines = []
        for name, val in traits:
            color = "var(--green)" if val > 0.3 else "var(--red)" if val < -0.3 else "var(--text-muted)"
            arrow = "\u2191" if val > 0.3 else "\u2193" if val < -0.3 else "\u2192"
            label = "High" if val > 0.3 else "Low" if val < -0.3 else "Neutral"
            trait_lines.append(
                f"{arrow} {name}: {val:+.2f} ({label})"
            )

        cluster_cards.append(html.Div([
            html.Div([
                html.Span(f"Cluster {c}",
                          style={"fontWeight": 700, "fontSize": "0.75rem",
                                 "color": "var(--slate-800)"}),
                html.Span(f"  {n_c:,} obs. ({pct_c:.1f}%)",
                          style={"fontSize": "0.6875rem",
                                 "color": "var(--text-muted)"}),
            ]),
            html.Div([
                html.Div(t, style={"fontSize": "0.625rem",
                                    "color": "var(--text-secondary)",
                                    "marginTop": 2, "fontFamily": "monospace"})
                for t in trait_lines
            ], style={"marginTop": 4}),
        ], style={
            "background": "var(--bg-muted)",
            "borderRadius": "var(--radius-sm)",
            "border": "1px solid var(--border-light)",
            "padding": "10px 12px",
            "marginBottom": 8,
        }))

    # ── Build panel ─────────────────────────────────────────────────────
    children = [
        html.Div("Cluster Analysis", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 10, "color": "var(--slate-800)",
        }),
        _row("Algorithm", _ALG_LABELS.get(algorithm, algorithm)),
        _row("Silhouette score", f"{sil:.4f}"),
        _row("Clusters", n_clust),
        html.Hr(className="section-divider"),
        html.Div("Per-cluster profile (standardised means):",
                 style={"fontWeight": 600, "fontSize": "0.75rem",
                        "color": "var(--text-secondary)", "marginBottom": 8}),
        *cluster_cards,
        html.Hr(className="section-divider"),
        html.Div([
            html.Span("\u26a0", style={"fontWeight": 700, "color": "#92400e",
                                        "fontSize": "0.625rem"}),
            html.Span(
                " Profiles above are in standard deviation units (mean=0, std=1). "
                "A value of +1.5 means the cluster is 1.5 std above the "
                "global average on that variable.",
                style={"fontSize": "0.625rem", "color": "#92400e"},
            ),
        ], style={"marginTop": 8}),
    ]

    # ── Algorithm-specific notes ───────────────────────────────────────
    desc = _ALG_DESCRIPTIONS.get(algorithm, "")
    if desc:
        children += [
            html.Hr(className="section-divider"),
            _tip_box(desc, kind="info"),
        ]

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Elbow curve interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def elbow_interpretation(result: dict) -> html.Div:
    """
    Interpretation panel for the elbow curve (inertia vs. k).

    Explains:
      - What the elbow method shows
      - Where the elbow is (rate-of-change analysis)
      - Whether the current k matches the elbow suggestion
      - Limitations of the elbow criterion
    """
    elbow     = result["elbow"]
    n_obs     = result["n_obs"]
    n_clust   = result["characterization"]["n_clusters"]
    algorithm = result["algorithm"]

    if not elbow["k_range"]:
        return html.Div(
            "Not enough data for elbow analysis.",
            style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16},
        )

    inertias    = elbow["inertias"]
    k_range     = elbow["k_range"]

    # ── Rate-of-change (first difference) ──────────────────────────────
    deltas = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    pct_deltas = [
        (inertias[i] - inertias[i + 1]) / max(inertias[i], 1e-12) * 100
        for i in range(len(inertias) - 1)
    ]

    # The "elbow" is where the marginal gain drops → suggested k
    # Simple heuristic: largest drop after k=2 has diminishing returns
    suggested_k = None
    if len(deltas) >= 2:
        for i in range(1, len(deltas)):
            if deltas[i] < deltas[i - 1] * 0.4:  # 60% drop in improvement
                suggested_k = k_range[i]
                break
    if suggested_k is None and k_range:
        suggested_k = k_range[-1]

    # ── Build panel ───────────────────────────────────────────────────
    children = [
        _interp_header(
            "Elbow Method — Expert Reading Guide",
            subtitle=(
                f"Inertia (within-cluster sum of squares) for k ∈ [{k_range[0]}, {k_range[-1]}]"
            ),
        ),
        _reading_grid([
            ("\u2198", "The curve always goes down",
             "Inertia decreases as k increases — more clusters = tighter fit. "
             "The question is: at what k does the improvement slow down?"),
            ("\ud83d\udccd", "The 'elbow' = optimal k",
             "Look for the bend where the curve flattens. "
             "Adding clusters beyond this point gives diminishing returns."),
            ("⚠", "The elbow can be ambiguous",
             "Many real datasets have no clear elbow. "
             "Cross-check with the Silhouette tab and domain knowledge."),
            ("\ud83d\udcca", "Compare with current k",
             f"Your current solution uses k={n_clust}. "
             f"{'Matches the elbow suggestion.' if suggested_k and n_clust == suggested_k else 'The elbow suggests k≈' + str(suggested_k) + '.' if suggested_k else ''}"),
        ]),
    ]

    # ── Rate-of-change table ──────────────────────────────────────────
    roc_rows = []
    for i, k in enumerate(k_range):
        inertia = inertias[i]
        if i < len(deltas):
            delta   = deltas[i]
            pct     = pct_deltas[i]
            if delta < max(deltas) * 0.3 if deltas else False:
                flag = html.Span("Elbow \u2190", style={"color": "var(--green)",
                                                         "fontWeight": 600})
            else:
                flag = ""
            roc_rows.append([
                f"k={k}",
                f"{inertia:.1f}",
                f"{delta:.1f}" if delta else "\u2014",
                f"{pct:.1f}%" if delta else "\u2014",
                flag,
            ])
        else:
            roc_rows.append([
                f"k={k}",
                f"{inertia:.1f}",
                "\u2014",
                "\u2014",
                "",
            ])

    children += [
        html.Div("Rate-of-change analysis:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["k", "Inertia", "\u0394 inertia", "\u0394 %", "Elbow?"],
            roc_rows,
            highlight_col=4,
        ),
        _tip_box(
            "How to choose k: (1) Find the elbow in the chart above. "
            "(2) Check the Silhouette tab for the k with the highest score. "
            "(3) If they disagree, prefer the silhouette k — it measures "
            "cluster quality directly. (4) Always validate with domain knowledge.",
            kind="expert",
        ),
        _tip_box(
            "Limitations: the elbow method assumes spherical clusters "
            "and works best when clusters are well separated. "
            "For DBSCAN, the notion of k is replaced by density parameters "
            "(eps, min_samples) — the elbow curve is only a diagnostic helper.",
            kind="warn",
        ),
    ]

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Silhouette plot interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def silhouette_interpretation(result: dict) -> html.Div:
    """
    Interpretation panel for the silhouette score plot.

    Explains:
      - What silhouette measures (cohesion vs. separation)
      - Which k is best according to silhouette
      - How the current k compares
      - Reading thresholds and limitations
    """
    sil_analysis = result["silhouette_analysis"]
    current_sil  = result["silhouette"]
    n_clust      = result["characterization"]["n_clusters"]
    algorithm    = result["algorithm"]

    if not sil_analysis["k_range"] or not sil_analysis["scores"]:
        return html.Div(
            "Not enough data for silhouette analysis.",
            style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16},
        )

    scores      = sil_analysis["scores"]
    k_range     = sil_analysis["k_range"]
    best_k      = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_k]
    ranked_by_sil = sorted(scores.items(), key=lambda x: -x[1])

    # ── Colour thresholds for best score ──────────────────────────────
    if best_score >= 0.50:
        verdict     = "good structure"
        verdict_col = "var(--green)"
        verdict_emoji = "\u2713"
    elif best_score >= 0.25:
        verdict     = "moderate structure"
        verdict_col = "var(--amber)"
        verdict_emoji = "~"
    else:
        verdict     = "weak / artificial structure"
        verdict_col = "var(--red)"
        verdict_emoji = "\u2717"

    # ── Build panel ───────────────────────────────────────────────────
    children = [
        _interp_header(
            "Silhouette Score — Expert Reading Guide",
            subtitle=(
                f"Score range: [-1, 1] · Best k = {best_k} "
                f"(score = {best_score:.4f}) · "
                f"Your k = {n_clust} ({current_sil:.4f})"
            ),
        ),
        _reading_grid([
            ("\ud83d\udcca", "Higher = better separation",
             "Silhouette measures how similar a point is to its own cluster "
             "vs. other clusters. +1 = well matched, 0 = on boundary, "
             "-1 = misclassified."),
            ("🎯", "Best k = highest bar (in red)",
             f"k={best_k} gives the best silhouette score ({best_score:.4f}). "
             f"This is {verdict_emoji} {verdict} in the data."),
            ("📏", "Interpretation thresholds",
             "> 0.70: Strong structure  ·  > 0.50: Reasonable  ·  "
             "> 0.25: Weak (may be artificial)  ·  < 0.25: No substantial structure."),
            ("\u26a0", "Silhouette favours spherical clusters",
             "Like K-Means, silhouette assumes convex clusters. "
             "DBSCAN may produce excellent clusters with low silhouette "
             "because of non-spherical shapes."),
        ]),
    ]

    # ── All k ranked ────────────────────────────────────────────────
    rank_rows = []
    for rank, (k, score) in enumerate(ranked_by_sil):
        if score >= 0.50:
            qual = html.Span("Good", style={"color": "var(--green)",
                                             "fontWeight": 600})
        elif score >= 0.25:
            qual = html.Span("Moderate", style={"color": "#92400e",
                                                  "fontWeight": 600})
        else:
            qual = html.Span("Weak", style={"color": "var(--red)",
                                             "fontWeight": 600})
        is_best = html.Span("← Best", style={"color": "var(--green)",
                                              "fontWeight": 700}) if k == best_k else ""
        is_current = html.Span("← Current", style={"color": "var(--blue)",
                                                    "fontWeight": 700,
                                                    "fontSize": "0.625rem"}) if k == n_clust and k != best_k else ""
        rank_rows.append([
            f"{rank + 1}",
            f"k={k}",
            f"{score:.4f}",
            qual,
            f"{is_best}{is_current}",
        ])

    children += [
        html.Div("All k ranked by silhouette score:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Rank", "k", "Score", "Quality", "Note"],
            rank_rows,
            highlight_col=3,
        ),
    ]

    # ── Recommendations ──────────────────────────────────────────────
    if n_clust == best_k:
        children += [
            _tip_box(
                f"✅ Your current k={n_clust} matches the silhouette optimum. "
                f"The silhouette score ({current_sil:.4f}) indicates {verdict}. "
                "This is a good sign that your choice of k is appropriate.",
                kind="success" if best_score >= 0.25 else "warn",
            ),
        ]
    else:
        children += [
            _tip_box(
                f"The silhouette method suggests k={best_k} "
                f"(score = {best_score:.4f}), but you are using k={n_clust} "
                f"(score = {current_sil:.4f}). "
                f"{'Consider trying k=' + str(best_k) + ' for better separation.' if best_score > current_sil + 0.05 else 'The difference is small — keep your current k if it makes sense for your domain.'}",
                kind="info",
            ),
        ]

    # ── DBSCAN-specific ─────────────────────────────────────────────
    if algorithm == "dbscan":
        children += [
            _tip_box(
                "DBSCAN silhouette is computed on core points only "
                "(noise label -1 is excluded by sklearn's silhouette_score). "
                "If noise ratio is high, the silhouette may overestimate "
                "cluster quality. Always check the number of noise points.",
                kind="warn",
            ),
        ]

    children += [
        _tip_box(
            "Final recommendation: use silhouette as a GUIDE, not a RULE. "
            "The best k depends on your analytical goal. "
            "If k=3 has 0.52 and k=5 has 0.48, both are plausible — "
            "choose based on interpretability, not the decimal.",
            kind="expert",
        ),
    ]

    return html.Div(children)
