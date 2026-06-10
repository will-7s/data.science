"""
ui_clustering.py  —  Dash HTML factories for the Clustering tab (v6.1).

FIX (v6.1)
----------
1. _discrimination_power() removed — now imported from clustering.py where
   it belongs (was the only computational function in a UI module).
2. cluster_scatter_interpretation() guards against NaN t-SNE values produced
   when subsampling is active (rows not in the subsample have tsne = NaN).
3. clustering_summary_panel() shows a t-SNE subsampling banner when
   result["tsne_subsampled"] is True.
4. Local duplicate helpers (_row, _section, _interp_header, _tip_box,
   _table, _reading_grid) retained verbatim — they differ slightly from
   ui_common.py in visual style (inline styles vs classNames); a future
   refactor can unify them.
"""
from __future__ import annotations
from dash import html
import numpy as np
from clustering import discrimination_power   # FIX: imported from computation module


# ── Private helpers ────────────────────────────────────────────────────────────

def _row(label: str, value, color: str = "var(--slate-700)") -> html.Div:
    return html.Div([
        html.Span(f"{label}: ", style={"fontWeight": 600, "color": color,
                                        "fontSize": "0.75rem"}),
        html.Span(value if hasattr(value, "to_plotly_json") else str(value),
                  style={"color": "var(--text-secondary)", "fontSize": "0.75rem"}),
    ], style={"marginBottom": 4})


def _section(title: str, children: list) -> html.Div:
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
    return html.Div([
        html.Div(title, style={"fontWeight": 700, "fontSize": "0.8125rem",
                                "marginBottom": 4, "color": "var(--slate-800)"}),
        *([html.Div(subtitle, style={"fontSize": "0.6875rem",
                                      "color": "var(--text-muted)",
                                      "marginBottom": 12})] if subtitle else []),
    ])


def _tip_box(text: str, kind: str = "info") -> html.Div:
    return html.Div(text, className=f"tip-box {kind}")


def _table(headers: list, rows: list, highlight_col: int = None) -> html.Table:
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


_ALG_LABELS = {
    "kmeans":       "K-Means",
    "dbscan":       "DBSCAN",
    "hierarchical": "Hierarchical (Agglomerative)",
}
_ALG_DESCRIPTIONS = {
    "kmeans": (
        "K-Means partitions the data into k spherical clusters by minimising "
        "within-cluster variance (inertia). Assumes isotropic, roughly equal-sized clusters."
    ),
    "dbscan": (
        "DBSCAN groups together densely connected points. Noise points (label=-1) "
        "are in low-density regions. k is data-driven — not specified in advance."
    ),
    "hierarchical": (
        "Hierarchical (agglomerative) clustering builds a merge tree. "
        "Ward linkage minimises within-cluster variance — similar to K-Means."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Summary sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def clustering_summary_panel(result: dict) -> html.Div:
    alg   = result["algorithm"]
    sil   = result["silhouette"]
    n_obs = result["n_obs"]
    char  = result["characterization"]

    if sil >= 0.50:
        sil_color = "var(--green)"; sil_label = f"{sil:.4f}  ✓ Good"
    elif sil >= 0.25:
        sil_color = "var(--amber)"; sil_label = f"{sil:.4f}  ~ Moderate"
    else:
        sil_color = "var(--red)"; sil_label = f"{sil:.4f}  ✗ Weak"

    children = [
        html.Div("Clustering Overview", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 10, "color": "var(--slate-800)",
        }),
        _row("Algorithm",  _ALG_LABELS.get(alg, alg)),
        _row("Observations", f"{n_obs:,}"),
        _row("Variables (numeric)", result["n_vars"]),
        _row("Clusters found", f"{char['n_clusters']}"),
        _row("Silhouette score",
             html.Span(sil_label, style={"color": sil_color, "fontWeight": 600}),
             color=sil_color),
        html.Hr(className="section-divider"),
    ]

    if alg == "dbscan":
        children += [
            _row("Noise points",  f"{result['n_noise']:,}"),
            _row("Noise ratio",   f"{result['n_noise'] / n_obs * 100:.1f}%"),
            _row("Epsilon (eps)", f"{result['eps']:.2f}"),
            _row("Min samples",   result["min_samples"]),
        ]
    elif alg == "kmeans":
        children += [
            _row("Selected k", result["n_clusters"]),
            _row("Inertia",    f"{result['inertia']:.1f}"),
        ]
    else:
        children += [
            _row("Selected k", result["n_clusters"]),
            _row("Linkage",    result["linkage"]),
        ]

    # FIX: surface t-SNE subsampling warning
    if result.get("tsne_subsampled"):
        from clustering import T_SNE_MAX_OBS
        children.append(_tip_box(
            f"⚡ t-SNE was computed on a random subsample of {T_SNE_MAX_OBS:,} observations "
            f"(dataset has {n_obs:,} rows). The scatter plot shows subsampled points only. "
            "Cluster profiles and statistics use all data.",
            kind="warn",
        ))

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
                "before clustering to remove scale effects.",
                style={"fontSize": "0.625rem", "color": "var(--blue-dark)"},
            ),
        ], className="tip-box info"),
    ]
    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cluster scatter interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_scatter_interpretation(result: dict) -> html.Div:
    char    = result["characterization"]
    alg     = result["algorithm"]
    labels  = result["labels"]
    tsne    = result["tsne"]
    n_obs   = result["n_obs"]
    sil     = result["silhouette"]
    n_clust = char["n_clusters"]
    sizes   = char["sizes"]

    # FIX: filter NaN rows from t-SNE (introduced by subsampling)
    valid_tsne = ~np.isnan(tsne[:, 0])
    tsne_valid  = tsne[valid_tsne]
    labels_valid = labels[valid_tsne]

    centers = np.array([
        tsne_valid[labels_valid == c].mean(axis=0)
        for c in sorted(sizes.keys())
        if (labels_valid == c).sum() > 0
    ])

    centroid_distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            centroid_distances.append((i, j, d))
    centroid_distances.sort(key=lambda x: -x[2])

    overlap_pairs = []
    sorted_keys = sorted(sizes.keys())
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pts_i = tsne_valid[labels_valid == sorted_keys[i]]
            pts_j = tsne_valid[labels_valid == sorted_keys[j]]
            if len(pts_i) < 2 or len(pts_j) < 2:
                continue
            dist_ij = float(np.linalg.norm(centers[i] - centers[j]))
            spread   = pts_i.std(axis=0).mean() + pts_j.std(axis=0).mean()
            ratio    = dist_ij / max(spread, 1e-12)
            overlap_pairs.append((i, j, ratio))
    overlap_pairs.sort(key=lambda x: x[2])

    best_sep    = overlap_pairs[-3:] if len(overlap_pairs) >= 3 else overlap_pairs
    most_over   = [p for p in overlap_pairs[:3] if p[2] < 1.5]
    n_noise     = int((labels == -1).sum()) if alg == "dbscan" else 0
    noise_pct   = n_noise / n_obs * 100 if n_obs > 0 else 0

    children = [
        _interp_header(
            "Cluster Scatter — Expert Reading Guide",
            subtitle=(
                f"t-SNE projection · {n_obs:,} observations · "
                f"{n_clust} clusters · Silhouette = {sil:.4f}"
                + (" · ⚡ subsampled" if result.get("tsne_subsampled") else "")
            ),
        ),
        _reading_grid([
            ("●", "Each dot = one observation",
             "Position reflects the t-SNE 2-D embedding. "
             "Nearby dots have similar profiles in the original space."),
            ("🎨", "Colour = cluster assignment",
             "Each colour is one cluster identified by the algorithm. "
             "Grey crosses (DBSCAN only) = noise points."),
            ("🔍", "t-SNE preserves LOCAL structure",
             "t-SNE focuses on keeping nearby points together. "
             "Global distances between clusters are NOT meaningful."),
            ("⚠", "Overlap in 2-D may be misleading",
             "Clusters overlapping in t-SNE may still be separable "
             "in the original high-dimensional space."),
        ]),
    ]

    if overlap_pairs:
        sep_rows = []
        for i, j, ratio in best_sep:
            sep_rows.append([
                f"C{sorted_keys[i]}", f"C{sorted_keys[j]}",
                f"{ratio:.2f}",
                html.Span("Well separated", style={"color": "var(--green)", "fontWeight": 600}),
            ])
        for i, j, ratio in most_over:
            sep_rows.append([
                f"C{sorted_keys[i]}", f"C{sorted_keys[j]}",
                f"{ratio:.2f}",
                html.Span("Overlapping", style={"color": "var(--red)", "fontWeight": 600}),
            ])
        if sep_rows:
            children += [
                html.Div("Cluster separation in t-SNE space:", style={
                    "fontWeight": 600, "fontSize": "0.75rem",
                    "color": "var(--text-secondary)", "marginBottom": 8, "marginTop": 4,
                }),
                _table(["Cluster A", "Cluster B", "Separation ratio", "Status"],
                       sep_rows, highlight_col=3),
                _tip_box(
                    "Separation ratio = centroid distance / (avg spread). "
                    "Ratio > 2.0 = well separated. "
                    "Remember: t-SNE can create false separation — "
                    "always cross-check with silhouette scores.",
                    kind="warn",
                ),
            ]

    if alg == "dbscan":
        children.append(_tip_box(
            f"DBSCAN found {n_noise:,} noise points ({noise_pct:.1f}% of data). "
            + ("High noise — consider lowering eps or increasing min_samples."
               if noise_pct > 30 else
               "Low noise — density parameters are well chosen."
               if noise_pct < 10 else
               "Moderate noise — fine-tune eps for cleaner clusters."),
            kind="warn" if noise_pct > 30 else "info",
        ))

    if sil >= 0.50:
        sil_msg = f"Silhouette = {sil:.4f} (good). Clusters are well separated."; sil_k = "success"
    elif sil >= 0.25:
        sil_msg = f"Silhouette = {sil:.4f} (moderate). Some overlap exists."; sil_k = "warn"
    else:
        sil_msg = f"Silhouette = {sil:.4f} (weak). Try different k or algorithm."; sil_k = "warn"

    children += [
        _tip_box(sil_msg, kind=sil_k),
        _tip_box(
            "How to improve: (1) Try the Elbow/Silhouette tabs for better k. "
            "(2) Try DBSCAN for non-spherical clusters. "
            "(3) Remove noise variables from the Profiles heatmap.",
            kind="expert",
        ),
    ]
    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Profiles heatmap interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def profiles_interpretation(result: dict) -> html.Div:
    char    = result["characterization"]
    n_clust = char["n_clusters"]
    sizes   = char["sizes"]
    prows   = char["profiles"]

    if n_clust < 2 or not prows:
        return html.Div(
            "Run clustering with ≥ 2 clusters to see the profile analysis.",
            style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16},
        )

    # FIX: use discrimination_power from clustering.py (not local function)
    dp    = discrimination_power(result["X_std"], result["labels"], char["col_names"])
    top3  = dp["top3"]
    weak3 = dp["weakest"]

    cluster_summaries = []
    for c in sorted(sizes.keys()):
        prof  = prows[c]
        n_c   = sizes[c]
        pct_c = n_c / result["n_obs"] * 100
        above = sorted([(n, v["mean"]) for n, v in prof.items()], key=lambda x: -x[1])[:3]
        below = sorted([(n, v["mean"]) for n, v in prof.items()], key=lambda x:  x[1])[:3]
        above_str = ", ".join(f"{n} ({v:+.2f})" for n, v in above if v > 0.3)
        below_str = ", ".join(f"{n} ({v:+.2f})" for n, v in below if v < -0.3)
        traits = []
        if above_str: traits.append(f"high on {above_str}")
        if below_str: traits.append(f"low on {below_str}")
        cluster_summaries.append({
            "cluster": c, "size": n_c, "pct": pct_c,
            "profile": " · ".join(traits) or "average on all variables",
        })

    children = [
        _interp_header(
            "Cluster Profiles — Expert Reading Guide",
            subtitle=(
                f"{n_clust} clusters · {len(char['col_names'])} variables · "
                "Values = standardised means (mean=0, std=1)"
            ),
        ),
        _reading_grid([
            ("🔴", "Red cell = above average",
             "Cluster is higher than the global mean on this variable."),
            ("🔵", "Blue cell = below average",
             "Cluster is lower than the global mean on this variable."),
            ("⚪", "White = near average",
             "Variable does NOT differentiate this cluster."),
            ("🎯", "Reading by ROW",
             "Read each row to understand what defines that cluster. "
             "Many strong colours = very distinctive profile."),
        ]),
    ]

    if top3:
        disc_rows = []
        for name, f_val in top3:
            best_c  = max(sizes.keys(), key=lambda c: prows[c][name]["mean"])
            worst_c = min(sizes.keys(), key=lambda c: prows[c][name]["mean"])
            disc_rows.append([
                name, f"{f_val:.1f}",
                f"C{best_c} ({prows[best_c][name]['mean']:+.2f})",
                f"C{worst_c} ({prows[worst_c][name]['mean']:+.2f})",
                "Strong" if f_val > 5 else "Moderate",
            ])
        children += [
            html.Div("Variables that most differentiate the clusters:", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "var(--text-secondary)", "marginBottom": 8,
            }),
            _table(["Variable", "F-ratio", "Highest cluster", "Lowest cluster", "Role"],
                   disc_rows, highlight_col=4),
            _tip_box(
                "F-ratio = between-cluster variance / within-cluster variance. "
                "High F → variable is crucial for cluster differences. "
                "Low F → similar across all clusters (noise variable).",
                kind="info",
            ),
        ]

    if cluster_summaries:
        children += [
            html.Div("What defines each cluster:", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "var(--text-secondary)", "marginBottom": 8,
            }),
            _table(
                ["Cluster", "Size", "Distinctive profile (std. mean)"],
                [[f"Cluster {s['cluster']}", f"{s['size']:,} ({s['pct']:.1f}%)", s["profile"]]
                 for s in cluster_summaries],
                highlight_col=2,
            ),
        ]

    if weak3:
        children.append(_tip_box(
            f"Variables barely differentiating clusters: "
            f"{', '.join(f'{n} (F={f:.1f})' for n, f in weak3)}. "
            "Consider removing them in a future run.",
            kind="warn",
        ))

    children.append(_tip_box(
        "Name each cluster based on its top discriminating variables. "
        "Use the t-SNE scatter to verify that named clusters form coherent regions.",
        kind="expert",
    ))
    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-cluster axis insight panel
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_axis_insight_panel(result: dict) -> html.Div:
    char      = result["characterization"]
    n_clust   = char["n_clusters"]
    sizes     = char["sizes"]
    prows     = char["profiles"]
    n_obs     = result["n_obs"]
    sil       = result["silhouette"]
    algorithm = result["algorithm"]

    if n_clust < 1 or not prows:
        return html.Div("Run clustering to see per-cluster analysis.",
                        style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16})

    # FIX: use imported discrimination_power
    dp        = discrimination_power(result["X_std"], result["labels"], char["col_names"])
    all_f     = dp["ranked"]

    cluster_cards = []
    for c in sorted(sizes.keys()):
        n_c   = sizes[c]
        pct_c = n_c / n_obs * 100
        prof  = prows[c]
        traits = sorted([(name, info["mean"]) for name, info in prof.items()],
                        key=lambda x: -abs(x[1]))[:4]
        trait_lines = []
        for name, val in traits:
            color  = "var(--green)" if val > 0.3 else "var(--red)" if val < -0.3 else "var(--text-muted)"
            arrow  = "\u2191" if val > 0.3 else "\u2193" if val < -0.3 else "\u2192"
            label  = "High" if val > 0.3 else "Low" if val < -0.3 else "Neutral"
            trait_lines.append(f"{arrow} {name}: {val:+.2f} ({label})")

        cluster_cards.append(html.Div([
            html.Div([
                html.Span(f"Cluster {c}",
                          style={"fontWeight": 700, "fontSize": "0.75rem",
                                 "color": "var(--slate-800)"}),
                html.Span(f"  {n_c:,} obs. ({pct_c:.1f}%)",
                          style={"fontSize": "0.6875rem", "color": "var(--text-muted)"}),
            ]),
            html.Div([
                html.Div(t, style={"fontSize": "0.625rem", "color": "var(--text-secondary)",
                                    "marginTop": 2, "fontFamily": "monospace"})
                for t in trait_lines
            ], style={"marginTop": 4}),
        ], style={
            "background": "var(--bg-muted)", "borderRadius": "var(--radius-sm)",
            "border": "1px solid var(--border-light)", "padding": "10px 12px",
            "marginBottom": 8,
        }))

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
            html.Span("⚠ ", style={"fontWeight": 700, "color": "#92400e", "fontSize": "0.625rem"}),
            html.Span(
                "Profiles above are in standard deviation units (mean=0, std=1). "
                "A value of +1.5 means the cluster is 1.5 std above the global average.",
                style={"fontSize": "0.625rem", "color": "#92400e"},
            ),
        ], style={"marginTop": 8}),
    ]

    if desc := _ALG_DESCRIPTIONS.get(algorithm, ""):
        children += [html.Hr(className="section-divider"), _tip_box(desc, kind="info")]

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Elbow interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def elbow_interpretation(result: dict) -> html.Div:
    elbow   = result["elbow"]
    n_clust = result["characterization"]["n_clusters"]

    if not elbow["k_range"]:
        return html.Div("Not enough data for elbow analysis.",
                        style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16})

    inertias = elbow["inertias"]
    k_range  = elbow["k_range"]
    deltas   = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    pct_d    = [(inertias[i]-inertias[i+1])/max(inertias[i],1e-12)*100
                for i in range(len(inertias)-1)]

    suggested_k = None
    for i in range(1, len(deltas)):
        if deltas[i] < deltas[i-1] * 0.4:
            suggested_k = k_range[i]; break
    if suggested_k is None and k_range:
        suggested_k = k_range[-1]

    children = [
        _interp_header(
            "Elbow Method — Expert Reading Guide",
            subtitle=f"Inertia for k ∈ [{k_range[0]}, {k_range[-1]}]",
        ),
        _reading_grid([
            ("\u2198", "The curve always goes down",
             "Inertia decreases as k increases — more clusters = tighter fit."),
            ("📍", "The 'elbow' = optimal k",
             "Look for the bend where the curve flattens."),
            ("⚠", "The elbow can be ambiguous",
             "Cross-check with the Silhouette tab and domain knowledge."),
            ("📊", f"Current k = {n_clust}",
             f"{'Matches the elbow suggestion.' if suggested_k and n_clust == suggested_k else 'Elbow suggests k≈' + str(suggested_k) + '.' if suggested_k else ''}"),
        ]),
    ]

    roc_rows = []
    for i, k in enumerate(k_range):
        if i < len(deltas):
            flag = html.Span("Elbow ←", style={"color": "var(--green)", "fontWeight": 600}) \
                   if (deltas and deltas[i] < max(deltas) * 0.3) else ""
            roc_rows.append([f"k={k}", f"{inertias[i]:.1f}", f"{deltas[i]:.1f}",
                              f"{pct_d[i]:.1f}%", flag])
        else:
            roc_rows.append([f"k={k}", f"{inertias[i]:.1f}", "—", "—", ""])

    children += [
        html.Div("Rate-of-change analysis:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(["k", "Inertia", "Δ inertia", "Δ %", "Elbow?"], roc_rows, highlight_col=4),
        _tip_box(
            "How to choose k: (1) Find the elbow. "
            "(2) Check the Silhouette tab — prefer the silhouette k. "
            "(3) Validate with domain knowledge.",
            kind="expert",
        ),
        _tip_box(
            "The elbow assumes spherical clusters. "
            "For DBSCAN, it is a diagnostic helper only.",
            kind="warn",
        ),
    ]
    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Silhouette interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def silhouette_interpretation(result: dict) -> html.Div:
    sil_analysis = result["silhouette_analysis"]
    current_sil  = result["silhouette"]
    n_clust      = result["characterization"]["n_clusters"]
    algorithm    = result["algorithm"]

    if not sil_analysis["k_range"] or not sil_analysis["scores"]:
        return html.Div("Not enough data for silhouette analysis.",
                        style={"color": "#9ca3af", "fontSize": "0.75rem", "padding": 16})

    scores   = sil_analysis["scores"]
    k_range  = sil_analysis["k_range"]
    best_k   = max(scores, key=lambda k: scores[k])
    best_sil = scores[best_k]
    ranked   = sorted(scores.items(), key=lambda x: -x[1])

    if best_sil >= 0.50:
        verdict = "good structure"; verdict_col = "var(--green)"
    elif best_sil >= 0.25:
        verdict = "moderate structure"; verdict_col = "var(--amber)"
    else:
        verdict = "weak structure"; verdict_col = "var(--red)"

    children = [
        _interp_header(
            "Silhouette Score — Expert Reading Guide",
            subtitle=(
                f"Score range: [-1, 1] · Best k = {best_k} "
                f"(score = {best_sil:.4f}) · "
                f"Your k = {n_clust} ({current_sil:.4f})"
            ),
        ),
        _reading_grid([
            ("📊", "Higher = better separation",
             "Silhouette measures cohesion vs. separation. +1 = perfect, 0 = boundary, -1 = wrong cluster."),
            ("🎯", f"Best k = {best_k}",
             f"Gives the best silhouette ({best_sil:.4f}): {verdict}."),
            ("📏", "Thresholds",
             "> 0.70: Strong · > 0.50: Reasonable · > 0.25: Weak · < 0.25: No structure."),
            ("⚠", "Favours spherical clusters",
             "DBSCAN may produce excellent clusters with low silhouette due to non-spherical shapes."),
        ]),
    ]

    rank_rows = []
    for rank, (k, score) in enumerate(ranked):
        qual = (html.Span("Good",     style={"color": "var(--green)",  "fontWeight": 600}) if score >= 0.50 else
                html.Span("Moderate", style={"color": "#92400e",       "fontWeight": 600}) if score >= 0.25 else
                html.Span("Weak",     style={"color": "var(--red)",    "fontWeight": 600}))
        note = ""
        if k == best_k:
            note = html.Span("← Best",    style={"color": "var(--green)", "fontWeight": 700})
        elif k == n_clust:
            note = html.Span("← Current", style={"color": "var(--blue)",  "fontWeight": 700, "fontSize": "0.625rem"})
        rank_rows.append([f"{rank+1}", f"k={k}", f"{score:.4f}", qual, note])

    children += [
        html.Div("All k ranked:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(["Rank", "k", "Score", "Quality", "Note"], rank_rows, highlight_col=3),
    ]

    if n_clust == best_k:
        children.append(_tip_box(
            f"✅ Your k={n_clust} matches the silhouette optimum ({current_sil:.4f}): {verdict}.",
            kind="success" if best_sil >= 0.25 else "warn",
        ))
    else:
        diff = best_sil - current_sil
        children.append(_tip_box(
            f"Silhouette suggests k={best_k} ({best_sil:.4f}), you use k={n_clust} ({current_sil:.4f}). "
            + (f"Consider k={best_k} for better separation." if diff > 0.05
               else "The difference is small — keep your k if it makes domain sense."),
            kind="info",
        ))

    if algorithm == "dbscan":
        children.append(_tip_box(
            "DBSCAN silhouette is computed on core points only (noise excluded). "
            "High noise ratio may overestimate quality.",
            kind="warn",
        ))

    children.append(_tip_box(
        "Use silhouette as a GUIDE. k=3 at 0.52 and k=5 at 0.48 are both plausible — "
        "choose based on interpretability.",
        kind="expert",
    ))
    return html.Div(children)
