"""
callbacks.py  —  Dash callback registration (v6 — PCA + Clustering).

Architecture
------------
All callbacks are registered inside register(app) to keep app.py clean.
Interaction callbacks use prevent_initial_call=True so they do not fire
on page load before a dataset is uploaded — this avoids empty-state errors.

Performance strategy
--------------------
- PCA: computed once on upload (on_upload), cached in store._pca_cache.
       Every PCA callback is a plain O(1) cache read — no recomputation.
- Clustering: computed once when the user clicks "Run clustering".
       Result cached in store._clustering_cache.
       Switching tabs does NOT recompute; only the button click triggers.

Edge-case handling
------------------
Every callback returns graceful empty states (html.Div() for text,
charts_pca.empty_pca() / charts_clustering.empty_clustering() for graphs)
when data is missing or the computation produced no meaningful result.
"""

from dash import html, dcc, Input, Output, State
import store, loader
import pca as pca_mod
import charts_pca, ui_pca
import clustering as cl_mod
import charts_clustering, ui_clustering


def register(app):

    # ═══════════════════════════════════════════════════════════════════════════
    # Upload — fires once when a file is dropped in the upload zone.
    # Returns:
    #   - status message
    #   - show/hide main content
    #   - PCA axis dropdown options (populated from numeric columns)
    # Clustering dropdowns have hardcoded options — no upload-time population.
    # ═══════════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("upload-status",       "children"),
         Output("main-content",        "style"),
         Output("pca-pc-x",            "options"),
         Output("pca-pc-x",            "value"),
         Output("pca-pc-y",            "options"),
         Output("pca-pc-y",            "value"),
         Output("pca-pc-insight",      "options"),
         Output("pca-pc-insight",      "value"),
         ],
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def on_upload(contents, filename):
        """
        Decode → parse → dedup → classify → PCA (one-shot).

        The _empty tuple is reused for the 3 PCA dropdown tuples
        when there is no data or no numeric columns.

        Improvement suggestion
        ----------------------
        Split this into two callbacks: one for load+parse (fast, runs first)
        and one for PCA computation (may be slower for wide data).
        Dash's multi-output model would show the status message immediately
        while PCA computes in the background.
        """
        # ── Empty state template for PCA dropdowns (options, value) × 3 ────
        _empty = ([], None, [], None, [], None)

        if contents is None:
            return "No file uploaded", {"display": "none"}, *_empty

        ok, msg = loader.load(contents, filename)
        if not ok:
            return msg, {"display": "none"}, *_empty

        # ── PCA: compute once, cache forever ───────────────────────────────
        result = pca_mod.run_pca(store.dataset, store.col_meta, store.col_stats)
        store.set_pca_cache(result)

        if result is None:
            # Dataset uploaded but no numeric columns → PCA impossible.
            return msg, {"display": "block"}, *_empty

        n_comp  = result["n_components"]
        pc_opts = [
            {"label": f"PC{i+1}  ({result['explained'][i]*100:.1f}%)", "value": i}
            for i in range(n_comp)
        ]

        return (msg, {"display": "block"},
                pc_opts, 0,      # PC-X  → first PC
                pc_opts, 1,      # PC-Y  → second PC
                pc_opts, 0)      # insight → first PC

    # ═══════════════════════════════════════════════════════════════════════════
    # PCA callbacks — all O(1) cache reads
    # ═══════════════════════════════════════════════════════════════════════════

    def _pca():
        """Shortcut — wraps store.get_pca_cache() for readability."""
        return store.get_pca_cache()

    # ── PCA summary sidebar ──────────────────────────────────────────────────

    @app.callback(
        Output("pca-summary", "children"),
        Input("pca-pc-x", "value"),
        prevent_initial_call=True,
    )
    def on_pca_summary(_):
        """Overview panel: n_obs, n_vars, recommended PCs, first-2-PC breakdown."""
        r = _pca()
        if r is None:
            return html.Div(
                "No PCA available. Upload a dataset with \u2265 2 numeric columns.",
                style={"color": "#9ca3af", "fontSize": "12px", "marginTop": "12px"},
            )
        return ui_pca.pca_summary_panel(r)

    # ── Scree plot + eigenvalue table ────────────────────────────────────────

    @app.callback(
        [Output("pca-scree",                "figure"),
         Output("pca-eigenvalue-table-wrap", "children")],
        Input("pca-pc-x", "value"),
        prevent_initial_call=True,
    )
    def on_pca_scree(_):
        """
        Variance explained per PC (bar) + cumulative (line).
        Eigenvalue table rendered as a dcc.Graph with a go.Table.
        """
        r = _pca()
        if r is None:
            return charts_pca.empty_pca(), html.Div()
        return charts_pca.scree_plot(r), dcc.Graph(figure=charts_pca.eigenvalue_table(r))

    # ── Correlation circle ───────────────────────────────────────────────────

    @app.callback(
        [Output("pca-circle",       "figure"),
         Output("pca-circle-interp","children")],
        [Input("pca-pc-x",        "value"),
         Input("pca-pc-y",        "value"),
         Input("pca-cos2-filter", "value")],
        prevent_initial_call=True,
    )
    def on_pca_circle(pc_x, pc_y, cos2_thresh):
        """
        Variable loadings projected on the unit circle.
        cos² filter dims poorly-represented variables.
        """
        r = _pca()
        if r is None or pc_x is None or pc_y is None:
            return charts_pca.empty_pca(), html.Div()
        if int(pc_x) == int(pc_y):
            return charts_pca.empty_pca("Please select two different axes."), html.Div()

        fig   = charts_pca.correlation_circle(r, int(pc_x), int(pc_y),
                                              cos2_threshold=float(cos2_thresh or 0))
        panel = ui_pca.circle_interpretation_panel(r, int(pc_x), int(pc_y))
        return fig, panel

    # ── Biplot ───────────────────────────────────────────────────────────────

    @app.callback(
        [Output("pca-biplot",       "figure"),
         Output("pca-biplot-interp","children")],
        [Input("pca-pc-x", "value"),
         Input("pca-pc-y", "value")],
        prevent_initial_call=True,
    )
    def on_pca_biplot(pc_x, pc_y):
        """Individuals (rescaled) + variable arrows in a single view."""
        r = _pca()
        if r is None or pc_x is None or pc_y is None:
            return charts_pca.empty_pca(), html.Div()
        if int(pc_x) == int(pc_y):
            return charts_pca.empty_pca("Please select two different axes."), html.Div()

        fig   = charts_pca.biplot(r, int(pc_x), int(pc_y))
        panel = ui_pca.biplot_interpretation_panel(r, int(pc_x), int(pc_y))
        return fig, panel

    # ── Individuals plane ────────────────────────────────────────────────────

    @app.callback(
        [Output("pca-individuals",       "figure"),
         Output("pca-ind-insight-panel", "children"),
         Output("pca-ind-interp",        "children")],
        [Input("pca-pc-x",     "value"),
         Input("pca-pc-y",     "value"),
         Input("pca-color-by", "value"),
         Input("pca-top-ind",  "value")],
        prevent_initial_call=True,
    )
    def on_pca_individuals(pc_x, pc_y, color_by, top_n):
        """
        Score scatter coloured by cos² / contribution.
        Top-n contributors can be overlaid with labels.
        """
        r = _pca()
        if r is None or pc_x is None or pc_y is None:
            return charts_pca.empty_pca(), html.Div(), html.Div()
        if int(pc_x) == int(pc_y):
            return charts_pca.empty_pca("Please select two different axes."), html.Div(), html.Div()

        fig    = charts_pca.individuals_plane(r, int(pc_x), int(pc_y),
                                              color_by=color_by or "cos2",
                                              top_contrib_n=int(top_n or 0))
        panel  = ui_pca.individuals_insight_panel(r, int(pc_x), int(pc_y))
        interp = ui_pca.individuals_interpretation_panel(r, int(pc_x), int(pc_y))
        return fig, panel, interp

    # ── Axis insight (sidebar) ───────────────────────────────────────────────

    @app.callback(
        Output("pca-axis-insight-panel", "children"),
        Input("pca-pc-insight", "value"),
        prevent_initial_call=True,
    )
    def on_pca_axis_insight(pc_idx):
        """Per-PC breakdown: contributions, cos², correlation, direction."""
        r = _pca()
        if r is None or pc_idx is None:
            return html.Div()
        return ui_pca.axis_insight_panel(pca_mod.axis_insight(r, int(pc_idx)))

    # ── Variable contributions ───────────────────────────────────────────────

    @app.callback(
        [Output("pca-contrib-bar",     "figure"),
         Output("pca-contrib-heatmap", "figure"),
         Output("pca-contrib-interp",  "children")],
        Input("pca-pc-insight", "value"),
        prevent_initial_call=True,
    )
    def on_pca_contributions(pc_idx):
        """Bar chart per axis + multi-axis heatmap of contributions."""
        r = _pca()
        if r is None or pc_idx is None:
            return charts_pca.empty_pca(), charts_pca.empty_pca(), html.Div()

        n_show = min(r["n_components"], 6)      # show max 6 PCs in heatmap
        interp = ui_pca.contributions_interpretation_panel(r, int(pc_idx))
        return (charts_pca.contributions_bar(r, int(pc_idx)),
                charts_pca.contributions_heatmap(r, n_pc=n_show),
                interp)

    # ── cos² quality heatmap ─────────────────────────────────────────────────

    @app.callback(
        [Output("pca-cos2-heatmap", "figure"),
         Output("pca-cos2-interp",  "children")],
        Input("pca-pc-x", "value"),
        prevent_initial_call=True,
    )
    def on_pca_cos2(_):
        """Quality of representation heatmap (variables × PCs)."""
        r = _pca()
        if r is None:
            return charts_pca.empty_pca(), html.Div()

        n_show = min(r["n_components"], 6)
        interp = ui_pca.cos2_interpretation_panel(r, n_pc=n_show)
        return charts_pca.cos2_heatmap(r, n_pc=n_show), interp

    # ═══════════════════════════════════════════════════════════════════════════
    # Clustering callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Show / hide DBSCAN-specific params ───────────────────────────────────

    @app.callback(
        Output("clustering-params", "style"),
        Input("clustering-algorithm", "value"),
        prevent_initial_call=True,
    )
    def on_clustering_params(alg):
        """
        DBSCAN has extra hyper-parameters (eps, min_samples) that are
        irrelevant for K-Means / Hierarchical.  This callback toggles
        their visibility in the sidebar.
        """
        if alg == "dbscan":
            return {"display": "block"}
        return {"display": "none"}

    # ── Run clustering (button click) ────────────────────────────────────────

    @app.callback(
        [Output("clustering-summary",            "children"),
         Output("clustering-scatter",             "figure"),
         Output("clustering-elbow",               "figure"),
         Output("clustering-silhouette",           "figure"),
         Output("clustering-profiles",             "figure"),
         Output("clustering-scatter-interp",      "children"),
         Output("clustering-elbow-interp",        "children"),
         Output("clustering-silhouette-interp",   "children"),
         Output("clustering-profiles-interp",     "children"),
         Output("clustering-axis-insight-panel",  "children")],
        Input("clustering-run", "n_clicks"),
        [State("clustering-algorithm",    "value"),
         State("clustering-n-clusters",   "value"),
         State("clustering-eps",          "value"),
         State("clustering-min-samples",  "value"),
         State("clustering-linkage",      "value"),
         State("clustering-perplexity",   "value")],
        prevent_initial_call=True,
    )
    def on_clustering_run(n_clicks, algorithm, n_clusters, eps, min_samples,
                          linkage, perplexity):
        """
        Full clustering pipeline triggered by the "Run clustering" button.

        Flow
        ----
        1. Guard: no dataset loaded → show "Upload first" message.
        2. Compute: cl_mod.run_clustering() → cached in store.
        3. Edge case: no numeric columns → show explanatory message.
        4. Edge case: only 1 cluster found → show summary + elbow + silhouette
           but skip profiles (profiles heatmap needs ≥ 2 clusters).
        5. Nominal: all 5 output charts rendered.

        Performance
        -----------
        This is the only "heavy" callback in the clustering tab.
        Typical runtime for n=10k, n_vars=8, k_max=8:
          - t-SNE:  ~2-5 s (Barnes-Hut)
          - K-Means sequential (elbow+silhouette): ~1-2 s
          - Total:  ~3-7 s

        Improvement suggestion
        ----------------------
        Add a dcc.Loading spinner (already wrapped in _loading() in app.py)
        so the user has visual feedback during the 3-7 s computation window.
        """
        # ── Empty state shortcuts ──────────────────────────────────────────
        ed = html.Div()                              # empty div for text
        ef = charts_clustering.empty_clustering()    # empty chart

        if not store.is_loaded():
            return (html.Div("Upload a dataset first.",
                             style={"color": "#9ca3af", "fontSize": "12px"}),
                    ef, ef, ef, ef, ed, ed, ed, ed, ed)

        # ── Compute ────────────────────────────────────────────────────────
        result = cl_mod.run_clustering(
            store.dataset, store.col_meta,
            algorithm=algorithm,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples,
            linkage=linkage,
            perplexity=perplexity,
        )
        store.set_clustering_cache(result)

        if result is None:
            msg = html.Div("Need at least 2 numeric columns with non-NaN values.",
                           style={"color": "#9ca3af", "fontSize": "12px"})
            return (msg, ef, ef, ef, ef, ed, ed, ed, ed, ed)

        n_clust = result["characterization"]["n_clusters"]

        # ── 1 cluster found → partial output (skip profiles heatmap) ───────
        if n_clust < 2:
            msg = html.Div("Only 1 cluster found — try different parameters.",
                           style={"color": "#9ca3af", "fontSize": "12px"})
            return (ui_clustering.clustering_summary_panel(result),
                    charts_clustering.cluster_scatter(result),
                    charts_clustering.elbow_plot(result),
                    charts_clustering.silhouette_plot(result),
                    ef,
                    ui_clustering.cluster_scatter_interpretation(result),
                    ui_clustering.elbow_interpretation(result),
                    ui_clustering.silhouette_interpretation(result),
                    ed,
                    ui_clustering.cluster_axis_insight_panel(result))

        # ── Nominal path — all charts + interpretation panels ──────────────
        return (ui_clustering.clustering_summary_panel(result),
                charts_clustering.cluster_scatter(result),
                charts_clustering.elbow_plot(result),
                charts_clustering.silhouette_plot(result),
                charts_clustering.cluster_profiles_heatmap(result),
                ui_clustering.cluster_scatter_interpretation(result),
                ui_clustering.elbow_interpretation(result),
                ui_clustering.silhouette_interpretation(result),
                ui_clustering.profiles_interpretation(result),
                ui_clustering.cluster_axis_insight_panel(result))
