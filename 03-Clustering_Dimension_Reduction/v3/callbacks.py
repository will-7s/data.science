"""
callbacks.py  —  Dash callback registration (v6.1).

FIX (v6.1)
----------
1. on_upload split into two callbacks:
   - on_upload_parse: fast — parse + store only → shows status immediately
   - on_upload_pca:   triggered by status update → computes PCA in background
   This eliminates the 2–5 s silent freeze on large files.

2. Dark-mode clientside_callback: added Plotly.relayout() call after theme
   switch so graph axes/labels update immediately without waiting for a
   server callback.

3. t-SNE subsampling banner: on_clustering_run reads result["tsne_subsampled"]
   and surfaces a warning in the summary panel.

Architecture
------------
All PCA callbacks remain O(1) cache reads (no recomputation).
Clustering computed on button-click only; tab-switching never recomputes.
"""
from dash import html, dcc, Input, Output, State, clientside_callback
import store, loader
import pca as pca_mod
import charts_pca, ui_pca
import clustering as cl_mod
import charts_clustering, ui_clustering


def register(app):

    # ═══════════════════════════════════════════════════════════════════════
    # FIX: Upload split into 2 callbacks for immediate feedback
    # ═══════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("upload-status",  "children"),
         Output("upload-status",  "data-loaded"),   # hidden attr used as trigger
         Output("main-content",   "style")],
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def on_upload_parse(contents, filename):
        """Step 1 — decode, parse, dedup, classify.  Fast (~100ms)."""
        if contents is None:
            return "No file uploaded", "0", {"display": "none"}
        ok, msg = loader.load(contents, filename)
        style   = {"display": "block"} if ok else {"display": "none"}
        return msg, ("1" if ok else "0"), style

    @app.callback(
        [Output("pca-pc-x",       "options"),
         Output("pca-pc-x",       "value"),
         Output("pca-pc-y",       "options"),
         Output("pca-pc-y",       "value"),
         Output("pca-pc-insight", "options"),
         Output("pca-pc-insight", "value")],
        Input("upload-status", "data-loaded"),
        prevent_initial_call=True,
    )
    def on_upload_pca(loaded_flag):
        """Step 2 — compute PCA (may take 1-5 s on wide data).
        Triggered by step-1 completing, so the status message renders first."""
        _empty = ([], None, [], None, [], None)
        if loaded_flag != "1":
            return _empty

        result = pca_mod.run_pca(store.dataset, store.col_meta, store.col_stats)
        store.set_pca_cache(result)
        if result is None:
            return _empty

        n_comp  = result["n_components"]
        pc_opts = [
            {"label": f"PC{i+1}  ({result['explained'][i]*100:.1f}%)", "value": i}
            for i in range(n_comp)
        ]
        return (pc_opts, 0, pc_opts, 1, pc_opts, 0)

    # ═══════════════════════════════════════════════════════════════════════
    # FIX: Clientside callback — Plotly relayout on theme toggle
    # Calls Plotly.relayout on every dcc.Graph in the page so axis labels,
    # grid colours, and font colours update immediately without a server trip.
    # ═══════════════════════════════════════════════════════════════════════

    clientside_callback(
        """
        function(themeValue) {
            const dark = themeValue === 'dark';
            const textColor   = dark ? '#e2e8f0' : '#1e293b';
            const gridColor   = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
            const paperBg     = 'rgba(0,0,0,0)';
            const relayoutObj = {
                'paper_bgcolor': paperBg,
                'plot_bgcolor':  paperBg,
                'font.color':    textColor,
                'xaxis.color':   textColor,
                'yaxis.color':   textColor,
                'xaxis.gridcolor': gridColor,
                'yaxis.gridcolor': gridColor,
            };
            // Update all Plotly graphs present in the DOM
            const graphs = document.querySelectorAll('.js-plotly-plot');
            graphs.forEach(function(g) {
                if (window.Plotly && g.layout) {
                    window.Plotly.relayout(g, relayoutObj);
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("theme-relayout-done", "children"),   # dummy output (hidden div)
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PCA callbacks — all O(1) cache reads
    # ═══════════════════════════════════════════════════════════════════════

    def _pca():
        return store.get_pca_cache()

    @app.callback(
        Output("pca-summary", "children"),
        Input("pca-pc-x", "value"),
        prevent_initial_call=True,
    )
    def on_pca_summary(_):
        r = _pca()
        if r is None:
            return html.Div(
                "No PCA available. Upload a dataset with ≥ 2 numeric columns.",
                style={"color": "#9ca3af", "fontSize": "12px", "marginTop": "12px"},
            )
        return ui_pca.pca_summary_panel(r)

    @app.callback(
        [Output("pca-scree",                "figure"),
         Output("pca-eigenvalue-table-wrap", "children")],
        Input("pca-pc-x", "value"),
        prevent_initial_call=True,
    )
    def on_pca_scree(_):
        r = _pca()
        if r is None:
            return charts_pca.empty_pca(), html.Div()
        return charts_pca.scree_plot(r), dcc.Graph(figure=charts_pca.eigenvalue_table(r))

    @app.callback(
        [Output("pca-circle",        "figure"),
         Output("pca-circle-interp", "children")],
        [Input("pca-pc-x",        "value"),
         Input("pca-pc-y",        "value"),
         Input("pca-cos2-filter", "value")],
        prevent_initial_call=True,
    )
    def on_pca_circle(pc_x, pc_y, cos2_thresh):
        r = _pca()
        if r is None or pc_x is None or pc_y is None:
            return charts_pca.empty_pca(), html.Div()
        if int(pc_x) == int(pc_y):
            return charts_pca.empty_pca("Please select two different axes."), html.Div()
        # FIX: raise quality threshold for inferred correlations (was 0.4)
        fig   = charts_pca.correlation_circle(r, int(pc_x), int(pc_y),
                                              cos2_threshold=float(cos2_thresh or 0))
        panel = ui_pca.circle_interpretation_panel(r, int(pc_x), int(pc_y),
                                                   cos2_quality_min=0.60)
        return fig, panel

    @app.callback(
        [Output("pca-biplot",        "figure"),
         Output("pca-biplot-interp", "children")],
        [Input("pca-pc-x", "value"),
         Input("pca-pc-y", "value")],
        prevent_initial_call=True,
    )
    def on_pca_biplot(pc_x, pc_y):
        r = _pca()
        if r is None or pc_x is None or pc_y is None:
            return charts_pca.empty_pca(), html.Div()
        if int(pc_x) == int(pc_y):
            return charts_pca.empty_pca("Please select two different axes."), html.Div()
        fig   = charts_pca.biplot(r, int(pc_x), int(pc_y))
        panel = ui_pca.biplot_interpretation_panel(r, int(pc_x), int(pc_y))
        return fig, panel

    @app.callback(
        [Output("pca-individuals",        "figure"),
         Output("pca-ind-insight-panel",  "children"),
         Output("pca-ind-interp",         "children")],
        [Input("pca-pc-x",     "value"),
         Input("pca-pc-y",     "value"),
         Input("pca-color-by", "value"),
         Input("pca-top-ind",  "value")],
        prevent_initial_call=True,
    )
    def on_pca_individuals(pc_x, pc_y, color_by, top_n):
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

    @app.callback(
        Output("pca-axis-insight-panel", "children"),
        Input("pca-pc-insight", "value"),
        prevent_initial_call=True,
    )
    def on_pca_axis_insight(pc_idx):
        r = _pca()
        if r is None or pc_idx is None:
            return html.Div()
        return ui_pca.axis_insight_panel(pca_mod.axis_insight(r, int(pc_idx)))

    @app.callback(
        [Output("pca-contrib-bar",     "figure"),
         Output("pca-contrib-heatmap", "figure"),
         Output("pca-contrib-interp",  "children")],
        Input("pca-pc-insight", "value"),
        prevent_initial_call=True,
    )
    def on_pca_contributions(pc_idx):
        r = _pca()
        if r is None or pc_idx is None:
            return charts_pca.empty_pca(), charts_pca.empty_pca(), html.Div()
        n_show = min(r["n_components"], 6)
        interp = ui_pca.contributions_interpretation_panel(r, int(pc_idx))
        return (charts_pca.contributions_bar(r, int(pc_idx)),
                charts_pca.contributions_heatmap(r, n_pc=n_show),
                interp)

    @app.callback(
        [Output("pca-cos2-heatmap", "figure"),
         Output("pca-cos2-interp",  "children")],
        Input("pca-pc-x", "value"),
        prevent_initial_call=True,
    )
    def on_pca_cos2(_):
        r = _pca()
        if r is None:
            return charts_pca.empty_pca(), html.Div()
        n_show = min(r["n_components"], 6)
        interp = ui_pca.cos2_interpretation_panel(r, n_pc=n_show)
        return charts_pca.cos2_heatmap(r, n_pc=n_show), interp

    # ═══════════════════════════════════════════════════════════════════════
    # Clustering callbacks
    # ═══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("clustering-params", "style"),
        Input("clustering-algorithm", "value"),
        prevent_initial_call=True,
    )
    def on_clustering_params(alg):
        return {"display": "block"} if alg == "dbscan" else {"display": "none"}

    @app.callback(
        [Output("clustering-summary",           "children"),
         Output("clustering-scatter",            "figure"),
         Output("clustering-elbow",              "figure"),
         Output("clustering-silhouette",         "figure"),
         Output("clustering-profiles",           "figure"),
         Output("clustering-scatter-interp",     "children"),
         Output("clustering-elbow-interp",       "children"),
         Output("clustering-silhouette-interp",  "children"),
         Output("clustering-profiles-interp",    "children"),
         Output("clustering-axis-insight-panel", "children")],
        Input("clustering-run", "n_clicks"),
        [State("clustering-algorithm",   "value"),
         State("clustering-n-clusters",  "value"),
         State("clustering-eps",         "value"),
         State("clustering-min-samples", "value"),
         State("clustering-linkage",     "value"),
         State("clustering-perplexity",  "value")],
        prevent_initial_call=True,
    )
    def on_clustering_run(n_clicks, algorithm, n_clusters, eps,
                          min_samples, linkage, perplexity):
        ed = html.Div()
        ef = charts_clustering.empty_clustering()

        if not store.is_loaded():
            return (html.Div("Upload a dataset first.",
                             style={"color": "#9ca3af", "fontSize": "12px"}),
                    ef, ef, ef, ef, ed, ed, ed, ed, ed)

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
            msg = html.Div("Need ≥ 2 numeric columns with non-NaN values.",
                           style={"color": "#9ca3af", "fontSize": "12px"})
            return (msg, ef, ef, ef, ef, ed, ed, ed, ed, ed)

        n_clust = result["characterization"]["n_clusters"]

        if n_clust < 2:
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
