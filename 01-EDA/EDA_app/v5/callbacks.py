"""
callbacks.py  —  Dash callback registration (v5 optimised for deployment).

Architecture changes vs previous version
-----------------------------------------
Univariate split into TWO callbacks:
    on_univariate_chart  →  figure only             (~12 ms)
    on_univariate_stats  →  stats panel + normality  (~8 ms, large n ~50 ms)

    The chart fires first so the user sees the graph immediately.
    Stats and normality load in parallel (Dash fires both at once but
    serialises separately, so the lighter one arrives first in practice).

Bivariate split into TWO callbacks:
    on_bivariate_chart   →  figure + corr matrix + badge   (~15 ms)
    on_bivariate_stats   →  test panel + insights           (~10-50 ms)

    This means the scatter/boxplot appears ~30-40 ms before heavy
    bivariate tests (Levene + ANOVA + Kruskal + multiple t-tests).

prevent_initial_call=True on all interaction callbacks
    Prevents empty renders on app load before data is uploaded.

PCA callbacks unchanged — already O(1) cache reads.
"""
import numpy as np
from dash import html, dcc, Input, Output, State
import store, loader, stats, charts, ui
import pca as pca_mod
import charts_pca, ui_pca


def register(app):

    # ── Upload ────────────────────────────────────────────────────────────────
    @app.callback(
        [Output("upload-status",       "children"),
         Output("main-content",        "style"),
         Output("univariate-variable", "options"),
         Output("univariate-variable", "value"),
         Output("bivariate-var1",      "options"),
         Output("bivariate-var1",      "value"),
         Output("bivariate-var2",      "options"),
         Output("bivariate-var2",      "value"),
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
        _empty = ([], None, [], None, [], None)
        if contents is None:
            return ("No file uploaded", {"display": "none"},
                    [], None, [], None, [], None, *_empty)
        ok, msg = loader.load(contents, filename)
        if not ok:
            return (msg, {"display": "none"},
                    [], None, [], None, [], None, *_empty)

        opts     = [{"label": c, "value": c} for c in store.all_cols]
        default1 = store.num_cols[0] if store.num_cols else store.all_cols[0]
        if store.num_cols and len(store.num_cols) > 1:
            default2 = store.num_cols[1] if store.num_cols[1] != default1 else store.num_cols[0]
        elif store.cat_cols:
            default2 = store.cat_cols[0]
        else:
            default2 = default1

        result = pca_mod.run_pca(store.dataset, store.col_meta, store.col_stats)
        store.set_pca_cache(result)

        if result is None:
            return (msg, {"display": "block"},
                    opts, default1, opts, default1, opts, default2, *_empty)

        n_comp  = result["n_components"]
        pc_opts = [
            {"label": f"PC{i+1}  ({result['explained'][i]*100:.1f}%)", "value": i}
            for i in range(n_comp)
        ]
        return (msg, {"display": "block"},
                opts, default1, opts, default1, opts, default2,
                pc_opts, 0, pc_opts, 1, pc_opts, 0)

    # ── Chart-type selector ───────────────────────────────────────────────────
    @app.callback(
        [Output("plot-type", "options"),
         Output("plot-type", "value")],
        Input("univariate-variable", "value"),
        State("plot-type", "value"),
        prevent_initial_call=True,
    )
    def on_variable_changed(variable, current_type):
        if not store.is_loaded() or variable not in store.dataset:
            return [{"label": "Histogram", "value": "hist"}], "hist"
        is_num = store.col_meta[variable] == "numeric"
        if is_num:
            opts  = [{"label": "Histogram + KDE", "value": "hist"},
                     {"label": "Box plot",         "value": "box"}]
            valid = {"hist", "box"}
        else:
            opts  = [{"label": "Bar chart", "value": "bar"},
                     {"label": "Pie chart",  "value": "pie"}]
            valid = {"bar", "pie"}
        chosen = current_type if current_type in valid else ("hist" if is_num else "bar")
        return opts, chosen

    # ── Univariate — chart only (fast path, ~12 ms) ───────────────────────────
    @app.callback(
        Output("univariate-plot", "figure"),
        [Input("univariate-variable", "value"),
         Input("plot-type",           "value")],
        prevent_initial_call=True,
    )
    def on_univariate_chart(variable, plot_type):
        """Renders the chart only — fires before stats for immediate visual feedback."""
        if not store.is_loaded() or not variable or variable not in store.dataset:
            return charts.empty()
        is_num = store.col_meta[variable] == "numeric"
        if is_num:
            clean = store.clean_arrays[variable]
            return (charts.histogram(clean, variable)
                    if plot_type == "hist" else charts.boxplot(clean, variable))
        arr = store.dataset[variable]
        return (charts.bar_categorical(arr, variable)
                if plot_type == "bar" else charts.pie_categorical(arr, variable))

    # ── Univariate — stats + normality (may be slower on large n) ────────────
    @app.callback(
        [Output("univariate-stats",     "children"),
         Output("univariate-normality", "children")],
        [Input("univariate-variable", "value"),
         Input("plot-type",           "value")],
        prevent_initial_call=True,
    )
    def on_univariate_stats(variable, plot_type):
        """Stats panel and normality battery — separated so chart renders first."""
        if not store.is_loaded() or not variable or variable not in store.dataset:
            return html.Div(), html.Div()
        is_num = store.col_meta[variable] == "numeric"
        if is_num:
            clean       = store.clean_arrays[variable]
            stats_panel = ui.descriptive_stats_panel(clean, variable)
            norm_panel  = ui.normality_battery_panel(clean)
        else:
            arr         = store.dataset[variable]
            stats_panel = ui.categorical_stats_panel(arr, variable)
            norm_panel  = html.Div()
        return stats_panel, norm_panel

    # ── Bivariate — chart + correlation matrix (fast path, ~15 ms) ───────────
    @app.callback(
        [Output("bivariate-plot",       "figure"),
         Output("bivariate-stats",      "children"),
         Output("pairwise-correlation", "figure"),
         Output("bivariate-pair-type",  "children")],
        [Input("bivariate-var1", "value"),
         Input("bivariate-var2", "value")],
        prevent_initial_call=True,
    )
    def on_bivariate_chart(var1, var2):
        """Chart + correlation matrix — fires before statistical tests."""
        ef = charts.empty()
        ed = html.Div()

        # Correlation matrix — O(1) cache read
        mat, corr_cols = store.get_corr_matrix()
        corr_fig = (charts.correlation_heatmap(mat, corr_cols)
                    if mat is not None
                    else charts.empty("Need at least 2 numeric columns"))

        if not store.is_loaded() or not var1 or not var2:
            return ef, ed, corr_fig, ed

        t1 = store.col_meta.get(var1)
        t2 = store.col_meta.get(var2)
        if t1 is None or t2 is None:
            return ef, ed, corr_fig, ed

        d1, d2 = store.dataset[var1], store.dataset[var2]

        if t1 == "numeric" and t2 == "numeric":
            fig   = charts.scatter(d1, d2, var1, var2)
            mask  = ~(np.isnan(d1) | np.isnan(d2))
            r_val = float(np.corrcoef(d1[mask], d2[mask])[0, 1]) if mask.sum() > 1 else 0.0
            sp    = ui.correlation_panel(r_val)
            badge = "Numeric × Numeric — Scatter plot"
        elif t1 != t2:
            num, cat, nl, cl = (d1, d2, var1, var2) if t1 == "numeric" else (d2, d1, var2, var1)
            fig   = charts.grouped_boxplot(num, cat, nl, cl)
            sp    = ui.group_stats_panel(num, cat, nl, cl)
            badge = "Numeric × Categorical — Box plots"
        else:
            fig   = charts.heatmap_categorical(d1, d2, var1, var2)
            sp    = ui.association_panel()
            badge = "Categorical × Categorical — Contingency heatmap"

        return (fig, sp, corr_fig,
                html.Div(badge, className="pair-badge"))

    # ── Bivariate — statistical tests + insights (may be slower) ─────────────
    @app.callback(
        [Output("bivariate-tests",      "children"),
         Output("correlation-insights", "children")],
        [Input("bivariate-var1", "value"),
         Input("bivariate-var2", "value")],
        prevent_initial_call=True,
    )
    def on_bivariate_stats(var1, var2):
        """Statistical tests — separated so chart appears before heavy tests."""
        ed = html.Div()

        # Correlation insights — O(1) cache read
        mat, corr_cols = store.get_corr_matrix()
        corr_insights  = (ui.correlation_insights_panel(mat, corr_cols)
                          if mat is not None
                          else html.Div("Not enough numeric columns."))

        if not store.is_loaded() or not var1 or not var2:
            return ed, corr_insights
        if var1 not in store.dataset or var2 not in store.dataset:
            return ed, corr_insights

        tests = stats.bivariate_test(var1, var2, store.dataset, store.col_meta)
        return ui.test_panel(tests), corr_insights

    # ═══════════════════════════════════════════════════════════════════════════
    # PCA callbacks — all O(1) cache reads
    # ═══════════════════════════════════════════════════════════════════════════

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
        [Output("pca-circle",       "figure"),
         Output("pca-circle-interp","children")],
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
        fig   = charts_pca.correlation_circle(r, int(pc_x), int(pc_y),
                                              cos2_threshold=float(cos2_thresh or 0))
        panel = ui_pca.circle_interpretation_panel(r, int(pc_x), int(pc_y))
        return fig, panel

    @app.callback(
        [Output("pca-biplot",       "figure"),
         Output("pca-biplot-interp","children")],
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
