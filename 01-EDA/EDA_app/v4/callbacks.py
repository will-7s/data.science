"""
callbacks.py  —  Dash callback registration.

Performance design
------------------
Correlation matrix
    store.get_corr_matrix() is O(1) — result computed once at upload time
    and cached.  on_bivariate reads from the cache on every call; the matrix
    re-renders only when the user changes var1/var2 (which is correct UX),
    but the underlying numpy work is never repeated.

    The previous split into a separate on_corr_matrix callback (triggered
    only by upload) broke the display: pairwise-correlation and
    correlation-insights were never populated when the user interacted with
    the bivariate dropdowns.  Merged back into on_bivariate — cost is
    negligible because the cache hit is O(1).

clean arrays
    on_univariate reads store.clean_arrays[variable] (pre-stripped at load
    time) instead of re-running drop_nan on every callback.

on_bivariate
    NaN mask computed once with a single boolean expression.
    Dead code (duplicate mask line) removed.
"""
import numpy as np
from dash import html, Input, Output, State
import store, loader, stats, charts, ui


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
         Output("bivariate-var2",      "value")],
        [Input("upload-data", "contents")],
        [State("upload-data", "filename")],
    )
    def on_upload(contents, filename):
        if contents is None:
            return "No file uploaded", {"display": "none"}, [], None, [], None, [], None
        ok, msg = loader.load(contents, filename)
        if not ok:
            return msg, {"display": "none"}, [], None, [], None, [], None
        opts     = [{"label": c, "value": c} for c in store.all_cols]
        default1 = store.num_cols[0] if store.num_cols else store.all_cols[0]
        if store.num_cols and len(store.num_cols) > 1:
            default2 = store.num_cols[1] if store.num_cols[1] != default1 else store.num_cols[0]
        elif store.cat_cols:
            default2 = store.cat_cols[0]
        else:
            default2 = default1
        return msg, {"display": "block"}, opts, default1, opts, default1, opts, default2

    # ── Chart-type selector ───────────────────────────────────────────────────
    @app.callback(
        [Output("plot-type", "options"),
         Output("plot-type", "value")],
        [Input("univariate-variable", "value")],
        [State("plot-type", "value")],
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

    # ── Univariate ────────────────────────────────────────────────────────────
    @app.callback(
        [Output("univariate-plot",      "figure"),
         Output("univariate-stats",     "children"),
         Output("univariate-normality", "children")],
        [Input("univariate-variable", "value"),
         Input("plot-type",           "value")],
    )
    def on_univariate(variable, plot_type):
        if not store.is_loaded() or variable not in store.dataset:
            return charts.empty(), html.Div(), html.Div()

        arr    = store.dataset[variable]
        is_num = store.col_meta[variable] == "numeric"

        if is_num:
            # Pre-stripped array from store — no re-scanning NaN on every call
            clean       = store.clean_arrays[variable]
            fig         = (charts.histogram(clean, variable)
                           if plot_type == "hist"
                           else charts.boxplot(clean, variable))
            stats_panel = ui.descriptive_stats_panel(clean, variable)
            norm_panel  = ui.normality_battery_panel(clean)
        else:
            fig         = (charts.bar_categorical(arr, variable)
                           if plot_type == "bar"
                           else charts.pie_categorical(arr, variable))
            stats_panel = ui.categorical_stats_panel(arr, variable)
            norm_panel  = html.Div()

        return fig, stats_panel, norm_panel

    # ── Bivariate ─────────────────────────────────────────────────────────────
    # Outputs include pairwise-correlation and correlation-insights so that
    # they are always visible in the bivariate tab regardless of which
    # variable pair is selected.  The correlation matrix itself is served
    # from store.get_corr_matrix() — O(1) cache hit, no numpy recomputation.
    @app.callback(
        [Output("bivariate-plot",       "figure"),
         Output("bivariate-stats",      "children"),
         Output("bivariate-tests",      "children"),
         Output("pairwise-correlation", "figure"),
         Output("correlation-insights", "children"),
         Output("bivariate-pair-type",  "children")],
        [Input("bivariate-var1", "value"),
         Input("bivariate-var2", "value")],
    )
    def on_bivariate(var1, var2):
        empty_fig = charts.empty()
        empty_div = html.Div()

        # ── Correlation matrix — O(1) cache read ─────────────────────────────
        mat, corr_cols = store.get_corr_matrix()
        if mat is not None:
            corr_fig    = charts.correlation_heatmap(mat, corr_cols)
            corr_insights = ui.correlation_insights_panel(mat, corr_cols)
        else:
            corr_fig    = charts.empty("Need at least 2 numeric columns")
            corr_insights = html.Div("Not enough numeric columns for correlation matrix.")

        if not store.is_loaded() or var1 is None or var2 is None:
            return empty_fig, empty_div, empty_div, corr_fig, corr_insights, empty_div

        t1, t2 = store.col_meta[var1], store.col_meta[var2]
        d1, d2 = store.dataset[var1],  store.dataset[var2]

        if t1 == "numeric" and t2 == "numeric":
            fig         = charts.scatter(d1, d2, var1, var2)
            # Single boolean mask — no duplicate computation
            mask        = ~(np.isnan(d1) | np.isnan(d2))
            r           = float(np.corrcoef(d1[mask], d2[mask])[0, 1]) if mask.sum() > 1 else 0.0
            stats_panel = ui.correlation_panel(r)
            badge       = "Numeric × Numeric — Scatter plot"

        elif t1 != t2:
            if t1 == "numeric":
                num, cat, numl, catl = d1, d2, var1, var2
            else:
                num, cat, numl, catl = d2, d1, var2, var1
            fig         = charts.grouped_boxplot(num, cat, numl, catl)
            stats_panel = ui.group_stats_panel(num, cat, numl, catl)
            badge       = "Numeric × Categorical — Box plots"

        else:
            fig         = charts.heatmap_categorical(d1, d2, var1, var2)
            stats_panel = ui.association_panel()
            badge       = "Categorical × Categorical — Contingency heatmap"

        tests = stats.bivariate_test(var1, var2, store.dataset, store.col_meta)

        return (
            fig,
            stats_panel,
            ui.test_panel(tests),
            corr_fig,
            corr_insights,
            html.Div(badge, style={"fontWeight": "bold", "marginTop": "8px"}),
        )
