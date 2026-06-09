"""
callbacks.py  —  Dash callback registration.

Split callbacks (chart → stats) for perceived performance.
Chart renders immediately (~ms), stats follow in parallel.
No dcc.Loading wrappers — fast enough to avoid spinner flash.
Figure caches in store.py for instant replay on variable switch.
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
         Output("bivariate-var2",      "value"),
         ],
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def on_upload(contents, filename):
        if contents is None:
            return ("No file uploaded", {"display": "none"},
                    [], None, [], None, [], None)
        ok, msg = loader.load(contents, filename)
        if not ok:
            return (msg, {"display": "none"},
                    [], None, [], None, [], None)

        store._plot_cache.clear()

        opts     = [{"label": c, "value": c} for c in store.all_cols]
        default1 = store.num_cols[0] if store.num_cols else store.all_cols[0]
        if store.num_cols and len(store.num_cols) > 1:
            default2 = store.num_cols[1] if store.num_cols[1] != default1 else store.num_cols[0]
        elif store.cat_cols:
            default2 = store.cat_cols[0]
        else:
            default2 = default1

        return (msg, {"display": "block"},
                opts, default1, opts, default1, opts, default2)

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
            opts  = [{"label": "Histogram", "value": "hist"},
                     {"label": "Box plot",   "value": "box"}]
            valid = {"hist", "box"}
        else:
            opts  = [{"label": "Bar chart", "value": "bar"},
                     {"label": "Pie chart",  "value": "pie"}]
            valid = {"bar", "pie"}
        chosen = current_type if current_type in valid else ("hist" if is_num else "bar")
        return opts, chosen

    # ── Univariate — chart only (fast path) ───────────────────────────────────
    @app.callback(
        Output("univariate-plot", "figure"),
        [Input("univariate-variable", "value"),
         Input("plot-type",           "value")],
        prevent_initial_call=True,
    )
    def on_univariate_chart(variable, plot_type):
        if not store.is_loaded() or not variable or variable not in store.dataset:
            return charts.empty()

        key = ('uni_chart', variable, plot_type)
        cached = store._plot_cache.get(key)
        if cached is not None:
            return cached

        is_num = store.col_meta[variable] == "numeric"
        if is_num:
            clean = store.clean_arrays[variable]
            fig = (charts.histogram(clean, variable)
                   if plot_type == "hist" else charts.boxplot(clean, variable))
        else:
            arr = store.dataset[variable]
            fig = (charts.bar_categorical(arr, variable)
                   if plot_type == "bar" else charts.pie_categorical(arr, variable))

        store._plot_cache[key] = fig
        return fig

    # ── Univariate — descriptive stats (fast path) ───────────────────────────
    @app.callback(
        Output("univariate-stats", "children"),
        [Input("univariate-variable", "value"),
         Input("plot-type",          "value")],
        prevent_initial_call=True,
    )
    def on_univariate_stats_fast(variable, plot_type):
        if not store.is_loaded() or not variable or variable not in store.dataset:
            return html.Div()

        key = ('uni_desc', variable)
        cached = store._plot_cache.get(key)
        if cached is not None:
            return cached

        is_num = store.col_meta[variable] == "numeric"
        if is_num:
            clean = store.clean_arrays[variable]
            panel = ui.descriptive_stats_panel(clean, variable)
        else:
            arr  = store.dataset[variable]
            panel = ui.categorical_stats_panel(arr, variable)

        store._plot_cache[key] = panel
        return panel

    # ── Univariate — normality battery (slower, runs in parallel) ────────────
    @app.callback(
        Output("univariate-normality", "children"),
        [Input("univariate-variable", "value"),
         Input("plot-type",           "value")],
        prevent_initial_call=True,
    )
    def on_univariate_normality(variable, plot_type):
        if not store.is_loaded() or not variable or variable not in store.dataset:
            return html.Div()

        is_num = store.col_meta[variable] == "numeric"
        if not is_num:
            return html.Div()

        key = ('uni_norm', variable)
        cached = store._plot_cache.get(key)
        if cached is not None:
            return cached

        clean = store.clean_arrays[variable]
        panel = ui.normality_battery_panel(clean)
        store._plot_cache[key] = panel
        return panel

    # ── Bivariate — chart + corr matrix (fast path) ──────────────────────────
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
        ef = charts.empty()
        ed = html.Div()

        mat, corr_cols = store.get_corr_matrix()
        corr_fig = (charts.correlation_heatmap(mat, corr_cols)
                    if mat is not None
                    else charts.empty("Need at least 2 numeric columns"))

        if not store.is_loaded() or not var1 or not var2:
            return ef, ed, corr_fig, ed

        key = ('bi_chart', var1, var2)
        cached = store._plot_cache.get(key)
        if cached is not None:
            return cached

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

        result = (fig, sp, corr_fig, html.Div(badge, className="pair-badge"))
        store._plot_cache[key] = result
        return result

    # ── Bivariate — statistical tests + insights (parallel) ──────────────────
    @app.callback(
        [Output("bivariate-tests",      "children"),
         Output("correlation-insights", "children")],
        [Input("bivariate-var1", "value"),
         Input("bivariate-var2", "value")],
        prevent_initial_call=True,
    )
    def on_bivariate_stats(var1, var2):
        ed = html.Div()

        mat, corr_cols = store.get_corr_matrix()
        corr_insights  = (ui.correlation_insights_panel(mat, corr_cols)
                          if mat is not None
                          else html.Div("Not enough numeric columns."))

        if not store.is_loaded() or not var1 or not var2:
            return ed, corr_insights
        if var1 not in store.dataset or var2 not in store.dataset:
            return ed, corr_insights

        key = ('bi_stats', var1, var2)
        cached = store._plot_cache.get(key)
        if cached is not None:
            return cached

        tests = stats.bivariate_test(var1, var2, store.dataset, store.col_meta)
        result = (ui.test_panel(tests), corr_insights)
        store._plot_cache[key] = result
        return result
