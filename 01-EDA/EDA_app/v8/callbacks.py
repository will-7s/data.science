"""
callbacks.py  —  Dash callback registration.

Split callbacks (chart → stats) for perceived performance.
Chart renders immediately (~ms), stats follow in parallel.
No dcc.Loading wrappers — fast enough to avoid spinner flash.
Figure caches in store.py for instant replay on variable switch.
"""
import numpy as np
import dash
from datetime import datetime
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import config
import store, loader, stats, charts, ui
from decorators import sample_random

_SAMPLE_N = config.SAMPLE_N

def _csv_quote(v: str) -> str:
    s = str(v)
    if ',' in s or '"' in s:
        escaped = s.replace('"', '""')
        return f'"{escaped}"'
    return s

# ── Computation kernels ────────────────────────────────────────────────────

def _normality_kernel(clean, col=""):
    return ui.normality_battery_panel(clean, col)


def _bivariate_chart_kernel(d1, d2, t1, t2, var1, var2):
    if t1 == "numeric" and t2 == "numeric":
        scatter_max = config.SCATTER_MAX_LIGHT if store.lightweight_mode else config.SCATTER_MAX
        fig   = charts.scatter(d1, d2, var1, var2, sampled=False, scatter_max=scatter_max)
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
    return fig, sp, html.Div(badge, className="pair-badge")


def _bivariate_stats_kernel(var1, var2, dataset=None, col_meta=None):
    ds = dataset if dataset is not None else store.dataset
    cm = col_meta if col_meta is not None else store.col_meta
    tests = stats.bivariate_test(var1, var2, ds, cm)
    return ui.test_panel(tests)


def register(app):

    # ── Upload ────────────────────────────────────────────────────────────────
    @app.callback(
        [Output("upload-status",       "children"),
         Output("overview-content",    "children"),
         Output("overview-content",    "style"),
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
            return ("No file uploaded", html.Div(), {"display": "none"}, {"display": "none"},
                    [], None, [], None, [], None)
        ok, msg = loader.load(contents, filename)
        if not ok:
            return (msg, html.Div(), {"display": "none"}, {"display": "none"},
                    [], None, [], None, [], None)

        store._plot_cache.clear()

        metadata = store.get_metadata()
        overview = ui.build_overview_card(metadata)

        opts     = [{"label": c, "value": c} for c in store.all_cols]
        default1 = store.num_cols[0] if store.num_cols else store.all_cols[0]
        if store.num_cols and len(store.num_cols) > 1:
            default2 = store.num_cols[1] if store.num_cols[1] != default1 else store.num_cols[0]
        elif store.cat_cols:
            default2 = store.cat_cols[0] if store.cat_cols[0] != default1 else (store.cat_cols[1] if len(store.cat_cols) > 1 else None)
        else:
            default2 = None

        status_msg = html.Div(f"✅ Loaded {filename}",
                              style={"color": "var(--green)", "fontSize": "0.875rem"})
        return (status_msg, overview, {"display": "block", "marginTop": 20}, {"display": "block"},
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

    # ── Univariate — chart + stats (fast, no normality delay) ──────────────────
    @app.callback(
        [Output("univariate-plot",  "figure"),
         Output("univariate-stats", "children")],
        [Input("univariate-variable", "value"),
         Input("plot-type",           "value")],
        prevent_initial_call=True,
    )
    def on_univariate(variable, plot_type):
        empty_fig = charts.empty()
        empty_div = html.Div()

        if not store.is_loaded() or not variable or variable not in store.dataset:
            return empty_fig, empty_div

        is_num = store.col_meta[variable] == "numeric"

        chart_key = ('uni_chart', variable, plot_type)
        cached = store._plot_cache.get(chart_key)
        if cached is not None:
            fig = cached
        else:
            if is_num:
                clean = store.clean_arrays[variable]
                fig = (charts.histogram(clean, variable)
                       if plot_type == "hist" else charts.boxplot(clean, variable))
            else:
                arr = store.dataset[variable]
                fig = (charts.bar_categorical(arr, variable)
                       if plot_type == "bar" else charts.pie_categorical(arr, variable))
            store._plot_cache[chart_key] = fig
            store.trim_plot_cache()

        stats_key = ('uni_desc', variable)
        cached_stats = store._plot_cache.get(stats_key)
        if cached_stats is not None:
            stats_panel = cached_stats
        else:
            if is_num:
                clean = store.clean_arrays[variable]
                stats_panel = ui.descriptive_stats_panel(clean, variable)
            else:
                arr = store.dataset[variable]
                stats_panel = ui.categorical_stats_panel(arr, variable)
            store._plot_cache[stats_key] = stats_panel
            store.trim_plot_cache()

        return fig, stats_panel

    # ── Univariate — normality (separate, computed after chart) ────────────────
    @app.callback(
        [Output("univariate-normality", "children"),
         Output("uni-norm-badge",       "children")],
        Input("univariate-variable", "value"),
        prevent_initial_call=True,
    )
    def on_univariate_normality(variable):
        empty_div = html.Div()
        empty_badge = html.Div()

        if not store.is_loaded() or not variable or variable not in store.dataset:
            return empty_div, empty_badge

        is_num = store.col_meta[variable] == "numeric"
        if not is_num:
            return empty_div, empty_badge

        clean = store.clean_arrays[variable]
        sampled = len(clean) > config.SAMPLE_THRESHOLD
        badge = ui.build_sampling_badge() if sampled else empty_badge
        store.track_computation_speed(sampled)

        norm_key = ('uni_norm', variable)
        cached = store._plot_cache.get(norm_key)
        if cached is not None:
            return cached, badge

        data = sample_random({variable: clean}, _SAMPLE_N)[variable] if sampled else clean
        try:
            norm_panel = _normality_kernel(data, variable)
        except Exception:
            norm_panel = html.Div([
                html.Div("Normality Tests", style={"fontWeight": 700, "fontSize": "0.8125rem", "marginBottom": 10}),
                html.Div("Could not compute", style={"color": "var(--red)", "fontSize": "0.75rem"}),
            ])
        store._plot_cache[norm_key] = norm_panel
        store.trim_plot_cache()
        return norm_panel, badge

    # ── Bivariate — chart + stats + correlation + insights (simplified) ────────
    @app.callback(
        [Output("bivariate-plot",       "figure"),
         Output("bivariate-stats",      "children"),
         Output("pairwise-correlation", "figure"),
         Output("bivariate-pair-type",  "children"),
         Output("bivariate-tests",      "children"),
         Output("correlation-insights", "children")],
        [Input("bivariate-var1", "value"),
         Input("bivariate-var2", "value")],
        prevent_initial_call=True,
    )
    def on_bivariate(var1, var2):
        ef = charts.empty()
        ed = html.Div()

        if not store.is_loaded() or not var1 or not var2:
            return ef, ed, ef, ed, ed, ed

        cached = store._plot_cache.get(('bi_combined', var1, var2))
        if cached is not None:
            return cached

        t1 = store.col_meta.get(var1)
        t2 = store.col_meta.get(var2)
        if t1 is None or t2 is None:
            return ef, ed, ef, ed, ed, ed

        d1, d2 = store.dataset[var1], store.dataset[var2]
        sampled = store.should_sample(d1)
        sampled_dict = {}
        if sampled:
            sampled_dict = sample_random({var1: d1, var2: d2}, _SAMPLE_N)
            d1, d2 = sampled_dict[var1], sampled_dict[var2]
        store.track_computation_speed(sampled)

        fig, sp, badge = _bivariate_chart_kernel(d1, d2, t1, t2, var1, var2)

        tests_panel = _bivariate_stats_kernel(var1, var2, dataset=sampled_dict if sampled else store.dataset)

        # Single corr matrix call for both heatmap and insights
        mat, cols = store.get_corr_matrix()
        corr_fig = charts.correlation_heatmap(mat, cols) if mat is not None else charts.empty("Need at least 2 numeric columns")
        corr_insights = (ui.correlation_insights_panel(mat, cols)
                         if mat is not None
                         else html.Div("Not enough numeric columns."))

        result = (fig, sp, corr_fig, badge, tests_panel, corr_insights)
        store._plot_cache[('bi_combined', var1, var2)] = result
        store.trim_plot_cache()
        return result

    # ── Computation state tracking ────────────────────────────────────────────
    @app.callback(
        Output("computation-state", "data"),
        Input("univariate-variable", "value"),
        Input("bivariate-var1", "value"),
        Input("bivariate-var2", "value"),
        prevent_initial_call=True,
    )
    def on_computation_start(*_):
        return "computing"

    @app.callback(
        Output("computation-state", "data", allow_duplicate=True),
        Input("univariate-plot", "figure"),
        Input("univariate-normality", "children"),
        Input("bivariate-plot", "figure"),
        Input("bivariate-tests", "children"),
        prevent_initial_call=True,
    )
    def on_computation_end(*_):
        return "idle"

    @app.callback(
        Output("cancel-btn", "disabled"),
        Input("computation-state", "data"),
    )
    def on_cancel_btn_state(state):
        return state != "computing"

    @app.callback(
        Output("cancel-status", "children"),
        Output("cancel-status", "style"),
        Input("cancel-btn", "n_clicks"),
        State("computation-state", "data"),
        prevent_initial_call=True,
    )
    def on_cancel_click(n_clicks, state):
        if not n_clicks:
            raise PreventUpdate
        store.cancel_event.set()
        return ui.build_cancelled_message(), {"display": "block", "textAlign": "center"},

    @app.callback(
        Output("cancel-status", "children", allow_duplicate=True),
        Output("cancel-status", "style", allow_duplicate=True),
        Input("computation-state", "data"),
        prevent_initial_call=True,
    )
    def on_state_reset(state):
        if state == "computing":
            return html.Div(), {"display": "none"}
        raise PreventUpdate

    # ── Lightweight mode badge ───────────────────────────────────────────────
    @app.callback(
        Output("lightweight-badge", "children"),
        Input("computation-state", "data"),
    )
    def on_lightweight_badge(state):
        if store.lightweight_mode:
            return ui.build_lightweight_badge()
        return html.Div()

    # ── Side panel collapse toggle ──────────────────────────────────────────
    @app.callback(
        Output("panel-collapsed", "data"),
        Output("uni-sidebar", "style"),
        Output("uni-graph", "width"),
        Input("uni-collapse-btn", "n_clicks"),
        State("panel-collapsed", "data"),
        prevent_initial_call=True,
    )
    def toggle_uni_sidebar(n_clicks, collapsed):
        new_state = not collapsed.get("univariate", False)
        hide_styles = {"display": "none", "overflow": "hidden", "padding": "0 !important"}
        return (
            {**collapsed, "univariate": new_state},
            hide_styles if new_state else {"overflow": "visible"},
            12 if new_state else 9,
        )

    @app.callback(
        Output("bi-sidebar", "style"),
        Output("bi-graph", "width"),
        Input("bi-collapse-btn", "n_clicks"),
        State("panel-collapsed", "data"),
        prevent_initial_call=True,
    )
    def toggle_bi_sidebar(n_clicks, collapsed):
        new_state = not collapsed.get("bivariate", False)
        collapsed["bivariate"] = new_state
        hide_styles = {"display": "none", "overflow": "hidden", "padding": "0 !important"}
        return (hide_styles if new_state else {"overflow": "visible"}), (12 if new_state else 9)

    # ── CSV export ──────────────────────────────────────────────────────────
    @app.callback(
        Output("download-csv", "data"),
        Input("uni-export-csv", "n_clicks"),
        State("univariate-variable", "value"),
        prevent_initial_call=True,
    )
    def export_uni_csv(n_clicks, variable):
        if not store.is_loaded() or not variable:
            return dash.no_update
        is_num = store.col_meta[variable] == "numeric"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_num:
            clean = store.clean_arrays[variable]
            s = stats.descriptive_stats(clean)
            pct = stats.outlier_percentage(clean)
            q1, q3 = np.quantile(clean, [0.25, 0.75])
            from scipy.stats import skew, kurtosis
            sk = float(skew(clean))
            ku = float(kurtosis(clean, fisher=True))
            n_missing = int(len(store.dataset[variable]) - s['n'])
            lines = [
                "Statistic,Value",
                f"N (non-missing),{s['n']}",
                f"N (missing),{n_missing}",
                f"Missing %,{100*n_missing/(s['n']+n_missing):.1f}",
                f"Mean,{s['mean']:.4f}",
                f"Median,{s['median']:.4f}",
                f"Mode,{s['mode']:.4f}",
                f"Mode frequency,{s['mode_freq']}",
                f"Std dev,{s['std']:.4f}",
                f"Min,{s['min']:.4f}",
                f"Max,{s['max']:.4f}",
                f"Q1,{q1:.4f}",
                f"Q3,{q3:.4f}",
                f"IQR,{q3-q1:.4f}",
                f"Skewness,{sk:.4f}",
                f"Kurtosis (excess),{ku:.4f}",
                f"Outliers %,{pct:.1f}",
            ]
            return dcc.send_string("\n".join(lines), f"{variable}_stats_{ts}.csv")
        else:
            arr = store.dataset[variable]
            vals, cnts = np.unique(arr, return_counts=True)
            order = np.argsort(cnts)[::-1]
            total = int(cnts.sum())
            lines = ["Category,Count,Percent"]
            for i in order:
                lines.append(f"{_csv_quote(vals[i])},{cnts[i]},{100*cnts[i]/total:.1f}")
            lines.append(f"Total,{total},100.0")
            return dcc.send_string("\n".join(lines), f"{variable}_freq_{ts}.csv")

    @app.callback(
        Output("download-csv", "data", allow_duplicate=True),
        Input("bi-export-csv", "n_clicks"),
        State("bivariate-var1", "value"),
        State("bivariate-var2", "value"),
        prevent_initial_call=True,
    )
    def export_bi_csv(n_clicks, var1, var2):
        if not store.is_loaded() or not var1 or not var2:
            return dash.no_update
        t1 = store.col_meta.get(var1)
        t2 = store.col_meta.get(var2)
        d1 = store.dataset[var1]
        d2 = store.dataset[var2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if t1 == "numeric" and t2 == "numeric":
            mask = ~(np.isnan(d1) | np.isnan(d2))
            n = int(mask.sum())
            if n < 2:
                return dash.no_update
            a, b = d1[mask], d2[mask]
            r_val   = float(np.corrcoef(a, b)[0, 1])
            cov_val = float(np.cov(a, b)[0, 1])
            from scipy.stats import pearsonr, spearmanr, kendalltau
            rp, pp  = pearsonr(a, b)
            rs, ps  = spearmanr(a, b)
            rk, pk  = kendalltau(a, b)
            lines = ["Variable1,Variable2,N,Pearson_r,Pearson_p,Spearman_rho,Spearman_p,Kendall_tau,Kendall_p,Covariance"]
            lines.append(f"{var1},{var2},{n},{rp:.6f},{pp:.6f},{rs:.6f},{ps:.6f},{rk:.6f},{pk:.6f},{cov_val:.6f}")
            return dcc.send_string("\n".join(lines), f"{var1}_x_{var2}_corr_{ts}.csv")
        elif t1 != t2:
            num, cat, nl, cl = (d1, d2, var1, var2) if t1 == "numeric" else (d2, d1, var2, var1)
            cat_uniq, cat_codes = np.unique(cat, return_inverse=True)
            sort_idx = np.argsort(cat_codes, kind='stable')
            num_sorted = num[sort_idx]
            cod_sorted = cat_codes[sort_idx]
            split_pts = np.where(np.diff(cod_sorted))[0] + 1
            groups = np.split(num_sorted, split_pts)
            lines = ["Group,N,Mean,Std,Min,Q1,Median,Q3,Max"]
            for c, g in zip(cat_uniq, groups):
                clean_g = g[~np.isnan(g)] if g.dtype.kind == 'f' else g
                if len(clean_g):
                    q1, q3 = np.quantile(clean_g, [0.25, 0.75])
                    lines.append(f"{_csv_quote(c)},{len(clean_g)},{clean_g.mean():.4f},{clean_g.std(ddof=1):.4f},{clean_g.min():.4f},{q1:.4f},{np.median(clean_g):.4f},{q3:.4f},{clean_g.max():.4f}")
            return dcc.send_string("\n".join(lines), f"{nl}_by_{cl}_groups_{ts}.csv")
        else:
            d1_u, d1_c = np.unique(d1, return_inverse=True)
            d2_u, d2_c = np.unique(d2, return_inverse=True)
            flat = np.ravel_multi_index((d1_c, d2_c), (len(d1_u), len(d2_u)))
            ct   = np.bincount(flat, minlength=len(d1_u)*len(d2_u)).reshape(len(d1_u), len(d2_u)).astype(int)
            lines = ["\\," + ",".join(_csv_quote(str(c)) for c in d2_u)]
            for i, r in enumerate(d1_u):
                lines.append(f"{_csv_quote(str(r))}," + ",".join(str(v) for v in ct[i]))
            return dcc.send_string("\n".join(lines), f"{var1}_x_{var2}_contingency_{ts}.csv")

    # ── Presentation mode ────────────────────────────────────────────────────
    @app.callback(
        Output("fullscreen-mode", "data"),
        Input("present-btn", "n_clicks"),
        State("fullscreen-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_presentation(n_clicks, fs_active):
        return not fs_active

    @app.callback(
        Output("panel-collapsed", "data", allow_duplicate=True),
        Output("uni-sidebar", "style", allow_duplicate=True),
        Output("uni-graph", "width", allow_duplicate=True),
        Output("bi-sidebar", "style", allow_duplicate=True),
        Output("bi-graph", "width", allow_duplicate=True),
        Input("fullscreen-mode", "data"),
        prevent_initial_call='initial_duplicate',
    )
    def on_fullscreen_layout(fs_active):
        if fs_active:
            hide = {"display": "none", "overflow": "hidden", "padding": "0 !important"}
            return ({"univariate": True, "bivariate": True}, hide, 12, hide, 12)
        return ({"univariate": False, "bivariate": False}, {}, 9, {}, 9)
