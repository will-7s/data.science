"""
callbacks.py
────────────
Dash callback registration — the orchestration layer.

Each callback does exactly three things:
  1. Read from `store` (the dataset).
  2. Call business-logic functions (stats / charts / ui).
  3. Return Dash component(s).

No computation happens here.  If a function call grows beyond one line,
it belongs in stats.py, charts.py, or ui.py instead.
"""

import numpy as np
from dash import html, Input, Output, State

import store
import loader
import stats
import charts
import ui


def register(app) -> None:
    """Attach all callbacks to *app*. Called once at startup."""

    # ── 1. File upload ────────────────────────────────────────────────────────

    @app.callback(
        [Output('upload-status',       'children'),
         Output('main-content',        'style'),
         Output('univariate-variable', 'options'),
         Output('univariate-variable', 'value'),
         Output('bivariate-var1',      'options'),
         Output('bivariate-var1',      'value'),
         Output('bivariate-var2',      'options'),
         Output('bivariate-var2',      'value')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')],
    )
    def on_upload(contents, filename):
        if contents is None:
            return "No file uploaded", _hidden, [], None, [], None, [], None

        success, message = loader.load(contents, filename)

        if not success:
            return message, _hidden, [], None, [], None, [], None

        options   = [{'label': c, 'value': c} for c in store.all_cols]
        default1  = store.num_cols[0] if store.num_cols else store.all_cols[0]
        default2  = _pick_second(default1)

        return (message, _visible,
                options, default1,
                options, default1,
                options, default2)

    # ── 2. Univariate tab ─────────────────────────────────────────────────────

    @app.callback(
        [Output('univariate-plot',  'figure'),
         Output('univariate-stats', 'children')],
        [Input('univariate-variable', 'value'),
         Input('plot-type',           'value')],
    )
    def on_univariate(variable, plot_type):
        if not store.is_loaded() or variable not in store.dataset:
            return charts.empty(), html.Div("No data loaded.")

        arr = store.dataset[variable]

        if store.col_meta[variable] == 'numeric':
            clean = arr[~np.isnan(arr)]
            fig   = _numeric_chart(clean, variable, plot_type)
            panel = ui.numeric_stats_panel(clean, variable)
        else:
            fig   = charts.bar_categorical(arr, variable)
            panel = ui.categorical_stats_panel(arr, variable)

        return fig, panel

    # ── 3. Bivariate tab ──────────────────────────────────────────────────────

    @app.callback(
        [Output('bivariate-plot',        'figure'),
         Output('bivariate-stats',       'children'),
         Output('bivariate-tests',       'children'),
         Output('normality-tests',       'children'),
         Output('pairwise-correlation',  'figure'),
         Output('correlation-insights',  'children')],
        [Input('bivariate-var1', 'value'),
         Input('bivariate-var2', 'value')],
    )
    def on_bivariate(var1, var2):
        _empty = charts.empty()
        _none  = html.Div()

        if not store.is_loaded() or var1 is None or var2 is None:
            return _empty, _none, _none, _none, _empty, _none

        t1 = store.col_meta.get(var1, 'categorical')
        t2 = store.col_meta.get(var2, 'categorical')
        d1, d2 = store.dataset[var1], store.dataset[var2]

        # ── bivariate figure + stats panel ────────────────────────────────
        biv_fig, stats_panel = _bivariate_chart_and_stats(var1, var2, t1, t2, d1, d2)

        # ── statistical test ──────────────────────────────────────────────
        test_result  = stats.bivariate_test(var1, var2, store.dataset, store.col_meta)
        tests_panel  = ui.test_panel(test_result)

        # ── normality of all numeric columns ─────────────────────────────
        norm_panel   = ui.normality_panel(store.dataset, store.num_cols)

        # ── correlation matrix ────────────────────────────────────────────
        matrix, cols = stats.correlation_matrix(store.dataset, store.num_cols)

        if matrix is not None:
            corr_fig      = charts.correlation_heatmap(matrix, cols)
            insights_panel = ui.correlation_insights_panel(matrix, cols)
        else:
            corr_fig      = charts.empty("Need 2+ numeric columns")
            insights_panel = html.Div("Not enough numeric columns.")

        return biv_fig, stats_panel, tests_panel, norm_panel, corr_fig, insights_panel


# ── private helpers ───────────────────────────────────────────────────────────

_hidden  = {'display': 'none'}
_visible = {'display': 'block'}


def _pick_second(default1: str) -> str:
    """Pick a sensible second variable different from default1."""
    for col in store.num_cols:
        if col != default1:
            return col
    for col in store.cat_cols:
        if col != default1:
            return col
    return default1   # only one column in the file


def _numeric_chart(arr, col, plot_type):
    dispatch = {
        'histogram': charts.histogram,
        'box':       charts.boxplot,
        'bar':       charts.bar_numeric,
    }
    return dispatch.get(plot_type, charts.histogram)(arr, col)


def _bivariate_chart_and_stats(var1, var2, t1, t2, d1, d2):
    """Return (figure, stats_panel) for the chosen variable pair."""

    # numeric × numeric
    if t1 == 'numeric' and t2 == 'numeric':
        fig   = charts.scatter(d1, d2, var1, var2)
        mask  = ~(np.isnan(d1) | np.isnan(d2))
        r     = np.corrcoef(d1[mask], d2[mask])[0, 1]
        panel = ui.correlation_panel(r)
        return fig, panel

    # numeric × categorical (either order)
    if t1 != t2:
        num_var = var1 if t1 == 'numeric' else var2
        cat_var = var2 if t1 == 'numeric' else var1
        num_d   = store.dataset[num_var]
        cat_d   = store.dataset[cat_var]
        fig     = charts.grouped_boxplot(num_d, cat_d, num_var, cat_var)
        panel   = ui.group_stats_panel(num_d, cat_d, num_var, cat_var)
        return fig, panel

    # categorical × categorical
    fig   = charts.heatmap_categorical(d1, d2, var1, var2)
    panel = ui.association_panel()
    return fig, panel
