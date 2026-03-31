"""
ui.py
─────
Reusable Dash HTML snippets.

Rules
-----
- No imports from store, stats, or charts.
- Pure functions: (data) → dash html component.
- Keep markup minimal; style belongs in the stylesheet.
"""

from dash import html
import numpy as np

import stats


# ── generic row ───────────────────────────────────────────────────────────────

def stat_row(label: str, value: str) -> html.Div:
    return html.Div([
        html.Span(f"{label}: ", className="stat-label"),
        html.Span(value, className="stat-value"),
    ])


# ── univariate panels ─────────────────────────────────────────────────────────

def numeric_stats_panel(arr: np.ndarray, col: str) -> html.Div:
    s   = stats.descriptive_stats(arr)
    pct = stats.outlier_percentage(arr)
    _, p_val = stats.normality_test(arr)

    return html.Div([
        html.H4(f"Statistics — {col}"),
        stat_row("Mean",             f"{s['mean']:.2f}"),
        stat_row("Median",           f"{s['median']:.2f}"),
        stat_row("Std dev",          f"{s['std']:.2f}"),
        stat_row("Min / Max",        f"{s['min']:.2f} / {s['max']:.2f}"),
        stat_row("N (non-missing)",  str(s['n'])),
        stat_row("Potential outliers (IQR ×1.5)", f"{pct:.1f} %"),
        html.Hr(),
        stat_row("Normality (Shapiro-Wilk)", stats.normality_label(p_val)),
    ])


def categorical_stats_panel(arr: np.ndarray, col: str) -> html.Div:
    values, counts = np.unique(arr, return_counts=True)
    top = np.argmax(counts)
    return html.Div([
        html.H4(f"Statistics — {col}"),
        stat_row("Unique categories", str(len(values))),
        stat_row("Most frequent",     f"{values[top]}  ({counts[top]} rows)"),
    ])


# ── bivariate panels ──────────────────────────────────────────────────────────

def correlation_panel(r: float) -> html.Div:
    strength = (
        "strong"   if abs(r) >= 0.7 else
        "moderate" if abs(r) >= 0.4 else
        "weak"
    )
    direction = "positive" if r >= 0 else "negative"
    return html.Div([
        html.H4("Pearson Correlation"),
        stat_row("r", f"{r:.3f}"),
        stat_row("Strength", f"{strength} {direction}"),
    ])


def group_stats_panel(num_arr: np.ndarray, cat_arr: np.ndarray,
                      num_col: str, cat_col: str) -> html.Div:
    means = stats.group_means(num_arr, cat_arr)
    return html.Div([
        html.H4(f"{num_col} by {cat_col}"),
        *[stat_row(cat, f"μ = {mean:.2f}") for cat, mean in means.items()],
    ])


def association_panel() -> html.Div:
    return html.Div([
        html.H4("Association"),
        html.Span("See Chi-square results →", className="stat-value"),
    ])


# ── statistical test panel ────────────────────────────────────────────────────

def test_panel(result: dict) -> html.Div:
    return html.Div([
        html.H4(result['test_name']),
        html.Div([
            html.Div(line, className="stat-value")
            for line in result['results']
        ]),
    ])


# ── normality panel (all numeric columns at once) ─────────────────────────────

def normality_panel(col_data: dict, num_cols: list[str]) -> html.Div:
    rows = []
    for col in num_cols:
        _, p = stats.normality_test(col_data[col])
        rows.append(stat_row(col, stats.normality_label(p)))
    return html.Div([html.H4("Normality Tests"), *rows])


# ── correlation insights ──────────────────────────────────────────────────────

def correlation_insights_panel(matrix: np.ndarray, col_names: list[str]) -> html.Div:
    n = len(col_names)
    pairs = sorted(
        [(col_names[i], col_names[j], matrix[i, j])
         for i in range(n) for j in range(i + 1, n)],
        key=lambda x: abs(x[2]),
        reverse=True,
    )
    return html.Div([
        html.H4("Top Correlations"),
        *[stat_row(f"{a} & {b}", f"{v:.3f}") for a, b, v in pairs[:5]],
    ])
