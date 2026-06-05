"""
ui.py  —  Reusable Dash HTML component factories.
No imports from store or charts. Pure (data → component) functions.
"""
from dash import html
import numpy as np
import stats
from utils import format_percent


# ── Generic building block ────────────────────────────────────────────────────

def stat_row(label: str, value) -> html.Div:
    return html.Div([
        html.Span(f"{label}: ", style={'fontWeight': '600', 'color': '#2c3e50'}),
        html.Span(str(value),   style={'color': '#555'}),
    ], style={'marginBottom': '4px', 'fontSize': '13px'})


# ── Univariate — numeric ──────────────────────────────────────────────────────

def descriptive_stats_panel(arr: np.ndarray, col: str) -> html.Div:
    """Descriptive stats shown below the chart in the right column."""
    s   = stats.descriptive_stats(arr)
    pct = stats.outlier_percentage(arr)
    if not s.get('n'):
        return html.Div("No data.")

    mode_str = f"{s['mode']:.4g}  (freq = {s['mode_freq']})"
    if s['mode_ties'] > 1:
        mode_str += f"  — {s['mode_ties']} tied modes (multimodal)"

    left_col = html.Div([
        stat_row("N",      s['n']),
        stat_row("Mean",   f"{s['mean']:.4f}"),
        stat_row("Median", f"{s['median']:.4f}"),
        stat_row("Mode",   mode_str),
    ], style={'flex': '1', 'minWidth': '180px'})

    right_col = html.Div([
        stat_row("Std dev", f"{s['std']:.4f}"),
        stat_row("Min",     f"{s['min']:.4f}"),
        stat_row("Max",     f"{s['max']:.4f}"),
        stat_row("Outliers (IQR × 1.5)", f"{pct:.1f} %"),
    ], style={'flex': '1', 'minWidth': '180px'})

    return html.Div([
        html.H5(f"Descriptive Statistics — {col}",
                style={'marginBottom': '10px', 'color': '#2c3e50', 'fontSize': '15px'}),
        html.Div([left_col, right_col],
                 style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '24px'}),
    ])


def normality_battery_panel(arr: np.ndarray) -> html.Div:
    """
    5-test normality battery shown in the left column below the chart-type radio.
    Skewness and kurtosis are displayed once at the top, not repeated per test.
    When n > _SUBSAMPLE_THRESHOLD, a subsampling info banner is shown.
    """
    results, shape = stats.run_normality_battery(arr)
    available   = [r for r in results if r.is_normal is not None]
    n_normal    = sum(1 for r in available if r.is_normal)
    n_run       = len(available)

    if n_run == 0:
        banner_text, banner_bg = "No test could run.", "#888"
    elif n_normal == n_run:
        banner_text, banner_bg = f"All {n_run} tests: Normal ✓", "#22c55e"
    elif n_normal == 0:
        banner_text, banner_bg = f"All {n_run} tests: Non-normal ✗", "#ef4444"
    else:
        banner_text = f"{n_normal}/{n_run} tests: Normal — inconclusive"
        banner_bg   = "#f59e0b"

    banner = html.Div(banner_text, style={
        'background': banner_bg, 'color': '#fff',
        'borderRadius': '5px', 'padding': '6px 10px',
        'fontWeight': '700', 'fontSize': '12px', 'marginBottom': '8px',
    })

    # ── Subsampling info banner (only when active) ────────────────────────────
    sub_banner = html.Div()
    if shape['subsampled']:
        sub_banner = html.Div([
            html.Span("⚡ Subsampling active — ", style={'fontWeight': '700'}),
            html.Span(
                f"n = {shape['n_full']:,} > {stats._SUBSAMPLE_THRESHOLD:,}. "
                f"Each test ran on {shape['subsample_k']} stratified subsamples "
                f"of {shape['subsample_n']:,} observations. "
                f"Statistic and p-value = median. "
                f"Skewness and kurtosis computed on the full array.",
            ),
        ], style={
            'background': '#fffbeb', 'border': '1px solid #fcd34d',
            'borderRadius': '5px', 'padding': '6px 10px',
            'fontSize': '11px', 'color': '#78350f', 'marginBottom': '8px',
            'lineHeight': '1.5',
        })

    # ── Shape moments row (always on full array) ──────────────────────────────
    shape_row = html.Div()
    if shape['skewness'] is not None:
        n_label = f"n = {shape['n_full']:,}"
        shape_row = html.Div([
            html.Span(f"{n_label}  |  Skewness: ",
                      style={'fontWeight': '600', 'fontSize': '11px', 'color': '#6b7280'}),
            html.Span(f"{shape['skewness']:+.3f}",
                      style={'fontSize': '11px', 'color': '#374151', 'fontFamily': 'monospace'}),
            html.Span("  |  Excess kurtosis: ",
                      style={'fontWeight': '600', 'fontSize': '11px', 'color': '#6b7280'}),
            html.Span(f"{shape['kurtosis']:+.3f}",
                      style={'fontSize': '11px', 'color': '#374151', 'fontFamily': 'monospace'}),
        ], style={
            'background': '#eef2ff', 'borderRadius': '4px',
            'padding': '4px 8px', 'marginBottom': '10px',
        })

    cards = [_normality_card(r) for r in results]
    return html.Div([
        html.Div("Normality Tests", style={
            'fontWeight': '700', 'fontSize': '13px',
            'marginBottom': '8px', 'color': '#2c3e50',
        }),
        banner,
        sub_banner,
        shape_row,
        *cards,
    ])


def _normality_card(r) -> html.Div:
    color = "#22c55e" if r.is_normal is True else "#ef4444" if r.is_normal is False else "#94a3b8"
    icon  = "✓" if r.is_normal is True else ("✗" if r.is_normal is False else "—")

    stat_line = ""
    if r.statistic is not None:
        stat_line = f"stat = {r.statistic:.4f}"
        if r.p_value is not None:
            stat_line += f",  p = {r.p_value:.4f}"

    children = [
        html.Div(f"{icon}  {r.name}",
                 style={'fontWeight': '700', 'fontSize': '12px', 'color': color}),
        html.Div(r.conclusion,
                 style={'fontSize': '11px', 'color': '#444', 'marginTop': '2px'}),
    ]
    if stat_line:
        children.append(html.Div(
            stat_line,
            style={'fontSize': '10px', 'color': '#777', 'marginTop': '2px'},
        ))
    if r.notes:
        children.append(html.Div(
            "  |  ".join(r.notes),
            style={'fontSize': '10px', 'color': '#aaa', 'marginTop': '2px'},
        ))

    return html.Div(children, style={
        'borderLeft':   f'4px solid {color}',
        'borderRadius': '4px',
        'padding':      '7px 10px',
        'marginBottom': '7px',
        'background':   '#f8f9fa',
    })


# ── Univariate — categorical ──────────────────────────────────────────────────

def categorical_stats_panel(arr: np.ndarray, col: str) -> html.Div:
    vals, cnts = np.unique(arr, return_counts=True)
    order      = np.argsort(cnts)[::-1]
    total      = int(cnts.sum())

    rows = [
        stat_row(str(vals[i]), f"{cnts[i]:,}  {format_percent(cnts[i], total)}")
        for i in order[:20]
    ]
    if len(vals) > 20:
        rows.append(html.Div(f"… and {len(vals)-20} more categories",
                             style={'color': '#999', 'fontSize': '11px', 'marginTop': '4px'}))

    return html.Div([
        html.H5(f"Statistics — {col}",
                style={'marginBottom': '10px', 'color': '#2c3e50', 'fontSize': '15px'}),
        stat_row("Unique categories",  len(vals)),
        stat_row("Total observations", f"{total:,}"),
        stat_row("Most frequent",
                 f"{vals[order[0]]}  ({cnts[order[0]]:,}  {format_percent(cnts[order[0]], total)})"),
        html.Hr(style={'margin': '10px 0'}),
        html.Div("Frequency Table",
                 style={'fontWeight': '700', 'fontSize': '13px', 'marginBottom': '6px'}),
        *rows,
    ])


# ── Bivariate panels ──────────────────────────────────────────────────────────

def correlation_panel(r: float) -> html.Div:
    strength  = "strong" if abs(r)>=0.7 else "moderate" if abs(r)>=0.4 else "weak"
    direction = "positive" if r >= 0 else "negative"
    return html.Div([
        html.H5("Pearson Correlation", style={'marginBottom': '8px'}),
        stat_row("r",        f"{r:.4f}"),
        stat_row("Strength", f"{strength} {direction}"),
    ])


def group_stats_panel(num: np.ndarray, cat: np.ndarray,
                      numl: str, catl: str) -> html.Div:
    from utils import drop_nan as dn
    rows = []
    for c in np.unique(cat):
        g = dn(num[cat == c])
        if g.size:
            rows.append(stat_row(str(c), f"μ = {g.mean():.4f}  (n = {len(g)})"))
    return html.Div([
        html.H5(f"{numl}  by  {catl}", style={'marginBottom': '8px'}),
        *rows,
    ])


def association_panel() -> html.Div:
    return html.Div([
        html.H5("Association", style={'marginBottom': '8px'}),
        html.Span("See statistical test results in the left panel.",
                  style={'fontSize': '13px', 'color': '#666'}),
    ])


# ── Statistical test panel ────────────────────────────────────────────────────

def test_panel(result: dict) -> html.Div:
    lines = []
    for line in result['results']:
        if line == "":
            lines.append(html.Br())

        elif line.startswith("GNORM_BANNER|"):
            # Format: GNORM_BANNER|colour_key|text
            _, colour_key, text = line.split("|", 2)
            bg_map = {"green": "#22c55e", "red": "#ef4444",
                      "orange": "#f59e0b", "grey": "#94a3b8"}
            bg = bg_map.get(colour_key, "#94a3b8")
            lines.append(html.Div(text, style={
                'background': bg, 'color': '#fff',
                'borderRadius': '5px', 'padding': '5px 10px',
                'fontWeight': '700', 'fontSize': '11px',
                'margin': '4px 0 4px 0',
            }))

        elif line.startswith("GNORM_CARD|"):
            # Format: GNORM_CARD|colour_key|title_line|note_line|shape_str
            parts = line.split("|", 4)
            _, colour_key, title, note, shape = parts
            border_map = {"green": "#22c55e", "red": "#ef4444",
                          "orange": "#f59e0b", "grey": "#94a3b8"}
            border = border_map.get(colour_key, "#94a3b8")
            txt_map = {"green": "#22c55e", "red": "#ef4444",
                       "orange": "#d97706", "grey": "#64748b"}
            txt_color = txt_map.get(colour_key, "#333")
            children = [
                html.Div(title, style={
                    'fontWeight': '700', 'fontSize': '11px', 'color': txt_color
                }),
            ]
            if note:
                children.append(html.Div(note, style={
                    'fontSize': '10px', 'color': '#444', 'marginTop': '1px'
                }))
            if shape:
                children.append(html.Div(shape, style={
                    'fontSize': '10px', 'color': '#888', 'marginTop': '1px'
                }))
            lines.append(html.Div(children, style={
                'borderLeft':   f'3px solid {border}',
                'borderRadius': '3px',
                'padding':      '5px 8px',
                'marginBottom': '4px',
                'background':   '#f8f9fa',
            }))

        elif line.startswith("──"):
            lines.append(html.Div(line, style={
                'fontWeight': '700', 'color': '#3b82f6',
                'marginTop': '8px', 'marginBottom': '2px', 'fontSize': '12px',
            }))
        elif line.startswith("  "):
            lines.append(html.Div(line, style={
                'fontSize': '11px', 'color': '#888', 'paddingLeft': '10px',
            }))
        else:
            color = '#22c55e' if '✓' in line else '#ef4444' if '✗' in line else '#333'
            lines.append(html.Div(line, style={
                'fontSize': '12px', 'fontFamily': 'monospace', 'color': color,
            }))
    return html.Div([
        html.H5(result['test_name'], style={'marginBottom': '8px', 'color': '#2c3e50'}),
        *lines,
    ])


# ── Normality sidebar (bivariate) ─────────────────────────────────────────────

def normality_panel(data: dict, num_cols: list) -> html.Div:
    """Compact Shapiro-Wilk summary for all numeric columns."""
    rows = []
    for col in num_cols:
        _, p = stats.normality_test(data[col])
        rows.append(stat_row(col, stats.normality_label(p)))
    return html.Div([
        html.Div("Normality (Shapiro-Wilk)", style={
            'fontWeight': '700', 'fontSize': '12px',
            'marginBottom': '6px', 'color': '#2c3e50',
        }),
        *rows,
    ])


# ── Correlation insights ──────────────────────────────────────────────────────

def correlation_insights_panel(mat: np.ndarray, cols: list) -> html.Div:
    n     = len(cols)
    pairs = sorted(
        [(cols[i], cols[j], mat[i, j]) for i in range(n) for j in range(i+1, n)],
        key=lambda x: abs(x[2]), reverse=True,
    )
    return html.Div([
        html.H5("Top Correlations", style={'marginBottom': '8px'}),
        *[stat_row(f"{a}  &  {b}", f"{v:+.4f}") for a, b, v in pairs[:5]],
    ])
