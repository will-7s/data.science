from dash import html
import numpy as np
import stats
from utils import format_percent, drop_nan as dn


def stat_row(label: str, value):
    return html.Div([
        html.Span(f"{label}:", className="label"),
        html.Span(str(value), className="value"),
    ], className="stat-row")


def descriptive_stats_panel(arr: np.ndarray, col: str) -> html.Div:
    s = stats.descriptive_stats(arr)
    pct = stats.outlier_percentage(arr)
    if not s.get('n'):
        return html.Div("No data.")

    mode_str = f"{s['mode']:.4g}  (freq = {s['mode_freq']})"
    if s['mode_ties'] > 1:
        mode_str += f"  — {s['mode_ties']} tied modes (multimodal)"

    left_col = html.Div([
        stat_row("N", s['n']),
        stat_row("Mean", f"{s['mean']:.4f}"),
        stat_row("Median", f"{s['median']:.4f}"),
        stat_row("Mode", mode_str),
    ], style={'flex': '1', 'minWidth': '180px'})

    right_col = html.Div([
        stat_row("Std dev", f"{s['std']:.4f}"),
        stat_row("Min", f"{s['min']:.4f}"),
        stat_row("Max", f"{s['max']:.4f}"),
        stat_row("Outliers (IQR × 1.5)", f"{pct:.1f} %"),
    ], style={'flex': '1', 'minWidth': '180px'})

    return html.Div([
        html.H5(f"Descriptive Statistics — {col}",
                style={'marginBottom': 12, 'color': 'var(--slate-800)',
                       'fontSize': '0.9375rem', 'fontWeight': 600}),
        html.Div([left_col, right_col],
                 style={'display': 'flex', 'flexWrap': 'wrap', 'gap': 24}),
    ])


def normality_battery_panel(arr: np.ndarray) -> html.Div:
    results, shape = stats.run_normality_battery(arr)
    available = [r for r in results if r.is_normal is not None]
    n_normal = sum(1 for r in available if r.is_normal)
    n_run = len(available)

    if n_run == 0:
        banner_text, banner_bg = "No test could run.", "#64748b"
    elif n_normal == n_run:
        banner_text, banner_bg = "All tests: Normal ✓", "var(--green)"
    elif n_normal == 0:
        banner_text, banner_bg = "All tests: Non-normal ✗", "var(--red)"
    else:
        banner_text = f"{n_normal}/{n_run} tests: Normal — inconclusive"
        banner_bg = "var(--amber)"

    banner = html.Div(banner_text, className="banner",
                      style={"background": banner_bg, "color": "#fff"})

    sub_banner = html.Div()
    if shape['subsampled']:
        sub_banner = html.Div([
            html.Span("Subsampling active — ", style={"fontWeight": 700}),
            html.Span(
                f"n = {shape['n_full']:,} > {stats._SUBSAMPLE_THRESHOLD:,}. "
                f"Each test ran on {shape['subsample_k']} stratified subsamples "
                f"of {shape['subsample_n']:,} observations. "
                f"Statistic and p-value = median. "
                f"Skewness and kurtosis computed on the full array.",
            ),
        ], className="sub-banner")

    shape_row = html.Div()
    if shape['skewness'] is not None:
        n_label = f"n = {shape['n_full']:,}"
        shape_row = html.Div([
            html.Span(f"{n_label}  |  Skewness: ",
                      style={"fontWeight": 600, "fontSize": "0.6875rem",
                             "color": "var(--text-muted)"}),
            html.Span(f"{shape['skewness']:+.3f}",
                      style={"fontSize": "0.6875rem", "color": "var(--text-secondary)",
                             "fontFamily": "monospace"}),
            html.Span("  |  Excess kurtosis: ",
                      style={"fontWeight": 600, "fontSize": "0.6875rem",
                             "color": "var(--text-muted)"}),
            html.Span(f"{shape['kurtosis']:+.3f}",
                      style={"fontSize": "0.6875rem", "color": "var(--text-secondary)",
                             "fontFamily": "monospace"}),
        ], className="shape-row")

    cards = [_normality_card(r) for r in results]
    return html.Div([
        html.Div("Normality Tests", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 10, "color": "var(--slate-800)",
        }),
        banner, sub_banner, shape_row, *cards,
    ])


def _normality_card(r):
    cls = "green" if r.is_normal is True else "red" if r.is_normal is False else "grey"
    icon = "\u2713" if r.is_normal is True else ("\u2717" if r.is_normal is False else "\u2014")

    stat_line = ""
    if r.statistic is not None:
        stat_line = f"stat = {r.statistic:.4f}"
        if r.p_value is not None:
            stat_line += f",  p = {r.p_value:.4f}"

    children = [
        html.Div(f"{icon}  {r.name}", className="title"),
        html.Div(r.conclusion, className="note"),
    ]
    if stat_line:
        children.append(html.Div(stat_line, className="meta"))
    if r.notes:
        children.append(html.Div("  |  ".join(r.notes), className="meta"))

    return html.Div(children, className=f"stat-card {cls}")


def categorical_stats_panel(arr: np.ndarray, col: str) -> html.Div:
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    total = int(cnts.sum())

    rows = [
        stat_row(str(vals[i]), f"{cnts[i]:,}  {format_percent(cnts[i], total)}")
        for i in order[:20]
    ]
    if len(vals) > 20:
        rows.append(html.Div(f"\u2026 and {len(vals)-20} more categories",
                             style={"color": "var(--text-muted)", "fontSize": "0.6875rem",
                                    "marginTop": 4}))

    return html.Div([
        html.H5(f"Statistics — {col}",
                style={'marginBottom': 12, 'color': 'var(--slate-800)',
                       'fontSize': '0.9375rem', 'fontWeight': 600}),
        stat_row("Unique categories", len(vals)),
        stat_row("Total observations", f"{total:,}"),
        stat_row("Most frequent",
                 f"{vals[order[0]]}  ({cnts[order[0]]:,}  {format_percent(cnts[order[0]], total)})"),
        html.Hr(className="section-divider"),
        html.Div("Frequency Table",
                 style={"fontWeight": 700, "fontSize": "0.8125rem",
                        "marginBottom": 6, "color": "var(--slate-800)"}),
        *rows,
    ])


def correlation_panel(r: float) -> html.Div:
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
    direction = "positive" if r >= 0 else "negative"
    return html.Div([
        html.H5("Pearson Correlation",
                style={"marginBottom": 8, "fontSize": "0.8125rem", "fontWeight": 600}),
        stat_row("r", f"{r:.4f}"),
        stat_row("Strength", f"{strength} {direction}"),
    ])


def group_stats_panel(num: np.ndarray, cat: np.ndarray,
                      numl: str, catl: str) -> html.Div:
    rows = []
    for c in np.unique(cat):
        g = dn(num[cat == c])
        if g.size:
            rows.append(stat_row(str(c), f"\u03bc = {g.mean():.4f}  (n = {len(g)})"))
    return html.Div([
        html.H5(f"{numl}  by  {catl}",
                style={"marginBottom": 8, "fontSize": "0.8125rem", "fontWeight": 600}),
        *rows,
    ])


def association_panel() -> html.Div:
    return html.Div([
        html.H5("Association",
                style={"marginBottom": 8, "fontSize": "0.8125rem", "fontWeight": 600}),
        html.Span("See statistical test results in the left panel.",
                  style={"fontSize": "0.8125rem", "color": "var(--text-muted)"}),
    ])


def test_panel(result: dict) -> html.Div:
    lines = []
    for line in result['results']:
        if line == "":
            lines.append(html.Br())

        elif line.startswith("GNORM_BANNER|"):
            _, colour_key, text = line.split("|", 2)
            bg_map = {"green": "var(--green)", "red": "var(--red)",
                      "orange": "var(--amber)", "grey": "var(--text-muted)"}
            bg = bg_map.get(colour_key, "var(--text-muted)")
            lines.append(html.Div(text, className="banner",
                                  style={"background": bg, "color": "#fff"}))

        elif line.startswith("GNORM_CARD|"):
            parts = line.split("|", 4)
            _, colour_key, title, note, shape_str = parts
            cls = {"green": "green", "red": "red",
                   "orange": "amber", "grey": "grey"}.get(colour_key, "grey")
            children = [html.Div(title, className="title")]
            if note:
                children.append(html.Div(note, className="note"))
            if shape_str:
                children.append(html.Div(shape_str, className="meta"))
            lines.append(html.Div(children, className=f"stat-card {cls}"))

        elif line.startswith("\u2500\u2500"):
            lines.append(html.Div(line, style={
                "fontWeight": 700, "color": "var(--blue)",
                "marginTop": 10, "marginBottom": 4, "fontSize": "0.75rem",
            }))
        elif line.startswith("  "):
            lines.append(html.Div(line, style={
                "fontSize": "0.6875rem", "color": "var(--text-muted)",
                "paddingLeft": 10,
            }))
        else:
            color = "var(--green)" if "\u2713" in line else "var(--red)" if "\u2717" in line else "var(--text-primary)"
            lines.append(html.Div(line, style={
                "fontSize": "0.75rem", "fontFamily": "monospace", "color": color,
            }))
    return html.Div([
        html.H5(result['test_name'],
                style={"marginBottom": 10, "color": "var(--slate-800)",
                       "fontSize": "0.8125rem", "fontWeight": 600}),
        *lines,
    ])


def normality_panel(data: dict, num_cols: list) -> html.Div:
    rows = []
    for col in num_cols:
        _, p = stats.normality_test(data[col])
        rows.append(stat_row(col, stats.normality_label(p)))
    return html.Div([
        html.Div("Normality (Shapiro-Wilk)", style={
            "fontWeight": 700, "fontSize": "0.75rem",
            "marginBottom": 6, "color": "var(--slate-800)",
        }),
        *rows,
    ])


def correlation_insights_panel(mat: np.ndarray, cols: list) -> html.Div:
    n = len(cols)
    pairs = sorted(
        [(cols[i], cols[j], mat[i, j]) for i in range(n) for j in range(i+1, n)],
        key=lambda x: abs(x[2]), reverse=True,
    )
    return html.Div([
        html.H5("Top Correlations",
                style={"marginBottom": 10, "fontSize": "0.8125rem", "fontWeight": 600,
                       "color": "var(--slate-800)"}),
        *[stat_row(f"{a}  &  {b}", f"{v:+.4f}") for a, b, v in pairs[:5]],
    ])
