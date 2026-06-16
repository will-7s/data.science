from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import config
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
        return html.Div("No data.", **{"data-coltype": "numeric"})

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
    ], **{"data-coltype": "numeric"})


def normality_battery_panel(arr: np.ndarray, col: str = "") -> html.Div:
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
                f"n = {shape['n_full']:,} > {config.SUBSAMPLE_THRESHOLD:,}. "
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
    ], **{"data-coltype": "numeric"})


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
    ], **{"data-coltype": "categorical"})


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
    cat_uniq, cat_codes = np.unique(cat, return_inverse=True)
    sort_idx   = np.argsort(cat_codes, kind='stable')
    num_sorted = num[sort_idx]
    cod_sorted = cat_codes[sort_idx]
    split_pts  = np.where(np.diff(cod_sorted))[0] + 1
    groups     = np.split(num_sorted, split_pts)
    rows = []
    for c, g in zip(cat_uniq, groups):
        clean_g = dn(g)
        if clean_g.size:
            rows.append(stat_row(str(c), f"\u03bc = {clean_g.mean():.4f}  (n = {len(clean_g)})"))
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


def correlation_insights_panel(mat: np.ndarray, cols: list) -> html.Div:
    n = len(cols)
    tri = np.triu_indices(n, k=1)
    vals = mat[tri]
    top_k = 5
    if len(vals) <= top_k:
        idx = np.argsort(np.abs(vals))[::-1]
    else:
        idx = np.argpartition(np.abs(vals), -top_k)[-top_k:]
        idx = idx[np.argsort(np.abs(vals)[idx])[::-1]]
    pairs = [(cols[tri[0][i]], cols[tri[1][i]], vals[i]) for i in idx]
    return html.Div([
        html.H5("Top Correlations",
                style={"marginBottom": 10, "fontSize": "0.8125rem", "fontWeight": 600,
                       "color": "var(--slate-800)"}),
        *[stat_row(f"{a}  &  {b}", f"{v:+.4f}") for a, b, v in pairs],
    ])


def _human_bytes(n: int) -> str:
    n = max(n, 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def build_overview_card(metadata: dict) -> html.Div:
    ct = metadata["col_types"]
    type_items = html.Div([
        html.Div([
            html.Span(f'{v} ', style={"fontSize": "1.5rem", "fontWeight": 700}),
            html.Span(k.capitalize(), className="overview-type-label"),
        ], style={"display": "flex", "alignItems": "baseline", "gap": 4, "justifyContent": "center"},
           **{"data-coltype": k})
        for k, v in sorted(ct.items())
    ], style={"display": "flex", "gap": 20, "flexWrap": "wrap", "justifyContent": "center"})

    stat_cards = dbc.Row([
        dbc.Col(dbc.Card([
            html.Div(f'{metadata["n_rows"]:,}', className="overview-stat-value"),
            html.Div("Rows", className="overview-stat-label"),
        ], className="overview-stat-card"), width=2),
        dbc.Col(dbc.Card([
            html.Div(str(metadata["n_cols"]), className="overview-stat-value"),
            html.Div("Columns", className="overview-stat-label"),
        ], className="overview-stat-card"), width=2),
        dbc.Col(dbc.Card([
            html.Div(_human_bytes(metadata["memory_bytes"]), className="overview-stat-value"),
            html.Div("Memory", className="overview-stat-label"),
        ], className="overview-stat-card"), width=2),
        dbc.Col(dbc.Card([
            html.Div(type_items, className="overview-stat-value"),
            html.Div("Column Types", className="overview-stat-label"),
        ], className="overview-stat-card"), width=4),
    ], className="mt-3 overview-stat-row", style={"marginBottom": 20})

    missing_section = html.Div()
    top = metadata["top_missing"]
    if top:
        missing_section = html.Div([
            html.H5("Top Missing Values",
                    style={"fontSize": "0.875rem", "fontWeight": 600,
                           "marginBottom": 8, "color": "var(--slate-800)"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Column", style={"textAlign": "left", "padding": "4px 8px"}),
                    html.Th("Missing", style={"textAlign": "right", "padding": "4px 8px"}),
                    html.Th("%", style={"textAlign": "right", "padding": "4px 8px"}),
                ], style={"fontSize": "0.75rem", "color": "var(--text-muted)"})),
                html.Tbody([
                    html.Tr([
                        html.Td(t["col"], style={"padding": "2px 8px",
                               "fontSize": "0.8125rem"}),
                        html.Td(f"{t['n_missing']:,}",
                                style={"textAlign": "right", "padding": "2px 8px",
                                       "fontSize": "0.8125rem",
                                       "fontFamily": "monospace"}),
                        html.Td(f"{t['pct']}%",
                                style={"textAlign": "right", "padding": "2px 8px",
                                       "fontSize": "0.8125rem",
                                       "fontFamily": "monospace"}),
                    ]) for t in top
                ]),
            ], style={"width": "100%", "maxWidth": 500}),
        ], style={"marginTop": 12})
    else:
        missing_section = html.Div("No missing values found.",
                                   style={"color": "var(--text-muted)",
                                          "fontSize": "0.8125rem", "marginTop": 12})

    return html.Div([
        html.H5("Dataset Overview",
                style={"fontSize": "1rem", "fontWeight": 600, "marginBottom": 4,
                       "color": "var(--slate-800)"}),
        html.Div(metadata.get("source", ""),
                 style={"fontSize": "0.8125rem", "color": "var(--text-secondary)",
                        "marginBottom": 8}),
        stat_cards,
        missing_section,
    ])


def build_sampling_badge(refresh_id: str = "") -> html.Div:
    kw = {"id": refresh_id} if refresh_id else {}
    return html.Div(
        "Sampled — approximate results. Click for exact.",
        className="sampling-badge",
        **kw,
        style={"cursor": "pointer", "fontSize": "0.75rem",
               "color": "var(--amber)", "marginTop": 4,
               "padding": "4px 8px", "borderRadius": "4px",
               "background": "var(--warning-light)"},
    )


def build_lightweight_badge() -> html.Div:
    return html.Div(
        "Auto-adjusted for speed",
        className="lightweight-badge",
        title="Heavy features have been limited to keep the app fast. "
              "Select specific columns for detailed analysis.",
        style={"fontSize": "0.8rem", "padding": "4px 10px",
               "borderRadius": "4px", "marginTop": 4,
               "background": "var(--info-light, #dbeafe)",
               "color": "var(--info, #2563eb)", "display": "inline-block"},
    )


def build_cancel_button() -> dbc.Button:
    return dbc.Button(
        "Cancel", id="cancel-btn", color="danger", size="sm",
        disabled=True,
        style={"position": "fixed", "bottom": 20, "right": 20,
               "zIndex": 1000, "display": "none"},
    )


def build_column_selector(columns: list[str], selector_id: str = "corr-col-selector",
                          placeholder: str = "Select columns...") -> html.Div:
    return html.Div([
        html.P("Select fewer columns or use a smaller dataset.",
               style={"fontSize": "0.85rem", "color": "var(--text-secondary)", "marginBottom": 8}),
        dcc.Dropdown(
            id=selector_id, options=[{"label": c, "value": c} for c in columns],
            value=columns[:min(5, len(columns))], multi=True,
            placeholder=placeholder,
        ),
    ], style={"padding": 12, "background": "var(--bg-card, #fff)",
              "borderRadius": "var(--radius-lg, 8px)"})


def build_cancelled_message() -> html.Div:
    return html.Div(
        "Cancelled — adjust your selection",
        className="cancel-msg",
        style={"textAlign": "center", "padding": 40,
               "fontSize": "0.95rem", "color": "var(--text-secondary)"},
    )
