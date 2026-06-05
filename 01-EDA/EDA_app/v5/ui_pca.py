"""
ui_pca.py  —  Dash HTML component factories for the PCA tab.
Pure (data → component). No store, no charts imports.
"""
from dash import html
import numpy as np


def _row(label, value, color="#2c3e50"):
    return html.Div([
        html.Span(f"{label}: ", style={"fontWeight": "600", "color": color}),
        html.Span(str(value),   style={"color": "#555"}),
    ], style={"marginBottom": "3px", "fontSize": "12px"})


def _section(title, children):
    return html.Div([
        html.Div(title, style={
            "fontWeight": "700", "fontSize": "12px", "color": "#3b82f6",
            "marginTop": "10px", "marginBottom": "4px",
            "borderBottom": "1px solid #e5e7eb", "paddingBottom": "3px",
        }),
        *children,
    ])


def pca_summary_panel(result: dict) -> html.Div:
    n_opt   = result["n_optimal"]
    cum_opt = result["cumulative"][n_opt - 1] * 100
    exp2    = result["explained"][:2] * 100
    cum2    = result["cumulative"][1] * 100 if result["n_components"] >= 2 else 0.0
    missing = result["missing_rows"]
    color_opt = "#22c55e" if cum_opt >= 70 else "#f59e0b" if cum_opt >= 50 else "#ef4444"

    return html.Div([
        html.Div("PCA Overview", style={
            "fontWeight": "700", "fontSize": "13px",
            "marginBottom": "8px", "color": "#2c3e50",
        }),
        _row("Observations",        f"{result['n_obs']:,}"),
        _row("Numeric variables",   result["n_vars"]),
        _row("Total components",    result["n_components"]),
        *([ _row("Excluded (NaN rows)", missing, "#e74c3c") ] if missing > 0 else []),
        html.Hr(style={"margin": "8px 0"}),
        _section("Recommendation", [
            html.Div([
                html.Span(f"{n_opt} PCs", style={
                    "background": color_opt, "color": "#fff",
                    "borderRadius": "4px", "padding": "2px 8px",
                    "fontSize": "11px", "fontWeight": "700", "marginRight": "5px",
                }),
                html.Span(f"explain {cum_opt:.1f}% of variance",
                          style={"fontSize": "11px", "color": "#555"}),
            ], style={"marginBottom": "4px"}),
            html.Div("Kaiser (λ>1) · Scree elbow · 70% threshold → majority vote",
                     style={"fontSize": "10px", "color": "#9ca3af"}),
        ]),
        html.Hr(style={"margin": "8px 0"}),
        _section("First two PCs", [
            _row("PC1", f"{exp2[0]:.2f}% variance"),
            _row("PC2", f"{exp2[1]:.2f}% variance" if len(exp2) > 1 else "—"),
            _row("PC1+PC2 cumul.", f"{cum2:.2f}%"),
        ]),
        html.Hr(style={"margin": "8px 0"}),
        html.Div([
            html.Span("Note: ", style={"fontWeight": "700", "fontSize": "10px", "color": "#1e40af"}),
            html.Span(
                "All numeric variables are standardised (mean=0, std=1) "
                "before PCA to remove size/unit effects.",
                style={"fontSize": "10px", "color": "#1e40af"},
            ),
        ], style={
            "background": "#eff6ff", "border": "1px solid #bfdbfe",
            "borderRadius": "4px", "padding": "6px 8px",
        }),
    ])


def axis_insight_panel(insight: dict) -> html.Div:
    pc_label   = insight["pc_label"]
    exp_pct    = insight["explained"] * 100
    eigenvalue = insight["eigenvalue"]
    threshold  = insight["uniform_threshold"]
    rows       = insight["rows"]
    above      = insight["above_threshold"]

    def _card(r):
        color = "#22c55e" if r["contrib"] >= threshold else "#94a3b8"
        icon_color = "#2980b9" if r["direction"] == "+" else "#e74c3c"
        return html.Div([
            html.Div([
                html.Span(f"{r['rank']}. ",
                          style={"color": "#9ca3af", "fontSize": "10px"}),
                html.Span(r["variable"],
                          style={"fontWeight": "600", "fontSize": "11px", "color": "#2c3e50"}),
                html.Span(f" {r['direction']}",
                          style={"color": icon_color, "fontSize": "11px", "fontWeight": "700"}),
            ]),
            html.Div([
                html.Span(f"Contrib: {r['contrib']:.1f}%  ",
                          style={"fontSize": "10px", "color": color, "fontWeight": "600"}),
                html.Span(f"cos²: {r['cos2']:.3f}  ",
                          style={"fontSize": "10px", "color": "#6b7280"}),
                html.Span(f"corr: {r['corr']:+.3f}",
                          style={"fontSize": "10px", "color": "#6b7280",
                                 "fontFamily": "monospace"}),
            ]),
        ], style={
            "borderLeft": f"3px solid {color}", "padding": "4px 8px",
            "marginBottom": "4px", "background": "#f9fafb", "borderRadius": "3px",
        })

    return html.Div([
        html.Div(f"Axis insight — {pc_label}", style={
            "fontWeight": "700", "fontSize": "13px",
            "marginBottom": "6px", "color": "#2c3e50",
        }),
        _row("Eigenvalue",         f"{eigenvalue:.4f}"),
        _row("Variance explained", f"{exp_pct:.2f}%"),
        _row("Uniform threshold",  f"{threshold:.2f}%"),
        html.Div(
            f"{len(above)} variable(s) above uniform threshold:",
            style={"fontSize": "11px", "color": "#6b7280",
                   "margin": "6px 0 4px"},
        ),
        *[_card(r) for r in rows],
        html.Div([
            html.Span("⚠ Contribution ≠ Correlation. ",
                      style={"fontWeight": "700", "color": "#92400e", "fontSize": "10px"}),
            html.Span(
                "Contribution measures how much a variable built this axis. "
                "Correlation (corr_circle) reflects its angle. "
                "Always inspect the contributions table first.",
                style={"fontSize": "10px", "color": "#92400e"},
            ),
        ], style={
            "background": "#fffbeb", "border": "1px solid #fcd34d",
            "borderRadius": "4px", "padding": "6px 8px", "marginTop": "8px",
        }),
    ])


def individuals_insight_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    scores  = result["scores"]
    contrib = result["contributions_ind"]
    cos2    = result["cos2_ind"]
    n_obs   = result["n_obs"]
    exp_x   = result["explained"][pc_x] * 100
    exp_y   = result["explained"][pc_y] * 100
    sx, sy  = scores[:, pc_x], scores[:, pc_y]
    combined = contrib[:, pc_x] + contrib[:, pc_y]
    top5     = np.argsort(combined)[::-1][:5]

    def _card(i, rank):
        cv  = combined[i]
        cs  = cos2[i, pc_x] + cos2[i, pc_y]
        q   = "well" if cs >= 0.6 else "moderate" if cs >= 0.3 else "poorly"
        col = "#22c55e" if cs >= 0.6 else "#f59e0b" if cs >= 0.3 else "#ef4444"
        return html.Div([
            html.Span(f"{rank}. {result['row_labels'][i]}",
                      style={"fontWeight": "600", "fontSize": "11px", "color": "#2c3e50"}),
            html.Div(f"Contrib: {cv:.2f}%  cos²: {cs:.3f} ({q} repr.)",
                     style={"fontSize": "10px", "color": "#6b7280"}),
        ], style={
            "borderLeft": f"3px solid {col}", "padding": "4px 8px",
            "marginBottom": "3px", "background": "#f9fafb", "borderRadius": "3px",
        })

    return html.Div([
        html.Div("Individuals — plane summary", style={
            "fontWeight": "700", "fontSize": "13px",
            "marginBottom": "6px", "color": "#2c3e50",
        }),
        _row("n", f"{n_obs:,}"),
        _row("Plane", f"PC{pc_x+1} ({exp_x:.1f}%) × PC{pc_y+1} ({exp_y:.1f}%)"),
        html.Hr(style={"margin": "6px 0"}),
        _section(f"PC{pc_x+1} scores",   [
            _row("Mean",  f"{sx.mean():.4f}"),
            _row("Std",   f"{sx.std():.4f}"),
            _row("Range", f"[{sx.min():.2f}, {sx.max():.2f}]"),
        ]),
        _section(f"PC{pc_y+1} scores",   [
            _row("Mean",  f"{sy.mean():.4f}"),
            _row("Std",   f"{sy.std():.4f}"),
            _row("Range", f"[{sy.min():.2f}, {sy.max():.2f}]"),
        ]),
        html.Hr(style={"margin": "6px 0"}),
        html.Div("Top 5 contributors to this plane:",
                 style={"fontSize": "11px", "color": "#6b7280", "marginBottom": "4px"}),
        *[_card(i, r + 1) for r, i in enumerate(top5)],
        html.Div([
            html.Span("Reading tip: ",
                      style={"fontWeight": "700", "color": "#1e40af", "fontSize": "10px"}),
            html.Span(
                "Nearby individuals share similar profiles on the variables "
                "that built these axes. Colour by cos² to check which individuals "
                "are faithfully represented on this plane.",
                style={"fontSize": "10px", "color": "#1e40af"},
            ),
        ], style={
            "background": "#eff6ff", "border": "1px solid #bfdbfe",
            "borderRadius": "4px", "padding": "6px 8px", "marginTop": "8px",
        }),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Expert interpretation panels — dynamic, data-driven commentary
# ═══════════════════════════════════════════════════════════════════════════════

def _interp_header(title: str, subtitle: str = "") -> html.Div:
    return html.Div([
        html.Div([
            html.Span("📊 ", style={"fontSize": "16px"}),
            html.Span(title, style={
                "fontWeight": "700", "fontSize": "14px", "color": "#1e293b",
            }),
        ]),
        *([ html.Div(subtitle, style={
            "fontSize": "11px", "color": "#64748b", "marginTop": "2px",
        }) ] if subtitle else []),
    ], style={"marginBottom": "12px", "borderBottom": "2px solid #e2e8f0", "paddingBottom": "8px"})


def _tip_box(text: str, kind: str = "info") -> html.Div:
    cfg = {
        "info":    ("#eff6ff", "#1d4ed8", "#bfdbfe", "ℹ"),
        "warn":    ("#fffbeb", "#92400e", "#fcd34d", "⚠"),
        "success": ("#f0fdf4", "#166534", "#bbf7d0", "✓"),
        "expert":  ("#faf5ff", "#6b21a8", "#e9d5ff", "🎓"),
    }
    bg, fg, border, icon = cfg.get(kind, cfg["info"])
    return html.Div([
        html.Span(f"{icon} ", style={"fontWeight": "700"}),
        html.Span(text, style={"fontSize": "11px"}),
    ], style={
        "background": bg, "color": fg,
        "border": f"1px solid {border}", "borderRadius": "5px",
        "padding": "7px 10px", "marginBottom": "8px", "lineHeight": "1.5",
    })


def _table(headers: list, rows: list, highlight_col: int = None) -> html.Table:
    """Compact HTML table for expert panels."""
    th_style = {
        "background": "#1e293b", "color": "#fff",
        "padding": "5px 10px", "fontSize": "11px",
        "fontWeight": "600", "textAlign": "left", "whiteSpace": "nowrap",
    }
    td_base = {"padding": "4px 10px", "fontSize": "11px",
               "borderBottom": "1px solid #e5e7eb", "verticalAlign": "top"}

    header_row = html.Tr([html.Th(h, style=th_style) for h in headers])
    data_rows  = []
    for i, row in enumerate(rows):
        tds = []
        for j, cell in enumerate(row):
            extra = {"fontWeight": "600", "color": "#1e40af"} if j == highlight_col else {}
            tds.append(html.Td(cell, style={**td_base, **extra,
                                            "background": "#f8fafc" if i % 2 == 0 else "#fff"}))
        data_rows.append(html.Tr(tds))

    return html.Table(
        [html.Thead(header_row), html.Tbody(data_rows)],
        style={"width": "100%", "borderCollapse": "collapse",
               "border": "1px solid #e5e7eb", "borderRadius": "6px",
               "overflow": "hidden", "marginBottom": "12px"},
    )


def _reading_grid(items: list) -> html.Div:
    """
    items: list of (icon, label, description) tuples
    Renders as a 2-column grid of reading-guide cards.
    """
    cards = []
    for icon, label, desc in items:
        cards.append(html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "18px", "marginRight": "6px"}),
                html.Span(label, style={"fontWeight": "700", "fontSize": "12px",
                                        "color": "#1e293b"}),
            ]),
            html.Div(desc, style={"fontSize": "10px", "color": "#64748b",
                                  "marginTop": "3px", "lineHeight": "1.45"}),
        ], style={
            "background": "#f8fafc", "border": "1px solid #e2e8f0",
            "borderRadius": "6px", "padding": "8px 10px",
        }))

    return html.Div(cards, style={
        "display": "grid", "gridTemplateColumns": "1fr 1fr",
        "gap": "8px", "marginBottom": "12px",
    })


# ── Correlation circle interpretation ─────────────────────────────────────────

def circle_interpretation_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    """
    Dynamic expert commentary for the correlation circle.
    Analyses the actual data: groups, oppositions, poorly represented variables.
    """
    variables = result["variables"]
    corr      = result["corr_circle"]
    cos2      = result["cos2_var"]
    exp_x     = result["explained"][pc_x] * 100
    exp_y     = result["explained"][pc_y] * 100
    n_vars    = len(variables)

    cx = corr[:, pc_x]
    cy = corr[:, pc_y]
    cs = cos2[:, pc_x] + cos2[:, pc_y]

    # Classify variables
    well_repr  = [(variables[j], float(cs[j])) for j in range(n_vars) if cs[j] >= 0.6]
    poor_repr  = [(variables[j], float(cs[j])) for j in range(n_vars) if cs[j] < 0.3]
    pos_x      = sorted([(variables[j], float(cx[j])) for j in range(n_vars) if cx[j] > 0.4],
                        key=lambda x: -x[1])
    neg_x      = sorted([(variables[j], float(cx[j])) for j in range(n_vars) if cx[j] < -0.4],
                        key=lambda x: x[1])
    pos_y      = sorted([(variables[j], float(cy[j])) for j in range(n_vars) if cy[j] > 0.4],
                        key=lambda x: -x[1])
    neg_y      = sorted([(variables[j], float(cy[j])) for j in range(n_vars) if cy[j] < -0.4],
                        key=lambda x: x[1])

    # Detect correlated pairs (same quadrant, cos² both > 0.4)
    corr_pairs = []
    for j in range(n_vars):
        for k in range(j + 1, n_vars):
            if cs[j] >= 0.4 and cs[k] >= 0.4:
                dot = float(cx[j] * cx[k] + cy[j] * cy[k])
                if dot > 0.6:
                    corr_pairs.append((variables[j], variables[k], dot))
    corr_pairs.sort(key=lambda x: -x[2])

    # Detect opposed pairs
    opp_pairs = []
    for j in range(n_vars):
        for k in range(j + 1, n_vars):
            if cs[j] >= 0.4 and cs[k] >= 0.4:
                dot = float(cx[j] * cx[k] + cy[j] * cy[k])
                if dot < -0.5:
                    opp_pairs.append((variables[j], variables[k], dot))
    opp_pairs.sort(key=lambda x: x[2])

    children = [
        _interp_header(
            "Correlation Circle — Expert Reading Guide",
            subtitle=f"PC{pc_x+1} ({exp_x:.1f}%) × PC{pc_y+1} ({exp_y:.1f}%) — "
                     f"plane captures {exp_x+exp_y:.1f}% of total variance",
        ),

        # Reading rules
        _reading_grid([
            ("↔", "Arrow length = quality",
             "A long arrow (reaching the circle) means the variable is well "
             "represented on this plane. A short arrow = look at other axes."),
            ("∠", "Angle = correlation",
             "Small angle between two arrows → positively correlated. "
             "Opposite directions → negatively correlated. "
             "90° → independent on this plane."),
            ("📍", "Position on axis",
             "The projection of the arrow tip onto an axis gives the correlation "
             "with that axis. Use this to name/label each PC."),
            ("⊙", "Circle boundary = perfect repr.",
             "Variables exactly on the unit circle have cos²=1: "
             "100% of their variance is explained by these two PCs."),
        ]),

        # Axis polarity table
        html.Div("Axis polarity — which variables define each axis:", style={
            "fontWeight": "600", "fontSize": "12px", "color": "#374151",
            "marginBottom": "6px",
        }),
        _table(
            ["Axis", "Positive pole (+)", "Negative pole (−)"],
            [
                [f"PC{pc_x+1}",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in pos_x[:4]) or "—",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in neg_x[:4]) or "—"],
                [f"PC{pc_y+1}",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in pos_y[:4]) or "—",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in neg_y[:4]) or "—"],
            ],
            highlight_col=0,
        ),
    ]

    # Well/poorly represented
    if well_repr or poor_repr:
        repr_rows = []
        for v, cs_val in sorted(well_repr + poor_repr, key=lambda x: -x[1]):
            status = "✓ Well repr." if cs_val >= 0.6 else ("~ Moderate" if cs_val >= 0.3 else "✗ Poor repr.")
            color  = "#166534" if cs_val >= 0.6 else ("#92400e" if cs_val >= 0.3 else "#991b1b")
            repr_rows.append([v, f"{cs_val:.3f}", html.Span(status, style={"color": color, "fontWeight": "600", "fontSize": "11px"})])
        children.append(html.Div("Quality of representation on this plane (cos²):", style={
            "fontWeight": "600", "fontSize": "12px", "color": "#374151", "marginBottom": "6px",
        }))
        children.append(_table(["Variable", "cos² (plane)", "Status"], repr_rows, highlight_col=2))

    # Correlated groups
    if corr_pairs:
        children.append(_tip_box(
            "Positively correlated pairs detected on this plane: "
            + " · ".join(f"{a} & {b} (dot={d:.2f})" for a, b, d in corr_pairs[:4])
            + ". These variables carry similar information — consider dimensionality reduction or "
              "interpret them as a group.",
            kind="info",
        ))

    if opp_pairs:
        children.append(_tip_box(
            "Opposed pairs (negative correlation): "
            + " · ".join(f"{a} vs {b}" for a, b, _ in opp_pairs[:3])
            + ". High values on one tend to accompany low values on the other.",
            kind="warn",
        ))

    if poor_repr:
        children.append(_tip_box(
            f"Variables poorly represented on this plane: "
            f"{', '.join(v for v, _ in poor_repr)}. "
            "Their information lies on other axes — consult the cos² heatmap tab "
            "to find which PC captures them.",
            kind="warn",
        ))

    children.append(_tip_box(
        "Expert rule: before interpreting correlations on the circle, always check the "
        "arrow length (cos²). Two short arrows that appear perpendicular may actually be "
        "correlated on a third axis not shown here.",
        kind="expert",
    ))

    return html.Div(children)


# ── Biplot interpretation ─────────────────────────────────────────────────────

def biplot_interpretation_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    """
    Dynamic expert commentary for the biplot.
    """
    variables = result["variables"]
    corr      = result["corr_circle"]
    cos2_var  = result["cos2_var"]
    exp_x     = result["explained"][pc_x] * 100
    exp_y     = result["explained"][pc_y] * 100
    n_obs     = result["n_obs"]

    cx = corr[:, pc_x]
    cy = corr[:, pc_y]
    cs = cos2_var[:, pc_x] + cos2_var[:, pc_y]

    # Variables with strong loading on each axis
    top_pc_x = sorted(range(len(variables)), key=lambda j: abs(cx[j]), reverse=True)[:3]
    top_pc_y = sorted(range(len(variables)), key=lambda j: abs(cy[j]), reverse=True)[:3]

    children = [
        _interp_header(
            "Biplot — Expert Reading Guide",
            subtitle=f"Individuals + variables overlaid — PC{pc_x+1} ({exp_x:.1f}%) × "
                     f"PC{pc_y+1} ({exp_y:.1f}%)",
        ),

        _reading_grid([
            ("🔵", "Individuals (dots)",
             "Each dot is one observation. Position reflects its score on both PCs. "
             "Nearby dots share a similar profile on the variables that built these axes."),
            ("➡", "Variable arrows",
             "Same as the correlation circle: direction = correlation with axes, "
             "length = quality of representation. The unit circle is drawn for reference."),
            ("🎯", "Individual near arrow tip",
             "An observation near the tip of a variable's arrow has a high value on that "
             "variable (after standardisation). Near the base = low value."),
            ("↔", "Perpendicular = independent",
             "An individual on the perpendicular bisector of a variable arrow has an "
             "average value for that variable — neither high nor low."),
        ]),

        _tip_box(
            "Biplot vs. Correlation circle: the biplot overlays both representations "
            "in a single view, enabling you to identify which individuals are "
            "characterised by which variables. The trade-off is visual density — "
            "use the Correlation Circle for clean variable analysis and the "
            "Individuals tab for clean individual analysis.",
            kind="expert",
        ),

        # Key variables per axis
        html.Div("Variables driving each axis (highest |correlation|):", style={
            "fontWeight": "600", "fontSize": "12px", "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Axis", "Top variable", "Corr", "2nd variable", "Corr", "3rd variable", "Corr"],
            [
                [f"PC{pc_x+1}"] + [item for j in top_pc_x for item in
                                    (variables[j], f"{cx[j]:+.2f}")],
                [f"PC{pc_y+1}"] + [item for j in top_pc_y for item in
                                    (variables[j], f"{cy[j]:+.2f}")],
            ],
            highlight_col=0,
        ),

        _tip_box(
            f"This biplot shows {min(n_obs, 1500):,} of {n_obs:,} individuals "
            f"(subsampled for rendering performance). "
            "Variable arrows are drawn at their true correlation-circle scale. "
            "Interpret individual positions relative to the arrows, not absolute coordinates.",
            kind="info",
        ),

        _tip_box(
            "Professional caution: the dual scaling of a biplot (individuals rescaled "
            "inside the circle, variables at correlation scale) makes exact numerical "
            "reading impossible. Use the biplot for visual patterns and clusters; "
            "read exact values in the Individuals and Contributions sub-tabs.",
            kind="warn",
        ),
    ]
    return html.Div(children)


# ── Individuals interpretation ────────────────────────────────────────────────

def individuals_interpretation_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    """
    Dynamic expert commentary for the individuals plane.
    Includes: distribution stats, spread, centring check, density regions.
    """
    scores  = result["scores"]
    cos2    = result["cos2_ind"]
    contrib = result["contributions_ind"]
    exp_x   = result["explained"][pc_x] * 100
    exp_y   = result["explained"][pc_y] * 100
    n_obs   = result["n_obs"]

    sx = scores[:, pc_x]
    sy = scores[:, pc_y]

    # Quadrant counts
    q1 = int(((sx > 0) & (sy > 0)).sum())
    q2 = int(((sx < 0) & (sy > 0)).sum())
    q3 = int(((sx < 0) & (sy < 0)).sum())
    q4 = int(((sx > 0) & (sy < 0)).sum())

    # Quality distribution
    cs_plane = cos2[:, pc_x] + cos2[:, pc_y]
    n_well   = int((cs_plane >= 0.6).sum())
    n_poor   = int((cs_plane <  0.3).sum())
    pct_well = n_well / n_obs * 100
    pct_poor = n_poor / n_obs * 100

    # Outliers on each axis (beyond 2.5 std)
    std_x = sx.std()
    std_y = sy.std()
    out_x = int((np.abs(sx) > 2.5 * std_x).sum())
    out_y = int((np.abs(sy) > 2.5 * std_y).sum())

    # Top contributors summary
    combined   = contrib[:, pc_x] + contrib[:, pc_y]
    top5_idx   = np.argsort(combined)[::-1][:5]
    top5_rows  = [
        [result["row_labels"][i],
         f"{contrib[i, pc_x]:.2f}%",
         f"{contrib[i, pc_y]:.2f}%",
         f"{combined[i]:.2f}%",
         f"{(cos2[i, pc_x]+cos2[i, pc_y]):.3f}"]
        for i in top5_idx
    ]

    children = [
        _interp_header(
            "Individuals Plane — Expert Reading Guide",
            subtitle=f"PC{pc_x+1} ({exp_x:.1f}%) × PC{pc_y+1} ({exp_y:.1f}%) — "
                     f"n = {n_obs:,} observations",
        ),

        _reading_grid([
            ("📍", "Position = profile",
             "An individual's coordinates are its scores on PC1 and PC2. "
             "Similar positions → similar profiles on the variables that built these axes."),
            ("↔", "Distance = dissimilarity",
             "The further apart two points, the more their profiles differ "
             "along the dimensions captured by these two PCs."),
            ("🎨", "Colour = quality (cos²)",
             "Dark colour → the individual is well represented on this plane. "
             "Light colour → most of its variability lies on other axes."),
            ("⭕", "Outliers on the periphery",
             "Individuals far from the centre have atypical profiles. "
             "Check their cos² before concluding: low cos² = artefact of this plane."),
        ]),

        # Distribution summary
        html.Div("Cloud distribution summary:", style={
            "fontWeight": "600", "fontSize": "12px", "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Metric", f"PC{pc_x+1}", f"PC{pc_y+1}"],
            [
                ["Mean (≈0 expected)", f"{sx.mean():.4f}", f"{sy.mean():.4f}"],
                ["Std dev",            f"{sx.std():.4f}",  f"{sy.std():.4f}"],
                ["Min / Max",
                 f"{sx.min():.2f} / {sx.max():.2f}",
                 f"{sy.min():.2f} / {sy.max():.2f}"],
                ["Outliers (>2.5σ)",   str(out_x), str(out_y)],
            ],
            highlight_col=0,
        ),

        # Quadrant table
        html.Div("Individuals per quadrant:", style={
            "fontWeight": "600", "fontSize": "12px", "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Quadrant", "PC_x", "PC_y", "Count", "%"],
            [
                [f"Q1 (+,+)", "+", "+", q1, f"{q1/n_obs*100:.1f}%"],
                [f"Q2 (−,+)", "−", "+", q2, f"{q2/n_obs*100:.1f}%"],
                [f"Q3 (−,−)", "−", "−", q3, f"{q3/n_obs*100:.1f}%"],
                [f"Q4 (+,−)", "+", "−", q4, f"{q4/n_obs*100:.1f}%"],
            ],
            highlight_col=0,
        ),

        # Quality summary
        _tip_box(
            f"{pct_well:.1f}% of individuals are well represented on this plane (cos² ≥ 0.6). "
            f"{pct_poor:.1f}% are poorly represented (cos² < 0.3) — "
            "interpret their position with caution.",
            kind="success" if pct_well >= 60 else "warn",
        ),

        # Top contributors table
        html.Div("Top 5 individuals contributing to this plane:", style={
            "fontWeight": "600", "fontSize": "12px", "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Individual", f"Contrib PC{pc_x+1}", f"Contrib PC{pc_y+1}",
             "Total contrib", "cos² (plane)"],
            top5_rows,
            highlight_col=3,
        ),

        _tip_box(
            "Professional interpretation workflow: (1) identify clusters of nearby individuals, "
            "(2) look at the variables in the Correlation Circle to understand what differentiates "
            "each cluster, (3) flag outliers and check their cos² — low cos² means the outlier "
            "position is an artefact of this projection.",
            kind="expert",
        ),

        _tip_box(
            "Mean scores are close to 0 and standard deviations close to √eigenvalue — "
            "expected after standardisation. Large deviations signal data anomalies "
            "(duplicate rows, scale errors, or very skewed distributions).",
            kind="info",
        ),
    ]
    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# Contributions panel interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def contributions_interpretation_panel(result: dict, pc_idx: int) -> html.Div:
    """
    Expert interpretation panel for the variable contributions sub-tab.
    Data-driven: all thresholds, flags and narratives computed from result.

    Covers:
    - Uniform threshold and how to read the bar chart
    - Variable ranking with contribution vs cos² vs correlation distinction
    - Axis naming suggestion based on top contributors
    - Detection of diffuse vs concentrated axes
    - Redundancy detection (high-cos² pairs on same axis)
    - R² cumulative per variable across retained PCs
    - FactoMineR-style supplementary metrics
    """
    variables  = result["variables"]
    contrib    = result["contributions_var"]
    cos2       = result["cos2_var"]
    corr       = result["corr_circle"]
    eig        = result["eigenvalues"]
    exp        = result["explained"]
    n_opt      = result["n_optimal"]
    p          = len(variables)
    k          = result["n_components"]
    pc         = pc_idx
    threshold  = 100.0 / p
    exp_pc     = exp[pc] * 100

    # ── Classify variables on this axis ──────────────────────────────────────
    order      = np.argsort(contrib[:, pc])[::-1]
    above      = [j for j in order if contrib[j, pc] >= threshold]
    below      = [j for j in order if contrib[j, pc] < threshold]

    # Axis concentration: Herfindahl-Hirschman Index on contributions
    # HHI = Σ(contrib_j²) / 100²  — 1/p = perfectly uniform, 1 = monopoly
    hhi        = float(np.sum((contrib[:, pc] / 100) ** 2))
    hhi_min    = 1.0 / p
    hhi_norm   = (hhi - hhi_min) / (1.0 - hhi_min)   # 0=uniform, 1=monopoly
    if hhi_norm < 0.2:
        concentration = "diffuse"
        conc_color    = "#f59e0b"
        conc_tip      = (
            "The axis is diffuse — many variables contribute similarly. "
            "This axis captures a general dimension of variability, "
            "not a specific phenomenon. Naming it is harder; "
            "look at the sign pattern rather than individual variables."
        )
    elif hhi_norm < 0.5:
        concentration = "moderate"
        conc_color    = "#3b82f6"
        conc_tip      = (
            "A moderate number of variables drive this axis. "
            "Focus on those above the uniform threshold "
            "and verify their cos² before interpreting."
        )
    else:
        concentration = "concentrated"
        conc_color    = "#22c55e"
        conc_tip      = (
            "The axis is concentrated — a small group of variables dominates. "
            "This is the easiest case to interpret: name the axis "
            "after the top contributors, checking their sign (+ or −)."
        )

    # ── Axis naming suggestion ────────────────────────────────────────────────
    pos_drivers = [(variables[j], float(corr[j, pc]), float(contrib[j, pc]))
                   for j in above if corr[j, pc] > 0]
    neg_drivers = [(variables[j], float(corr[j, pc]), float(contrib[j, pc]))
                   for j in above if corr[j, pc] < 0]
    pos_drivers.sort(key=lambda x: -x[2])
    neg_drivers.sort(key=lambda x: -x[2])

    # ── Redundancy detection ─────────────────────────────────────────────────
    # Two variables are "redundant on this axis" if both have cos² > 0.6
    # and their loadings have the same sign.
    redundant_pairs = []
    for j in range(p):
        for l in range(j + 1, p):
            if cos2[j, pc] >= 0.6 and cos2[l, pc] >= 0.6:
                same_sign = (corr[j, pc] * corr[l, pc]) > 0
                redundant_pairs.append((
                    variables[j], variables[l],
                    float(cos2[j, pc]), float(cos2[l, pc]),
                    "same direction" if same_sign else "opposite direction",
                ))

    # ── R² cumulative per variable across n_opt axes ─────────────────────────
    r2_opt  = cos2[:, :n_opt].sum(axis=1)
    r2_rows = []
    for j in np.argsort(r2_opt)[::-1]:
        r2   = float(r2_opt[j])
        flag = "✓" if r2 >= 0.7 else "~" if r2 >= 0.5 else "✗"
        col  = "#166534" if r2 >= 0.7 else "#92400e" if r2 >= 0.5 else "#991b1b"
        r2_rows.append([
            variables[j],
            html.Span(flag, style={"color": col, "fontWeight": "700"}),
            f"{r2:.3f}",
            f"{r2*100:.1f} %",
        ])

    # ── Build detailed variable table ─────────────────────────────────────────
    detail_rows = []
    for rank, j in enumerate(order):
        c_val  = float(contrib[j, pc])
        cs_val = float(cos2[j, pc])
        cr_val = float(corr[j, pc])
        above_flag = "▲" if c_val >= threshold else "▽"
        flag_col   = "#22c55e" if c_val >= threshold else "#94a3b8"
        direction  = "+" if cr_val >= 0 else "−"
        dir_col    = "#2563eb" if cr_val >= 0 else "#dc2626"
        detail_rows.append([
            f"{rank+1}. {variables[j]}",
            html.Span(f"{above_flag} {c_val:.2f}%",
                      style={"color": flag_col, "fontWeight": "600"}),
            f"{cs_val:.3f}",
            html.Span(f"{direction}  {abs(cr_val):.3f}",
                      style={"color": dir_col, "fontFamily": "monospace"}),
        ])

    # ── Children ─────────────────────────────────────────────────────────────
    children = [
        _interp_header(
            f"Contributions — Expert Analysis of PC{pc+1}",
            subtitle=(
                f"PC{pc+1} explains {exp_pc:.1f}% of total variance · "
                f"Eigenvalue λ = {eig[pc]:.3f} · "
                f"Uniform threshold = {threshold:.1f}% · "
                f"Axis type: {concentration}"
            ),
        ),

        # Reading guide
        _reading_grid([
            ("📊", "What contribution measures",
             "The percentage of the axis's variance that a variable built. "
             "Σ contributions = 100% per PC. "
             "Variables above the uniform threshold (1/p) matter."),
            ("🎯", "Uniform threshold = 1/p",
             f"With {p} variables, the threshold is {threshold:.1f}%. "
             "A variable above it contributed more than average. "
             "Variables below it are passengers, not drivers."),
            ("⚠", "Contribution ≠ Correlation",
             "A variable can have a high correlation with an axis "
             "without contributing much to its construction. "
             "Always read contributions FIRST, then correlations."),
            ("🔬", "HHI concentration index",
             f"This axis is {concentration} (HHI = {hhi_norm:.2f}). "
             "Concentrated axes are easier to name and interpret. "
             "Diffuse axes reflect general variance, not specific signals."),
        ]),

        _tip_box(conc_tip,
                 kind="success" if hhi_norm >= 0.5 else "warn" if hhi_norm < 0.2 else "info"),

        # Axis polarity
        html.Div("Axis polarity — drivers by sign:", style={
            "fontWeight": "600", "fontSize": "12px",
            "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Pole", "Variables (ranked by contribution)", "Interpretation hint"],
            [
                ["➕ Positive",
                 ", ".join(f"{v} ({c:.1f}%)" for v, _, c in pos_drivers[:5]) or "—",
                 f"High scores on PC{pc+1} → high {', '.join(v for v, _, _ in pos_drivers[:2])}"
                 if pos_drivers else "—"],
                ["➖ Negative",
                 ", ".join(f"{v} ({c:.1f}%)" for v, _, c in neg_drivers[:5]) or "—",
                 f"High scores on PC{pc+1} → low {', '.join(v for v, _, _ in neg_drivers[:2])}"
                 if neg_drivers else "—"],
            ],
            highlight_col=0,
        ),

        # Detailed variable table
        html.Div(f"All {p} variables ranked by contribution to PC{pc+1}:", style={
            "fontWeight": "600", "fontSize": "12px",
            "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Variable", "Contribution", "cos²", "Direction & |corr|"],
            detail_rows,
            highlight_col=1,
        ),

        # Redundancy
        *([ html.Div("Redundant variable pairs on this axis (both cos² ≥ 0.6):", style={
                "fontWeight": "600", "fontSize": "12px",
                "color": "#374151", "marginBottom": "6px",
            }),
            _table(
                ["Variable A", "Variable B", "cos² A", "cos² B", "Relationship"],
                [[a, b, f"{ca:.3f}", f"{cb:.3f}", rel]
                 for a, b, ca, cb, rel in redundant_pairs],
                highlight_col=4,
            ),
            _tip_box(
                "Redundant variables carry similar information on this axis. "
                "In a predictive model, consider keeping only one per pair "
                "or constructing a composite index. In interpretation, "
                "treat them as a single conceptual dimension.",
                kind="warn",
            ),
        ] if redundant_pairs else []),

        # R2 cumulative table
        html.Div(f"Global quality of representation across {n_opt} retained PCs (R²):", style={
            "fontWeight": "600", "fontSize": "12px",
            "color": "#374151", "marginBottom": "6px",
            "marginTop": "8px",
        }),
        _table(
            ["Variable", "Status", "Σ cos²", "% variance retained"],
            r2_rows,
            highlight_col=2,
        ),
        _tip_box(
            f"Variables with Σcos² < 0.5 over {n_opt} PCs are poorly captured "
            "by the retained solution. Their variance lies on discarded axes. "
            "Consider increasing the number of retained components, "
            "or flag these variables as poorly integrated in the PCA.",
            kind="warn",
        ),

        _tip_box(
            "FactoMineR reading rule: a variable is 'active and well-integrated' "
            "if (1) it contributes above 1/p to at least one axis, "
            "(2) its cos² > 0.6 on that axis, and "
            "(3) its cumulative R² over retained PCs exceeds 0.7. "
            "Variables failing all three criteria may be candidates "
            "for supplementary (illustrative) variable status.",
            kind="expert",
        ),
    ]
    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════════════════
# cos² quality panel interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def cos2_interpretation_panel(result: dict, n_pc: int = 5) -> html.Div:
    """
    Expert interpretation panel for the cos² heatmap sub-tab.

    Covers:
    - What cos² means geometrically and statistically
    - Per-variable quality classification (well / moderate / poor)
    - Per-axis coverage analysis
    - Variables that "escape" the retained solution
    - Relationship between cos², contribution, and correlation
    - FactoMineR-style supplementary variable detection
    - Recommendations for variable selection / axis count revision
    """
    variables  = result["variables"]
    cos2       = result["cos2_var"]
    contrib    = result["contributions_var"]
    corr       = result["corr_circle"]
    exp        = result["explained"]
    eig        = result["eigenvalues"]
    n_opt      = result["n_optimal"]
    p          = len(variables)
    k          = result["n_components"]
    n_pc_show  = min(n_pc, k)
    threshold  = 100.0 / p

    # ── Per-variable quality over retained PCs ────────────────────────────────
    r2_opt    = cos2[:, :n_opt].sum(axis=1)
    r2_all    = cos2.sum(axis=1)             # should be ≈ 1 for standardised data

    # Best axis per variable
    best_axis  = np.argmax(cos2, axis=1)
    best_cos2  = cos2[np.arange(p), best_axis]

    # Classification
    well       = [j for j in range(p) if r2_opt[j] >= 0.7]
    moderate   = [j for j in range(p) if 0.5 <= r2_opt[j] < 0.7]
    poor       = [j for j in range(p) if r2_opt[j] < 0.5]

    # ── Per-axis discriminating power ─────────────────────────────────────────
    # How many variables does each axis represent well (cos² > 0.5)?
    axis_well  = [(int((cos2[:, k_] >= 0.5).sum()), float(exp[k_] * 100))
                  for k_ in range(n_pc_show)]

    # ── Variables poorly represented on ALL retained axes (escaped) ───────────
    escaped    = [variables[j] for j in poor
                  if cos2[j, :n_opt].max() < 0.4]

    # ── Correlation vs cos² dissociation (interesting cases) ─────────────────
    # Variables where |corr| is high but cos² is low → "misleading arrow"
    misleading = []
    for k_ in range(min(3, k)):
        for j in range(p):
            if abs(corr[j, k_]) >= 0.45 and cos2[j, k_] < 0.25:
                misleading.append((variables[j], k_, float(corr[j, k_]), float(cos2[j, k_])))

    # ── Full per-variable diagnostic table ───────────────────────────────────
    diag_rows = []
    for j in np.argsort(r2_opt)[::-1]:
        r2        = float(r2_opt[j])
        bk        = int(best_axis[j])
        bc        = float(best_cos2[j])
        contrib_bk = float(contrib[j, bk])
        flag      = "✓" if r2 >= 0.7 else "~" if r2 >= 0.5 else "✗"
        flag_col  = "#166534" if r2 >= 0.7 else "#92400e" if r2 >= 0.5 else "#991b1b"
        diag_rows.append([
            variables[j],
            html.Span(f"{flag} {r2:.3f}",
                      style={"color": flag_col, "fontWeight": "600",
                             "fontFamily": "monospace"}),
            f"PC{bk+1} ({bc:.2f})",
            f"{contrib_bk:.1f}%",
            "Active" if r2 >= 0.7 else ("Supplementary?" if r2 < 0.5 else "Borderline"),
        ])

    # ── Per-axis cos² coverage rows ───────────────────────────────────────────
    axis_rows = []
    for k_ in range(n_pc_show):
        n_well_k   = int((cos2[:, k_] >= 0.5).sum())
        n_above_k  = int((cos2[:, k_] >= 0.3).sum())
        top2       = ", ".join(variables[j] for j in np.argsort(cos2[:, k_])[::-1][:2])
        axis_rows.append([
            f"PC{k_+1}",
            f"{exp[k_]*100:.1f}%",
            f"{eig[k_]:.3f}",
            str(n_well_k),
            str(n_above_k),
            top2,
        ])

    children = [
        _interp_header(
            "cos² Quality of Representation — Expert Analysis",
            subtitle=(
                f"{p} variables · {n_opt} retained PCs · "
                f"{len(well)} well-repr. ({len(well)/p*100:.0f}%) · "
                f"{len(poor)} poorly-repr. ({len(poor)/p*100:.0f}%)"
            ),
        ),

        _reading_grid([
            ("📐", "Geometric meaning",
             "cos²(variable, axis) is the squared cosine of the angle "
             "between the variable vector and the axis in individual space. "
             "cos² = 1 → perfectly aligned. cos² = 0 → perpendicular."),
            ("📊", "Statistical meaning",
             "cos²(j, k) is the proportion of variable j's variance "
             "explained by axis k. "
             "Σ_k cos²(j,k) = 1 over all axes (standardised data)."),
            ("🎯", "Reading threshold",
             "cos² ≥ 0.7: excellent · 0.5–0.7: good · "
             "0.3–0.5: moderate · < 0.3: poor. "
             "For a plane (2 axes), add the two cos² values."),
            ("⚠", "cos² ≠ contribution",
             "A variable can have high cos² (well represented) "
             "without high contribution (it didn't build the axis). "
             "cos² answers 'how well is this variable shown here?' "
             "Contribution answers 'did this variable create this axis?'"),
        ]),

        # Overall quality summary
        html.Div("Global representation quality across retained PCs:", style={
            "fontWeight": "600", "fontSize": "12px",
            "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Category", "Count", "Variables", "Action"],
            [
                [html.Span("✓ Well repr. (Σcos² ≥ 0.7)",
                           style={"color": "#166534", "fontWeight": "600"}),
                 str(len(well)),
                 ", ".join(variables[j] for j in well) or "—",
                 "Retain and interpret freely"],
                [html.Span("~ Moderate (0.5 ≤ Σcos² < 0.7)",
                           style={"color": "#92400e", "fontWeight": "600"}),
                 str(len(moderate)),
                 ", ".join(variables[j] for j in moderate) or "—",
                 "Interpret with caution"],
                [html.Span("✗ Poor (Σcos² < 0.5)",
                           style={"color": "#991b1b", "fontWeight": "600"}),
                 str(len(poor)),
                 ", ".join(variables[j] for j in poor) or "—",
                 "Consider supplementary status or more PCs"],
            ],
            highlight_col=0,
        ),

        # Per-variable diagnostic
        html.Div("Per-variable diagnostic (sorted by Σcos² over retained PCs):", style={
            "fontWeight": "600", "fontSize": "12px",
            "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Variable", f"Σcos² ({n_opt} PCs)", "Best axis (cos²)", "Contrib on best", "Status"],
            diag_rows,
            highlight_col=1,
        ),

        # Per-axis coverage
        html.Div(f"Per-axis discriminating power (first {n_pc_show} PCs):", style={
            "fontWeight": "600", "fontSize": "12px",
            "color": "#374151", "marginBottom": "6px",
        }),
        _table(
            ["Axis", "Variance %", "Eigenvalue", "Vars cos²≥0.5", "Vars cos²≥0.3", "Top 2 vars"],
            axis_rows,
            highlight_col=3,
        ),

        # Escaped variables
        *([ _tip_box(
                "Variables poorly represented on ALL retained axes (max cos² < 0.4): "
                + ", ".join(escaped) + ". "
                "These variables have most of their variance on discarded PCs. "
                "Options: (1) increase the number of retained components, "
                "(2) treat them as supplementary / illustrative variables, "
                "(3) investigate whether they measure a different construct "
                "not captured by the retained solution.",
                kind="warn",
            ),
        ] if escaped else [
            _tip_box(
                f"All variables have at least one axis with cos² ≥ 0.4. "
                f"The retained solution of {n_opt} PCs provides "
                "reasonable coverage across all variables.",
                kind="success",
            ),
        ]),

        # Misleading arrows warning
        *([ html.Div("⚠ Misleading arrows — high |corr| but low cos²:", style={
                "fontWeight": "600", "fontSize": "12px",
                "color": "#92400e", "marginBottom": "6px",
            }),
            _table(
                ["Variable", "Axis", "|corr|", "cos²", "Risk"],
                [[v, f"PC{k_+1}", f"{abs(cr):.3f}", f"{cs:.3f}",
                  "Arrow direction misleading — low representativity"]
                 for v, k_, cr, cs in misleading[:6]],
                highlight_col=4,
            ),
            _tip_box(
                "These variables appear in the correlation circle with a "
                "non-negligible arrow, but their cos² is low. "
                "The arrow direction is unreliable — most of the variable's "
                "variance lies on axes not shown. "
                "Do NOT interpret their position in the circle as meaningful.",
                kind="warn",
            ),
        ] if misleading else []),

        _tip_box(
            "FactoMineR interpretation protocol: "
            "(1) Read the heatmap column by column to identify which axis "
            "represents each variable best (darkest cell per row). "
            "(2) A variable should appear dark on exactly one or two axes "
            "in a clean PCA. If it's uniformly pale everywhere, "
            "it's a noise variable. "
            "(3) If two variables always show high cos² on the same axes, "
            "they are measuring the same latent dimension — "
            "consider removing one or creating a composite index.",
            kind="expert",
        ),

        *([_tip_box(
            "Increasing retained PCs: each additional PC adds "
            f"{exp[n_opt]*100:.1f}% variance (PC{n_opt+1}) to the solution. "
            f"Retaining {n_opt+1} PCs would raise cumulative variance from "
            f"{result['cumulative'][n_opt-1]*100:.1f}% to "
            f"{result['cumulative'][n_opt]*100:.1f}%. "
            "Justified if poorly-represented variables are theoretically important.",
            kind="info",
        )] if n_opt < k else []),
    ]
    return html.Div(children)
