from dash import html
import numpy as np


def _row(label, value, color="var(--slate-700)"):
    return html.Div([
        html.Span(f"{label}: ",
                  style={"fontWeight": 600, "color": color, "fontSize": "0.75rem"}),
        html.Span(str(value),
                  style={"color": "var(--text-secondary)", "fontSize": "0.75rem"}),
    ], style={"marginBottom": 4})


def _section(title, children):
    return html.Div([
        html.Div([
            html.Span("\u25b6", style={"fontSize": "0.625rem", "color": "var(--blue)"}),
            html.Span(title, style={"fontSize": "0.75rem", "fontWeight": 700,
                                    "color": "var(--blue)"}),
        ], className="section-header"),
        *children,
    ])


def _axis_card(r, threshold):
    cls = "insight-card above" if r["contrib"] >= threshold else "insight-card"
    icon_color = "var(--blue)" if r["direction"] == "+" else "var(--red)"
    return html.Div([
        html.Div([
            html.Span(f"{r['rank']}. ", className="variable-rank"),
            html.Span(r["variable"], className="variable-name"),
            html.Span(f" {r['direction']}",
                      className="direction", style={"color": icon_color}),
        ]),
        html.Div([
            html.Span(f"Contrib: {r['contrib']:.1f}%  ",
                      style={"fontSize": "0.625rem",
                             "color": "var(--green)", "fontWeight": 600}),
            html.Span(f"cos\u00b2: {r['cos2']:.3f}  ",
                      style={"fontSize": "0.625rem", "color": "var(--text-muted)"}),
            html.Span(f"corr: {r['corr']:+.3f}",
                      style={"fontSize": "0.625rem", "color": "var(--text-muted)",
                             "fontFamily": "monospace"}),
        ], className="metric"),
    ], className=cls)


def _individual_card(i, rank, combined, cos2, contrib, pc_x, pc_y, row_labels):
    cv = combined[i]
    cs = cos2[i, pc_x] + cos2[i, pc_y]
    q = "well" if cs >= 0.6 else "moderate" if cs >= 0.3 else "poorly"
    col = "var(--green)" if cs >= 0.6 else "var(--amber)" if cs >= 0.3 else "var(--red)"
    return html.Div([
        html.Span(f"{rank}. {row_labels[i]}",
                  style={"fontWeight": 600, "fontSize": "0.6875rem",
                         "color": "var(--slate-800)"}),
        html.Div(f"Contrib: {cv:.2f}%  cos\u00b2: {cs:.3f} ({q} repr.)",
                 style={"fontSize": "0.625rem", "color": "var(--text-muted)"}),
    ], style={
        "borderLeft": f"3px solid {col}", "padding": "5px 10px",
        "marginBottom": 4, "background": "var(--bg-muted)",
        "borderRadius": "var(--radius-sm)",
        "transition": "all 150ms cubic-bezier(0.4,0,0.2,1)",
    })


def pca_summary_panel(result: dict) -> html.Div:
    n_opt = result["n_optimal"]
    cum_opt = result["cumulative"][n_opt - 1] * 100
    exp2 = result["explained"][:2] * 100
    cum2 = result["cumulative"][1] * 100 if result["n_components"] >= 2 else 0.0
    missing = result["missing_rows"]
    color_opt = "var(--green)" if cum_opt >= 70 else "var(--amber)" if cum_opt >= 50 else "var(--red)"

    return html.Div([
        html.Div("PCA Overview", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 10, "color": "var(--slate-800)",
        }),
        _row("Observations", f"{result['n_obs']:,}"),
        _row("Numeric variables", result["n_vars"]),
        _row("Total components", result["n_components"]),
        *([_row("Excluded (NaN rows)", missing, "var(--red)")] if missing > 0 else []),
        html.Hr(className="section-divider"),
        _section("Recommendation", [
            html.Div([
                html.Span(f"{n_opt} PCs", className="badge-custom",
                          style={"background": color_opt}),
                html.Span(f" explain {cum_opt:.1f}% of variance",
                          style={"fontSize": "0.6875rem", "color": "var(--text-secondary)",
                                 "marginLeft": 6}),
            ], style={"marginBottom": 4}),
            html.Div("Kaiser (\u03bb>1) · Scree elbow · 70% threshold \u2192 majority vote",
                     style={"fontSize": "0.625rem", "color": "var(--text-muted)"}),
        ]),
        html.Hr(className="section-divider"),
        _section("First two PCs", [
            _row("PC1", f"{exp2[0]:.2f}% variance"),
            _row("PC2", f"{exp2[1]:.2f}% variance" if len(exp2) > 1 else "\u2014"),
            _row("PC1+PC2 cumul.", f"{cum2:.2f}%"),
        ]),
        html.Hr(className="section-divider"),
        html.Div([
            html.Span("Note: ", style={"fontWeight": 700, "fontSize": "0.625rem",
                                        "color": "var(--blue-dark)"}),
            html.Span(
                "All numeric variables are standardised (mean=0, std=1) "
                "before PCA to remove size/unit effects.",
                style={"fontSize": "0.625rem", "color": "var(--blue-dark)"},
            ),
        ], className="tip-box info"),
    ])


def axis_insight_panel(insight: dict) -> html.Div:
    pc_label = insight["pc_label"]
    exp_pct = insight["explained"] * 100
    eigenvalue = insight["eigenvalue"]
    threshold = insight["uniform_threshold"]
    rows_data = insight["rows"]
    above = insight["above_threshold"]

    return html.Div([
        html.Div(f"Axis insight \u2014 {pc_label}", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 8, "color": "var(--slate-800)",
        }),
        _row("Eigenvalue", f"{eigenvalue:.4f}"),
        _row("Variance explained", f"{exp_pct:.2f}%"),
        _row("Uniform threshold", f"{threshold:.2f}%"),
        html.Div(
            f"{len(above)} variable(s) above uniform threshold:",
            style={"fontSize": "0.6875rem", "color": "var(--text-muted)",
                   "margin": "8px 0 6px"},
        ),
        *[_axis_card(r, threshold) for r in rows_data],
        html.Div([
            html.Span("\u26a0 Contribution \u2260 Correlation. ",
                      style={"fontWeight": 700, "color": "#92400e",
                             "fontSize": "0.625rem"}),
            html.Span(
                "Contribution measures how much a variable built this axis. "
                "Correlation (corr_circle) reflects its angle. "
                "Always inspect the contributions table first.",
                style={"fontSize": "0.625rem", "color": "#92400e"},
            ),
        ], className="tip-box warn"),
    ])


def individuals_insight_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    scores = result["scores"]
    contrib = result["contributions_ind"]
    cos2 = result["cos2_ind"]
    n_obs = result["n_obs"]
    exp_x = result["explained"][pc_x] * 100
    exp_y = result["explained"][pc_y] * 100
    sx, sy = scores[:, pc_x], scores[:, pc_y]
    combined = contrib[:, pc_x] + contrib[:, pc_y]
    top5 = np.argsort(combined)[::-1][:5]

    return html.Div([
        html.Div("Individuals \u2014 plane summary", style={
            "fontWeight": 700, "fontSize": "0.8125rem",
            "marginBottom": 8, "color": "var(--slate-800)",
        }),
        _row("n", f"{n_obs:,}"),
        _row("Plane", f"PC{pc_x+1} ({exp_x:.1f}%) \u00d7 PC{pc_y+1} ({exp_y:.1f}%)"),
        html.Hr(className="section-divider"),
        _section(f"PC{pc_x+1} scores", [
            _row("Mean", f"{sx.mean():.4f}"),
            _row("Std", f"{sx.std():.4f}"),
            _row("Range", f"[{sx.min():.2f}, {sx.max():.2f}]"),
        ]),
        _section(f"PC{pc_y+1} scores", [
            _row("Mean", f"{sy.mean():.4f}"),
            _row("Std", f"{sy.std():.4f}"),
            _row("Range", f"[{sy.min():.2f}, {sy.max():.2f}]"),
        ]),
        html.Hr(className="section-divider"),
        html.Div("Top 5 contributors to this plane:",
                 style={"fontSize": "0.6875rem", "color": "var(--text-muted)",
                        "marginBottom": 6}),
        *[_individual_card(i, r + 1, combined, cos2, contrib, pc_x, pc_y, result['row_labels'])
          for r, i in enumerate(top5)],
        html.Div([
            html.Span("Reading tip: ",
                      style={"fontWeight": 700, "color": "var(--blue-dark)",
                             "fontSize": "0.625rem"}),
            html.Span(
                "Nearby individuals share similar profiles on the variables "
                "that built these axes. Colour by cos\u00b2 to check which individuals "
                "are faithfully represented on this plane.",
                style={"fontSize": "0.625rem", "color": "var(--blue-dark)"},
            ),
        ], className="tip-box info"),
    ])


def _interp_header(title: str, subtitle: str = ""):
    return html.Div([
        html.Div(title, className="title"),
        *([html.Div(subtitle, className="subtitle")] if subtitle else []),
    ], className="panel-header")


def _tip_box(text: str, kind: str = "info"):
    return html.Div(text, className=f"tip-box {kind}")


def _table(headers: list, rows: list, highlight_col: int = None):
    th_style = {"background": "var(--slate-800)", "color": "#fff",
                "padding": "6px 12px", "fontSize": "0.6875rem",
                "fontWeight": 600, "textAlign": "left", "whiteSpace": "nowrap"}
    td_base = {"padding": "5px 12px", "fontSize": "0.6875rem",
               "borderBottom": "1px solid var(--border-light)",
               "verticalAlign": "top"}

    header_row = html.Tr([html.Th(h, style=th_style) for h in headers])
    data_rows = []
    for i, row in enumerate(rows):
        tds = []
        for j, cell in enumerate(row):
            extra = {"fontWeight": 600, "color": "var(--blue)"} if j == highlight_col else {}
            tds.append(html.Td(cell, style={**td_base, **extra,
                                            "background": "var(--slate-50)" if i % 2 == 0 else "#fff"}))
        data_rows.append(html.Tr(tds))

    return html.Table(
        [html.Thead(header_row), html.Tbody(data_rows)],
        className="modern-table",
    )


def _reading_grid(items: list):
    cards = []
    for icon, label, desc in items:
        cards.append(html.Div([
            html.Div([
                html.Span(icon, className="icon"),
                html.Span(label, className="label"),
            ]),
            html.Div(desc, className="desc"),
        ], className="card"))
    return html.Div(cards, className="reading-grid")


def circle_interpretation_panel(result: dict, pc_x: int, pc_y: int,
                                   cos2_quality_min: float = 0.60) -> html.Div:
    variables = result["variables"]
    corr = result["corr_circle"]
    cos2 = result["cos2_var"]
    exp_x = result["explained"][pc_x] * 100
    exp_y = result["explained"][pc_y] * 100
    n_vars = len(variables)

    cx = corr[:, pc_x]
    cy = corr[:, pc_y]
    cs = cos2[:, pc_x] + cos2[:, pc_y]

    well_repr = [(variables[j], float(cs[j])) for j in range(n_vars) if cs[j] >= 0.6]
    poor_repr = [(variables[j], float(cs[j])) for j in range(n_vars) if cs[j] < 0.3]
    pos_x = sorted([(variables[j], float(cx[j])) for j in range(n_vars) if cx[j] > 0.4],
                   key=lambda x: -x[1])
    neg_x = sorted([(variables[j], float(cx[j])) for j in range(n_vars) if cx[j] < -0.4],
                   key=lambda x: x[1])
    pos_y = sorted([(variables[j], float(cy[j])) for j in range(n_vars) if cy[j] > 0.4],
                   key=lambda x: -x[1])
    neg_y = sorted([(variables[j], float(cy[j])) for j in range(n_vars) if cy[j] < -0.4],
                   key=lambda x: x[1])

    corr_pairs = []
    for j in range(n_vars):
        for k in range(j + 1, n_vars):
            if cs[j] >= cos2_quality_min and cs[k] >= cos2_quality_min:
                dot = float(cx[j] * cx[k] + cy[j] * cy[k])
                if dot > 0.6:
                    corr_pairs.append((variables[j], variables[k], dot))
    corr_pairs.sort(key=lambda x: -x[2])

    opp_pairs = []
    for j in range(n_vars):
        for k in range(j + 1, n_vars):
            if cs[j] >= cos2_quality_min and cs[k] >= cos2_quality_min:
                dot = float(cx[j] * cx[k] + cy[j] * cy[k])
                if dot < -0.5:
                    opp_pairs.append((variables[j], variables[k], dot))
    opp_pairs.sort(key=lambda x: x[2])

    # FIX: add low-plane-variance disclaimer when plane captures < 50% variance
    low_plane = (exp_x + exp_y) < 50.0

    children = [
        _interp_header(
            "Correlation Circle \u2014 Expert Reading Guide",
            subtitle=f"PC{pc_x+1} ({exp_x:.1f}%) \u00d7 PC{pc_y+1} ({exp_y:.1f}%) \u2014 "
                     f"plane captures {exp_x+exp_y:.1f}% of total variance",
        ),
        _reading_grid([
            ("\u2194", "Arrow length = quality",
             "A long arrow (reaching the circle) means the variable is well "
             "represented on this plane. A short arrow = look at other axes."),
            ("\u2220", "Angle = correlation",
             "Small angle between two arrows \u2192 positively correlated. "
             "Opposite directions \u2192 negatively correlated. "
             "90\u00b0 \u2192 independent on this plane."),
            ("📍", "Position on axis",
             "The projection of the arrow tip onto an axis gives the correlation "
             "with that axis. Use this to name/label each PC."),
            ("\u2299", "Circle boundary = perfect repr.",
             "Variables exactly on the unit circle have cos\u00b2=1: "
             "100% of their variance is explained by these two PCs."),
        ]),
        *([_tip_box(
            f"This plane captures only {exp_x+exp_y:.1f}% of total variance. "
            "Correlations inferred from the circle may be unreliable — "
            "important relationships may lie on other axes. "
            "Increase the number of displayed PCs to get a more complete picture.",
            kind="warn",
        )] if low_plane else []),
        html.Div("Axis polarity \u2014 which variables define each axis:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Axis", "Positive pole (+)", "Negative pole (\u2212)"],
            [
                [f"PC{pc_x+1}",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in pos_x[:4]) or "\u2014",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in neg_x[:4]) or "\u2014"],
                [f"PC{pc_y+1}",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in pos_y[:4]) or "\u2014",
                 ", ".join(f"{v} ({c:+.2f})" for v, c in neg_y[:4]) or "\u2014"],
            ],
            highlight_col=0,
        ),
    ]

    if well_repr or poor_repr:
        repr_rows = []
        for v, cs_val in sorted(well_repr + poor_repr, key=lambda x: -x[1]):
            status = "\u2713 Well repr." if cs_val >= 0.6 else ("~ Moderate" if cs_val >= 0.3 else "\u2717 Poor repr.")
            color = "var(--green)" if cs_val >= 0.6 else ("#92400e" if cs_val >= 0.3 else "var(--red)")
            repr_rows.append([v, f"{cs_val:.3f}",
                              html.Span(status, style={"color": color, "fontWeight": 600,
                                                       "fontSize": "0.6875rem"})])
        children.append(html.Div("Quality of representation on this plane (cos\u00b2):", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }))
        children.append(_table(["Variable", "cos\u00b2 (plane)", "Status"], repr_rows, highlight_col=2))

    if corr_pairs:
        children.append(_tip_box(
            "Positively correlated pairs detected on this plane: "
            + " \u00b7 ".join(f"{a} & {b} (dot={d:.2f})" for a, b, d in corr_pairs[:4])
            + ". These variables carry similar information \u2014 consider dimensionality reduction or "
              "interpret them as a group.",
            kind="info",
        ))

    if opp_pairs:
        children.append(_tip_box(
            "Opposed pairs (negative correlation): "
            + " \u00b7 ".join(f"{a} vs {b}" for a, b, _ in opp_pairs[:3])
            + ". High values on one tend to accompany low values on the other.",
            kind="warn",
        ))

    if poor_repr:
        children.append(_tip_box(
            f"Variables poorly represented on this plane: "
            f"{', '.join(v for v, _ in poor_repr)}. "
            "Their information lies on other axes \u2014 consult the cos\u00b2 heatmap tab "
            "to find which PC captures them.",
            kind="warn",
        ))

    children.append(_tip_box(
        "Expert rule: before interpreting correlations on the circle, always check the "
        "arrow length (cos\u00b2). Two short arrows that appear perpendicular may actually be "
        "correlated on a third axis not shown here.",
        kind="expert",
    ))

    return html.Div(children)


def biplot_interpretation_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    variables = result["variables"]
    corr = result["corr_circle"]
    cos2_var = result["cos2_var"]
    exp_x = result["explained"][pc_x] * 100
    exp_y = result["explained"][pc_y] * 100
    n_obs = result["n_obs"]

    cx = corr[:, pc_x]
    cy = corr[:, pc_y]
    cs = cos2_var[:, pc_x] + cos2_var[:, pc_y]

    top_pc_x = sorted(range(len(variables)), key=lambda j: abs(cx[j]), reverse=True)[:3]
    top_pc_y = sorted(range(len(variables)), key=lambda j: abs(cy[j]), reverse=True)[:3]

    children = [
        _interp_header(
            "Biplot \u2014 Expert Reading Guide",
            subtitle=f"Individuals + variables overlaid \u2014 PC{pc_x+1} ({exp_x:.1f}%) \u00d7 "
                     f"PC{pc_y+1} ({exp_y:.1f}%)",
        ),
        _reading_grid([
            ("🔵", "Individuals (dots)",
             "Each dot is one observation. Position reflects its score on both PCs. "
             "Nearby dots share a similar profile on the variables that built these axes."),
            ("\u27a1", "Variable arrows",
             "Same as the correlation circle: direction = correlation with axes, "
             "length = quality of representation. The unit circle is drawn for reference."),
            ("🎯", "Individual near arrow tip",
             "An observation near the tip of a variable's arrow has a high value on that "
             "variable (after standardisation). Near the base = low value."),
            ("\u2194", "Perpendicular = independent",
             "An individual on the perpendicular bisector of a variable arrow has an "
             "average value for that variable \u2014 neither high nor low."),
        ]),
        _tip_box(
            "Biplot vs. Correlation circle: the biplot overlays both representations "
            "in a single view, enabling you to identify which individuals are "
            "characterised by which variables. The trade-off is visual density \u2014 "
            "use the Correlation Circle for clean variable analysis and the "
            "Individuals tab for clean individual analysis.",
            kind="expert",
        ),
        html.Div("Variables driving each axis (highest |correlation|):", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
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


def individuals_interpretation_panel(result: dict, pc_x: int, pc_y: int) -> html.Div:
    scores = result["scores"]
    cos2 = result["cos2_ind"]
    contrib = result["contributions_ind"]
    exp_x = result["explained"][pc_x] * 100
    exp_y = result["explained"][pc_y] * 100
    n_obs = result["n_obs"]

    sx = scores[:, pc_x]
    sy = scores[:, pc_y]

    q1 = int(((sx > 0) & (sy > 0)).sum())
    q2 = int(((sx < 0) & (sy > 0)).sum())
    q3 = int(((sx < 0) & (sy < 0)).sum())
    q4 = int(((sx > 0) & (sy < 0)).sum())

    cs_plane = cos2[:, pc_x] + cos2[:, pc_y]
    n_well = int((cs_plane >= 0.6).sum())
    n_poor = int((cs_plane < 0.3).sum())
    pct_well = n_well / n_obs * 100
    pct_poor = n_poor / n_obs * 100

    std_x = sx.std()
    std_y = sy.std()
    out_x = int((np.abs(sx) > 2.5 * std_x).sum())
    out_y = int((np.abs(sy) > 2.5 * std_y).sum())

    combined = contrib[:, pc_x] + contrib[:, pc_y]
    top5_idx = np.argsort(combined)[::-1][:5]
    top5_rows = [
        [result["row_labels"][i],
         f"{contrib[i, pc_x]:.2f}%",
         f"{contrib[i, pc_y]:.2f}%",
         f"{combined[i]:.2f}%",
         f"{(cos2[i, pc_x]+cos2[i, pc_y]):.3f}"]
        for i in top5_idx
    ]

    children = [
        _interp_header(
            "Individuals Plane \u2014 Expert Reading Guide",
            subtitle=f"PC{pc_x+1} ({exp_x:.1f}%) \u00d7 PC{pc_y+1} ({exp_y:.1f}%) \u2014 "
                     f"n = {n_obs:,} observations",
        ),
        _reading_grid([
            ("📍", "Position = profile",
             "An individual's coordinates are its scores on PC1 and PC2. "
             "Similar positions \u2192 similar profiles on the variables that built these axes."),
            ("\u2194", "Distance = dissimilarity",
             "The further apart two points, the more their profiles differ "
             "along the dimensions captured by these two PCs."),
            ("🎨", "Colour = quality (cos\u00b2)",
             "Dark colour \u2192 the individual is well represented on this plane. "
             "Light colour \u2192 most of its variability lies on other axes."),
            ("\u2B55", "Outliers on the periphery",
             "Individuals far from the centre have atypical profiles. "
             "Check their cos\u00b2 before concluding: low cos\u00b2 = artefact of this plane."),
        ]),
        html.Div("Cloud distribution summary:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Metric", f"PC{pc_x+1}", f"PC{pc_y+1}"],
            [
                ["Mean (\u22480 expected)", f"{sx.mean():.4f}", f"{sy.mean():.4f}"],
                ["Std dev", f"{sx.std():.4f}", f"{sy.std():.4f}"],
                ["Min / Max",
                 f"{sx.min():.2f} / {sx.max():.2f}",
                 f"{sy.min():.2f} / {sy.max():.2f}"],
                ["Outliers (>2.5\u03c3)", str(out_x), str(out_y)],
            ],
            highlight_col=0,
        ),
        html.Div("Individuals per quadrant:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Quadrant", "PC_x", "PC_y", "Count", "%"],
            [
                [f"Q1 (+,+)", "+", "+", q1, f"{q1/n_obs*100:.1f}%"],
                [f"Q2 (\u2212,+)", "\u2212", "+", q2, f"{q2/n_obs*100:.1f}%"],
                [f"Q3 (\u2212,\u2212)", "\u2212", "\u2212", q3, f"{q3/n_obs*100:.1f}%"],
                [f"Q4 (+,\u2212)", "+", "\u2212", q4, f"{q4/n_obs*100:.1f}%"],
            ],
            highlight_col=0,
        ),
        _tip_box(
            f"{pct_well:.1f}% of individuals are well represented on this plane (cos\u00b2 \u2265 0.6). "
            f"{pct_poor:.1f}% are poorly represented (cos\u00b2 < 0.3) \u2014 "
            "interpret their position with caution.",
            kind="success" if pct_well >= 60 else "warn",
        ),
        html.Div("Top 5 individuals contributing to this plane:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Individual", f"Contrib PC{pc_x+1}", f"Contrib PC{pc_y+1}",
             "Total contrib", "cos\u00b2 (plane)"],
            top5_rows,
            highlight_col=3,
        ),
        _tip_box(
            "Professional interpretation workflow: (1) identify clusters of nearby individuals, "
            "(2) look at the variables in the Correlation Circle to understand what differentiates "
            "each cluster, (3) flag outliers and check their cos\u00b2 \u2014 low cos\u00b2 means the outlier "
            "position is an artefact of this projection.",
            kind="expert",
        ),
        _tip_box(
            "Mean scores are close to 0 and standard deviations close to \u221aeigenvalue \u2014 "
            "expected after standardisation. Large deviations signal data anomalies "
            "(duplicate rows, scale errors, or very skewed distributions).",
            kind="info",
        ),
    ]
    return html.Div(children)


def contributions_interpretation_panel(result: dict, pc_idx: int) -> html.Div:
    variables = result["variables"]
    contrib = result["contributions_var"]
    cos2 = result["cos2_var"]
    corr = result["corr_circle"]
    eig = result["eigenvalues"]
    exp = result["explained"]
    n_opt = result["n_optimal"]
    p = len(variables)
    k = result["n_components"]
    pc = pc_idx
    threshold = 100.0 / p
    exp_pc = exp[pc] * 100

    order = np.argsort(contrib[:, pc])[::-1]
    above = [j for j in order if contrib[j, pc] >= threshold]
    below = [j for j in order if contrib[j, pc] < threshold]

    hhi = float(np.sum((contrib[:, pc] / 100) ** 2))
    hhi_min = 1.0 / p
    hhi_norm = (hhi - hhi_min) / (1.0 - hhi_min)
    if hhi_norm < 0.2:
        concentration = "diffuse"
        conc_color = "var(--amber)"
        conc_tip = (
            "The axis is diffuse \u2014 many variables contribute similarly. "
            "This axis captures a general dimension of variability, "
            "not a specific phenomenon. Naming it is harder; "
            "look at the sign pattern rather than individual variables."
        )
    elif hhi_norm < 0.5:
        concentration = "moderate"
        conc_color = "var(--blue)"
        conc_tip = (
            "A moderate number of variables drive this axis. "
            "Focus on those above the uniform threshold "
            "and verify their cos\u00b2 before interpreting."
        )
    else:
        concentration = "concentrated"
        conc_color = "var(--green)"
        conc_tip = (
            "The axis is concentrated \u2014 a small group of variables dominates. "
            "This is the easiest case to interpret: name the axis "
            "after the top contributors, checking their sign (+ or \u2212)."
        )

    pos_drivers = [(variables[j], float(corr[j, pc]), float(contrib[j, pc]))
                   for j in above if corr[j, pc] > 0]
    neg_drivers = [(variables[j], float(corr[j, pc]), float(contrib[j, pc]))
                   for j in above if corr[j, pc] < 0]
    pos_drivers.sort(key=lambda x: -x[2])
    neg_drivers.sort(key=lambda x: -x[2])

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

    r2_opt = cos2[:, :n_opt].sum(axis=1)
    r2_rows = []
    for j in np.argsort(r2_opt)[::-1]:
        r2 = float(r2_opt[j])
        flag = "\u2713" if r2 >= 0.7 else "~" if r2 >= 0.5 else "\u2717"
        col = "var(--green)" if r2 >= 0.7 else "#92400e" if r2 >= 0.5 else "var(--red)"
        r2_rows.append([
            variables[j],
            html.Span(flag, style={"color": col, "fontWeight": 700}),
            f"{r2:.3f}",
            f"{r2*100:.1f} %",
        ])

    detail_rows = []
    for rank, j in enumerate(order):
        c_val = float(contrib[j, pc])
        cs_val = float(cos2[j, pc])
        cr_val = float(corr[j, pc])
        above_flag = "\u25b2" if c_val >= threshold else "\u25bd"
        flag_col = "var(--green)" if c_val >= threshold else "var(--text-muted)"
        direction = "+" if cr_val >= 0 else "\u2212"
        dir_col = "var(--blue)" if cr_val >= 0 else "var(--red)"
        detail_rows.append([
            f"{rank+1}. {variables[j]}",
            html.Span(f"{above_flag} {c_val:.2f}%",
                      style={"color": flag_col, "fontWeight": 600}),
            f"{cs_val:.3f}",
            html.Span(f"{direction}  {abs(cr_val):.3f}",
                      style={"color": dir_col, "fontFamily": "monospace"}),
        ])

    children = [
        _interp_header(
            f"Contributions \u2014 Expert Analysis of PC{pc+1}",
            subtitle=(
                f"PC{pc+1} explains {exp_pc:.1f}% of total variance \u00b7 "
                f"Eigenvalue \u03bb = {eig[pc]:.3f} \u00b7 "
                f"Uniform threshold = {threshold:.1f}% \u00b7 "
                f"Axis type: {concentration}"
            ),
        ),
        _reading_grid([
            ("📊", "What contribution measures",
             "The percentage of the axis's variance that a variable built. "
             "\u03a3 contributions = 100% per PC. "
             "Variables above the uniform threshold (1/p) matter."),
            ("🎯", "Uniform threshold = 1/p",
             f"With {p} variables, the threshold is {threshold:.1f}%. "
             "A variable above it contributed more than average. "
             "Variables below it are passengers, not drivers."),
            ("\u26a0", "Contribution \u2260 Correlation",
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
        html.Div("Axis polarity \u2014 drivers by sign:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Pole", "Variables (ranked by contribution)", "Interpretation hint"],
            [
                ["\u2795 Positive",
                 ", ".join(f"{v} ({c:.1f}%)" for v, _, c in pos_drivers[:5]) or "\u2014",
                 f"High scores on PC{pc+1} \u2192 high {', '.join(v for v, _, _ in pos_drivers[:2])}"
                 if pos_drivers else "\u2014"],
                ["\u2796 Negative",
                 ", ".join(f"{v} ({c:.1f}%)" for v, _, c in neg_drivers[:5]) or "\u2014",
                 f"High scores on PC{pc+1} \u2192 low {', '.join(v for v, _, _ in neg_drivers[:2])}"
                 if neg_drivers else "\u2014"],
            ],
            highlight_col=0,
        ),
        html.Div(f"All {p} variables ranked by contribution to PC{pc+1}:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Variable", "Contribution", "cos\u00b2", "Direction & |corr|"],
            detail_rows,
            highlight_col=1,
        ),
        *([html.Div("Redundant variable pairs on this axis (both cos\u00b2 \u2265 0.6):", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "var(--text-secondary)", "marginBottom": 8,
            }),
            _table(
                ["Variable A", "Variable B", "cos\u00b2 A", "cos\u00b2 B", "Relationship"],
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
        html.Div(f"Global quality of representation across {n_opt} retained PCs (R\u00b2):", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
            "marginTop": 10,
        }),
        _table(
            ["Variable", "Status", "\u03a3 cos\u00b2", "% variance retained"],
            r2_rows,
            highlight_col=2,
        ),
        _tip_box(
            f"Variables with \u03a3cos\u00b2 < 0.5 over {n_opt} PCs are poorly captured "
            "by the retained solution. Their variance lies on discarded axes. "
            "Consider increasing the number of retained components, "
            "or flag these variables as poorly integrated in the PCA.",
            kind="warn",
        ),
        _tip_box(
            "FactoMineR reading rule: a variable is 'active and well-integrated' "
            "if (1) it contributes above 1/p to at least one axis, "
            "(2) its cos\u00b2 > 0.6 on that axis, and "
            "(3) its cumulative R\u00b2 over retained PCs exceeds 0.7. "
            "Variables failing all three criteria may be candidates "
            "for supplementary (illustrative) variable status.",
            kind="expert",
        ),
    ]
    return html.Div(children)


def cos2_interpretation_panel(result: dict, n_pc: int = 5) -> html.Div:
    variables = result["variables"]
    cos2 = result["cos2_var"]
    contrib = result["contributions_var"]
    corr = result["corr_circle"]
    exp = result["explained"]
    eig = result["eigenvalues"]
    n_opt = result["n_optimal"]
    p = len(variables)
    k = result["n_components"]
    n_pc_show = min(n_pc, k)

    r2_opt = cos2[:, :n_opt].sum(axis=1)
    r2_all = cos2.sum(axis=1)

    best_axis = np.argmax(cos2, axis=1)
    best_cos2 = cos2[np.arange(p), best_axis]

    well = [j for j in range(p) if r2_opt[j] >= 0.7]
    moderate = [j for j in range(p) if 0.5 <= r2_opt[j] < 0.7]
    poor = [j for j in range(p) if r2_opt[j] < 0.5]

    axis_well = [(int((cos2[:, k_] >= 0.5).sum()), float(exp[k_] * 100))
                 for k_ in range(n_pc_show)]

    escaped = [variables[j] for j in poor
               if cos2[j, :n_opt].max() < 0.4]

    misleading = []
    for k_ in range(min(3, k)):
        for j in range(p):
            if abs(corr[j, k_]) >= 0.45 and cos2[j, k_] < 0.25:
                misleading.append((variables[j], k_, float(corr[j, k_]), float(cos2[j, k_])))

    diag_rows = []
    for j in np.argsort(r2_opt)[::-1]:
        r2 = float(r2_opt[j])
        bk = int(best_axis[j])
        bc = float(best_cos2[j])
        contrib_bk = float(contrib[j, bk])
        flag = "\u2713" if r2 >= 0.7 else "~" if r2 >= 0.5 else "\u2717"
        flag_col = "var(--green)" if r2 >= 0.7 else "#92400e" if r2 >= 0.5 else "var(--red)"
        diag_rows.append([
            variables[j],
            html.Span(f"{flag} {r2:.3f}",
                      style={"color": flag_col, "fontWeight": 600,
                             "fontFamily": "monospace"}),
            f"PC{bk+1} ({bc:.2f})",
            f"{contrib_bk:.1f}%",
            "Active" if r2 >= 0.7 else ("Supplementary?" if r2 < 0.5 else "Borderline"),
        ])

    axis_rows = []
    for k_ in range(n_pc_show):
        n_well_k = int((cos2[:, k_] >= 0.5).sum())
        n_above_k = int((cos2[:, k_] >= 0.3).sum())
        top2 = ", ".join(variables[j] for j in np.argsort(cos2[:, k_])[::-1][:2])
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
            "cos\u00b2 Quality of Representation \u2014 Expert Analysis",
            subtitle=(
                f"{p} variables \u00b7 {n_opt} retained PCs \u00b7 "
                f"{len(well)} well-repr. ({len(well)/p*100:.0f}%) \u00b7 "
                f"{len(poor)} poorly-repr. ({len(poor)/p*100:.0f}%)"
            ),
        ),
        _reading_grid([
            ("📐", "Geometric meaning",
             "cos\u00b2(variable, axis) is the squared cosine of the angle "
             "between the variable vector and the axis in individual space. "
             "cos\u00b2 = 1 \u2192 perfectly aligned. cos\u00b2 = 0 \u2192 perpendicular."),
            ("📊", "Statistical meaning",
             "cos\u00b2(j, k) is the proportion of variable j's variance "
             "explained by axis k. "
             "\u03a3_k cos\u00b2(j,k) = 1 over all axes (standardised data)."),
            ("🎯", "Reading threshold",
             "cos\u00b2 \u2265 0.7: excellent \u00b7 0.5\u20130.7: good \u00b7 "
             "0.3\u20130.5: moderate \u00b7 < 0.3: poor. "
             "For a plane (2 axes), add the two cos\u00b2 values."),
            ("\u26a0", "cos\u00b2 \u2260 contribution",
             "A variable can have high cos\u00b2 (well represented) "
             "without high contribution (it didn't build the axis). "
             "cos\u00b2 answers 'how well is this variable shown here?' "
             "Contribution answers 'did this variable create this axis?'"),
        ]),
        html.Div("Global representation quality across retained PCs:", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Category", "Count", "Variables", "Action"],
            [
                [html.Span("\u2713 Well repr. (\u03a3cos\u00b2 \u2265 0.7)",
                           style={"color": "var(--green)", "fontWeight": 600}),
                 str(len(well)),
                 ", ".join(variables[j] for j in well) or "\u2014",
                 "Retain and interpret freely"],
                [html.Span("~ Moderate (0.5 \u2264 \u03a3cos\u00b2 < 0.7)",
                           style={"color": "#92400e", "fontWeight": 600}),
                 str(len(moderate)),
                 ", ".join(variables[j] for j in moderate) or "\u2014",
                 "Interpret with caution"],
                [html.Span("\u2717 Poor (\u03a3cos\u00b2 < 0.5)",
                           style={"color": "var(--red)", "fontWeight": 600}),
                 str(len(poor)),
                 ", ".join(variables[j] for j in poor) or "\u2014",
                 "Consider supplementary status or more PCs"],
            ],
            highlight_col=0,
        ),
        html.Div(f"Per-variable diagnostic (sorted by \u03a3cos\u00b2 over retained PCs):", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Variable", f"\u03a3cos\u00b2 ({n_opt} PCs)", "Best axis (cos\u00b2)",
             "Contrib on best", "Status"],
            diag_rows,
            highlight_col=1,
        ),
        html.Div(f"Per-axis discriminating power (first {n_pc_show} PCs):", style={
            "fontWeight": 600, "fontSize": "0.75rem",
            "color": "var(--text-secondary)", "marginBottom": 8,
        }),
        _table(
            ["Axis", "Variance %", "Eigenvalue", "Vars cos\u00b2\u22650.5",
             "Vars cos\u00b2\u22650.3", "Top 2 vars"],
            axis_rows,
            highlight_col=3,
        ),
        *([_tip_box(
                "Variables poorly represented on ALL retained axes (max cos\u00b2 < 0.4): "
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
                f"All variables have at least one axis with cos\u00b2 \u2265 0.4. "
                f"The retained solution of {n_opt} PCs provides "
                "reasonable coverage across all variables.",
                kind="success",
            ),
        ]),
        *([html.Div("\u26a0 Misleading arrows \u2014 high |corr| but low cos\u00b2:", style={
                "fontWeight": 600, "fontSize": "0.75rem",
                "color": "#92400e", "marginBottom": 8,
            }),
            _table(
                ["Variable", "Axis", "|corr|", "cos\u00b2", "Risk"],
                [[v, f"PC{k_+1}", f"{abs(cr):.3f}", f"{cs:.3f}",
                  "Arrow direction misleading \u2014 low representativity"]
                 for v, k_, cr, cs in misleading[:6]],
                highlight_col=4,
            ),
            _tip_box(
                "These variables appear in the correlation circle with a "
                "non-negligible arrow, but their cos\u00b2 is low. "
                "The arrow direction is unreliable \u2014 most of the variable's "
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
            "(3) If two variables always show high cos\u00b2 on the same axes, "
            "they are measuring the same latent dimension \u2014 "
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
