"""
charts_pca.py  —  Plotly figure factories for the PCA tab.
All _LAYOUT overrides strip conflicting keys before spreading.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_LAYOUT = dict(
    template="plotly_white",
    height=460,
    margin=dict(l=50, r=50, t=40, b=50),
    font=dict(family="sans-serif", size=12),
)

_BLUE   = "#3498db"
_RED    = "#e74c3c"
_GREEN  = "#27ae60"
_GREY   = "#95a5a6"
_ORANGE = "#e67e22"

_VAR_COLORS = [
    "#3498db","#e74c3c","#27ae60","#e67e22","#8e44ad",
    "#16a085","#d35400","#2980b9","#c0392b","#1abc9c",
    "#f39c12","#7f8c8d","#6c3483","#117864","#1a5276",
]


def _layout(**overrides):
    """Return _LAYOUT merged with overrides, removing duplicate keys first."""
    base = {k: v for k, v in _LAYOUT.items() if k not in overrides}
    return {**base, **overrides}


# ── Scree plot ────────────────────────────────────────────────────────────────

def scree_plot(result: dict) -> go.Figure:
    explained  = result["explained"] * 100
    cumulative = result["cumulative"] * 100
    n_comp     = result["n_components"]
    n_opt      = result["n_optimal"]
    labels     = [f"PC{i+1}" for i in range(n_comp)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=labels, y=explained,
        name="Individual (%)",
        marker_color=[_BLUE if i < n_opt else _GREY for i in range(n_comp)],
        opacity=0.85,
        text=[f"{v:.1f}" for v in explained],
        textposition="outside",
        textfont=dict(size=9),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=cumulative,
        name="Cumulative (%)",
        mode="lines+markers",
        line=dict(color=_RED, width=2),
        marker=dict(size=6, color=_RED),
    ), secondary_y=True)

    fig.add_vline(
        x=n_opt - 0.5, line_dash="dash",
        line_color=_ORANGE, line_width=1.5,
        annotation_text=f"  Recommended: {n_opt} PCs",
        annotation_font_size=10, annotation_font_color=_ORANGE,
        annotation_position="top right",
    )
    fig.add_hline(
        y=70, line_dash="dot", line_color=_GREEN, line_width=1,
        secondary_y=True,
        annotation_text="70 %", annotation_font_size=9,
        annotation_font_color=_GREEN, annotation_position="right",
    )

    fig.update_layout(
        **_layout(),
        title=dict(text="Scree plot — variance explained per component", font_size=13),
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Individual variance (%)",
                     secondary_y=False, range=[0, max(explained) * 1.3])
    fig.update_yaxes(title_text="Cumulative variance (%)",
                     secondary_y=True, range=[0, 110])
    return fig


# ── Eigenvalue table ──────────────────────────────────────────────────────────

def eigenvalue_table(result: dict) -> go.Figure:
    n     = result["n_components"]
    n_opt = result["n_optimal"]
    pcs   = [f"PC{i+1}" for i in range(n)]
    eig   = [f"{v:.4f}" for v in result["eigenvalues"]]
    var   = [f"{v*100:.2f} %" for v in result["explained"]]
    cum   = [f"{v*100:.2f} %" for v in result["cumulative"]]
    keep  = ["✓ keep" if i < n_opt else "" for i in range(n)]
    fill  = ["#eaf4fb" if i < n_opt else "#ffffff" for i in range(n)]

    fig = go.Figure(go.Table(
        header=dict(
            values=["PC", "Eigenvalue", "Variance explained", "Cumulative", "Recommendation"],
            fill_color="#2c3e50", font_color="white", font_size=11,
            align="center", height=28,
        ),
        cells=dict(
            values=[pcs, eig, var, cum, keep],
            fill_color=fill, align="center", font_size=10, height=24,
        ),
    ))
    fig.update_layout(
        height=max(250, n * 26 + 80),
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text="Eigenvalue table", font_size=13),
    )
    return fig


# ── Correlation circle ────────────────────────────────────────────────────────

def correlation_circle(result: dict, pc_x: int = 0, pc_y: int = 1,
                       cos2_threshold: float = 0.0) -> go.Figure:
    variables = result["variables"]
    corr      = result["corr_circle"]
    cos2      = result["cos2_var"]
    exp_x     = result["explained"][pc_x] * 100
    exp_y     = result["explained"][pc_y] * 100

    fig = go.Figure()

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line=dict(color=_GREY, width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Axes
    for xy in [([-1.15, 1.15], [0, 0]), ([0, 0], [-1.15, 1.15])]:
        fig.add_trace(go.Scatter(
            x=xy[0], y=xy[1], mode="lines",
            line=dict(color=_GREY, width=0.5),
            showlegend=False, hoverinfo="skip",
        ))

    for j, var in enumerate(variables):
        cx  = float(corr[j, pc_x])
        cy  = float(corr[j, pc_y])
        cs  = float(cos2[j, pc_x] + cos2[j, pc_y])
        quality = "well repr." if cs >= 0.6 else "poorly repr." if cs < 0.3 else "moderate repr."
        opacity = max(0.2, cs) if (cos2_threshold > 0 and cs < cos2_threshold) else 1.0
        color   = _VAR_COLORS[j % len(_VAR_COLORS)]

        fig.add_annotation(
            x=cx, y=cy, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.8, arrowcolor=color, opacity=opacity,
        )
        off = 0.07
        lx  = cx + off * (1 if cx >= 0 else -1)
        ly  = cy + off * (1 if cy >= 0 else -1)
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=7, color=color),
            name=var,
            opacity=opacity,
            hovertemplate=(
                f"<b>{var}</b><br>"
                f"Corr PC{pc_x+1}: {cx:.3f}<br>"
                f"Corr PC{pc_y+1}: {cy:.3f}<br>"
                f"cos² (plane): {cs:.3f} — {quality}"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="text",
            text=[var], textfont=dict(size=10, color=color),
            opacity=opacity, showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        **_layout(),
        title=dict(
            text=f"Correlation circle — PC{pc_x+1} ({exp_x:.1f}%) × PC{pc_y+1} ({exp_y:.1f}%)",
            font_size=13,
        ),
        xaxis=dict(
            title=f"PC{pc_x+1} ({exp_x:.1f}%)", range=[-1.35, 1.35],
            zeroline=False, scaleanchor="y",
        ),
        yaxis=dict(
            title=f"PC{pc_y+1} ({exp_y:.1f}%)", range=[-1.35, 1.35],
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5, font_size=10),
    )
    return fig


# ── Biplot ────────────────────────────────────────────────────────────────────

def biplot(result: dict, pc_x: int = 0, pc_y: int = 1) -> go.Figure:
    scores    = result["scores"]
    corr      = result["corr_circle"]
    variables = result["variables"]
    cos2_var  = result["cos2_var"]
    exp_x     = result["explained"][pc_x] * 100
    exp_y     = result["explained"][pc_y] * 100
    n_obs     = result["n_obs"]

    sx = scores[:, pc_x]
    sy = scores[:, pc_y]
    scale = max(np.abs(sx).max(), np.abs(sy).max(), 1e-9)
    sx = sx / scale * 0.88
    sy = sy / scale * 0.88

    idx_show = np.arange(n_obs)
    if n_obs > 1500:
        idx_show = np.random.default_rng(0).choice(n_obs, 1500, replace=False)

    fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line=dict(color=_GREY, width=0.8, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=sx[idx_show], y=sy[idx_show], mode="markers",
        marker=dict(size=4, color=_BLUE, opacity=0.35),
        name="Individuals",
    ))

    for j, var in enumerate(variables):
        cx = float(corr[j, pc_x])
        cy = float(corr[j, pc_y])
        cs = float(cos2_var[j, pc_x] + cos2_var[j, pc_y])
        color = _VAR_COLORS[j % len(_VAR_COLORS)]
        fig.add_annotation(
            x=cx, y=cy, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=2, arrowcolor=color,
        )
        fig.add_trace(go.Scatter(
            x=[cx * 1.09], y=[cy * 1.09], mode="text",
            text=[var], textfont=dict(size=9, color=color),
            showlegend=False,
            hovertemplate=f"<b>{var}</b><br>cos²={cs:.3f}<extra></extra>",
        ))

    fig.update_layout(
        **_layout(),
        title=dict(
            text=f"Biplot — PC{pc_x+1} ({exp_x:.1f}%) × PC{pc_y+1} ({exp_y:.1f}%)",
            font_size=13,
        ),
        xaxis=dict(title=f"PC{pc_x+1} ({exp_x:.1f}%)", range=[-1.3, 1.3],
                   zeroline=True, zerolinewidth=0.5, zerolinecolor=_GREY),
        yaxis=dict(title=f"PC{pc_y+1} ({exp_y:.1f}%)", range=[-1.3, 1.3],
                   zeroline=True, zerolinewidth=0.5, zerolinecolor=_GREY,
                   scaleanchor="x"),
    )
    return fig


# ── Individuals plane ─────────────────────────────────────────────────────────

def individuals_plane(result: dict, pc_x: int = 0, pc_y: int = 1,
                      color_by: str = "cos2",
                      top_contrib_n: int = 0) -> go.Figure:
    scores  = result["scores"]
    cos2    = result["cos2_ind"]
    contrib = result["contributions_ind"]
    labels  = result["row_labels"]
    exp_x   = result["explained"][pc_x] * 100
    exp_y   = result["explained"][pc_y] * 100
    n_obs   = result["n_obs"]

    if color_by == "cos2":
        color_vals  = cos2[:, pc_x] + cos2[:, pc_y]
        color_label = "cos² (quality)"
        colorscale  = "Blues"
    elif color_by == "contrib":
        color_vals  = contrib[:, pc_x] + contrib[:, pc_y]
        color_label = "Contribution (%)"
        colorscale  = "Oranges"
    else:
        color_vals  = np.zeros(n_obs)
        color_label = ""
        colorscale  = [[0, _BLUE], [1, _BLUE]]

    note = ""
    idx_show = np.arange(n_obs)
    if n_obs > 2000:
        idx_show = np.random.default_rng(0).choice(n_obs, 2000, replace=False)
        note = f" (sample 2 000 / {n_obs:,})"

    x_data = scores[:, pc_x]
    y_data = scores[:, pc_y]

    hover = [
        f"<b>{labels[i]}</b><br>"
        f"PC{pc_x+1}: {x_data[i]:.3f}<br>"
        f"PC{pc_y+1}: {y_data[i]:.3f}<br>"
        f"cos² (plane): {(cos2[i,pc_x]+cos2[i,pc_y]):.3f}<br>"
        f"Contrib (%): {(contrib[i,pc_x]+contrib[i,pc_y]):.2f}"
        for i in idx_show
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data[idx_show], y=y_data[idx_show],
        mode="markers",
        marker=dict(
            size=5, opacity=0.65,
            color=color_vals[idx_show],
            colorscale=colorscale,
            showscale=(color_by != "none"),
            colorbar=dict(title=color_label, thickness=12, len=0.6),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        showlegend=False,
    ))

    if top_contrib_n > 0:
        combined = contrib[:, pc_x] + contrib[:, pc_y]
        top_idx  = np.argsort(combined)[::-1][:top_contrib_n]
        fig.add_trace(go.Scatter(
            x=x_data[top_idx], y=y_data[top_idx],
            mode="markers+text",
            marker=dict(size=9, color=_RED, symbol="circle-open", line_width=1.5),
            text=[labels[i] for i in top_idx],
            textfont=dict(size=8, color=_RED),
            textposition="top center",
            name=f"Top {top_contrib_n} contributors",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=_GREY, line_width=0.8)
    fig.add_vline(x=0, line_dash="dot", line_color=_GREY, line_width=0.8)

    fig.update_layout(
        **_layout(),
        title=dict(
            text=f"Individuals — PC{pc_x+1} ({exp_x:.1f}%) × PC{pc_y+1} ({exp_y:.1f}%){note}",
            font_size=13,
        ),
        xaxis_title=f"PC{pc_x+1} ({exp_x:.1f}%)",
        yaxis_title=f"PC{pc_y+1} ({exp_y:.1f}%)",
    )
    return fig


# ── Contributions bar ─────────────────────────────────────────────────────────

def contributions_bar(result: dict, pc_idx: int) -> go.Figure:
    variables  = result["variables"]
    contrib    = result["contributions_var"][:, pc_idx]
    cos2       = result["cos2_var"][:, pc_idx]
    corr       = result["corr_circle"][:, pc_idx]
    threshold  = 100.0 / len(variables)
    exp        = result["explained"][pc_idx] * 100
    order      = np.argsort(contrib)

    sv = [variables[i] for i in order]
    sc = contrib[order]
    scos2 = cos2[order]
    scorr = corr[order]
    colors = [_BLUE if v >= threshold else _GREY for v in sc]

    fig = go.Figure(go.Bar(
        y=sv, x=sc, orientation="h",
        marker_color=colors,
        customdata=np.column_stack([scos2, scorr]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution: %{x:.2f}%<br>"
            "cos²: %{customdata[0]:.3f}<br>"
            "Corr: %{customdata[1]:.3f}"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=threshold, line_dash="dash",
        line_color=_RED, line_width=1.5,
        annotation_text=f"Uniform = {threshold:.1f}%",
        annotation_font_size=9, annotation_font_color=_RED,
        annotation_position="top right",
    )
    h = max(300, len(variables) * 28 + 80)
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=120, r=60, t=50, b=50),
        font=dict(family="sans-serif", size=12),
        title=dict(text=f"Variable contributions — PC{pc_idx+1} ({exp:.1f}%)", font_size=13),
        xaxis_title="Contribution (%)",
        yaxis_title="",
    )
    return fig


# ── cos² heatmap ──────────────────────────────────────────────────────────────

def cos2_heatmap(result: dict, n_pc: int = 5) -> go.Figure:
    n_pc      = min(n_pc, result["n_components"])
    variables = result["variables"]
    cos2      = result["cos2_var"][:, :n_pc]
    exp       = result["explained"][:n_pc] * 100
    col_labels = [f"PC{i+1} ({exp[i]:.1f}%)" for i in range(n_pc)]

    fig = px.imshow(
        cos2, x=col_labels, y=variables,
        color_continuous_scale="Blues", zmin=0, zmax=1,
        text_auto=".2f", aspect="auto",
        labels=dict(color="cos²"),
    )
    h = max(300, len(variables) * 30 + 100)
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=130, r=50, t=50, b=80),
        font=dict(family="sans-serif", size=12),
        title=dict(text="Quality of representation (cos²) — variables × PCs", font_size=13),
    )
    fig.update_traces(textfont_size=9)
    return fig


# ── Contributions heatmap ─────────────────────────────────────────────────────

def contributions_heatmap(result: dict, n_pc: int = 5) -> go.Figure:
    n_pc      = min(n_pc, result["n_components"])
    variables = result["variables"]
    contrib   = result["contributions_var"][:, :n_pc]
    exp       = result["explained"][:n_pc] * 100
    col_labels = [f"PC{i+1} ({exp[i]:.1f}%)" for i in range(n_pc)]

    fig = px.imshow(
        contrib, x=col_labels, y=variables,
        color_continuous_scale="Oranges",
        text_auto=".1f", aspect="auto",
        labels=dict(color="Contrib. (%)"),
    )
    h = max(300, len(variables) * 30 + 100)
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=130, r=50, t=50, b=80),
        font=dict(family="sans-serif", size=12),
        title=dict(text="Variable contributions (%) — variables × PCs", font_size=13),
    )
    fig.update_traces(textfont_size=9)
    return fig


# ── Empty placeholder ─────────────────────────────────────────────────────────

def empty_pca(msg: str = "Upload a dataset with at least 2 numeric columns.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#7f8c8d"),
    )
    fig.update_layout(height=300, template="plotly_white")
    return fig
