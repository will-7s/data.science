"""
charts.py  —  Plotly figure factories with professional colour palette.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import drop_nan

# ── Professional colour palette (aligned with app design system) ──────────────
_PRIMARY     = '#6366f1'
_SECONDARY   = '#0ea5e9'
_ACCENT      = '#10b981'
_DANGER      = '#ef4444'
_WARNING     = '#f59e0b'
_PURPLE      = '#8b5cf6'
_PINK        = '#ec4899'
_CYAN        = '#06b6d4'
_LIME        = '#84cc16'
_ORANGE      = '#f97316'

_CATEGORICAL = [_PRIMARY, _SECONDARY, _ACCENT, _WARNING, _DANGER,
                _PURPLE, _PINK, _CYAN, _LIME, _ORANGE]

_SEQUENTIAL = ['#eef2ff', '#c7d2fe', '#a5b4fc', '#818cf8', _PRIMARY,
               '#4f46e5', '#4338ca']

_LAYOUT = dict(
    template='plotly_white', height=460,
    margin=dict(l=44, r=44, t=28, b=44),
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
              size=12),
    hoverlabel=dict(
        font_family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        font_size=12,
        bordercolor='rgba(0,0,0,0.1)',
    ),
)
_SCATTER_MAX = 10_000
_BAR_MAX_CATS= 50
_PIE_MAX_CATS= 20


def histogram(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()

    n      = len(clean)
    nbins  = min(int(np.sqrt(n)), 50)
    counts, edges = np.histogram(clean, bins=nbins)
    midpoints     = (edges[:-1] + edges[1:]) / 2
    widths        = np.diff(edges)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=midpoints, y=counts, width=widths,
        marker_color='rgba(99,102,241,0.65)', opacity=1,
        marker_line=dict(color=_PRIMARY, width=1.5),
        hovertemplate='%{y:,} observations<extra></extra>',
    ))
    fig.update_layout(**_LAYOUT, xaxis_title=col, yaxis_title='Count',
                      bargap=0.02)
    return fig


def boxplot(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()
    bp  = 'suspectedoutliers' if len(clean) > 50_000 else 'outliers'
    fig = go.Figure(go.Box(
        y=clean, name=col,
        marker=dict(color=_DANGER, size=4, opacity=0.8),
        line=dict(color=_PRIMARY, width=2),
        fillcolor='rgba(99,102,241,0.15)',
        boxmean='sd', boxpoints=bp,
        whiskerwidth=0.6,
    ))
    fig.update_layout(**_LAYOUT, yaxis_title=col)
    return fig


def bar_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    order      = np.argsort(cnts)[::-1]
    sv         = vals[order].astype(str)
    sc         = cnts[order]
    total      = int(sc.sum())

    if len(sv) > _BAR_MAX_CATS:
        others_n   = int(sc[_BAR_MAX_CATS:].sum())
        others_cnt = len(sv) - _BAR_MAX_CATS
        sv = np.append(sv[:_BAR_MAX_CATS], f"({others_cnt} others)")
        sc = np.append(sc[:_BAR_MAX_CATS], others_n)

    pcts   = sc / total * 100
    labels = [f"{c:,} ({p:.1f}%)" for c, p in zip(sc, pcts)]

    fig = go.Figure(go.Bar(
        x=sv, y=sc, text=labels,
        textposition='outside', marker_color=_PRIMARY,
        marker_line=dict(color=_PRIMARY, width=1),
        hovertemplate='%{x}: %{y:,}<extra></extra>',
    ))
    fig.update_layout(**_LAYOUT, xaxis_title=col, yaxis_title='Count',
                      yaxis_range=[0, sc.max() * 1.18],
                      xaxis_tickangle=-30 if len(sv) > 10 else 0)
    return fig


def pie_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    order      = np.argsort(cnts)[::-1]
    sv         = vals[order].astype(str)
    sc         = cnts[order]

    if len(sv) > _PIE_MAX_CATS:
        other_sum = int(sc[_PIE_MAX_CATS:].sum())
        sv = np.append(sv[:_PIE_MAX_CATS], 'Other')
        sc = np.append(sc[:_PIE_MAX_CATS], other_sum)

    colors = _CATEGORICAL[:len(sv)]

    fig = go.Figure(go.Pie(
        labels=sv, values=sc, hole=0.35,
        marker=dict(colors=colors, line=dict(color='#fff', width=1.5)),
        textinfo='label+percent',
        textfont=dict(size=11),
        hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>',
        pull=[0.03 if i == 0 else 0 for i in range(len(sv))],
    ))
    fig.update_layout(**_LAYOUT, title=f'Distribution of {col}')
    return fig


def scatter(x: np.ndarray, y: np.ndarray, xl: str, yl: str) -> go.Figure:
    mask   = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 2:
        return empty('Not enough points')

    note = ''
    if len(xm) > _SCATTER_MAX:
        idx    = np.random.default_rng(0).choice(len(xm), _SCATTER_MAX, replace=False)
        xm, ym = xm[idx], ym[idx]
        note   = f' (sample of {_SCATTER_MAX:,})'

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xm, y=ym, mode='markers',
        marker=dict(color=_PRIMARY, size=5, opacity=0.6,
                    line=dict(color='rgba(255,255,255,0.3)', width=0.5)),
        name='Data',
        hovertemplate=f'{xl}: %{{x:.3f}}<br>{yl}: %{{y:.3f}}<extra></extra>',
    ))

    if len(xm) >= 2:
        try:
            coeffs = np.polyfit(xm, ym, 1)
            x_line = np.array([xm.min(), xm.max()])
            y_line = np.polyval(coeffs, x_line)
            r2     = float(np.corrcoef(xm, ym)[0, 1] ** 2)
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines',
                line=dict(color=_DANGER, width=2, dash='dash'),
                name=f'OLS  R²={r2:.3f}',
                hovertemplate='OLS: %{y:.3f}<extra></extra>',
            ))
        except np.linalg.LinAlgError:
            pass

    fig.update_layout(
        **_LAYOUT,
        xaxis_title=xl + note,
        yaxis_title=yl,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def grouped_boxplot(num: np.ndarray, cat: np.ndarray,
                    numl: str, catl: str) -> go.Figure:
    cat_uniq, cat_codes = np.unique(cat, return_inverse=True)

    sort_idx   = np.argsort(cat_codes, kind='stable')
    num_sorted = num[sort_idx]
    cod_sorted = cat_codes[sort_idx]

    split_pts  = np.where(np.diff(cod_sorted))[0] + 1
    groups     = np.split(num_sorted, split_pts)

    bp = 'suspectedoutliers' if len(num) > 50_000 else 'outliers'
    traces = []
    for i, (label, grp) in enumerate(zip(cat_uniq, groups)):
        clean_grp = drop_nan(grp)
        if clean_grp.size > 0:
            clr = _CATEGORICAL[i % len(_CATEGORICAL)]
            r,g,b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
            traces.append(go.Box(
                y=clean_grp, name=str(label),
                marker=dict(color=clr, size=4, opacity=0.8),
                line=dict(color=clr, width=2),
                fillcolor=f'rgba({r},{g},{b},0.15)',
                boxmean='sd', boxpoints=bp,
                whiskerwidth=0.6,
            ))

    if not traces:
        return empty('No valid groups')

    fig = go.Figure(data=traces)
    fig.update_layout(**_LAYOUT, xaxis_title=catl, yaxis_title=numl)
    return fig


def heatmap_categorical(a: np.ndarray, b: np.ndarray,
                        al: str, bl: str) -> go.Figure:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    k, r   = len(ua), len(ub)
    flat   = np.ravel_multi_index((ia, ib), (k, r))
    counts = np.bincount(flat, minlength=k * r).reshape(k, r).astype(float)
    row_sums            = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pct    = counts / row_sums * 100

    fig = px.imshow(
        pct, x=ub.astype(str), y=ua.astype(str),
        labels={'x': bl, 'y': al, 'color': '%'},
        text_auto='.1f' if k * r <= 400 else False,
        color_continuous_scale=_SEQUENTIAL, aspect='auto',
    )
    fig.update_layout(**_LAYOUT)
    return fig


def correlation_heatmap(mat: np.ndarray, cols: list[str]) -> go.Figure:
    n = len(cols)
    fig = px.imshow(
        mat, x=cols, y=cols,
        color_continuous_scale='RdYlBu_r', zmin=-1, zmax=1,
        text_auto='.2f' if n <= 15 else False,
        aspect='auto',
    )
    fig.update_layout(**_LAYOUT)
    return fig


def empty(msg: str = 'No data') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref='paper', yref='paper',
                       x=0.5, y=0.5, showarrow=False, font_size=14)
    fig.update_layout(height=400, template='plotly_white',
                      font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"))
    return fig
