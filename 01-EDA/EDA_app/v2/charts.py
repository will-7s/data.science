"""
charts.py
─────────
All Plotly figure factories — no Dash, no store, no stats logic here.

Every function takes plain NumPy arrays + labels and returns a go.Figure.
Keeping charts pure makes them composable and easy to test in isolation.

Palette
-------
BLUE    #3498db   numeric distributions
RED     #e74c3c   box plots
GREEN   #2ecc71   bar charts
PURPLE  #9b59b6   scatter plots
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── shared style helpers ──────────────────────────────────────────────────────

_LAYOUT = dict(template='plotly_white', height=460, margin=dict(l=44, r=44, t=28, b=44))

def _fig(traces=(), **layout_kwargs) -> go.Figure:
    """Base figure with consistent styling."""
    fig = go.Figure(data=list(traces))
    fig.update_layout(**_LAYOUT, **layout_kwargs)
    return fig


# ── univariate — numeric ──────────────────────────────────────────────────────

def histogram(arr: np.ndarray, col: str) -> go.Figure:
    fig = px.histogram(x=arr, color_discrete_sequence=['#3498db'],
                       labels={'x': col, 'y': 'Count'})
    fig.update_layout(**_LAYOUT)
    return fig


def boxplot(arr: np.ndarray, col: str) -> go.Figure:
    return _fig([go.Box(y=arr, name=col, marker_color='#e74c3c',
                        boxmean='sd', boxpoints='outliers')],
                yaxis_title=col)


def bar_numeric(arr: np.ndarray, col: str, bins: int = 20) -> go.Figure:
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return _fig([go.Bar(x=centers, y=counts, marker_color='#2ecc71',
                        name=col)],
                xaxis_title=col, yaxis_title='Count')


# ── univariate — categorical ──────────────────────────────────────────────────

def bar_categorical(arr: np.ndarray, col: str) -> go.Figure:
    values, counts = np.unique(arr, return_counts=True)
    order = np.argsort(counts)[::-1]   # most frequent first
    return _fig([go.Bar(x=values[order].astype(str), y=counts[order],
                        marker_color='#3498db', name=col)],
                xaxis_title=col, yaxis_title='Count')


# ── bivariate ─────────────────────────────────────────────────────────────────

def scatter(x: np.ndarray, y: np.ndarray, x_col: str, y_col: str) -> go.Figure:
    mask = ~(np.isnan(x) | np.isnan(y))
    fig = px.scatter(x=x[mask], y=y[mask],
                     labels={'x': x_col, 'y': y_col},
                     opacity=0.55,
                     trendline='ols',
                     color_discrete_sequence=['#9b59b6'])
    fig.update_layout(**_LAYOUT)
    return fig


def grouped_boxplot(num: np.ndarray, cat: np.ndarray,
                    num_col: str, cat_col: str) -> go.Figure:
    """One box per category value."""
    traces = []
    for label in np.unique(cat):
        values = num[cat == label]
        if values.dtype.kind == 'f':
            values = values[~np.isnan(values)]
        if len(values) > 0:
            traces.append(go.Box(y=values, name=str(label), boxmean='sd'))
    return _fig(traces, xaxis_title=cat_col, yaxis_title=num_col)


def heatmap_categorical(d1: np.ndarray, d2: np.ndarray,
                        col1: str, col2: str) -> go.Figure:
    """Row-percentage contingency heatmap for two categorical variables."""
    u1, u2 = np.unique(d1), np.unique(d2)
    matrix = np.array([
        [np.sum((d1 == v1) & (d2 == v2)) for v2 in u2]
        for v1 in u1
    ], dtype=float)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pct = matrix / row_sums * 100

    fig = px.imshow(pct,
                    x=u2.astype(str), y=u1.astype(str),
                    labels=dict(x=col2, y=col1, color='%'),
                    color_continuous_scale='Blues',
                    text_auto='.1f', aspect='auto')
    fig.update_layout(**_LAYOUT)
    return fig


def correlation_heatmap(matrix: np.ndarray, columns: list[str]) -> go.Figure:
    """Symmetric correlation matrix with diverging colour scale."""
    fig = px.imshow(matrix,
                    x=columns, y=columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    text_auto='.2f', aspect='auto',
                    labels=dict(color='r'))
    fig.update_layout(**_LAYOUT)
    return fig


# ── empty placeholder ─────────────────────────────────────────────────────────

def empty(message: str = 'No data') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref='paper', yref='paper',
                       x=0.5, y=0.5, showarrow=False, font_size=14)
    fig.update_layout(height=400, template='plotly_white')
    return fig
