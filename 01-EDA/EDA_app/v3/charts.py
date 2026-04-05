import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import drop_nan

_LAYOUT = dict(template='plotly_white', height=460, margin=dict(l=44, r=44, t=28, b=44))

def histogram(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()
    n = len(clean)
    nbins = min(int(np.sqrt(n)), 50)
    sigma = clean.std()
    iqr = np.percentile(clean, 75) - np.percentile(clean, 25)
    bw = 0.9 * min(sigma, iqr / 1.34) * n ** (-0.2)
    x_kde = np.linspace(clean.min() - 2*bw, clean.max() + 2*bw, 300)
    diff = (x_kde[:, None] - clean[None, :]) / bw
    kde = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2*np.pi))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x=clean, nbinsx=nbins, name='Count',
                               marker_color='#3498db', opacity=0.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_kde, y=kde, mode='lines', name='Density',
                             line=dict(color='#e74c3c', width=2.5)), secondary_y=True)
    fig.update_layout(**_LAYOUT, xaxis_title=col, bargap=0.02)
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Density", secondary_y=True, showgrid=False)
    return fig

def boxplot(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()
    fig = go.Figure(go.Box(y=clean, name=col, marker_color='#e74c3c',
                           boxmean='sd', boxpoints='outliers'))
    fig.update_layout(**_LAYOUT, yaxis_title=col)
    return fig

def bar_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    sorted_vals = vals[order].astype(str)
    sorted_cnts = cnts[order]
    total = sorted_cnts.sum()
    labels = [f"{c} ({c/total*100:.1f}%)" for c in sorted_cnts]
    fig = go.Figure(go.Bar(x=sorted_vals, y=sorted_cnts, text=labels,
                           textposition='outside', marker_color='#3498db'))
    fig.update_layout(**_LAYOUT, xaxis_title=col, yaxis_title='Count')
    fig.update_layout(yaxis_range=[0, sorted_cnts.max() * 1.15])
    return fig

def pie_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    fig = go.Figure(go.Pie(labels=vals.astype(str), values=cnts,
                           hole=0.3, textinfo='label+percent'))
    fig.update_layout(**_LAYOUT, title=f"Distribution of {col}")
    return fig

def scatter(x: np.ndarray, y: np.ndarray, xl: str, yl: str) -> go.Figure:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return empty("Not enough points")
    fig = px.scatter(x=x[mask], y=y[mask], trendline='ols',
                     labels={'x': xl, 'y': yl})
    fig.update_layout(**_LAYOUT)
    return fig

def grouped_boxplot(num: np.ndarray, cat: np.ndarray, numl: str, catl: str) -> go.Figure:
    cats = np.unique(cat)
    traces = []
    for c in cats:
        values = drop_nan(num[cat == c])
        if values.size > 0:
            traces.append(go.Box(y=values, name=str(c),
                                 boxmean='sd', boxpoints='outliers'))
    if not traces:
        return empty("No valid groups")
    fig = go.Figure(data=traces)
    fig.update_layout(**_LAYOUT, xaxis_title=catl, yaxis_title=numl)
    return fig

def heatmap_categorical(a: np.ndarray, b: np.ndarray, al: str, bl: str) -> go.Figure:
    ua, ub = np.unique(a), np.unique(b)
    mat = np.array([[np.sum((a == va) & (b == vb)) for vb in ub] for va in ua], dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pct = mat / row_sums * 100
    fig = px.imshow(pct, x=ub.astype(str), y=ua.astype(str),
                    labels={'x': bl, 'y': al, 'color': '%'},
                    text_auto='.1f', color_continuous_scale='Blues', aspect='auto')
    fig.update_layout(**_LAYOUT)
    return fig

def correlation_heatmap(mat: np.ndarray, cols: list[str]) -> go.Figure:
    fig = px.imshow(mat, x=cols, y=cols, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, text_auto='.2f', aspect='auto')
    fig.update_layout(**_LAYOUT)
    return fig

def empty(msg: str = 'No data') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref='paper', yref='paper',
                       x=0.5, y=0.5, showarrow=False, font_size=14)
    fig.update_layout(height=400, template='plotly_white')
    return fig