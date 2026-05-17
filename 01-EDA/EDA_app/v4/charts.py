"""
charts.py  —  Plotly figure factories.

Performance notes
-----------------
KDE (histogram)
    Old: dense (300 × n) broadcast → O(300n) RAM and ops.
    New: FFT-based KDE via np.fft → O(n log n).  For n > 20 000 the input
    is first compressed to a grid of 512 points before the FFT so memory
    stays bounded regardless of n.

Scatter
    Plotly renders all points on the client; above 10 000 the plot becomes
    sluggish with no visual gain.  We subsample to 10 000 with a
    random seed-stable draw and add a note to the figure title.

Contingency heatmap
    Old: double Python loop O(k × r × n).
    New: np.ravel_multi_index + np.bincount → single vectorised pass O(n).
"""
from __future__ import annotations
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import drop_nan

_LAYOUT       = dict(template='plotly_white', height=460,
                     margin=dict(l=44, r=44, t=28, b=44))
_SCATTER_MAX  = 10_000   # max points rendered in scatter
_KDE_GRID     = 512      # FFT grid size (power of 2 → fast FFT)
_KDE_COMPRESS = 20_000   # above this n, bin into _KDE_GRID before FFT


# ── KDE helpers ───────────────────────────────────────────────────────────────

def _silverman_bw(c: np.ndarray) -> float:
    """Silverman's rule of thumb — O(n) single pass."""
    n     = len(c)
    sigma = c.std(ddof=1)
    iqr   = np.subtract(*np.percentile(c, [75, 25]))
    return 0.9 * min(sigma, iqr / 1.34) * n ** (-0.2)


def _kde_fft(c: np.ndarray, bw: float, n_grid: int = _KDE_GRID
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT-based Gaussian KDE.  O(n + g·log g) where g = n_grid.

    For n > _KDE_COMPRESS the data is first binned into n_grid counts
    (linear binning, O(n)) — this is the standard trick for fast KDE on
    large datasets (Wand & Jones 1995).

    Returns (x_grid, density).
    """
    lo = c.min() - 3 * bw
    hi = c.max() + 3 * bw

    # ── Bin data onto uniform grid ───────────────────────────────────────────
    dx = (hi - lo) / (n_grid - 1)
    if len(c) > _KDE_COMPRESS:
        # Linear binning: each point contributes fractionally to two adjacent bins
        grid_idx = (c - lo) / (hi - lo) * (n_grid - 1)
        lo_idx   = np.floor(grid_idx).astype(int).clip(0, n_grid - 2)
        hi_idx   = lo_idx + 1
        hi_frac  = grid_idx - lo_idx
        counts   = np.zeros(n_grid)
        np.add.at(counts, lo_idx, 1.0 - hi_frac)
        np.add.at(counts, hi_idx, hi_frac)
    else:
        # Simple nearest-bin assignment for smaller arrays
        grid_idx = ((c - lo) / (hi - lo) * (n_grid - 1)).clip(0, n_grid - 1)
        lo_idx   = np.floor(grid_idx).astype(int).clip(0, n_grid - 2)
        hi_idx   = lo_idx + 1
        hi_frac  = grid_idx - lo_idx
        counts   = np.zeros(n_grid)
        np.add.at(counts, lo_idx, 1.0 - hi_frac)
        np.add.at(counts, hi_idx, hi_frac)

    # ── Gaussian kernel via FFT convolution ──────────────────────────────────
    # Build kernel centred at 0 in wrap-around order (standard for FFT convolution)
    dx      = (hi - lo) / (n_grid - 1)
    x_grid  = np.linspace(lo, hi, n_grid)
    t       = np.fft.fftfreq(n_grid, d=1.0 / n_grid) * dx   # grid spacings, wrap-around
    kernel  = np.exp(-0.5 * (t / bw) ** 2)
    kernel /= kernel.sum()   # normalise so convolution preserves total count

    density = np.fft.ifft(np.fft.fft(counts) * np.fft.fft(kernel)).real
    density = np.maximum(density / (len(c) * dx), 0)

    return x_grid, density


# ── Public figure factories ───────────────────────────────────────────────────

def histogram(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()
    n    = len(clean)
    bw   = _silverman_bw(clean)
    nbins = min(int(np.sqrt(n)), 50)

    x_kde, kde = _kde_fft(clean, bw)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x=clean, nbinsx=nbins, name='Count',
                               marker_color='#3498db', opacity=0.75),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x_kde, y=kde, mode='lines', name='Density',
                             line=dict(color='#e74c3c', width=2.5)),
                  secondary_y=True)
    fig.update_layout(**_LAYOUT, xaxis_title=col, bargap=0.02)
    fig.update_yaxes(title_text="Count",   secondary_y=False)
    fig.update_yaxes(title_text="Density", secondary_y=True, showgrid=False)
    return fig


def boxplot(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()
    # For very large n, boxpoints='outliers' sends every outlier to the client.
    # Switch to 'suspectedoutliers' above 50k to cap payload.
    bp = 'suspectedoutliers' if len(clean) > 50_000 else 'outliers'
    fig = go.Figure(go.Box(y=clean, name=col, marker_color='#e74c3c',
                           boxmean='sd', boxpoints=bp))
    fig.update_layout(**_LAYOUT, yaxis_title=col)
    return fig


def bar_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    order      = np.argsort(cnts)[::-1]
    sv, sc     = vals[order].astype(str), cnts[order]
    total      = int(sc.sum())
    pcts       = sc / total * 100
    labels     = [f"{c:,} ({p:.1f}%)" for c, p in zip(sc, pcts)]
    fig = go.Figure(go.Bar(x=sv, y=sc, text=labels,
                           textposition='outside', marker_color='#3498db'))
    fig.update_layout(**_LAYOUT, xaxis_title=col, yaxis_title='Count',
                      yaxis_range=[0, sc.max() * 1.15])
    return fig


def pie_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    fig = go.Figure(go.Pie(labels=vals.astype(str), values=cnts,
                           hole=0.3, textinfo='label+percent'))
    fig.update_layout(**_LAYOUT, title=f"Distribution of {col}")
    return fig


def scatter(x: np.ndarray, y: np.ndarray, xl: str, yl: str) -> go.Figure:
    mask = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 2:
        return empty("Not enough points")

    note = ""
    if len(xm) > _SCATTER_MAX:
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(xm), _SCATTER_MAX, replace=False)
        xm, ym = xm[idx], ym[idx]
        note = f" (sample of {_SCATTER_MAX:,})"

    fig = px.scatter(x=xm, y=ym, trendline='ols',
                     labels={'x': xl + note, 'y': yl})
    fig.update_layout(**_LAYOUT)
    return fig


def grouped_boxplot(num: np.ndarray, cat: np.ndarray,
                    numl: str, catl: str) -> go.Figure:
    cats   = np.unique(cat)
    bp     = 'suspectedoutliers' if len(num) > 50_000 else 'outliers'
    traces = [
        go.Box(y=drop_nan(num[cat == c]), name=str(c),
               boxmean='sd', boxpoints=bp)
        for c in cats
        if drop_nan(num[cat == c]).size > 0
    ]
    if not traces:
        return empty("No valid groups")
    fig = go.Figure(data=traces)
    fig.update_layout(**_LAYOUT, xaxis_title=catl, yaxis_title=numl)
    return fig


def heatmap_categorical(a: np.ndarray, b: np.ndarray,
                        al: str, bl: str) -> go.Figure:
    """
    Vectorised contingency table via np.ravel_multi_index + np.bincount.
    O(n) instead of the previous O(k × r × n) double loop.
    """
    ua, ia = np.unique(a, return_inverse=True)   # ia: integer codes for a
    ub, ib = np.unique(b, return_inverse=True)   # ib: integer codes for b
    k, r   = len(ua), len(ub)

    flat   = np.ravel_multi_index((ia, ib), (k, r))
    counts = np.bincount(flat, minlength=k * r).reshape(k, r).astype(float)

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pct = counts / row_sums * 100

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
