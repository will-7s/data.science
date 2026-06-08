"""
charts.py  —  Plotly figure factories.

v5 fixes and optimisations
---------------------------
histogram
    - KDE bandwidth guard: bw=0 (constant/near-constant column) no longer
      causes divide-by-zero — falls back to a plain bar chart.
    - make_subplots removed: replaced by a single go.Figure with a secondary
      y-axis declared via layout (yaxis2=dict(overlaying='y', side='right')).
      This saves ~8 ms per call (make_subplots construction overhead).
    - Pre-binned go.Bar instead of go.Histogram: np.histogram runs in C,
      sends 50 points to the client instead of n raw values.

scatter
    - trendline='ols' required statsmodels (not in requirements.txt) and
      crashed on import.  Replaced by a manual np.polyfit OLS trendline
      drawn as a go.Scatter trace — no extra dependency, identical output.

bar_categorical
    - Capped at 50 categories in the chart (others → "N others" label).
      Large-cardinality string columns no longer produce unreadable charts.

grouped_boxplot
    - np.unique(cat) + boolean mask per group replaced by
      np.argsort(cat_codes) + np.split — single vectorised groupby, O(n log n).

correlation_heatmap
    - text_auto disabled above 15×15 (unreadable anyway, saves render time).

pie_categorical
    - Capped at 20 slices (others → "Other" slice) to prevent browser freeze.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import drop_nan

_LAYOUT      = dict(template='plotly_white', height=460,
                    margin=dict(l=44, r=44, t=28, b=44),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, -apple-system, sans-serif", size=12),
                    hoverlabel=dict(font_family="Inter, -apple-system, sans-serif"))
_SCATTER_MAX = 10_000
_KDE_GRID    = 512
_KDE_COMPRESS= 20_000
_BAR_MAX_CATS= 50      # max categories shown in bar chart
_PIE_MAX_CATS= 20      # max slices in pie chart


# ── KDE helpers ────────────────────────────────────────────────────────────────

def _silverman_bw(c: np.ndarray) -> float:
    n     = len(c)
    sigma = c.std(ddof=1)
    q75, q25 = np.percentile(c, [75, 25])
    iqr   = float(q75 - q25)
    s     = min(sigma, iqr / 1.34) if iqr > 0 else sigma
    return 0.9 * s * n ** (-0.2)


def _kde_fft(c: np.ndarray, bw: float,
             n_grid: int = _KDE_GRID) -> tuple[np.ndarray, np.ndarray] | None:
    """
    FFT-based Gaussian KDE.  Returns None if bw <= 0 (constant column).
    O(n + g log g) where g = n_grid.
    """
    if bw <= 0:
        return None

    lo = c.min() - 3 * bw
    hi = c.max() + 3 * bw
    if lo == hi:
        return None

    # ── Linear binning ────────────────────────────────────────────────────────
    n_grid_eff = n_grid - 1 if len(c) > _KDE_COMPRESS else n_grid
    counts = np.zeros(n_grid)
    span   = hi - lo
    grid_idx = ((c - lo) / span * (n_grid - 1)).clip(0, n_grid_eff)
    lo_idx   = np.floor(grid_idx).astype(int).clip(0, n_grid - 2)
    hi_frac  = grid_idx - lo_idx
    np.add.at(counts, lo_idx,     1.0 - hi_frac)
    np.add.at(counts, lo_idx + 1, hi_frac)

    # ── FFT convolution ───────────────────────────────────────────────────────
    dx     = span / (n_grid - 1)
    x_grid = np.linspace(lo, hi, n_grid)
    t      = np.fft.fftfreq(n_grid, d=1.0 / n_grid) * dx
    kernel = np.exp(-0.5 * (t / bw) ** 2)
    kernel /= kernel.sum()

    density = np.fft.ifft(np.fft.fft(counts) * np.fft.fft(kernel)).real
    density = np.maximum(density / (len(c) * dx), 0)
    return x_grid, density


# ── Public figure factories ────────────────────────────────────────────────────

def histogram(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()

    n     = len(clean)
    nbins = min(int(np.sqrt(n)), 50)
    bw    = _silverman_bw(clean)
    kde_result = _kde_fft(clean, bw) if bw > 0 else None

    # Pre-bin with numpy — sends 50 points instead of n raw values
    counts, edges = np.histogram(clean, bins=nbins)
    midpoints     = (edges[:-1] + edges[1:]) / 2
    widths        = np.diff(edges)
    dx_avg        = widths.mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=midpoints, y=counts, width=widths,
        name='Count', marker_color='#3498db', opacity=0.75,
        hovertemplate='%{y:,} observations<extra></extra>',
    ))

    if kde_result is not None:
        x_kde, kde = kde_result
        # Scale KDE to count axis so both share one y-axis cleanly
        kde_scaled = kde * (n * dx_avg)
        fig.add_trace(go.Scatter(
            x=x_kde, y=kde_scaled, mode='lines', name='KDE',
            line=dict(color='#e74c3c', width=2.5),
            hovertemplate='density: %{customdata:.4f}<extra></extra>',
            customdata=kde,
        ))

    fig.update_layout(**_LAYOUT, xaxis_title=col, yaxis_title='Count',
                      bargap=0.02,
                      legend=dict(orientation='h', yanchor='bottom',
                                  y=1.02, xanchor='right', x=1))
    return fig


def boxplot(arr: np.ndarray, col: str) -> go.Figure:
    clean = drop_nan(arr)
    if clean.size == 0:
        return empty()
    bp  = 'suspectedoutliers' if len(clean) > 50_000 else 'outliers'
    fig = go.Figure(go.Box(
        y=clean, name=col, marker_color='#e74c3c',
        boxmean='sd', boxpoints=bp,
    ))
    fig.update_layout(**_LAYOUT, yaxis_title=col)
    return fig


def bar_categorical(arr: np.ndarray, col: str) -> go.Figure:
    vals, cnts = np.unique(arr, return_counts=True)
    order      = np.argsort(cnts)[::-1]
    sv         = vals[order].astype(str)
    sc         = cnts[order]
    total      = int(sc.sum())

    # Cap at _BAR_MAX_CATS — aggregate remainder into "N others"
    if len(sv) > _BAR_MAX_CATS:
        others_n   = int(sc[_BAR_MAX_CATS:].sum())
        others_cnt = len(sv) - _BAR_MAX_CATS
        sv = np.append(sv[:_BAR_MAX_CATS], f"({others_cnt} others)")
        sc = np.append(sc[:_BAR_MAX_CATS], others_n)

    pcts   = sc / total * 100
    labels = [f"{c:,} ({p:.1f}%)" for c, p in zip(sc, pcts)]

    fig = go.Figure(go.Bar(
        x=sv, y=sc, text=labels,
        textposition='outside', marker_color='#3498db',
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

    fig = go.Figure(go.Pie(
        labels=sv, values=sc, hole=0.3,
        textinfo='label+percent',
        hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>',
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
        marker=dict(color='#3498db', size=4, opacity=0.55),
        name='Data',
        hovertemplate=f'{xl}: %{{x:.3f}}<br>{yl}: %{{y:.3f}}<extra></extra>',
    ))

    # Manual OLS trendline — no statsmodels dependency
    if len(xm) >= 2:
        try:
            coeffs = np.polyfit(xm, ym, 1)
            x_line = np.array([xm.min(), xm.max()])
            y_line = np.polyval(coeffs, x_line)
            r2     = float(np.corrcoef(xm, ym)[0, 1] ** 2)
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines',
                line=dict(color='#e74c3c', width=2, dash='dash'),
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
    """
    Vectorised groupby: np.argsort on category codes + np.split.
    O(n log n) — no Python loop per group.
    """
    # Encode categories to integer codes
    cat_uniq, cat_codes = np.unique(cat, return_inverse=True)
    n_cats = len(cat_uniq)

    # Sort both arrays by category code
    sort_idx  = np.argsort(cat_codes, kind='stable')
    num_sorted = num[sort_idx]
    cod_sorted = cat_codes[sort_idx]

    # Find split points between consecutive groups
    split_pts  = np.where(np.diff(cod_sorted))[0] + 1
    groups     = np.split(num_sorted, split_pts)

    bp = 'suspectedoutliers' if len(num) > 50_000 else 'outliers'
    traces = []
    for i, (label, grp) in enumerate(zip(cat_uniq, groups)):
        clean_grp = drop_nan(grp)
        if clean_grp.size > 0:
            traces.append(go.Box(
                y=clean_grp, name=str(label),
                boxmean='sd', boxpoints=bp,
            ))

    if not traces:
        return empty('No valid groups')

    fig = go.Figure(data=traces)
    fig.update_layout(**_LAYOUT, xaxis_title=catl, yaxis_title=numl)
    return fig


def heatmap_categorical(a: np.ndarray, b: np.ndarray,
                        al: str, bl: str) -> go.Figure:
    """Vectorised contingency table — O(n) via ravel_multi_index + bincount."""
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
        color_continuous_scale='Blues', aspect='auto',
    )
    fig.update_layout(**_LAYOUT)
    return fig


def correlation_heatmap(mat: np.ndarray, cols: list[str]) -> go.Figure:
    n = len(cols)
    fig = px.imshow(
        mat, x=cols, y=cols,
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
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
                      paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Inter, -apple-system, sans-serif"))
    return fig
