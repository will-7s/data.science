# EDA Dashboard

Interactive exploratory data analysis dashboard built with **Dash** and **Plotly**.
Fast, professional, dual‑theme (light/dark).

## Features

- **Univariate analysis** — histograms, box plots, bar charts, pie charts
- **Bivariate analysis** — scatter plots (with OLS), grouped box plots, categorical heatmaps
- **Pairwise correlation matrix** — diverging heatmap with `RdYlBu_r` scale (visible in both themes)
- **Descriptive statistics** — mean, median, std, skewness, kurtosis, normality tests
- **Statistical tests** — t‑test, Mann‑Whitney, ANOVA, Kruskal‑Wallis, chi‑square
- **Dual theme** — light/dark toggle persisted in `localStorage`

## Performance

| Optimisation | Detail |
|---|---|
| **PCA removed** | ~1 500 lines deleted (slow to render, blank graphs) |
| **No `dcc.Loading`** | Eliminates spinner flash + latency overhead |
| **No Google Fonts** | System fonts only — no render‑blocking `@import` |
| **No CSS animations** | `gradient-shift`, `backdrop-filter`, keyframes removed |
| **Vectorised NumPy** | Contingency tables via `ravel_multi_index`+`bincount`; groups via `argsort`+`split` |
| **KDE removed** | FFT‑based kernel density was the heaviest per‑chart cost |
| **Figure cache** | `store._plot_cache` — instant replay on variable switch |
| **`responsive: False`** | No Plotly resize listener |
| **Triple parallel callbacks** | Chart + descriptive stats + normality all fire in parallel |
| **Subsampled normality** | `sp.shapiro` capped at 5 000 points — prevents multi‑second block in `_test_num_num` |
| **Dead code removed** | `_stratified_subsample`, `correlation_matrix`, `normality_panel` deleted |
| **No theme callback overhead** | Theme toggle is clientside only — zero server round‑trips |
| **`prevent_initial_call`** | On all interaction callbacks |

## Colour palette

```
_PRIMARY     #6366f1   indigo
_SECONDARY   #0ea5e9   sky
_ACCENT      #10b981   emerald
_DANGER      #ef4444   red
_WARNING     #f59e0b   amber
_PURPLE      #8b5cf6   violet
_PINK        #ec4899   pink
_CYAN        #06b6d4   cyan
_LIME        #84cc16   lime
_ORANGE      #f97316   orange
```

## Project structure

```
├── eda_app.py          Dash app layout + theme toggle (clientside only)
├── callbacks.py        Callback registration (upload, charts, stats)
├── charts.py           Plotly figure factories (7 chart types)
├── stats.py            Statistical computations (vectorised, subsampling)
├── store.py            Dataset, metadata, correlation cache, figure cache
├── ui.py               Dash HTML component builders
├── loader.py           File parsing (CSV, Excel, Parquet, Feather)
├── parsers.py          Low‑level file readers
├── utils.py            NaN / inf helpers
└── assets/
    └── style.css       Full theme system (CSS custom properties, Plotly overrides)
```

## Quick start

```bash
pip install dash plotly numpy pandas scipy openpyxl pyarrow
python eda_app.py
```

Open `http://127.0.0.1:8050` in your browser.

## Architecture

### Callback flow (univariate)

```
variable change
      │
      ├──► on_univariate_chart      → figure          (ms)
      ├──► on_univariate_stats_fast → stats panel     (O(n))
      └──► on_univariate_normality  → normality tests (O(n log n) for n > 5 000)
```

All three fire simultaneously. The chart and descriptive stats appear in the same
frame; normality test results populate a fraction of a second later.

### Subsampling

For `n > 5 000`:
- **Normality battery** (`run_normality_battery`): one `argsort` → K stratified
  subsamples → 5 tests each → median aggregation. Avoids over‑rejection
  from trivial deviations.
- **`normality_test`** (used in bivariate correlation): random subsample at 5 000
  before `sp.shapiro`. Prevents the 1+ second block on large arrays.

### Theme system

CSS custom properties in `assets/style.css`. Light mode = `:root`, dark mode =
`[data-theme="dark"]`. Toggle is a **clientside callback** that never touches
the server; chart backgrounds are transparent so CSS handles everything.

## Dependencies

```
dash        ≥ 2.14
plotly      ≥ 5.18
numpy       ≥ 1.24
pandas      ≥ 2.0
scipy       ≥ 1.10
openpyxl              (Excel support)
pyarrow               (Parquet / Feather support)
```
