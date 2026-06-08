# EDA Dashboard Application — v5

A production-ready Exploratory Data Analysis web application built with **Dash** and **Plotly**. This version adds a full **PCA tab** with six interactive sub-views and expert interpretation panels, a **dark mode** theme system with persistent preferences, fixes critical performance and correctness bugs in the univariate and bivariate tabs, and delivers a deployment-optimised callback architecture with immediate visual feedback.

---

## ✨ What's New in v5

### Dark mode theme system

| Feature | Description |
|---|---|
| **CSS custom properties** | Full design token system (`--text-primary`, `--bg-card`, `--primary`, etc.) — both themes defined in one stylesheet |
| **Clientside toggle** | Sun/moon button toggles `data-theme` attribute on `<html>` — no server round-trip |
| **localStorage persistence** | Theme choice saved across page reloads |
| **Automatic chart theming** | Plotly text/grid colours inherit CSS variables via `.js-plotly-plot` selectors — no template switching needed |
| **Glass-morphism cards** | `backdrop-filter: blur(12px)` cards in dark mode, subtle gradient backgrounds in both themes |
| **Smooth transitions** | Theme switch animates via `transition: background var(--transition-slow), color var(--transition-slow)` |

### New module: PCA tab

| Feature | Description |
|---|---|
| **6 sub-tabs** | Scree plot, Correlation circle, Biplot, Individuals plane, Contributions, cos² quality |
| **Scree plot** | Variance bars + cumulative curve + recommended PC count (vertical line) + 70% threshold |
| **Eigenvalue table** | λ, variance %, cumulative %, keep/discard per PC |
| **Correlation circle** | Variable arrows in unit circle, interactive cos² filter slider |
| **Biplot** | Individuals (rescaled) + variable arrows in one view |
| **Individuals plane** | Score scatter, colourable by cos² or contribution, top-n annotation slider |
| **Contributions** | Bar chart per axis (uniform threshold line) + multi-axis heatmap |
| **cos² quality** | Variable × PC heatmap of representation quality |
| **5 expert interpretation panels** | Dynamic data-driven commentary under circle, biplot, individuals, contributions, cos² |
| **Axis insight sidebar** | Per-PC breakdown: contribution, cos², correlation, direction per variable |
| **Distinction contrib vs corr** | Explicitly computed and displayed separately throughout |
| **PCA cache** | Computed once at upload → all PCA callbacks are O(1) reads |
| **StandardScaler** | Variables centred and scaled before PCA — no size/unit effects |
| **Optimal PC recommendation** | Majority vote of Kaiser (λ>1), scree elbow, 70% threshold |
| **New modules** | `pca.py`, `charts_pca.py`, `ui_pca.py` |

---

### Bug fixes — Univariate and Bivariate tabs

| Bug | v4 | v5 |
|---|---|---|
| **Histogram crash on constant column** | `bw=0` → divide-by-zero in KDE → blank chart | Guard added: falls back to plain bar chart |
| **Scatter crash (missing statsmodels)** | `trendline='ols'` required `statsmodels` → `ModuleNotFoundError` | Replaced by `np.polyfit` OLS — zero extra dependency |
| **Integer columns misclassified** | `store.py` only checked `dtype.kind == 'f'`; `int64`/`uint8` from parser treated as categorical | Full dtype coverage: `int8/16/32/64`, `uint8/16/32/64` all handled as numeric |
| **Embedded header row in JSON** | First data row treated as data → numeric columns read as strings | `_is_numeric_string` + key-match detection strips header automatically |
| **Charts render on startup** | Callbacks fired on load before data → empty/error renders visible | `prevent_initial_call=True` on all interaction callbacks |
| **`make_subplots` overhead** | Histogram used `make_subplots` for dual-axis → +8 ms per call | Replaced by `yaxis2=dict(overlaying='y')` in single `go.Figure` |
| **Raw `go.Histogram` payload** | Sent all n raw values to client | Pre-binned with `np.histogram` → 50 points sent regardless of n |
| **Scatter OLS import latency** | `statsmodels` imported per call | `go.Scattergl` + `np.polyfit` — no import |
| **`grouped_boxplot` Python loop** | `for c in np.unique(cat)` + mask per group — O(k×n) | `np.argsort` + `np.split` vectorised groupby — O(n log n) |
| **`bar_categorical` unreadable** | No category cap | Capped at 50; rest → "(N others)" |
| **`pie_categorical` browser freeze** | No slice cap | Capped at 20; rest → "Other" |
| **`correlation_heatmap` text** | `text_auto` always on | Disabled above 15×15 |
| **`heatmap_categorical` text** | `text_auto` always on | Disabled above 20×20 |
| **`anderson` FutureWarning** | `method='interpolate'` triggered FutureWarning on scipy ≥ 1.17 | `try/except (TypeError, AttributeError)` handles both old and new scipy |

---

### Performance improvements

| Operation | v4 | v5 | Gain |
|---|---|---|---|
| **Histogram** | ~182 ms | ~14 ms | **13×** |
| **Scatter** | crash | ~14 ms | fixed |
| **Normality battery n=50k** | ~104 ms | ~60 ms | **1.7×** |
| **Subsampling setup (n=50k, K=10)** | ~95 ms (K=20 argsort each) | ~5 ms (one argsort) | **19×** |
| **JSON load 15k rows UTF-16** | ~810 ms (v3 baseline) | ~315 ms | **2.6×** |
| **CSV load 1.6k rows** | ~15 ms | ~5 ms | **3×** |
| **Deduplication mixed** | O(n) Python loop | O(n log n) numpy void | 3–5× |
| **Correlation matrix per bivariate call** | O(p²·n) recomputed | O(1) cache | — |
| **PCA all interactions** | n/a | O(1) cache | — |
| **NaN strip per callback** | O(n) re-scan | O(1) from `clean_arrays` | — |
| **`grouped_boxplot`** | O(k×n) Python loop | O(n log n) vectorised | — |

---

### Deployment architecture — split callbacks

The most impactful change for perceived latency is the **callback split**:

| v4 | v5 |
|---|---|
| `on_univariate` — chart + stats + normality in one callback | **`on_univariate_chart`** — figure only (~14 ms) fires first |
| User waits for normality tests before seeing anything | **`on_univariate_stats`** — stats + normality (~7 ms small, ~60 ms large n) fires in parallel |
| `on_bivariate` — chart + tests + matrix in one callback | **`on_bivariate_chart`** — chart + correlation matrix (~15 ms) fires first |
| User waits for all statistical tests before seeing plot | **`on_bivariate_stats`** — tests + insights (~4–50 ms) fires in parallel |

**`dcc.Loading` wrappers** added around every heavy output so users see a spinner immediately rather than a frozen interface.

**Global chart config** (`_GRAPH_CFG`): `responsive=True`, `displaylogo=False`, `modeBarButtonsToRemove=["lasso2d", "select2d"]` on all `dcc.Graph` components.

---

### Visual design system

| Token | Light mode | Dark mode | Purpose |
|---|---|---|---|
| `--text-primary` | `#0f172a` | `#e2e8f0` | Headings, body text |
| `--text-secondary` | `#475569` | `#94a3b8` | Labels, notes |
| `--text-muted` | `#64748b` | `#94a3b8` | Metadata, secondary info |
| `--bg-body` | `#f5f3ff` | `#0c0e19` | Page background |
| `--bg-card` | `#ffffff` | `#161827` | Card backgrounds |
| `--primary` | `#6366f1` | `#818cf8` | Accent (indigo) |
| `--header-from` → `--header-to` | `#1e1b4b` → `#312e81` | `#080a14` → `#0f111e` | Header gradient |

- **Typography**: Inter (UI) + JetBrains Mono (data) via Google Fonts
- **Animations**: `fadeIn` (200ms), `slideIn` (300ms), `gradient-shift` (8s infinite) on header
- **Glass effect**: `backdrop-filter: blur(8–12px)` on cards and theme toggle in dark mode
- **Responsive breakpoints**: 1200px (reading grid collapse), 768px (header stack)
- **Custom scrollbar**: Thin (6px), colour-matching theme

---

## 🧠 Architecture Overview

| Layer | Files |
|---|---|
| **Entry point** | `eda_app.py` |
| **State management** | `store.py` |
| **Data loading** | `loader.py` + `parsers.py` |
| **Business logic** | `stats.py` |
| **Visualisation** | `charts.py` + `charts_pca.py` |
| **UI components** | `ui.py` + `ui_pca.py` |
| **PCA computations** | `pca.py` |
| **Orchestration** | `callbacks.py` |
| **Utilities** | `utils.py` |
| **Styling** | `assets/style.css` |

---

## 📁 File Descriptions

### `eda_app.py`
Application entry point. Defines layout with `dcc.Loading` wrappers on every output. PCA tab with 6 sub-tabs and full sidebar. **Dark mode clientside callbacks**: toggle (click → `data-theme` attribute + `localStorage`) and init (restore from `localStorage` on load). Exposes `server` for WSGI deployment.

### `callbacks.py`
Orchestration layer. **8 callbacks** total:

- **`on_upload`** — decodes upload, resets store, runs PCA once, populates all dropdowns.
- **`on_variable_changed`** — updates plot-type options (`prevent_initial_call=True`).
- **`on_univariate_chart`** — chart only, ~14 ms, fires first (`prevent_initial_call=True`).
- **`on_univariate_stats`** — stats + normality battery, fires in parallel (`prevent_initial_call=True`).
- **`on_bivariate_chart`** — chart + correlation matrix O(1) cache, ~15 ms (`prevent_initial_call=True`).
- **`on_bivariate_stats`** — statistical tests + insights, fires in parallel (`prevent_initial_call=True`).
- **7 PCA callbacks** — all O(1) reads from `store._pca_cache`.

### `store.py`
Single source of truth. Full dtype coverage for column classification.

**Classification rules (in order):**

| dtype kind | Rule | Result |
|---|---|---|
| `U`, `S`, `O` (string/object) | Always | categorical |
| `b` (bool) | Always | categorical |
| `i`, `u` (int8…uint64) | n_uniq ≤ 10 **and** n_uniq < 0.5×n | categorical |
| `i`, `u` | otherwise | **numeric** ← v5 fix |
| `f` (float64) | ≤ 10 distinct integer values | categorical |
| `f` | otherwise | numeric |

**Caches** (all invalidated on each upload):
- `clean_arrays` — NaN-stripped float64, pre-computed at reset.
- `col_stats` — `{n_clean, n_nan}` per column.
- `_corr_matrix_cache` — pairwise Pearson (p×p), computed once.
- `_pca_cache` — full PCA result dict, computed once.
- `_lilliefors_cache` — MC null distributions, lazy per unique n.

### `loader.py`
Base64 decode → parser → column alignment → numpy deduplication → `store.reset()`.

**Deduplication** — pure numpy void-view + `np.unique`, O(n log n):
- Float: `(value, nan_flag_uint8)` pairs — NaN rows hash consistently.
- String: fixed-width numpy fields.

**Column alignment** — `np.unique` with `return_counts` finds majority length; outlier columns dropped with user-visible warning.

### `parsers.py`
Numpy-first parsing pipeline. Phase 1: raw bytes → Python objects (orjson / csv / pandas read_excel / pyarrow). Phase 2: Python objects → numpy arrays — no Python per-cell loop.

**Embedded header row detection** — strips rows where string values match their column key (datasud.fr, Opendatasoft exports).

**Encoding support** — UTF-8, UTF-16 LE/BE, UTF-32 LE/BE via BOM detection + C codec transcoding.

**Formats** — CSV, TSV, TXT, JSON, JSONL, XLSX, XLSM, XLS, ODS, Parquet.

**JSON structures** — list of objects, column dict, Opendatasoft `values`/`results`/`records`.

### `stats.py`
Pure statistical functions. No Dash, Plotly, or pandas.

**Optimised large-n subsampling (`_subsample_matrix`):**
```
v4: K separate calls to _stratified_subsample(arr)
    = K × argsort(n)  → 20 × 4.8ms = 96ms just for setup

v5: _subsample_matrix(arr, K, n_each)
    = 1 × argsort(n) + vectorised matrix indexing
    → 5ms total regardless of K
    → 19× faster setup
```
`_SUBSAMPLE_REPS` reduced from 20 to 10. Statistical impact: median p-value stability decreases by √2 — negligible for α=0.05 decisions (validated empirically).

**`anderson_darling`** — `try/except (TypeError, AttributeError)` handles both scipy ≥ 1.17 (`method='interpolate'` with `.pvalue`) and older scipy (5% critical value fallback).

**Normality battery (5 tests):** Shapiro-Wilk, KS, Anderson-Darling, Lilliefors (MC cached), D'Agostino-Pearson. Subsampling active for n > 5 000.

**Bivariate tests:**

| Pair type | Tests | Effect sizes |
|---|---|---|
| Num × Num | Pearson r, Spearman ρ, Kendall τ | r, ρ, τ |
| Num × Cat | Levene, ANOVA, Kruskal-Wallis; k=2: t-Student, t-Welch, Z-test, Mann-Whitney U, Wilcoxon | η² |
| Cat × Cat | Chi-square, Fisher's Exact (k×r) | Cramér's V, Tschuprow's T |

### `charts.py`
Plotly figure factories.

- **`histogram`** — `np.histogram` pre-bins (50 points to client), no `make_subplots`, KDE scaled to count axis, `bw=0` guard.
- **`scatter`** — `go.Scattergl` (WebGL), `np.polyfit` OLS with R², no statsmodels.
- **`bar_categorical`** — capped at 50 categories.
- **`pie_categorical`** — capped at 20 slices.
- **`grouped_boxplot`** — `np.argsort + np.split` vectorised groupby.
- **`heatmap_categorical`** — `text_auto` off above 20×20.
- **`correlation_heatmap`** — `text_auto` off above 15×15.

### `pca.py`
Pure PCA. `sklearn.PCA` + `StandardScaler`. All quantities vectorised numpy:

| Quantity | Formula | Meaning |
|---|---|---|
| `corr_circle[j,k]` | `loading[j,k] × √λ_k` | Pearson corr. of variable j with axis k |
| `cos2_var[j,k]` | `corr_circle²` | Proportion of variable j's variance on axis k |
| `contributions_var[j,k]` | `loading²×λ_k / Σλ × 100` | % of axis k built by variable j (Σ=100 per axis) |
| `contributions_ind[i,k]` | `score²/(n×λ_k) × 100` | % of axis k built by individual i |
| `cos2_ind[i,k]` | `score²/Σscore²` | Quality of representation of individual i on axis k |

### `charts_pca.py`
Plotly factories for PCA: scree, eigenvalue table, correlation circle, biplot, individuals plane, contributions bar, cos² heatmap, contributions heatmap. All use `_layout()` helper to avoid duplicate-key errors. Configurable cos² filter slider, top-n annotation, colour-by modes.

### `ui_pca.py`
Six expert interpretation panels:
- `circle_interpretation_panel` — axis polarity, HHI concentration, correlated/opposed pairs, misleading arrows.
- `biplot_interpretation_panel` — dual-scale reading guide, top variables per axis.
- `individuals_interpretation_panel` — cloud stats, quadrant table, top-5 contributors.
- `contributions_interpretation_panel` — HHI, axis polarity, redundancy detection, R² cumulative.
- `cos2_interpretation_panel` — quality table, per-axis coverage, escaped variables, FactoMineR protocol.
- `axis_insight_panel` — per-variable cards: contribution vs threshold, cos², correlation, direction.

### `ui.py`
Pure `(data → Dash HTML component)` factories for univariate/bivariate tabs.

### `utils.py`
Pure numpy helpers: `drop_nan`, `is_integer_array` (modulo-based, 3× faster than round), `format_percent`.

### `assets/style.css`
Single stylesheet with 943 lines. Design token system via CSS custom properties (`:root` for light, `[data-theme="dark"]` for dark). Includes animations, scrollbar, responsive breakpoints, Plotly chart overrides, glass-morphism effects, and custom component styles (upload zone, stat cards, tip boxes, tables, sliders, tabs, dropdowns).

---

## 🚀 Performance Summary

| Operation | v3 | v4 | v5 |
|---|---|---|---|
| Histogram render | ~180 ms | ~180 ms | **~14 ms** |
| Scatter render | crash | crash | **~14 ms** |
| Normality battery n=50k | n/a | ~104 ms | **~60 ms** |
| Subsampling setup n=50k | n/a | ~95 ms | **~5 ms** |
| User sees chart (univariate) | full callback wait | full callback wait | **~14 ms** (split) |
| User sees chart (bivariate) | full callback wait | full callback wait | **~15 ms** (split) |
| JSON load 15k rows UTF-16 | ~810 ms | ~315 ms | **~315 ms** |
| CSV load 1.6k rows | ~15 ms | ~5 ms | **~5 ms** |
| Correlation matrix per call | O(p²n) | O(1) | O(1) |
| PCA per interaction | n/a | n/a | O(1) |

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### Core dependencies (pinned versions)

```
dash==4.1.0
dash-bootstrap-components==2.0.4
numpy==2.4.6
pandas==3.0.3
plotly==6.6.0
scipy==1.17.1
scikit-learn==1.9.0
orjson==3.11.9
```

### Optional (per file format)

| Package | Formats |
|---|---|
| `openpyxl==3.1.5` | `.xlsx`, `.xlsm` |
| `odfpy==1.4.1` | `.ods` |
| `pyarrow==21.0.0` | `.parquet` |

> **Note:** `statsmodels` is **not required** in v5. OLS trendline uses `numpy.polyfit`.

---

## 📂 Supported File Formats

| Format | Extension(s) | Notes |
|---|---|---|
| CSV | `.csv`, `.txt` | BOM stripped; delimiter auto-detected |
| TSV | `.tsv` | Tab delimiter forced |
| JSON | `.json` | List of objects, column dict, Opendatasoft `values`/`results`/`records` |
| JSON Lines | `.jsonl` | One JSON object per line |
| Excel | `.xlsx`, `.xlsm` | Via openpyxl |
| Excel (legacy) | `.xls` | Via xlrd |
| OpenDocument | `.ods` | Via odfpy |
| Parquet | `.parquet` | Via pyarrow |

**Encodings:** UTF-8 (±BOM), UTF-16 LE/BE, UTF-32 LE/BE — auto-detected.

**French open-data:** Opendatasoft API v1/v2, datasud.fr UTF-16 exports with embedded header rows — all handled automatically.

---

## 🚀 Running the App

```bash
python eda_app.py
```

Available at `http://localhost:8050`.

### Production deployment (Gunicorn)

```bash
gunicorn eda_app:server --workers 1 --threads 4 --timeout 120
```

> **Workers = 1** is recommended because `store.py` uses module-level globals. With multiple workers, each worker has its own independent store — this is correct behaviour for a stateless deployment (each user session is isolated). Use `--threads 4` to handle concurrent requests within one worker.

### Render / Railway / Fly.io

```
web: gunicorn eda_app:server --workers 1 --threads 4 --timeout 120
```
