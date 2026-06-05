# EDA Dashboard Application — v5

A production-ready Exploratory Data Analysis web application built with **Dash** and **Plotly**. This version adds a full **PCA tab** with six interactive sub-views, fixes critical bugs in the univariate and bivariate tabs, achieves a **13× histogram speed-up**, eliminates all external dependencies that caused crashes (`statsmodels`), and delivers a numpy-first loading pipeline with automatic handling of UTF-16 files and embedded header rows.

EDA app Link : https://huggingface.co/spaces/will-7s/eda_app 

---

## ✨ What's New in v5

### New: PCA Tab

| Feature | Description |
|---|---|
| **Scree plot** | Individual variance % (bars) + cumulative % (line) + optimal PC recommendation (vertical line) + 70% threshold (horizontal line) |
| **Eigenvalue table** | Full table: eigenvalue, variance %, cumulative %, keep/discard recommendation |
| **Correlation circle** | Variable arrows in unit circle, interactive cos² filter slider, colour-coded by variable |
| **Biplot** | Individuals (scaled) + variable arrows overlaid in one view |
| **Individuals plane** | Score scatter coloured by cos² or contribution; top-n annotation slider |
| **Contributions** | Horizontal bar chart per axis (uniform threshold line) + multi-axis heatmap |
| **cos² quality** | Variable × PC heatmap of representation quality |
| **Expert interpretation panels** | Dynamic data-driven commentary under every sub-tab (circle, biplot, individuals, contributions, cos²) |
| **Axis insight sidebar** | Per-PC breakdown: contribution, cos², correlation, direction for every variable |
| **Optimal PC recommendation** | Majority vote of Kaiser (λ>1), scree elbow (max 2nd derivative), 70% variance threshold |
| **Distinction contrib vs corr** | Explicitly documented and displayed separately throughout |
| **PCA cache** | Computed once at upload (`store._pca_cache`), all PCA callbacks are O(1) cache reads |
| **StandardScaler** | All numeric variables centred and scaled before PCA — avoids size/unit effects |

### Bug fixes — Univariate and Bivariate tabs

| Bug | v4 | v5 |
|---|---|---|
| **Histogram crash on constant column** | `bw=0` → divide-by-zero in KDE → blank chart | `bw=0` guard → falls back to plain bar chart |
| **Scatter crash** | `trendline='ols'` required `statsmodels` (not installed) → `ModuleNotFoundError` | Replaced by `np.polyfit` manual OLS — no extra dependency |
| **Integer columns classified as categorical** | `store.py` only checked `dtype.kind == 'f'`; `int64`/`uint8` from `parsers.py` v5 were silently treated as categorical | Full dtype coverage: `int8/16/32/64`, `uint8/16/32/64` all handled as numeric |
| **Embedded header row in JSON** | First data row treated as data → all numeric columns read as strings | `_is_numeric_string` + key-match detection strips the header row automatically |
| **Charts render on startup** | Univariate/bivariate callbacks fired immediately on load (before data available) → empty/error renders | `prevent_initial_call=True` on all three callbacks |
| **`make_subplots` overhead** | Used for histogram dual-axis → +8 ms per render | Replaced by `yaxis2=dict(overlaying='y')` in single `go.Figure` |
| **Raw `go.Histogram` payload** | Sent all n raw data points to client | Pre-binned with `np.histogram` → sends 50 points regardless of n |
| **`px.scatter` OLS import latency** | `statsmodels` imported on every scatter call (even if available) | Single `go.Scattergl` + manual polyfit — faster and dependency-free |
| **`grouped_boxplot` Python loop** | `for c in np.unique(cat)` + boolean mask per group — O(k×n) | `np.argsort` + `np.split` vectorised groupby — O(n log n) |
| **`bar_categorical` unreadable** | No cap on categories — 200+ categories produced unreadable chart | Capped at 50; remainder aggregated into "(N others)" bar |
| **`pie_categorical` browser freeze** | No cap on slices | Capped at 20; remainder → "Other" slice |
| **`correlation_heatmap` text overflow** | `text_auto='.2f'` on large matrices made labels unreadable | Disabled above 15×15 |
| **`heatmap_categorical` text overflow** | `text_auto='.1f'` unconditional | Disabled above 20×20 (k×r > 400) |

### Performance improvements

| Operation | v4 | v5 | Gain |
|---|---|---|---|
| **Histogram** | ~182 ms | ~14 ms | **13×** |
| **Scatter** | crashed | ~13 ms | fixed |
| **Boxplot** | ~12 ms | ~11 ms | 1.1× |
| **bar_categorical** | ~13 ms | ~13 ms | — |
| **JSON load (15k rows, UTF-16)** | ~810 ms (v3 baseline) | ~315 ms | **2.6×** |
| **CSV load (1.6k rows)** | ~15 ms | ~5 ms | **3×** |
| **Deduplication mixed** | O(n) Python loop | O(n log n) numpy void | 3–5× |
| **Correlation matrix per bivariate call** | O(p²·n) recomputed | O(1) cache | — |
| **PCA all sub-tab interactions** | n/a | O(1) cache reads | — |

---

## 🧠 Architecture Overview

| Layer | Files |
|---|---|
| **Entry point** | `eda_app.py` |
| **State management** | `store.py` |
| **Data loading** | `loader.py` + `parsers.py` |
| **Business logic** | `stats.py` |
| **Visualization** | `charts.py` + `charts_pca.py` |
| **UI components** | `ui.py` + `ui_pca.py` |
| **PCA computations** | `pca.py` |
| **Orchestration** | `callbacks.py` |
| **Utilities** | `utils.py` |

---

## 📁 File Descriptions

### `eda_app.py`
Application entry point. Defines layout: upload card, Univariate tab, Bivariate tab, PCA tab (with 6 sub-tabs and a full sidebar of controls). Exposes `server` for WSGI deployment.

### `callbacks.py`
Orchestration layer. All callbacks:

- **`on_upload`** — decodes upload, populates all dropdowns (univariate, bivariate, all PCA axes), runs PCA and caches the result.
- **`on_variable_changed`** — updates plot-type radio options (`prevent_initial_call=True`).
- **`on_univariate`** — histogram/boxplot/bar/pie + stats + normality battery (`prevent_initial_call=True`). Reads from `store.clean_arrays`.
- **`on_bivariate`** — scatter/grouped box/contingency heatmap + correlation matrix O(1) cache (`prevent_initial_call=True`).
- **7 PCA callbacks** — all O(1) cache reads from `store._pca_cache`:
  - `on_pca_summary`, `on_pca_scree`, `on_pca_circle`, `on_pca_biplot`, `on_pca_individuals`, `on_pca_axis_insight`, `on_pca_contributions`, `on_pca_cos2`.

### `store.py`
Single source of truth. All caches invalidated on upload.

**Column classification (v5 — full dtype coverage):**

| dtype kind | Rule | Result |
|---|---|---|
| `U`, `S`, `O` (string/object) | Always | categorical |
| `b` (bool) | Always | categorical |
| `i`, `u` (int8…int64, uint8…uint64) | n_uniq ≤ 10 AND n_uniq < 0.5×n | categorical |
| `i`, `u` | otherwise | **numeric** ← v5 fix |
| `f` (float64) | ≤ 10 distinct integer values | categorical |
| `f` | otherwise | numeric |

Integer-coded categorical columns (e.g. `quality = 5`) are converted to string for display.

**Caches:**
- `clean_arrays` — NaN-stripped float64, pre-computed at reset, O(1) per callback.
- `col_stats` — `{n_clean, n_nan}` per column.
- `_corr_matrix_cache` — pairwise Pearson (p × p), computed once.
- `_pca_cache` — full PCA result dict, computed once.
- `_lilliefors_cache` — MC null distributions, lazy per unique n.

### `loader.py`
Decodes base64 uploads → parser → column alignment → numpy deduplication → `store.reset()`.

**Deduplication** — pure numpy void-view + `np.unique`, O(n log n):
- Float columns: `(value, nan_flag_uint8)` pairs so NaN rows compare equal.
- String columns: fixed-width numpy string fields.
- Original row order preserved by `np.sort(idx)` after `np.unique`.

**Column alignment** — `np.unique` with `return_counts` finds the majority length; outlier columns dropped with a user-visible warning.

### `parsers.py`
Numpy-first parsing pipeline.

**Phase 1 — Raw bytes → Python objects:**
- JSON/JSONL: `orjson` (Rust C-extension, 3–5× faster than stdlib `json`).
- Encoding: `_to_utf8()` detects UTF-32/UTF-16/UTF-8-BOM via BOM bytes and transcodes in one C codec call.
- CSV: stdlib `csv` C reader.
- Excel/ODS: `pd.read_excel` (unavoidable — only reliable cross-format Excel parser).
- Parquet: `pyarrow` → numpy directly, no pandas.

**Phase 2 — Python objects → numpy arrays (`_infer_numpy_col`):**
- None mask via `np.equal(arr, None)` — vectorised, no Python loop.
- Numeric fast path: checks `isinstance(first_non_null, (int, float))` → single `astype(float64)` call.
- String path: tries `astype(float64)` on string array; falls back to str with `None → ''`.
- No Python per-cell loop anywhere.

**Embedded header row detection (v5 fix):**
- Detects rows where string values match their column key (e.g. `{"ANNEE": "ANNEE", "CONSO": "CONSOA"}`).
- `_is_numeric_string()` helper distinguishes label strings from numeric strings.
- Handles formats from datasud.fr, Opendatasoft, and other French open-data portals.

**Supported formats:** CSV, TSV, TXT, JSON, JSONL, XLSX, XLSM, XLS, ODS, Parquet.

**Recognised JSON structures:** list of objects, column dict, Opendatasoft `values`/`results`/`records` wrappers.

### `stats.py`
Pure statistical functions. No Dash, Plotly, or pandas.

**Normality battery (5 tests)** with large-sample stratified subsampling (n > 5 000):
1. Draw 20 stratified subsamples of 2 000 observations (quantile bins, vectorised).
2. Run all 5 tests on each subsample.
3. Aggregate via `np.nanmedian` over (20 × 5) matrix.
4. `is_normal` by majority vote. Skewness/kurtosis always on full array.

| Test | Min n | Notes |
|---|---|---|
| Shapiro-Wilk | 3 | Most powerful for n ≤ 5 000 |
| Kolmogorov-Smirnov | 3 | Conservative |
| Anderson-Darling | 7 | Emphasises tails; `method='interpolate'` when available |
| Lilliefors | 4 | MC cache from `store`; corrects KS for estimated parameters |
| D'Agostino-Pearson | 8 | Omnibus K² = Z²(skew) + Z²(kurt) ~ χ²(2) |

**Bivariate tests:**

| Pair type | Tests | Effect sizes |
|---|---|---|
| Num × Num | Pearson r, Spearman ρ, Kendall τ | r, ρ, τ |
| Num × Cat | Levene, ANOVA, Kruskal-Wallis; k=2: Student t, Welch t, Z-test, Mann-Whitney U, Wilcoxon | η² |
| Cat × Cat | Chi-square, Fisher's Exact (k×r) | Cramér's V, Tschuprow's T |

### `charts.py`
Plotly figure factories. All v5 fixes applied.

**`histogram`:** `np.histogram` pre-bins data (50 points sent to client, not n). Single `go.Figure` with `yaxis2` overlay (no `make_subplots`). KDE scaled to count axis. `bw=0` guard for constant columns.

**`scatter`:** `go.Scattergl` (WebGL, faster for > 1 000 points). Manual `np.polyfit` OLS trendline with R² label — no `statsmodels`.

**`bar_categorical`:** capped at 50 categories; remainder → "(N others)". `xaxis_tickangle=-30` for many categories.

**`pie_categorical`:** capped at 20 slices; remainder → "Other".

**`grouped_boxplot`:** vectorised groupby — `np.argsort(cat_codes)` + `np.split` — O(n log n), no Python loop per group.

**`heatmap_categorical`:** `text_auto` disabled above 20×20.

**`correlation_heatmap`:** `text_auto` disabled above 15×15.

### `charts_pca.py`
Plotly factories for the PCA tab: `scree_plot`, `eigenvalue_table`, `correlation_circle`, `biplot`, `individuals_plane`, `contributions_bar`, `cos2_heatmap`, `contributions_heatmap`, `empty_pca`. All use `_layout()` helper to avoid duplicate key errors in `update_layout`.

### `pca.py`
Pure PCA computations. `sklearn.PCA` + `StandardScaler` for numerical stability. No Dash, no Plotly.

**Computed quantities (all vectorised numpy):**

| Quantity | Formula | Interpretation |
|---|---|---|
| `corr_circle[j,k]` | `loading[j,k] × √λ_k` | Pearson correlation of variable j with axis k |
| `cos2_var[j,k]` | `corr_circle[j,k]²` | Proportion of variable j's variance explained by axis k |
| `contributions_var[j,k]` | `loading[j,k]² × λ_k / Σλ × 100` | % of axis k's variance built by variable j (Σ=100 per axis) |
| `contributions_ind[i,k]` | `score[i,k]² / (n × λ_k) × 100` | % of axis k's variance built by individual i |
| `cos2_ind[i,k]` | `score[i,k]² / Σ_k score[i,k]²` | Quality of representation of individual i on axis k |

**Optimal PC count:** majority vote of Kaiser (λ>1), scree elbow (max 2nd difference of explained variance), 70% cumulative variance threshold. Result: `max(2, min(vote, n_components))`.

### `ui_pca.py`
Pure `(data → Dash HTML component)` factories. Six expert interpretation panels:

- **`circle_interpretation_panel`** — axis polarity table, HHI concentration index, correlated/opposed pair detection, misleading-arrow warnings.
- **`biplot_interpretation_panel`** — dual-scale reading guide, top variables per axis, professional cautions.
- **`individuals_interpretation_panel`** — cloud statistics, quadrant table, top-5 contributors, cos² quality summary.
- **`contributions_interpretation_panel`** — HHI concentration, axis polarity, redundancy detection, R² cumulative per variable, FactoMineR integration rules.
- **`cos2_interpretation_panel`** — per-variable quality table, per-axis coverage, escaped variables, misleading arrows, FactoMineR heatmap reading protocol.
- **`axis_insight_panel`** — per-variable cards: contribution vs threshold, cos², correlation, direction.

### `ui.py`
Pure `(data → Dash HTML component)` factories for univariate and bivariate tabs.

### `utils.py`
Pure numpy helpers. No Dash, Plotly, or scipy.

---

## 🚀 Performance Summary

| Operation | v3 | v4 | v5 |
|---|---|---|---|
| Histogram render | ~180 ms | ~180 ms | **~14 ms** (13×) |
| Scatter render | n/a (crash) | n/a (crash) | **~13 ms** |
| JSON load — 15k rows UTF-16 | ~810 ms | ~330 ms | **~315 ms** |
| CSV load — 1.6k rows | ~15 ms | ~6 ms | **~5 ms** |
| Deduplication mixed | O(n) Python | O(n log n) numpy | — |
| Correlation matrix per interaction | O(p²n) | O(1) | O(1) |
| PCA all interactions | n/a | n/a | O(1) |
| NaN strip per callback | O(n) | O(1) | O(1) |
| Lilliefors first call (per n) | n/a | ~20 ms | ~20 ms |
| Lilliefors subsequent calls | n/a | <1 ms | <1 ms |

---

## 📊 Statistical Tests Reference

### Normality Battery (Univariate tab sidebar)
Colour-coded banner (green/amber/red). Subsampling banner when n > 5 000. Skewness + excess kurtosis always on full array.

### Bivariate Test Reports (Bivariate tab sidebar)
**Num × Num:** Pearson/Spearman/Kendall with p-values. Per-variable normality check. Recommendation adapts to normality and n.

**Num × Cat:** Group normality (Shapiro-Wilk, subsampled for large groups). Levene, ANOVA, Kruskal-Wallis for any k. For k=2: Student t, Welch t, Z-test (n>60), Mann-Whitney U, Wilcoxon (equal sizes). Recommendation adapts to normality and homogeneity.

**Cat × Cat:** Chi-square with expected-cell diagnostics. Fisher's Exact (generalised k×r). Cramér's V + Tschuprow's T with strength labels.

### Cramér's V Strength Thresholds

| min(k,r) − 1 | Small | Medium | Large |
|---|---|---|---|
| 1 | 0.10 | 0.30 | 0.50 |
| 2 | 0.07 | 0.21 | 0.35 |
| 3 | 0.06 | 0.17 | 0.29 |

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### Core dependencies

```
dash>=2.0
dash-bootstrap-components>=1.0
numpy>=1.24
pandas>=2.0
plotly>=5.0
scipy>=1.10
scikit-learn>=1.3
orjson>=3.9
```

### Optional (per file format)

| Package | Formats |
|---|---|
| `openpyxl` | `.xlsx`, `.xlsm` (bundled with pandas) |
| `xlrd` | `.xls` |
| `odfpy` | `.ods` |
| `pyarrow` | `.parquet` |

> **Note:** `statsmodels` is **not required** in v5. The scatter OLS trendline uses `numpy.polyfit`.

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

**Encoding support:** UTF-8 (with/without BOM), UTF-16 LE/BE, UTF-32 LE/BE. Auto-detected from BOM bytes.

**French open-data formats:** Opendatasoft API v1/v2 (`values`, `results`, `records` wrappers), datasud.fr UTF-16 exports with embedded header rows — all handled automatically.

---

## 🚀 Running the App

```bash
python eda_app.py
```

Available at `http://localhost:8050`. For production WSGI deployment:

```bash
gunicorn eda_app:server
```
