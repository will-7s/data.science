# EDA Dashboard Application — v4

A production-ready Exploratory Data Analysis web application built with **Dash** and **Plotly**. This version builds on v3's advanced statistical foundation and introduces **large-sample subsampling for normality tests**, **FFT-based KDE**, **vectorised contingency heatmaps**, **a full correlation matrix cache**, and several correctness and performance fixes throughout the codebase.

---

## ✨ What's New in v4

| Feature | v3 (Previous) | v4 (This Version) |
|---|---|---|
| **Large-sample normality** | Not available | **Stratified subsampling** for n > 5 000 — avoids over-rejection; p-value = median across 20 × 2 000-obs. draws |
| **Group normality (bivariate)** | Not available | Shapiro-Wilk per group with the same subsampling logic for large groups |
| **KDE (histogram)** | Dense broadcast O(300n) | **FFT-based KDE** O(n log n); linear binning onto 512-point grid for n > 20 000 |
| **Contingency heatmap** | Double Python loop O(k × r × n) | **`np.ravel_multi_index` + `np.bincount`** — single vectorised pass O(n) |
| **Scatter plot** | All points sent to client | **Subsampled to 10 000 points** above threshold, seeded for reproducibility |
| **Boxplot payload** | All outliers sent | Switches to `suspectedoutliers` above 50 000 rows |
| **Correlation matrix** | Recomputed on every bivariate interaction | **Cached once at upload** — O(1) read on every subsequent callback |
| **Correlation insights** | Not available | Top-5 pairwise Pearson correlations displayed in the sidebar |
| **Deduplication** | Row-by-row Python tuple loop | **NumPy structured arrays** (value + NaN-flag pairs) — 3–5× faster |
| **`clean_arrays` cache** | Not available | NaN-stripped arrays pre-computed once at load time; no re-scan per callback |
| **`col_stats` cache** | Not available | `{n_clean, n_nan}` per column, available from the first callback |
| **Integer detection** | `round`-based | Modulo-based — 3× faster |
| **Lilliefors test** | Not available | MC null distribution, cached per unique n — ~20 ms first call, < 1 ms after |
| **Anderson-Darling** | Not available | `method='interpolate'` used when available; fallback to 5 % critical value |
| **Two-group tests** | t-test only | Added **Z-test** (n > 60), **Mann-Whitney U**, **Wilcoxon** (equal-size groups) |
| **Chi-square effect sizes** | Not available | **Cramér's V** + **Tschuprow's T** with strength labelling |
| **Fisher's Exact** | 2×2 only | **Generalised k×r** — triggered when Chi-square assumptions are weak or table is small |
| **Outlier detection** | Not available | IQR × 1.5 rule with percentage display |
| **Mode handling** | Simple mode | Frequency + tie detection (multimodal labelling) |
| **Column classification** | Simple dtype check | Float64 with ≤ 10 distinct integer values → categorical (binary flags, Likert scales, etc.) |
| **Callbacks architecture** | Correlation matrix in a separate callback | **Merged into `on_bivariate`** — fixes blank correlation panel on variable change |

---

## 🧠 Architecture Overview

| Layer | Files |
|---|---|
| **Entry point** | `eda_app.py` |
| **State management** | `store.py` |
| **Data loading** | `loader.py` + `parsers.py` |
| **Business logic** | `stats.py` |
| **Visualization** | `charts.py` |
| **UI components** | `ui.py` |
| **Orchestration** | `callbacks.py` |
| **Utilities** | `utils.py` |

---

## 📁 File Descriptions

### `eda_app.py`
Application entry point. Creates the Dash app, defines the layout (upload card, univariate tab, bivariate tab), and exposes `server` for WSGI deployment.

### `callbacks.py`
Orchestration layer. Four main callbacks:

- **`on_upload`** — decodes the upload, populates all dropdowns, resets the store.
- **`on_variable_changed`** — updates plot-type radio options based on column type (numeric → histogram/box; categorical → bar/pie).
- **`on_univariate`** — generates the chart, descriptive stats panel, and normality battery panel. Reads from `store.clean_arrays` — no NaN re-scan per call.
- **`on_bivariate`** — handles all three pair types (Num×Num, Num×Cat, Cat×Cat), renders the appropriate chart, runs bivariate tests, and **also populates the pairwise correlation matrix and top-5 insights on every call** via an O(1) cache read from `store.get_corr_matrix()`.

**Architecture fix:** in v3 the correlation matrix was produced by a separate callback triggered only at upload time. This left the `pairwise-correlation` and `correlation-insights` panels blank whenever the user changed a dropdown. Both outputs are now owned by `on_bivariate`, so they are always visible regardless of which variable pair is selected.

### `store.py`
Single source of truth. All state is reset on every upload; all caches are invalidated.

- **`dataset`** — raw column arrays (`dict[str, np.ndarray]`)
- **`clean_arrays`** — NaN-stripped arrays, computed once inside `reset()`
- **`col_stats`** — `{'n_clean', 'n_nan'}` per column
- **`col_meta`** — `'numeric'` or `'categorical'` per column
- **`num_cols` / `cat_cols` / `all_cols`** — ordered lists

**Column classification heuristic:** `float64` with ≤ 10 distinct integer values is treated as categorical (binary flags, Likert scales, ratings). All other `float64` columns are numeric; string/object columns are always categorical.

**Lilliefors MC cache** (`_lilliefors_cache`): built lazily on first access per unique `n_clean`. Uses 200 Monte Carlo reps with `scipy.special.ndtr` (3× faster than `sp.norm.cdf`). First call ~20 ms; subsequent calls < 1 ms.

**Correlation matrix cache** (`_corr_matrix_cache`): computed once after upload via `np.corrcoef` on a `(p × n_valid)` matrix aligned by a union NaN mask. Served from cache on every `on_bivariate` call; set back to `None` on new upload.

### `loader.py`
Decodes base64 Dash uploads, routes to the appropriate parser, removes duplicate rows, then commits to `store.reset()`.

**Deduplication strategy (`_dedup_rows`):**
- Float columns stored as `(value, nan_flag)` pairs so NaN rows hash consistently (IEEE 754 NaN ≠ NaN, but two `uint8` flags compare equal).
- String columns stored as fixed-length NumPy string fields.
- A `np.void` view + `np.unique` replaces the previous row-by-row Python tuple loop — **3–4× faster on mixed datasets, 5× on all-numeric datasets**. Original row order is preserved by sorting the returned indices.

### `parsers.py`
One function per format. Features: BOM stripping, automatic delimiter detection, missing-value normalisation (`''`, `na`, `n/a`, `null`, `none`, `nan`), date/datetime values passed through as strings.

Supported formats: CSV, TSV, TXT, JSON, JSONL, XLSX, XLSM, XLS, ODS, Parquet.

### `stats.py`
Pure statistical functions. No Dash or Plotly dependency. All functions accept NumPy arrays and return plain Python values or dicts.

**Large-sample subsampling** (threshold: n > 5 000):
Standard normality tests over-reject at large n because any trivial deviation from normality becomes statistically significant. The approach used in v4:
1. Draw `_SUBSAMPLE_REPS = 20` stratified subsamples of `_SUBSAMPLE_N = 2 000` observations each, using `_SUBSAMPLE_BINS = 20` quantile strata (vectorised — no Python loop on rows).
2. Run the full 5-test battery on each subsample.
3. Aggregate via column-wise `np.nanmedian` over a `(K × 5)` matrix — one NumPy call for all tests simultaneously.
4. `is_normal` determined by majority vote; ties → `None`.
5. Skewness and kurtosis are **always computed on the full array** (convergent estimators — precision improves with n).

The same subsampling logic is applied inside **`group_normality_report`** for individual groups in the Num×Cat bivariate path.

**Normality battery (5 tests):**

| Test | Minimum n | Notes |
|---|---|---|
| Shapiro-Wilk | 3 | Most powerful for n ≤ 5 000 |
| Kolmogorov-Smirnov | 3 | Conservative; sidebar note recommends Lilliefors when parameters are estimated |
| Anderson-Darling | 7 | Emphasises tail deviations; uses `method='interpolate'` when scipy supports it, otherwise falls back to 5 % critical value |
| Lilliefors | 4 | Corrects KS for estimated mean/std; MC null distribution from `store` cache |
| D'Agostino-Pearson | 8 | Omnibus K² = Z²(skew) + Z²(kurt) ~ χ²(2) under H₀ |

**Bivariate tests:**

| Pair type | Tests | Effect sizes |
|---|---|---|
| Num × Num | Pearson r, Spearman ρ, Kendall τ | r, ρ, τ |
| Num × Cat | Levene, one-way ANOVA, Kruskal-Wallis; for k = 2 groups: Student t, Welch t, Z-test (n₁+n₂ > 60), Mann-Whitney U, Wilcoxon (equal-size groups) | η² |
| Cat × Cat | Chi-square of independence, Fisher's Exact (k×r generalised) | Cramér's V, Tschuprow's T |

**Cramér's V strength thresholds (bias-corrected interpretation):**

| min(k,r) − 1 | Small | Medium | Large |
|---|---|---|---|
| 1 | 0.10 | 0.30 | 0.50 |
| 2 | 0.07 | 0.21 | 0.35 |
| 3 | 0.06 | 0.17 | 0.29 |

### `charts.py`
Plotly figure factories.

**FFT-based KDE (`histogram`):**
Replaces the previous dense O(300 × n) broadcast.
1. Silverman's bandwidth — O(n) single pass using `min(σ, IQR/1.34)`.
2. For n > 20 000: linear binning onto a 512-point grid (Wand & Jones 1995) — O(n).
3. Gaussian kernel built in frequency space; FFT convolution — O(g log g), g = 512.
4. Overall: O(n log n) vs O(300n) previously.

**Scatter subsampling:** above 10 000 points, `np.random.default_rng(0)` produces a stable subsample and a note is appended to the axis label.

**Contingency heatmap:** `np.ravel_multi_index` + `np.bincount` build the full contingency table in a single vectorised pass. Complexity drops from O(k × r × n) to O(n). Values are row-normalised to percentages before display.

**Boxplot:** switches to `boxpoints='suspectedoutliers'` above 50 000 rows to cap the client-side JSON payload.

### `ui.py`
Pure `(data → Dash HTML component)` functions. No dependency on `store` or `charts`.

Key panels:
- **`normality_battery_panel`** — colour-coded summary banner, optional subsampling info banner (shown only when active), full-array shape row (skewness + excess kurtosis), one card per test with statistic, p-value, and notes.
- **`test_panel`** — renders the flat-string result list produced by `stats.py`; handles `GNORM_BANNER|…` and `GNORM_CARD|…` tokens for rich inline group-normality cards.
- **`correlation_insights_panel`** — top-5 pairwise Pearson correlations sorted by absolute value.
- **`descriptive_stats_panel`** — N, mean, median, mode (with tie/multimodal count), std dev, min, max, IQR outlier %.
- **`categorical_stats_panel`** — unique count, most frequent category, frequency table (capped at 20 categories).

### `utils.py`
Pure NumPy helpers. No Dash, Plotly, or scipy dependency.

- **`drop_nan(arr)`** — NaN removal, safe for any dtype.
- **`is_integer_array(arr, tol=1e-9)`** — modulo-based (`abs(x % 1.0) < tol`), 3× faster than `round`.
- **`format_percent(value, total)`** — returns `"(xx.x%)"` string.

---

## 🚀 Performance Summary

| Operation | v3 | v4 |
|---|---|---|
| Deduplication — mixed types | baseline | 3–4× faster |
| Deduplication — numeric only | baseline | 5× faster |
| Integer detection | baseline | 3× faster |
| KDE computation | O(300n) | O(n log n) |
| Contingency heatmap | O(k × r × n) | O(n) |
| Scatter above 10 000 pts | all points rendered | capped at 10 000 |
| Boxplot above 50 000 rows | all outliers sent | payload capped |
| Correlation matrix per bivariate interaction | O(p² · n) recomputed | O(1) cache hit |
| NaN strip per callback | O(n) re-scan | O(1) from `clean_arrays` |
| Lilliefors (first call per unique n) | n/a | ~20 ms |
| Lilliefors (subsequent calls) | n/a | < 1 ms |

---

## 📊 Statistical Tests Reference

### Normality Battery (5 tests)
Shown in the left sidebar of the **Univariate** tab. A colour-coded banner summarises the overall verdict (green = all normal, amber = mixed, red = all non-normal). When n > 5 000, a subsampling info banner details the strategy used.

### Bivariate Test Reports
Shown in the left sidebar of the **Bivariate** tab.

**Num × Num — Correlation Analysis:** all three coefficients (Pearson, Spearman, Kendall) with p-values and significance flags. Per-variable normality check via Shapiro-Wilk. Recommendation generated from normality and sample size.

**Num × Cat — Group Comparison:** group normality via Shapiro-Wilk (with subsampling for large groups), displayed as colour-coded cards. Levene test for variance homogeneity. One-way ANOVA and Kruskal-Wallis for any number of groups. For k = 2 groups: additionally reports Student t, Welch t, Z-test (n₁+n₂ > 60 only), Mann-Whitney U, and Wilcoxon (equal-size groups only). A recommendation adapts to normality and homogeneity results.

**Cat × Cat — Association Tests:** Chi-square test of independence with expected-frequency diagnostics (min expected cell, % cells below 5). Fisher's Exact in its generalised k×r form is triggered when chi-square assumptions are weak or the table fits the size threshold. Cramér's V and Tschuprow's T with qualitative strength labelling.

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### Dependencies

```
dash==4.1.0
dash_bootstrap_components==2.0.4
numpy==2.4.4
pandas==3.0.2
plotly==6.6.0
pyarrow==21.0.0
scipy==1.17.1
```

Optional packages (required only for specific file formats):

| Package | Formats |
|---|---|
| `openpyxl` | `.xlsx`, `.xlsm` (bundled with pandas) |
| `xlrd` | `.xls` |
| `odfpy` | `.ods` |
| `pyarrow` | `.parquet` |

---

## 📂 Supported File Formats

| Format | Extension(s) | Notes |
|---|---|---|
| CSV | `.csv`, `.txt` | BOM stripped; delimiter auto-detected |
| TSV | `.tsv` | Tab delimiter forced |
| JSON | `.json` | List of objects or column dict |
| JSON Lines | `.jsonl` | One JSON object per line |
| Excel | `.xlsx`, `.xlsm` | Via openpyxl |
| Excel (legacy) | `.xls` | Via xlrd |
| OpenDocument | `.ods` | Via odfpy |
| Parquet | `.parquet` | Via pyarrow |

---

## 🚀 Running the App

```bash
python eda_app.py
```

The app is available at `http://localhost:8050`. For production WSGI deployment (Gunicorn, uWSGI), use the `server` object exported from `eda_app.py`:

```bash
gunicorn eda_app:server
```
