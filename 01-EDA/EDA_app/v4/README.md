# EDA Dashboard Application — v3

A production-ready Exploratory Data Analysis web application built with **Dash** and **Plotly**. This version introduces **advanced statistical testing**, **intelligent column type detection**, **optimized performance**, and a **completely redesigned architecture** focused on correctness and user experience.

---

## ✨ What's New in v3

| Feature | v2 (Previous) | v3 (This Version) |
|---------|---------------|-------------------|
| **Normality testing** | Shapiro‑Wilk only | **5-test battery**: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, Lilliefors, D'Agostino-Pearson |
| **Lilliefors test** | Not available | **MC‑based** with per‑n caching — < 1 ms after first call |
| **Column classification** | Simple dtype check | **Smart heuristics**: ≤10 distinct integer values → categorical |
| **Deduplication** | Row‑by‑row tuple loop | **NumPy structured arrays** — 3–5× faster |
| **Correlation matrix** | Not available | **Pairwise Pearson** + heatmap + top‑5 insights |
| **Bivariate test reports** | Basic output | **Rich formatted output** with recommendations |
| **Chi‑square effect sizes** | Not available | **Cramér's V** + **Tschuprow's T** |
| **Outlier detection** | Not available | **IQR × 1.5** rule with percentage display |
| **Mode handling** | Simple mode | **Frequency + tie detection** |

---

## 🧠 Architecture Overview

| Layer | Files |
|-------|-------|
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
Application entry point. Creates Dash app, defines layout (upload card, tabs), exposes `server` for WSGI deployment.

### `callbacks.py`
Orchestration layer. Four main callbacks:
- `on_upload`: loads data, initializes dropdowns
- `on_variable_changed`: updates plot type options
- `on_univariate`: generates charts + stats + normality tests
- `on_bivariate`: handles all pair types + correlation matrix

### `store.py`
Single source of truth. Maintains `dataset`, `col_meta`, `num_cols`, `cat_cols`. Includes Lilliefors MC cache with lazy evaluation.

### `loader.py`
Decodes base64 uploads, routes to appropriate parser, removes duplicate rows using NumPy structured arrays (3-5x faster), commits to store.

### `parsers.py`
One function per format: CSV, TSV, JSON, JSONL, XLSX, XLS, ODS, Parquet. Features BOM stripping, delimiter auto-detection, missing value handling.

### `stats.py`
Pure statistical functions. Includes:
- 5-test normality battery (Shapiro-Wilk, KS, Anderson-Darling, Lilliefors, D'Agostino-Pearson)
- Descriptive stats with mode tie detection
- Outlier detection (IQR × 1.5)
- Bivariate tests with effect sizes (Cramér's V, Tschuprow's T)

### `charts.py`
Plotly figure factories: histogram+KDE, box, bar, pie, scatter+OLS, grouped box, heatmap, correlation heatmap.

### `ui.py`
Dash HTML components: descriptive stats panels, normality battery cards, test panels, correlation insights.

### `utils.py`
Helpers: `drop_nan()`, `is_integer_array()` (3x faster), `format_percent()`.

---

## 🚀 Performance Optimizations

| Operation | Improvement |
|-----------|-------------|
| Deduplication (mixed types) | 3–4× faster |
| Deduplication (numeric only) | 5× faster |
| Lilliefors test (first call) | ~20 ms |
| Lilliefors test (subsequent) | < 1 ms (cached) |
| Integer detection | 3× faster |
| CDF computation | 3× faster |

---

## 📊 Statistical Tests

### Normality Battery (5 tests)
- **Shapiro-Wilk**: most powerful for n ≤ 5,000
- **Kolmogorov-Smirnov**: conservative
- **Anderson-Darling**: emphasizes tails
- **Lilliefors**: corrects KS, cached MC
- **D'Agostino-Pearson**: based on skewness + kurtosis

### Bivariate Tests

| Pair Type | Tests | Effect Sizes |
|-----------|-------|---------------|
| Num × Num | Pearson, Spearman, Kendall | r, ρ, τ |
| Num × Cat | Levene, ANOVA, Kruskal-Wallis, t-test | η² |
| Cat × Cat | Chi-square, Fisher's exact | Cramér's V, Tschuprow's T |

### Cramér's V Interpretation

| df | Small | Medium | Large |
|----|-------|--------|-------|
| 1 | 0.10 | 0.30 | 0.50 |
| 2 | 0.07 | 0.21 | 0.35 |
| 3 | 0.06 | 0.17 | 0.29 |

---

## 🔧 Installation

```bash
pip install -r requirements.txt