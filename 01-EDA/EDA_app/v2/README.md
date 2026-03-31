# EDA Dashboard — v2

A production-ready Exploratory Data Analysis web application built with **Dash** and **Plotly**. This version represents a complete architectural overhaul of the original prototype (v1), emphasizing **separation of concerns**, **testability**, **maintainability**, and **extensibility**.

---

## 🧠 Architecture Overview

The application follows a clean **modular architecture** where each file has a single, well-defined responsibility:

| Layer | Files |
|-------|-------|
| **Entry point** | `eda_app.py` |
| **State management** | `store.py` |
| **Data loading** | `loader.py` + `parsers.py` |
| **Business logic (stats)** | `stats.py` |
| **Visualization** | `charts.py` |
| **UI components** | `ui.py` |
| **Orchestration** | `callbacks.py` |

This separation makes the codebase:
- **Unit-testable** – each pure function can be tested in isolation
- **Reusable** – charts and stats functions work independently of Dash
- **Extensible** – adding a new file format requires only one change (in `parsers.py`)

---

## 📁 File-by-File Description

### `eda_app.py`
**Role:** Application entry point & layout definition

Creates the Dash application, defines the user interface (upload card, tabs for univariate/bivariate analysis, dropdowns, graph placeholders), and registers all callbacks. Exposes `server` for WSGI deployment (Gunicorn, uWSGI, etc.).

> *"Nothing else belongs here."* – the file itself states its philosophy.

---

### `callbacks.py`
**Role:** Orchestration layer — connects UI events to business logic

Each callback does exactly three things:
1. Reads from `store` (the dataset)
2. Calls business-logic functions (`stats.*`, `charts.*`, `ui.*`)
3. Returns Dash component updates

No computation happens here. This file is deliberately kept thin, delegating all real work to dedicated modules.

---

### `store.py`
**Role:** Single source of truth for the loaded dataset

Maintains the global state of the application:
- `dataset`: column name → NumPy array
- `col_meta`: column name → `'numeric'` or `'categorical'`
- Convenience lists: `all_cols`, `num_cols`, `cat_cols`

All other modules import from `store`, ensuring exactly one copy of the data in memory and no stale caches.

---

### `loader.py`
**Role:** Bridge between Dash Upload component and the parsing system

Decodes the base64 payload from `dcc.Upload`, routes to the appropriate parser based on file extension, removes duplicate rows (pure NumPy, no pandas), and commits the cleaned dataset to `store`. Returns a `(success, message)` tuple for UI feedback.

---

### `parsers.py`
**Role:** One pure function per file format

Each parser receives raw bytes and returns a `dict[str, np.ndarray]` (column name → 1‑D array, dtype `float64` or `str`). Supports:

| Format | Extension | Dependency |
|--------|-----------|------------|
| CSV / TXT | `.csv`, `.txt` | none (pure Python) |
| TSV | `.tsv` | none |
| JSON | `.json` | none |
| JSONL | `.jsonl` | none |
| Excel (.xlsx, .xlsm) | `.xlsx`, `.xlsm` | `openpyxl` (optional, falls back to pandas) |
| Legacy Excel | `.xls` | `xlrd` + `pandas` |
| OpenDocument | `.ods` | `odfpy` + `pandas` |
| Parquet | `.parquet` | `pyarrow` |

Adding a new format: write `_parse_<ext>()` and register it in `PARSERS` – no other file changes.

---

### `stats.py`
**Role:** Pure statistical computations

Every function takes plain NumPy arrays and returns plain Python values (floats, strings, dicts). No Dash, no Plotly, no UI concerns. Functions include:
- `normality_test()` – Shapiro‑Wilk
- `descriptive_stats()` – mean, median, std, min, max, N
- `outlier_percentage()` – IQR‑based (×1.5 rule)
- `group_means()` – for numeric vs categorical
- `correlation_matrix()` – pairwise Pearson
- `bivariate_test()` – auto‑selects Pearson/Spearman, ANOVA/Kruskal‑Wallis, or Chi‑square

---

### `charts.py`
**Role:** All Plotly figure factories

Pure visualisation layer. Every function takes NumPy arrays + labels and returns a `go.Figure`. No Dash, no store, no stats logic. Chart types include:
- Histogram, box plot, bar chart (numeric binned)
- Bar chart (categorical)
- Scatter plot with OLS trendline
- Grouped box plot (numeric × categorical)
- Heatmap (categorical × categorical, row percentages)
- Correlation matrix heatmap (diverging RdBu_r)

Consistent styling (`_LAYOUT`) and a reusable `_fig()` constructor.

---

### `ui.py`
**Role:** Reusable Dash HTML snippets

Pure UI component factories. Each function takes data (NumPy arrays, dicts, floats) and returns a Dash `html.Div` with formatted statistics. No imports from `store`, `stats`, or `charts` – all dependencies are passed as arguments. Functions include:
- `numeric_stats_panel()`, `categorical_stats_panel()`
- `correlation_panel()`, `group_stats_panel()`
- `test_panel()`, `normality_panel()`, `correlation_insights_panel()`

---

## 🚀 Improvements Over the Previous Version

| Aspect | Previous Version | This Version |
|--------|------------------|--------------|
| **Architecture** | Monolithic – all logic in `callbacks.py` and `eda_app.py` | Modular – 8 focused files with clear separation of concerns |
| **State management** | Module‑level globals scattered across files | Centralised `store.py` – single source of truth |
| **File format support** | CSV + Excel only | CSV, TSV, JSON, JSONL, XLSX, XLSM, XLS, ODS, Parquet |
| **Parsing robustness** | `np.genfromtxt` with hardcoded assumptions | BOM stripping, line‑ending normalisation, delimiter auto‑detection, date handling |
| **Deduplication** | `np.unique(axis=0)` – unreliable for mixed types | Row‑by‑row tuple deduplication – works for any types |
| **Statistical tests** | Mixed responsibility inside callbacks | Pure `stats.py` – each test is a standalone function |
| **Charts** | Inline Plotly calls inside callbacks | Dedicated `charts.py` – reusable, testable figure factories |
| **UI components** | Inline HTML in callbacks | `ui.py` – reusable panels with consistent styling |
| **Error handling** | Basic try/except | Graceful fallbacks for missing dependencies (openpyxl → pandas, etc.) |
| **Extensibility** | Adding a format required changes in multiple files | One function + one line in `PARSERS` |
| **Testability** | Difficult – logic embedded in Dash callbacks | Easy – pure functions in `stats.py`, `charts.py`, `parsers.py` can be unit‑tested without Dash |
| **Deployment** | Not explicitly supported | Exposes `server` for WSGI (Gunicorn ready) |
| **Documentation** | Minimal | Comprehensive docstrings and inline comments explaining *why* |
