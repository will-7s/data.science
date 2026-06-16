# EDA v8 — Exploratory Data Analysis Dashboard

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![Dash 4.1.0](https://img.shields.io/badge/Dash-4.1.0-purple)](https://dash.plotly.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Interactive web-based EDA tool built with Dash 4, Plotly 6, and NumPy. Designed for datasets from 1 KB to 150+ MB with adaptive performance and rich statistical output.

---

## Features

### Upload & Overview
- **Multi-format upload** — CSV, TSV, JSON, JSONL, Excel (`.xls`/`.xlsx`), ODS, Parquet
- **Encoding detection** — automatic BOM detection for UTF-32/UTF-16/UTF-8 + CSV Sniffer for delimiter detection
- **Instant dataset overview** — rows, columns, memory usage, column type distribution, top-5 missing values — computed in < 1 s for up to 1M rows
- **Column classification** — automatic detection of numeric, categorical, and temporal columns using NumPy dtype heuristics; integers with ≤10 distinct values flagged as categorical
- **Deduplication** — numpy void-view + `np.unique` for all-dtype row dedup; no pandas in hot path

### Univariate Analysis
- **Chart types per column type:**
  - Numeric: histogram or box plot
  - Categorical: bar chart or pie chart (with "others" grouping beyond 50/20 categories)
- **Descriptive statistics:** N, mean, median, mode (with tie detection), std, min, max, IQR, outlier % (IQR × 1.5)
- **Normality test battery** (sidebar, async — chart renders immediately):
  - Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling (scipy ≥1.17 p-value via interpolation), Lilliefors (MC-cached < 1 ms), D'Agostino-Pearson
  - Skewness, excess kurtosis
  - Consolidated verdict banner: "All tests: Normal ✓" / "All tests: Non-normal ✗" / "N/M tests: Normal — inconclusive"
  - Stratified subsampling for n > 5,000 — single argsort for all K subsamples, median aggregation across repetitions
  - Pre-computed Lilliefors Monte Carlo null distributions per unique sample size

### Bivariate Analysis
- **Auto-detected chart type:**
  - Numeric × Numeric: scatter plot with OLS trend line + R²
  - Numeric × Categorical: grouped box plots (semantic colour per group)
  - Categorical × Categorical: contingency heatmap (row %)
- **Sidebar tests:**
  - Numeric × Numeric: Pearson, Spearman, Kendall correlations + normality assessment + recommended test
  - Numeric × Categorical: ANOVA, Kruskal-Wallis, Levene, normality per group, two-group tests (Student/Welch t-test, Mann-Whitney U, Wilcoxon, Z-test)
  - Categorical × Categorical: Chi-square, Fisher's Exact (k×r generalised), Cramér's V, Tschuprow's T, normality of marginal counts
- **Pairwise correlation matrix** — heatmap of all numeric columns (`Plotly.react`, single `get_corr_matrix()` call)
- **Top correlations insights** — ranked list of strongest pairwise relationships

### Performance & Adaptive Mode
- **Adaptive sampling** — auto-switches to random sampling when dataset exceeds threshold; `track_computation_speed()` monitors 2 consecutive slow computations
- **Cancel button** — fixed-position button during long computations; clean stop via `threading.Event`
- **Lightweight mode** — auto-activated after 2 consecutive slow sampled computations; limits scatter to 3,000 points, disables heavy features; visible badge: "Auto-adjusted for speed"
- **5‑test normality subsampling** — stratified subsampling with single `O(n log n)` argsort for all repetitions; median aggregation avoids over-rejection
- **LRU plot cache** — 50‑figure LRU cache; instant chart replay on variable switch
- **Lilliefors MC cache** — pre-computed Monte Carlo null distributions per unique sample size (< 1 ms per call)
- **Pre-warmed lazy imports** — `_sp()`, `_go()`, `_px()` warmed at startup so first callback isn't slowed by scipy/plotly import

### Export
- **PNG export** — one-click download for both univariate and bivariate charts (2400 × 1200 @ 2× scale); uses `Plotly.downloadImage` with correct `.js-plotly-plot` DOM selector (Dash wrapper fix, Jun 16)
- **CSV export:**
  - Univariate numeric: all descriptive stats in CSV (N, mean, median, mode, std, min, max, Q1, Q3, IQR, skewness, kurtosis, outlier %)
  - Univariate categorical: frequency table with counts and percentages
  - Bivariate numeric: Pearson/Spearman/Kendall coefficients + covariance
  - Bivariate mixed: group statistics (mean, std, min, Q1, median, Q3, max per category)
  - Bivariate categorical: contingency table
- **CSV quoting** — categories with commas or quotes properly escaped via `_csv_quote()`
- **Filename pattern:** `{analysis}_{timestamp}.png`, `{column}_{type}_{timestamp}.csv`

### Presentation Mode
- **Toggle button** — hides both sidebars for full‑width chart display
- **Escape key exit** — `fullscreenchange` event listener restores layout
- **CSS class** `.presentation-mode` applied to `<body>` for custom styling
- **Clientside fullscreen** — `dash_clientside.set_props` toggles `fullscreen-mode` store

### Visual Design
- **Dark/light mode** — persistent via `localStorage`; system-wide `data-theme` attribute on `<html>`; clientside callback (no server round-trip)
- **Semantic colour system** — CSS custom properties: `--primary`, `--slate-*`, `--bg-card`, `--text-primary`, `--radius-*`, `--info`/`--info-light` (both themes)
- **Collapsible side panels** — independent collapse per tab (univariate/bivariate); button `◀` toggles sidebar width from 3 → 0
- **Professional Plotly palette** — 10-category palette, sequential indigo scale, consistent layout/font configuration
- **Micro-interactions:** hover/focus states, transition effects, box shadows
- **Responsive sidebar** — stacks on narrow viewports instead of hiding

---

## Architecture

```
eda_app.py      → Entry point, layout, clientside callbacks (theme, PNG, fullscreen)
callbacks.py    → All Dash callback registration (register(app))
ui.py           → Pure component builders (descriptive stats, normality cards, badges, overview)
store.py        → Global module-level state + caches (LRU plot cache, MC cache, corr matrix cache)
stats.py        → Statistical computations (5-test battery, subsampling, group tests, bivariate tests)
charts.py       → Plotly chart factories (histogram, box, scatter, bar, pie, heatmap)
decorators.py   → Sampling helpers (sample_random, sample_stratified, sample_rows)
loader.py       → Upload → parse → dedup → store pipeline
parsers.py      → Low-level CSV/JSON/JSONL/Excel/ODS/Parquet parsers with BOM encoding detection
config.py       → Tunable parameters (thresholds, sizes, seeds)
utils.py        → Shared NumPy helpers (drop_nan, is_integer_array)
app.py          → WSGI entry point (from eda_app import server)
gunicorn.conf.py → Production / HF Spaces deployment config
edge-cases.json → Documented edge cases with trigger conditions and guard snippets
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **NumPy-only hot path** (pandas only at parse bridge) | Predictable memory/perf for 151 MB creditcard.csv; no pandas overhead |
| **Split chart/stats/normality calls** | Chart renders in ~ms, stats follow in parallel — perceived performance |
| **No `dcc.Loading`** | Fast enough to avoid spinner flash; normality is the only slow path and renders async |
| **Clientside callbacks** | Theme toggle immediate (no round-trip); PNG export via `Plotly.downloadImage`; fullscreen via `dash_clientside.set_props` |
| **Single argsort subsampling** | All K stratified subsamples from one `O(n log n)` argsort — 19× faster than K×argsort |
| **Lazy imports** (`_sp()`, `_go()`, `_px()`) | Pre-warmed at startup — first callback isn't slowed by scipy/plotly import |
| **`dcc.send_string` for CSV** | Pure client-side download; no temp files |
| **`_csv_quote()` for export** | Properly escapes commas and quotes in categorical values |

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements

```
dash==4.1.0
dash-bootstrap-components==2.0.4
numpy==2.4.6
plotly==6.6.0
scipy==1.17.1
orjson==3.11.9          # Fast JSON parsing (falls back to stdlib)
pandas==3.0.3           # Parse bridge only (JSON→numpy transpose, Excel)
pyarrow==21.0.0         # Parquet support
openpyxl==3.1.5         # Excel .xlsx support
odfpy==1.4.1            # OpenDocument Spreadsheet (.ods) support
gunicorn==23.0.0        # Production WSGI server
```

---

## Usage

```bash
python eda_app.py
```

Open `http://127.0.0.1:8050` in your browser. Upload any supported file via drag‑and‑drop or the file browser.

### Production / HuggingFace Spaces

```bash
gunicorn app:server -c gunicorn.conf.py
```

The `gunicorn.conf.py` binds to `0.0.0.0:7860` with 1 worker and a 600 s timeout.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Escape` | Exit presentation mode |

---

## Configuration

All tunable parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_THRESHOLD` | 5 000 | Sampling trigger (rows) |
| `SAMPLE_N` | 5 000 | Random sample size |
| `SAMPLE_SEED` | 42 | Global RNG seed for sampling |
| `SUBSAMPLE_THRESHOLD` | 5 000 | Normality subsampling trigger |
| `SUBSAMPLE_N` | 2 000 | Subsample size (was 10 000) |
| `SUBSAMPLE_REPS` | 5 | Subsample repetitions |
| `SUBSAMPLE_BINS` | 20 | Quantile strata for draw |
| `MC_REPS` | 100 | Lilliefors MC null reps |
| `SCATTER_MAX` | 10 000 | Full scatter plot points |
| `SCATTER_MAX_LIGHT` | 3 000 | Lightweight mode scatter points |
| `BAR_MAX_CATS` | 50 | Max categories in bar chart (remainder → "others") |
| `PIE_MAX_CATS` | 20 | Max categories in pie chart |
| `ALPHA` | 0.05 | Hypothesis test significance level |
| `LIGHTWEIGHT_SLOW_COUNT` | 2 | Slow computations before lightweight mode |

---

## Tests

```bash
# Unit tests (no browser)
pytest tests/ -v --ignore=tests/test_playwright.py

# Integration tests (Playwright — install once)
pip install pytest-playwright
playwright install chromium
pytest tests/test_playwright.py -v
```

**54 tests** — 45 unit + 9 Playwright integration:
- **Unit (45):** sampling helpers (11), metadata cache (11), temporal classification (7), CSV export formatting (9), integration/HTTP endpoints (7)
- **Playwright integration (9):** upload → status, chart render, stats appearance, CSV button, PNG download, theme toggle/cycle, bivariate tab, sidebar collapse

---

## Edge Cases

Documented in `edge-cases.json` — 13 identified edge cases with trigger conditions, guard snippets, and potential consequences covering:
- Empty CSV input (`StopIteration` in reader)
- UTF-16 BOM in CSV (data corruption via `utf-8` decode)
- String `"nan"` as legitimate category value
- Zero-variance numeric columns (NaN correlation matrix)
- Empty arrays in categorical stats panel
- `_subsample_matrix` with n < n_each
- Single-pair correlation (misleading r = 0.0)
- Unbounded plot cache growth

---

## Changelog

See `AGENTS.md` for the full post-delivery log. Key fixes:

- **PNG export (Jun 16 v2)** — root cause: `document.getElementById()` returns Dash wrapper, not Plotly div. Fixed with `document.querySelector('#univariate-plot .js-plotly-plot')`. Playwright confirmed: 75.85 % non-white pixels.
- **Sampling badge (Jun 16)** — `id=None` rejected by Dash 4. Fixed with conditional `**kw` in `build_sampling_badge`.
- **`track_computation_speed` activation (Jun 16)** — function was defined but never called; lightweight mode never activated automatically.
- **Code review (Jun 16)** — 5 patches: `sampled_dict` initialised, `np.isfinite` filter in scatter, `nan.astype` → `nan.view`, extension-detection before `csv.Sniffer`, `num_cols = []` not `.clear()`. `_csv_quote()` for categories with commas. CSS globals moved outside `@media`, `--info`/`--info-light` in both themes.
- **Perf fix v3 (Jun 15)** — sampling-store removed to fix cross-contamination; normality made async; `pairwise-correlation` converted to `figure` output for `Plotly.react`. `gunicorn timeout` 300→600 s for large files.
- **HF Spaces (Jun 15)** — threshold tuning, cache limits, OSError handling for read-only filesystem, responsive sidebar stacking.

---

## License

MIT
