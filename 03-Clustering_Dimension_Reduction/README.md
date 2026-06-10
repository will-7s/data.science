# Dimensionality Reduction & Clustering Dashboard

An interactive web application for **Principal Component Analysis (PCA)** and **clustering** (K-Means, DBSCAN, Hierarchical), built with **Dash** and **Plotly**. Upload a dataset and explore its structure through expert-guided visualisations and interpretation panels.

---

## Features

### Principal Component Analysis (PCA)

| Feature | Description |
|---------|-------------|
| **6 sub-tabs** | Scree plot, Correlation circle, Biplot, Individuals plane, Contributions, cos² quality |
| **Scree plot** | Variance bars + cumulative curve + recommended PC count + 70% threshold |
| **Eigenvalue table** | Eigenvalue (λ), variance %, cumulative %, keep/discard per PC |
| **Correlation circle** | Variable arrows in unit circle with interactive cos² quality filter |
| **Biplot** | Individuals (rescaled) + variable arrows overlaid in one view |
| **Individuals plane** | Score scatter coloured by cos² or contribution; top-n annotation slider |
| **Contributions bar & heatmap** | Per-axis contribution bar (uniform threshold line) + multi-PC heatmap |
| **cos² quality heatmap** | Variable × PC heatmap of representation quality |
| **5 expert interpretation panels** | Dynamic data-driven commentary for circle, biplot, individuals, contributions, cos² |
| **Axis insight sidebar** | Per-PC breakdown: contribution, cos², correlation, direction per variable |
| **Optimal PC recommendation** | Majority vote of Kaiser (λ>1), scree elbow, and 70% cumulative variance threshold |
| **HHI concentration index** | Herfindahl-Hirschman Index quantifies how concentrated/diffuse each axis is |
| **Misleading arrow detection** | Flags variables with high |correlation| but low cos² (unreliable on this plane) |
| **Redundancy detection** | Identifies variable pairs carrying similar information on the same axis |

### Clustering

| Feature | Description |
|---------|-------------|
| **3 algorithms** | K-Means, DBSCAN, Hierarchical (Ward/Complete/Average/Single linkage) |
| **t-SNE projection** | 2-D visualisation of cluster structure (capped at 15k observations) |
| **Elbow method** | Within-cluster inertia vs. k with rate-of-change analysis |
| **Silhouette analysis** | Silhouette score for k ∈ [2, 15] with ranking and quality thresholds |
| **Cluster profiles heatmap** | Standardised variable means per cluster (identifies what defines each cluster) |
| **Discrimination power** | F-ratio per variable (between/within cluster variance) ranks variables by importance |
| **Cluster separation analysis** | Centroid distance / spread ratio quantifies separation in t-SNE space |
| **Per-cluster axis insight** | Sidebar summary of each cluster's distinctive traits |
| **DBSCAN noise handling** | Noise points rendered as grey crosses; core-point-only silhouette |

### General

| Feature | Description |
|---------|-------------|
| **Dark mode** | CSS custom property theming with clientside toggle and `localStorage` persistence |
| **Multi-format upload** | CSV, TSV, JSON, JSONL, Excel (.xlsx/.xls/.xlsm/.ods), Parquet |
| **Numpy-first pipeline** | All computations vectorised NumPy — no Python per-cell loops after parsing |
| **O(1) PCA callbacks** | PCA computed once on upload; all tab-switches and parameter changes read from cache |
| **Interpretation panels** | Every chart comes with an expert reading guide explaining what to look for |
| **Responsive layout** | Dash Bootstrap Components with fluid container |

---

## Architecture

```
v3/
├── app.py                  Entry point — Dash layout + clientside dark mode
├── callbacks.py            Callback orchestration (16 callbacks)
├── store.py                Global state — dataset, caches (PCA, clustering)
├── loader.py               Upload decoding → parsing → alignment → dedup → store
├── parsers.py              File parsers (CSV, JSON, Excel, Parquet)
├── pca.py                  Pure PCA computations (sklearn wrapper + derived quantities)
├── clustering.py           Clustering algorithms + diagnostics + characterisation
├── charts_common.py        Shared Plotly constants and layout helper
├── charts_pca.py           Plotly figure factories — PCA tab
├── charts_clustering.py    Plotly figure factories — Clustering tab
├── ui_common.py            Shared Dash/HTML component helpers
├── ui_pca.py               PCA interpretation panels (6 expert guides)
├── ui_clustering.py        Clustering interpretation panels (5 expert guides)
├── utils.py                NumPy helpers (drop_nan, format_percent)
├── requirements.txt        Python dependencies
├── assets/
│   └── style.css           Full design system (CSS custom properties, light/dark)
```

### Data flow

```
Upload (base64) → loader.decode + parser → store.reset (align + dedup + classify)
                                                      │
            ┌─────────────────────────────────────────┼────────────────────────┐
            ▼                                         ▼                        ▼
      pca.run_pca()                            clustering.run_clustering()    UI callbacks
            │                                         │                        │
            ▼                                         ▼                        ▼
      store._pca_cache                         store._clustering_cache    store.dataset
            │                                         │                   (read-only)
            ▼                                         ▼
      charts_pca.* + ui_pca.*                   charts_clustering.* + ui_clustering.*
```

---

## File Descriptions

### `app.py`
Dash entry point. Defines the full layout: header with dark-mode toggle, upload zone, and two-tab main area (PCA + Clustering). Clientside callbacks for instant theme switching. All `dcc.Graph` components share a common config (`responsive=True`, `displaylogo=False`).

### `callbacks.py`
16 callbacks registered via `callbacks.register(app)`:

| Callback | Trigger | Outputs |
|----------|---------|---------|
| `on_upload_parse` | Upload contents | Status message, data-loaded flag, main-content visibility |
| `on_upload_pca` | data-loaded flag | PCA dropdown options + values (triggers PCA computation) |
| `on_pca_summary` | pc-x change | Sidebar summary panel |
| `on_pca_scree` | pc-x change | Scree plot + eigenvalue table |
| `on_pca_circle` | pc-x, pc-y, cos² filter | Correlation circle + interpretation |
| `on_pca_biplot` | pc-x, pc-y | Biplot + interpretation |
| `on_pca_individuals` | pc-x, pc-y, colour, top-n | Individuals plane + insight panel + interpretation |
| `on_pca_axis_insight` | pc-insight dropdown | Axis insight panel |
| `on_pca_contributions` | pc-insight dropdown | Contributions bar + heatmap + interpretation |
| `on_pca_cos2` | pc-x change | cos² heatmap + interpretation |
| `on_clustering_params` | Algorithm dropdown | DBSCAN params visibility |
| `on_clustering_run` | Run button | All 10 clustering outputs |

All PCA callbacks are O(1) cache reads — the PCA result is computed once in `on_upload_pca` and stored in `store._pca_cache`. Clustering is computed on demand (button click).

### `store.py`
Module-level singleton with public state and caches:

| Variable | Type | Purpose |
|----------|------|---------|
| `dataset` | `dict[str, ndarray]` | Raw data per column |
| `clean_arrays` | `dict[str, ndarray]` | NaN-stripped numeric arrays |
| `col_stats` | `dict[str, dict]` | `{n_clean, n_nan}` per column |
| `col_meta` | `dict[str, str]` | Column type: `numeric`, `categorical`, or `datetime` |
| `num_cols`, `cat_cols`, `datetime_cols` | `list[str]` | Typed column names |
| `_pca_cache` | `dict \| None` | Computed PCA result |
| `_clustering_cache` | `dict \| None` | Computed clustering result |

Column classification (in `_classify_column`):

| dtype kind | Condition | Result |
|------------|-----------|--------|
| `U`, `S`, `O` | Always | categorical |
| `i`, `u` | ≤ 10 unique values, < 50% of rows | categorical |
| `i`, `u` | Otherwise | numeric |
| `f` | ≤ 10 unique integer values | categorical |
| `f` | Otherwise | numeric |

### `loader.py`
Upload pipeline: base64 decode → format dispatch → column alignment → numpy deduplication → `store.reset()`. 

**Column alignment**: `np.unique` with `return_counts` finds the majority length; columns with mismatched lengths are dropped with a user-visible warning.

**Deduplication**: Pure numpy void-view approach — each row becomes an opaque byte sequence via a structured dtype, then `np.unique` on the void view removes duplicates in O(n log n). Float NaN values are handled via a separate `uint8` flag to ensure consistent hashing.

### `parsers.py`
Two-phase parsing:
1. **Raw bytes → Python objects**: language-specific C libraries (orjson for JSON, stdlib csv, pandas read_excel for Excel, pyarrow for Parquet)
2. **Python objects → numpy arrays**: vectorised `_infer_numpy_col` with no Python per-cell loop

| Format | Engine | Phase 1 |
|--------|--------|---------|
| CSV/TXT | `csv.reader` (C) | Rows → column lists |
| TSV | `csv.reader(delim='\t')` | Same |
| JSON | `orjson` (Rust) → `_extract_rows` | list[dict] → pandas DataFrame bridge → numpy |
| JSONL | `orjson` per line | Same |
| XLSX/XLSM | `openpyxl` via pandas | DataFrame → numpy |
| XLS | `xlrd` via pandas | Same |
| ODS | `odf` via pandas | Same |
| Parquet | `pyarrow` | Arrow table → `to_pylist` → numpy |

**Embedded header row detection**: Strips rows where string values match their column key (datasud.fr, Opendatasoft exports).

**Encoding support**: UTF-8 (±BOM), UTF-16 LE/BE, UTF-32 LE/BE — auto-detected via BOM.

### `pca.py`
Pure PCA computations. Wraps `sklearn.decomposition.PCA` with `StandardScaler` and computes all derived quantities vectorised:

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| `corr_circle[j,k]` | `loading[j,k] × √λ_k` | Pearson correlation of variable j with principal axis k |
| `cos2_var[j,k]` | `corr_circle²` | Proportion of variable j's variance explained by axis k |
| `contributions_var[j,k]` | `loading² × 100` | % of axis k built by variable j (Σ = 100 per axis) |
| `contributions_ind[i,k]` | `score² / (n × λ_k) × 100` | % of axis k built by individual i |
| `cos2_ind[i,k]` | `score² / Σ score²` | Quality of representation of individual i on axis k |

**Row label detection**: Searches for an ID-like column (name, code, label, key, index) with ≥50% unique values — priority-ordered by column name keywords.

**Optimal PC count**: Majority vote of Kaiser criterion (λ>1), scree elbow (max of second derivative), and 70% cumulative variance threshold.

### `clustering.py`
Clustering algorithms and diagnostics.

**Pipeline** (`run_clustering`):
1. `_get_X` — NaN filtering + StandardScaler → (X, valid_mask)
2. `run_tsne` — 2-D t-SNE projection (subsampled to 15k if n > 15k)
3. Algorithm runner — `run_kmeans` / `run_dbscan` / `run_hierarchical`
4. `characterize_clusters` — per-cluster mean/std profiles (vectorised)
5. `_run_diagnostics` — fused elbow + silhouette in one KMeans loop (7 fits)
6. `discrimination_power` — F-ratio per variable (fully vectorised using SS_total = SS_between + SS_within)

| Algorithm | Key details |
|-----------|-------------|
| K-Means | `n_init="auto"`, `random_state=0` |
| DBSCAN | Configurable eps + min_samples; silhouette on core points only |
| Hierarchical | Configurable linkage; exposes `children_` for potential dendrogram |

### `charts_pca.py`
Plotly figure factories for PCA. Shared layout and colour constants from `charts_common.py`. Functions:

- `scree_plot` — Bar + cumulative curve with dual y-axes, recommended PC marker, 70% threshold line
- `eigenvalue_table` — `go.Table` with keep/discard recommendations
- `correlation_circle` — Unit circle + variable arrows (annotations) + interactive cos² filter
- `biplot` — Overlaid individual scores + variable arrows in unit circle
- `individuals_plane` — Score scatter coloured by cos²/contrib with top-n annotation
- `contributions_bar` — Horizontal bar per axis with uniform threshold line
- `contributions_heatmap` — Multi-PC heatmap via `px.imshow`
- `cos2_heatmap` — Quality of representation heatmap via `px.imshow`

### `charts_clustering.py`
Plotly figure factories for clustering:

- `cluster_scatter` — t-SNE scatter coloured by cluster label (boolean mask for subsampling, no `np.intersect1d`)
- `elbow_plot` — Inertia vs. k with markers
- `silhouette_plot` — Silhouette score per k with best-k highlight
- `cluster_profiles_heatmap` — Cluster × variable means via `px.imshow`

### `ui_pca.py`
Six expert interpretation panels (Dash HTML factories):

| Panel | Content |
|-------|---------|
| `pca_summary_panel` | Overview: n, variables, components, recommendation, NaN count |
| `axis_insight_panel` | Per-variable cards: rank, contribution, cos², corr, direction |
| `circle_interpretation_panel` | Axis polarity, quality table, correlated/opposed pairs, low-plane warning |
| `biplot_interpretation_panel` | Top variables per axis, dual-scale reading guide |
| `individuals_interpretation_panel` | Cloud stats, quadrant distribution, top-5 contributors, quality summary |
| `contributions_interpretation_panel` | HHI concentration, axis polarity, redundant pairs, R² cumulative, FactoMineR protocol |
| `cos2_interpretation_panel` | Per-variable diagnostic, per-axis coverage, misleading arrows, escaped variables |

### `ui_clustering.py`
Five expert interpretation panels:

| Panel | Content |
|-------|---------|
| `clustering_summary_panel` | Algorithm, silhouette score, cluster sizes, t-SNE subsampling warning |
| `cluster_scatter_interpretation` | Separation ratio analysis, overlap detection, noise % (DBSCAN) |
| `elbow_interpretation` | Rate-of-change analysis with suggested k |
| `silhouette_interpretation` | All k ranked with quality labels |
| `profiles_interpretation` | Top discriminating variables (F-ratio), per-cluster distinctive traits |
| `cluster_axis_insight_panel` | Per-cluster profile cards with trait highlights |

### `utils.py`
Pure NumPy helpers: `drop_nan`, `format_percent`.

### `charts_common.py`
Shared Plotly constants: `LAYOUT` dict, `THETA` (unit circle angles), colour palettes (`CLUSTER_COLORS`, `VAR_COLORS`, single colours). Reduces duplication between chart modules.

### `ui_common.py`
Shared Dash/HTML component factories: `_table`, `_row`, `_section`, `_interp_header`, `_tip_box`, `_reading_grid`. Used by interpretation panels for consistent visual styling.

---

## Performance Optimisations

### Algorithmic

| Module | Technique | Complexity |
|--------|-----------|------------|
| `discrimination_power` | Uses SS_total = SS_between + SS_within identity — fully vectorised | **O(n×p)** instead of O(k×n×p) |
| `characterize_clusters` | Vectorised mean/std per axis (not per variable) | **O(k + n/k)** instead of O(k×p) |
| `cluster_scatter` | Boolean mask instead of `np.intersect1d` | **O(n)** per cluster instead of O(n log n) |
| `circle_interpretation_panel` | Fused 6 list comprehensions → 1 pass through variables | **O(v)** instead of 6× O(v) |
| `corr_pairs` + `opp_pairs` | Merged into single loop with early-continue | **1× O(v²)** instead of 2× |
| `_run_diagnostics` for DBSCAN | Guarded — 0 fits instead of 7 | **0** instead of 7 KMeans |
| `characterize_clusters` call | Reused via optional `char` parameter | **1×** instead of 2× per run |

### Memory

| Module | Technique | Saving |
|--------|-----------|--------|
| `pca.py`, `clustering.py` | Removed redundant `.astype(float)` on already-float matrices | **O(n×p)** copy eliminated |
| `loader.py` | `reshape(-1)` instead of `ravel()` for guaranteed view | Prevents hidden copy |
| `charts_pca.py` | Merged marker + text Plotly traces | **33% fewer DOM nodes** |
| `pca.py` | `.tolist()` instead of `list(arr.astype(str))` | Single C-level copy |

### Caching

| Cache | Scope | Invalidated |
|-------|-------|-------------|
| `store._pca_cache` | Full PCA result | On new upload |
| `store._clustering_cache` | Full clustering result | On new upload (but computed on demand) |
| `store.clean_arrays` | NaN-free numeric columns | On new upload |
| `store.col_stats` | Per-column NaN counts | On new upload |

### Redundancy eliminated

- `scipy` dependency removed (was never imported)
- `_layout()` function extracted from 2 files into shared `charts_common.py`
- `THETA = np.linspace(0, 2π, 300)` computed once in `charts_common.py`
- `is_integer_array` import removed from `store.py`

---

## Installation

```bash
pip install -r requirements.txt
```

### Core dependencies

```
dash==4.1.0
dash-bootstrap-components==2.0.4
numpy==2.4.6
pandas==3.0.3
plotly==6.6.0
scikit-learn==1.9.0
orjson==3.11.9
```

### Optional (file format support)

| Package | Formats |
|---------|---------|
| `openpyxl==3.1.5` | `.xlsx`, `.xlsm` |
| `odfpy==1.4.1` | `.ods` |
| `pyarrow==21.0.0` | `.parquet` |

> `orjson` is optional — falls back to stdlib `json` if unavailable. Installed by default for 3–5× faster JSON parsing.

---

## Supported File Formats

| Format | Extension(s) | Notes |
|--------|--------------|-------|
| CSV | `.csv`, `.txt` | BOM stripped; delimiter auto-detected via `csv.Sniffer` |
| TSV | `.tsv` | Tab delimiter forced |
| JSON | `.json` | List of objects, column dict, Opendatasoft `values`/`results`/`records` |
| JSON Lines | `.jsonl` | One JSON object per line |
| Excel | `.xlsx`, `.xlsm`, `.xls` | Via openpyxl / xlrd |
| OpenDocument | `.ods` | Via odfpy |
| Parquet | `.parquet` | Via pyarrow |

**Encodings:** UTF-8 (±BOM), UTF-16 LE/BE, UTF-32 LE/BE — auto-detected.

**French open-data formats:** Opendatasoft API v1/v2, datasud.fr UTF-16 exports with embedded header rows — handled automatically.

---

## Running the App

```bash
python app.py
```

Available at `http://localhost:8050`.

### Production deployment (Gunicorn)

```bash
gunicorn app:server --workers 1 --threads 4 --timeout 120
```

> **Workers = 1** is recommended because `store.py` uses module-level globals. Each worker has its own independent store — correct for stateless deployment. Use `--threads 4` for concurrent requests within one worker.

---

## Project Structure

```
v3/
├── app.py                 # Dash entry point
├── callbacks.py           # 16 callbacks
├── store.py               # Global state + caches
├── loader.py              # Upload handling
├── parsers.py             # File format parsers
├── pca.py                 # PCA computations
├── clustering.py          # Clustering computations
├── charts_common.py       # Shared Plotly constants
├── charts_pca.py          # PCA chart factories
├── charts_clustering.py   # Clustering chart factories
├── ui_common.py           # Shared Dash component helpers
├── ui_pca.py              # PCA interpretation panels
├── ui_clustering.py       # Clustering interpretation panels
├── utils.py               # NumPy helpers
├── requirements.txt       # Dependencies
├── README.md              # This file
├── CHANGELOG.md           # Version history
└── assets/
    └── style.css          # Full design system
```
