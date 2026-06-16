# v8 — Exploratory Data Analysis Dashboard

## Stack
- **Dash 4.1.0** + **dash-bootstrap-components 2.0.4**
- **NumPy** only for analysis (no pandas in hot path)
- **Plotly 6.6.0** for charts
- **Scipy** for statistical tests
- **pytest** for testing

## Architecture
```
eda_app.py      → Entry point, layout, clientside callbacks
callbacks.py    → All Dash callback registration (register(app))
ui.py           → Pure component builders (no callback logic)
store.py        → Global module-level state + caches (singleton pattern)
stats.py        → Statistical computations
charts.py       → Plotly chart factories
decorators.py   → @time_budget decorator + sampling helpers
loader.py       → Upload → parse → store pipeline
parsers.py      → Low-level CSV/JSON/Excel/Parquet parsers
utils.py        → Shared NumPy helpers (drop_nan, is_integer_array)
assets/style.css → Full theme system (~1100 lines)
```

## BMad AI Agent Framework
```
_bmad/              → Core platform (v6.8.0): modules, config, scripts, skills
  _config/            → Installation registry: manifest, skill/file manifests, help index
  config.toml         → Project config (name, output paths, module settings)
  config.user.toml    → Personal overrides (user_name, language, skill_level)
  custom/             → Team/user customization overrides (gitignored user override)
  scripts/            → Config/customization resolution pipeline (resolve_config.py, resolve_customization.py)
  core/               → Built-in module (12 skills: help, spec, brainstorming, review, etc.)
  bmm/                → BMad Method (42 skills): analysis → planning → solutioning → implementation
  bmb/                → BMad Builder (5 skills): module/workflow/agent/skill authoring
  tea/                → Test Architecture (9 skills): ATDD, CI, framework, NFR, trace, review
  cis/                → Creative Intelligence Suite (8 skills): storytelling, design thinking, brainstorming
  gds/                → Game Dev Studio (23 skills): game-specific agents + workflows
  automator/          → Story Automator (2 skills): autonomous story build cycle
  wds/                → Web Design System (20 skills): UX → scenarios → dev → assets → design system
_bmad-output/       → Generated artifacts from BMad sessions
  planning-artifacts/  → PRD, architecture, epics, decision logs
  implementation-artifacts/ → Epic 1-3 specs (performance, redesign, export)
  brainstorming/       → Session outputs
  test-artifacts/      → QA plans
.agents/skills/     → 109 OpenCode-compatible skill copies (synced from _bmad/)
.opencode/          → OpenCode IDE plugin (@opencode-ai/plugin@1.17.6)
```
BMad provides the full AI collaboration pipeline: **WDS agents** (Freya UX, Saga analyst, Mimir builder) design → spec → build; **BMM agents** (Mary analyst, John PM, Sally UX, Winston architect, Amelia dev) handle discovery → planning → architecture → delivery; **TEA** (Murat) ensures QA coverage; **CIS** adds creative methods (storytelling, brainstorming, design thinking). All 120 skills are available as `.agents/skills/*/SKILL.md` for OpenCode agent loading.

## Conventions
- `snake_case` for functions and variables
- `build_*` for UI factories, `on_*` for callbacks
- No relative imports (run as `python eda_app.py`)
- Partial type hints (modern style: `list[str]`)
- Descriptive module docstrings, sparse per-function docstrings

## Key State (store.py globals)
- `dataset: dict[str, np.ndarray]` — raw column arrays
- `clean_arrays: dict[str, np.ndarray]` — NaN-stripped float arrays
- `col_meta: dict[str, str]` — `"numeric"` / `"categorical"` / `"temporal"`
- `lightweight_mode: bool` — adaptive speed limiter
- `cancel_event: threading.Event` — computation cancellation

## Callback Pattern
- `register(app)` called once from `eda_app.py`
- Three parallel callbacks on univariate variable change (chart, stats, normality)
- Support for sampling, cancellation, lightweight mode throughout
- `@time_budget(threshold=2)` decorator on heavy kernels
- Clientside callbacks for: theme toggle, PNG export, fullscreen

## Tests
```bash
# Unit tests
pytest tests/ -v --ignore=tests/test_playwright.py

# Integration tests (Playwright — install once)
pip install pytest-playwright
playwright install chromium
pytest tests/test_playwright.py -v
```
**53 tests** — 44 unit + 9 Playwright integration:
- **Unit**: time_budget decorator, sampling helpers, metadata cache, temporal classification, CSV export formatting, HTTP endpoints
- **Integration** (Playwright): upload → chart render, stats appearance, CSV button, PNG download, theme toggle/cycle, bivariate tab, sidebar collapse

## Status
All 3 epics (8 FRs) delivered, plus post-delivery refinements:
- Epic 1: Performance (sampling, cancel, lightweight mode, instant overview)
- Epic 2: Visual Redesign (semantic colors, collapsible panels, micro-interactions)
- Epic 3: Export & Presentation (PNG/CSV export, presentation mode)
- Post-delivery: Overview tab removed, overview card now below upload section. Normality tests appear only in sidebar. Bivariate chart callback fixed (decorator misplacement). Presentation mode: toggle button + Escape exit, `_dashprivate_layout` replaced with `dash_clientside.set_props`.
- HF Spaces optimisations (Jun 15): TIME_BUDGET_THRESHOLD 5→2, SUBSAMPLE_REPS 10→5, MC_REPS 200→100, `_corr_output` appel unique, `_plot_cache` LRU max 50, SCATTER_MAX_LIGHT=3000, `store.trim_plot_cache()`.
- HF Spaces correctifs (Jun 15, v2): `_log()` catch OSError/PermissionError (filesystem read-only), `_corr_output()` déplacé après la garde, discriminants sampling-store (short-circuit si refresh key mismatch), CSS responsive sidebar (stack au lieu de hide), TIME_BUDGET_THRESHOLD 2→5, workers 2→1.
- **Perf fix (Jun 15, v3)**: Suppression du `sampling-store` + badge outputs → plus de cross-contamination entre callbacks univariée/bivariée. Normality séparée dans son propre callback (chart s'affiche immédiatement, normality async). `pairwise-correlation` changé de `children`→`figure` (Plotly.react au lieu de recréer le dcc.Graph). Cache simplifié, `get_corr_matrix()` appel unique. Layout: `dcc.Graph(id="pairwise-correlation")` au lieu de `html.Div`. `gunicorn timeout` 300→600s pour créditcard.csv (151 MB, transfert ~180-200s).
- **PNG fix (Jun 16)**: Gardes `_fullData`/`_fullLayout` supprimées — le wrapper `dcc.Graph` de Dash 4 ne les expose pas. `Plotly.toImage` + lien manuel. Feedback bouton ajouté.
- **PNG fix v2 (Jun 16)**: La vraie cause — `document.getElementById()` renvoie le wrapper Dash (`dash-graph`), pas le div Plotly (`.js-plotly-plot`). Le wrapper n'a pas `.data` ni `.layout` → `Plotly.downloadImage` rend un graphique vide (axes sans courbes). Corrigé avec `document.querySelector('#univariate-plot .js-plotly-plot')`. Test Playwright confirmé : 75.85% pixels non-blancs (vs ~1% avant).
- **Sampling badge fix (Jun 16)**: `build_sampling_badge()` plantait sur les gros datasets — `id=None` rejeté par Dash 4. Corrigé avec `**kw` conditionnel. `track_computation_speed()` activée.
- **Normality callback fix (Jun 16)**: `build_sampling_badge(id=None)` TypeError dans `on_univariate_normality` → callback ne retournait jamais le panneau de normalité. `try/except` ajouté. Corrigé définitivement avec le fix `id=None` ci-dessus.
- **Code review (Jun 16)**: 5 patches appliqués : `sampled_dict` initialisé, filtre `np.isfinite` dans scatter, `nan.astype` au lieu de `nan.view`, extension-detection avant `csv.Sniffer`, `num_cols = []` au lieu de `.clear()`. 45/45 OK.
- **Code review v2 (Jun 16)**: CSV quoting `_csv_quote()` pour catégories avec virgules. 4 defer documentés.
- **Code review v3 (Jun 16)**: CSS : règles globales sorties du `@media`, variables `--info`/`--info-light` ajoutées aux deux thèmes.
