"""
app.py  —  Dash entry point (v6 — PCA + Clustering).

Layout architecture
-------------------
The UI is split into three vertical sections:

  1. Header (full width)        — title, dark-mode toggle
  2. Upload zone (full width)   — file drag-and-drop + status
  3. Main content (hidden until upload) — dbc.Tabs with two tabs:
       a) PCA tab        — sidebar (controls) + main area (sub-tabs)
       b) Clustering tab — sidebar (params)  + main area (sub-tabs)

Component naming convention
---------------------------
Every Dash component id is snake_case and prefixed by its domain:
  - pca-*         → PCA tab
  - upload-*      → upload zone
  - theme-*       → dark mode
  - clustering-*  → clustering tab

Note on dcc.Slider
------------------
Dash 4.1.0 removed the `style` kwarg from dcc.Slider.
All sliders below are wrapped in a `html.Div(style=...)` instead.

Improvement suggestions
-----------------------
  a) Lazy tab rendering: use dcc.Tab(children=[]) with a clientside
     callback that only populates the tab DOM when the user clicks on it.
     This would reduce the initial page-weight by ~60 % for users who
     only care about PCA.

  b) Loading states: dcc.Loading is currently wrapped around every
     output.  A more sparing approach would wrap the main content area
     once with a single dcc.Loading — fewer DOM nodes.

  c) Responsive breakpoints: the sidebar has fixed width=3 (25 %).
     On narrow screens (tablets, small laptops) the sidebar becomes
     overly narrow.  A dbc.Col(xs=12, md=3, lg=3) pattern would
     stack the sidebar below the charts on small viewports.

  d) Accessibility: all labels are html.Label or plain text — good.
     But radio items and sliders lack aria attributes.
     Adding aria-label / aria-labelledby would improve screen-reader
     navigation.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import callbacks
from parsers import SUPPORTED_EXTENSIONS

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,   # tabs add/remove components at runtime
)
server = app.server

# ── Clientside dark-mode toggle ───────────────────────────────────────────────
# Two callbacks:
#   1. theme-toggle button → switch light/dark + set attribute + localStorage
#   2. theme-init on load → read localStorage & apply saved theme
# Using clientside avoids the round-trip latency of a Python callback.

_ = app.clientside_callback(
    """
    function(n_clicks, current) {
        if (!n_clicks) return 'light';
        var next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('dr-theme', next);
        return next;
    }
    """,
    dash.Output("theme-store", "data"),
    dash.Input("theme-toggle", "n_clicks"),
    dash.State("theme-store", "data"),
    prevent_initial_call=True,
)
_ = app.clientside_callback(
    """
    function(data) {
        var saved = localStorage.getItem('dr-theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);
        return saved;
    }
    """,
    dash.Output("theme-store", "data"),
    dash.Input("theme-init", "children"),
)

# ── Accepted file extensions (displayed in upload zone) ──────────────────────
_ACCEPTED = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTENSIONS))

# ── Shared Plotly config (removes modebar clutter) ──────────────────────────
_GRAPH_CFG = {"responsive": True, "displaylogo": False,
              "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ── Helper factories ──────────────────────────────────────────────────────────

def _pc_dd(id_: str, label: str) -> html.Div:
    """
    Reusable dropdown labelled widget used for PCA axis selection.

    Every PCA tab sub-view needs two axis dropdowns (X, Y); this factory
    avoids repeating the layout code.

    Improvement suggestion
    ----------------------
    Add `id_` validation at dev time (e.g. assert id_.startswith('pca-'))
    to catch naming typos early.
    """
    return html.Div([
        html.Label(label, style={"fontSize": "0.75rem", "fontWeight": 600,
                                 "color": "var(--slate-700)", "marginBottom": 4,
                                 "display": "block"}),
        dcc.Dropdown(id=id_, clearable=False, style={"fontSize": "0.75rem"}),
    ], style={"marginBottom": 10})


def _loading(children) -> dcc.Loading:
    """
    Wrap children in a dcc.Loading spinner.

    All plot/layout outputs in this app are wrapped with _loading() so
    the user sees a visual indicator during computation.

    Improvement suggestion
    ----------------------
    Consider a single wrapping at the per-tab level rather than
    per-output; fewer DOM mutations when multiple outputs update.
    """
    return dcc.Loading(children=children, type="dot",
                       color="var(--primary)",
                       style={"minHeight": 30})


DD_STYLE = {"fontSize": "0.75rem", "marginBottom": 10}


# ── Layout ─────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([

    # ── Theme store (silent) ──────────────────────────────────────────────
    dcc.Store(id="theme-store", data="light"),
    html.Div(id="theme-init", style={"display": "none"}, children="init"),
    html.Div(id="theme-relayout-done", style={"display": "none"}),

    # ── Header row ────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div([
                html.H1("Dimensionality Reduction & Clustering"),
                html.Div(
                    "Upload a dataset — analyse it with PCA, K-Means, "
                    "DBSCAN, hierarchical clustering, and t-SNE",
                    className="subtitle"),
            ], className="header-left"),
            html.Button(id="theme-toggle", className="theme-toggle-btn",
                        title="Toggle dark mode"),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "width": "100%"}),
    ], className="app-header"),

    # ── Upload zone ──────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.Div([dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("[+]", style={"fontSize": 32, "fontWeight": 300,
                                       "color": "var(--slate-400)", "marginBottom": 8}),
                html.Div(["Drag & drop or ", html.A("browse a file")],
                         style={"fontSize": "0.875rem", "color": "var(--text-secondary)"}),
                html.Div(_ACCEPTED, style={"fontSize": "0.75rem",
                                           "color": "var(--text-muted)", "marginTop": 4}),
            ]),
            accept=_ACCEPTED, multiple=False,
        )], className="upload-zone"),
        _loading(html.Div(id="upload-status", className="upload-status")),
    ], style={"background": "var(--bg-card)", "borderRadius": "var(--radius-lg)",
              "padding": 24, "boxShadow": "var(--shadow-sm)",
              "border": "1px solid var(--border)"}))),

    # ── Main content (hidden until upload) ────────────────────────────────
    html.Div(id="main-content", style={"display": "none"}, children=[
        dbc.Tabs([
            # ═══════════════════════════════════════════════════════════════
            # PCA Tab
            # ═══════════════════════════════════════════════════════════════
            dbc.Tab(label="Principal Component Analysis", children=[
                dbc.Row([
                    # ── PCA sidebar (controls) ────────────────────────────
                    dbc.Col([
                        _loading(html.Div(id="pca-summary",
                                          className="sidebar-scroll")),
                        html.Hr(className="section-divider"),
                        _pc_dd("pca-pc-x", "Axis X (horizontal)"),
                        _pc_dd("pca-pc-y", "Axis Y (vertical)"),
                        html.Hr(className="section-divider"),
                        html.Label("Axis insight",
                                   style={"fontSize": "0.75rem", "fontWeight": 600,
                                          "color": "var(--slate-700)", "display": "block"}),
                        dcc.Dropdown(id="pca-pc-insight", clearable=False,
                                     style={"fontSize": "0.75rem", "marginTop": 4,
                                            "marginBottom": 10}),
                        html.Hr(className="section-divider"),
                        html.Label("Circle: min cos\u00b2 filter",
                                   style={"fontSize": "0.6875rem",
                                          "color": "var(--text-muted)", "display": "block"}),
                        html.Div(  # wrapper needed — dcc.Slider has no `style` in Dash 4.1
                            dcc.Slider(id="pca-cos2-filter", min=0, max=0.9, step=0.1,
                                       value=0,
                                       marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
                                       tooltip={"placement": "bottom"}),
                        ),
                        html.Hr(className="section-divider"),
                        html.Label("Individuals: annotate top-n",
                                   style={"fontSize": "0.6875rem",
                                          "color": "var(--text-muted)", "display": "block"}),
                        html.Div(
                            dcc.Slider(id="pca-top-ind", min=0, max=20, step=5, value=0,
                                       marks={i: str(i) for i in [0, 5, 10, 15, 20]},
                                       tooltip={"placement": "bottom"}),
                        ),
                        html.Hr(className="section-divider"),
                        html.Label("Individuals: colour by",
                                   style={"fontSize": "0.6875rem",
                                          "color": "var(--text-muted)", "display": "block"}),
                        dcc.RadioItems(
                            id="pca-color-by",
                            options=[
                                {"label": " cos\u00b2 (quality)",  "value": "cos2"},
                                {"label": " Contribution",    "value": "contrib"},
                                {"label": " None",            "value": "none"},
                            ],
                            value="cos2",
                            inputStyle={"marginRight": 4},
                            labelStyle={"display": "block", "fontSize": "0.6875rem",
                                        "color": "var(--text-secondary)",
                                        "marginBottom": 2},
                        ),
                        html.Hr(className="section-divider"),
                        _loading(html.Div(id="pca-axis-insight-panel",
                                          style={"overflowY": "auto",
                                                 "maxHeight": "45vh",
                                                 "fontSize": "0.6875rem"})),
                    ], width=3),

                    # ── PCA main area (sub-tabs) ──────────────────────────
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(label="Scree plot", children=[
                                _loading(dcc.Graph(id="pca-scree", config=_GRAPH_CFG)),
                                _loading(html.Div(id="pca-eigenvalue-table-wrap",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 16, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Correlation circle", children=[
                                _loading(dcc.Graph(id="pca-circle", config=_GRAPH_CFG)),
                                _loading(html.Div(id="pca-circle-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Biplot", children=[
                                _loading(dcc.Graph(id="pca-biplot", config=_GRAPH_CFG)),
                                _loading(html.Div(id="pca-biplot-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Individuals", children=[
                                _loading(dcc.Graph(id="pca-individuals", config=_GRAPH_CFG)),
                                _loading(html.Div(id="pca-ind-insight-panel",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                                _loading(html.Div(id="pca-ind-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Contributions", children=[
                                _loading(dcc.Graph(id="pca-contrib-bar", config=_GRAPH_CFG)),
                                _loading(dcc.Graph(id="pca-contrib-heatmap", config=_GRAPH_CFG)),
                                _loading(html.Div(id="pca-contrib-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="cos\u00b2 quality", children=[
                                _loading(dcc.Graph(id="pca-cos2-heatmap", config=_GRAPH_CFG)),
                                _loading(html.Div(id="pca-cos2-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                        ]),
                    ], width=9),
                ], className="mt-3"),
            ]),

            # ═══════════════════════════════════════════════════════════════
            # Clustering Tab
            # ═══════════════════════════════════════════════════════════════
            dbc.Tab(label="Clustering", children=[
                dbc.Row([
                    # ── Clustering sidebar (controls) ─────────────────────
                    dbc.Col([
                        _loading(html.Div(id="clustering-summary",
                                          className="sidebar-scroll")),
                        html.Hr(className="section-divider"),

                        html.Label("Algorithm",
                                   style={"fontSize": "0.75rem", "fontWeight": 600,
                                          "color": "var(--slate-700)", "display": "block"}),
                        dcc.Dropdown(id="clustering-algorithm", clearable=False,
                                     options=[
                                         {"label": "K-Means",      "value": "kmeans"},
                                         {"label": "DBSCAN",       "value": "dbscan"},
                                         {"label": "Hierarchical", "value": "hierarchical"},
                                     ],
                                     value="kmeans",
                                     style=DD_STYLE),

                        html.Label("Number of clusters (k)",
                                   style={"fontSize": "0.75rem", "fontWeight": 600,
                                          "color": "var(--slate-700)", "display": "block",
                                          "marginTop": 8}),
                        html.Div(  # wrapper needed — dcc.Slider has no `style` in Dash 4.1
                            dcc.Slider(id="clustering-n-clusters", min=2, max=15, step=1,
                                       value=3,
                                       marks={i: str(i) for i in [2, 3, 5, 8, 10, 12, 15]},
                                       tooltip={"placement": "bottom"}),
                            style={"marginBottom": 10},
                        ),

                        # ── DBSCAN-specific params (hidden for kmeans/hier) ──
                        html.Div(id="clustering-params", children=[
                            html.Label("DBSCAN eps",
                                       style={"fontSize": "0.75rem", "fontWeight": 600,
                                              "color": "var(--slate-700)", "display": "block"}),
                            html.Div(
                                dcc.Slider(id="clustering-eps", min=0.1, max=2.0, step=0.1,
                                           value=0.5,
                                           marks={i/10: f"{i/10:.1f}" for i in [1, 5, 10, 15, 20]},
                                           tooltip={"placement": "bottom"}),
                                style={"marginBottom": 10},
                            ),
                            html.Label("DBSCAN min samples",
                                       style={"fontSize": "0.75rem", "fontWeight": 600,
                                              "color": "var(--slate-700)", "display": "block"}),
                            html.Div(
                                dcc.Slider(id="clustering-min-samples", min=2, max=20, step=1,
                                           value=5,
                                           marks={i: str(i) for i in [2, 5, 10, 15, 20]},
                                           tooltip={"placement": "bottom"}),
                                style={"marginBottom": 10},
                            ),
                        ], style={"display": "none"}),

                        html.Label("Linkage (hierarchical)",
                                   style={"fontSize": "0.75rem", "fontWeight": 600,
                                          "color": "var(--slate-700)", "display": "block"}),
                        dcc.Dropdown(id="clustering-linkage", clearable=False,
                                     options=[
                                         {"label": "Ward",     "value": "ward"},
                                         {"label": "Complete", "value": "complete"},
                                         {"label": "Average",  "value": "average"},
                                         {"label": "Single",   "value": "single"},
                                     ],
                                     value="ward",
                                     style=DD_STYLE),

                        html.Label("t-SNE perplexity",
                                   style={"fontSize": "0.75rem", "fontWeight": 600,
                                          "color": "var(--slate-700)", "display": "block"}),
                        html.Div(
                            dcc.Slider(id="clustering-perplexity", min=5, max=50, step=5,
                                       value=30,
                                       marks={i: str(i) for i in [5, 15, 30, 50]},
                                       tooltip={"placement": "bottom"}),
                        ),

                        html.Hr(className="section-divider"),
                        html.Button("Run clustering", id="clustering-run",
                                    className="btn btn-primary btn-sm",
                                    style={"width": "100%", "fontSize": "0.75rem"}),

                        html.Hr(className="section-divider"),
                        _loading(html.Div(id="clustering-axis-insight-panel",
                                          style={"overflowY": "auto",
                                                 "maxHeight": "45vh",
                                                 "fontSize": "0.6875rem"})),
                    ], width=3),

                    # ── Clustering main area (sub-tabs) ─────────────────────
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(label="Cluster scatter", children=[
                                _loading(dcc.Graph(id="clustering-scatter",
                                                   config=_GRAPH_CFG)),
                                _loading(html.Div(id="clustering-scatter-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Elbow curve", children=[
                                _loading(dcc.Graph(id="clustering-elbow",
                                                   config=_GRAPH_CFG)),
                                _loading(html.Div(id="clustering-elbow-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Silhouette", children=[
                                _loading(dcc.Graph(id="clustering-silhouette",
                                                   config=_GRAPH_CFG)),
                                _loading(html.Div(id="clustering-silhouette-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                            dbc.Tab(label="Profiles", children=[
                                _loading(dcc.Graph(id="clustering-profiles",
                                                   config=_GRAPH_CFG)),
                                _loading(html.Div(id="clustering-profiles-interp",
                                                  style={"background": "var(--bg-card)",
                                                         "border": "1px solid var(--border)",
                                                         "borderRadius": "var(--radius-lg)",
                                                         "padding": 20, "marginTop": 16,
                                                         "boxShadow": "var(--shadow-sm)"})),
                            ]),
                        ]),
                    ], width=9),
                ], className="mt-3"),
            ]),
        ])
    ]),

], fluid=True, className="py-4")

# ── Register callbacks ────────────────────────────────────────────────────────
callbacks.register(app)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)
