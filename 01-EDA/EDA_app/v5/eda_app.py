import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import callbacks
from parsers import SUPPORTED_EXTENSIONS

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server
_ = app.clientside_callback(
    """
    function(n_clicks, current) {
        if (!n_clicks) return 'light';
        var next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('eda-theme', next);
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
        var saved = localStorage.getItem('eda-theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);
        return saved;
    }
    """,
    dash.Output("theme-store", "data"),
    dash.Input("theme-init", "children"),
)
_ACCEPTED = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTENSIONS))

_GRAPH_CFG = {"responsive": True, "displaylogo": False,
              "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _pc_dd(id_, label):
    return html.Div([
        html.Label(label, style={"fontSize": "0.75rem", "fontWeight": 600,
                                 "color": "var(--slate-700)", "marginBottom": 4,
                                 "display": "block"}),
        dcc.Dropdown(id=id_, clearable=False, style={"fontSize": "0.75rem"}),
    ], style={"marginBottom": 10})


def _loading(children):
    return dcc.Loading(children=children, type="dot",
                       color="var(--primary)",
                       style={"minHeight": 30})


app.layout = dbc.Container([

    dcc.Store(id="theme-store", data="light"),
    html.Div(id="theme-init", style={"display": "none"}, children="init"),

    html.Div([
        html.Div([
            html.Div([
                html.H1("Exploratory Data Analysis"),
                html.Div("Upload a dataset and explore it with interactive visualizations",
                         className="subtitle"),
            ], className="header-left"),
            html.Button(id="theme-toggle", className="theme-toggle-btn",
                        title="Toggle dark mode"),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "width": "100%"}),
    ], className="app-header"),

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
        )],
        className="upload-zone"),
        _loading(html.Div(id="upload-status", className="upload-status")),
    ], style={"background": "var(--bg-card)", "borderRadius": "var(--radius-lg)",
              "padding": 24, "boxShadow": "var(--shadow-sm)", "border": "1px solid var(--border)"}))),

    html.Div(id="main-content", style={"display": "none"}, children=[
        dbc.Tabs([
            dbc.Tab(label="Univariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Variable", className="mt-3",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.Dropdown(id="univariate-variable", clearable=False,
                                     style={"fontSize": "0.8125rem"}),
                        html.H5("Chart type", className="mt-3",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.RadioItems(id="plot-type", options=[], value=None,
                                       className="mt-1",
                                       inputStyle={"marginRight": 6}),
                        html.Hr(className="section-divider"),
                        _loading(html.Div(id="univariate-normality",
                                          className="sidebar-scroll")),
                    ], width=3),
                    dbc.Col([
                        _loading(dcc.Graph(id="univariate-plot", config=_GRAPH_CFG)),
                        _loading(html.Div(id="univariate-stats",
                                          style={"background": "var(--bg-card)",
                                                 "border": "1px solid var(--border)",
                                                 "borderRadius": "var(--radius-lg)",
                                                 "padding": 20, "marginTop": 16,
                                                 "boxShadow": "var(--shadow-sm)"})),
                    ], width=9),
                ], className="mt-3"),
            ]),
            dbc.Tab(label="Bivariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Variable 1", className="mt-3",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.Dropdown(id="bivariate-var1", clearable=False,
                                     style={"fontSize": "0.8125rem"}),
                        html.H5("Variable 2", className="mt-2",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.Dropdown(id="bivariate-var2", clearable=False,
                                     style={"fontSize": "0.8125rem"}),
                        _loading(html.Div(id="bivariate-pair-type")),
                        html.Hr(className="section-divider"),
                        _loading(html.Div(id="bivariate-tests",
                                          style={"overflowY": "auto",
                                                 "maxHeight": "100vh"})),
                        html.Hr(className="section-divider"),
                        _loading(html.Div(id="correlation-insights",
                                          style={"overflowY": "auto",
                                                 "maxHeight": "100vh",
                                                 "fontSize": "0.75rem"})),
                    ], width=3),
                    dbc.Col([
                        _loading(dcc.Graph(id="bivariate-plot", config=_GRAPH_CFG)),
                        _loading(html.Div(id="bivariate-stats",
                                          style={"background": "var(--bg-card)",
                                                 "border": "1px solid var(--border)",
                                                 "borderRadius": "var(--radius-lg)",
                                                 "padding": 20, "marginTop": 16,
                                                 "boxShadow": "var(--shadow-sm)"})),
                        html.Div([
                            html.H5("Pairwise Correlation Matrix",
                                    className="mt-4 mb-2",
                                    style={"fontSize": "0.875rem", "fontWeight": 600}),
                            _loading(dcc.Graph(id="pairwise-correlation", config=_GRAPH_CFG)),
                        ]),
                    ], width=9),
                ], className="mt-3"),
            ]),
            dbc.Tab(label="PCA", children=[
                dbc.Row([
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
                        html.Label("Circle: min cos² filter",
                                   style={"fontSize": "0.6875rem",
                                          "color": "var(--text-muted)",
                                          "display": "block"}),
                        dcc.Slider(id="pca-cos2-filter", min=0, max=0.9, step=0.1,
                                   value=0,
                                   marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
                                   tooltip={"placement": "bottom"}),
                        html.Hr(className="section-divider"),
                        html.Label("Individuals: annotate top-n",
                                   style={"fontSize": "0.6875rem",
                                          "color": "var(--text-muted)",
                                          "display": "block"}),
                        dcc.Slider(id="pca-top-ind", min=0, max=20, step=5, value=0,
                                   marks={i: str(i) for i in [0, 5, 10, 15, 20]},
                                   tooltip={"placement": "bottom"}),
                        html.Hr(className="section-divider"),
                        html.Label("Individuals: colour by",
                                   style={"fontSize": "0.6875rem",
                                          "color": "var(--text-muted)",
                                          "display": "block"}),
                        dcc.RadioItems(
                            id="pca-color-by",
                            options=[
                                {"label": " cos² (quality)",  "value": "cos2"},
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
                            dbc.Tab(label="cos² quality", children=[
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
        ])
    ]),

], fluid=True, className="py-4")

callbacks.register(app)

if __name__ == "__main__":
    app.run(debug=False)
