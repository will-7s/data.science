import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import callbacks
from parsers import SUPPORTED_EXTENSIONS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
_ACCEPTED = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTENSIONS))

_CARD    = {"border": "1px solid #dee2e6", "borderRadius": "8px",
            "padding": "16px", "marginTop": "12px", "background": "#ffffff"}
_SIDEBAR = {"overflowY": "auto", "maxHeight": "82vh"}


def _pc_dd(id_, label):
    return html.Div([
        html.Label(label, style={"fontSize": "12px", "fontWeight": "600",
                                 "color": "#2c3e50", "marginBottom": "2px",
                                 "display": "block"}),
        dcc.Dropdown(id=id_, clearable=False, style={"fontSize": "12px"}),
    ], style={"marginBottom": "8px"})


app.layout = dbc.Container([

    dbc.Row(dbc.Col([
        html.H1("Exploratory Data Analysis",
                className="text-center text-primary mb-2"),
        html.Hr(),
    ])),

    dbc.Row(dbc.Col(html.Div([
        html.H4("Upload a dataset", className="text-center"),
        html.P(_ACCEPTED, className="text-center text-muted small"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag & drop or ", html.A("browse a file")]),
            style={"border": "2px dashed #3498db", "borderRadius": "10px",
                   "padding": "28px", "textAlign": "center", "cursor": "pointer"},
            accept=_ACCEPTED, multiple=False,
        ),
        html.Div(id="upload-status", className="text-center mt-3 fw-semibold"),
    ], style=_CARD))),

    html.Div(id="main-content", style={"display": "none"}, children=[
        dbc.Tabs([

            # ── Univariate ────────────────────────────────────────────────────
            dbc.Tab(label="Univariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Variable", className="mt-3"),
                        dcc.Dropdown(id="univariate-variable", clearable=False),
                        html.H5("Chart type", className="mt-3"),
                        dcc.RadioItems(id="plot-type", options=[], value=None,
                                       className="mt-1",
                                       inputStyle={"marginRight": "6px"}),
                        html.Hr(className="mt-3 mb-2"),
                        html.Div(id="univariate-normality",
                                 style={"overflowY": "auto", "maxHeight": "57vh"}),
                    ], width=3),
                    dbc.Col([
                        dcc.Graph(id="univariate-plot"),
                        html.Div(id="univariate-stats", style=_CARD),
                    ], width=9),
                ], className="mt-3"),
            ]),

            # ── Bivariate ─────────────────────────────────────────────────────
            dbc.Tab(label="Bivariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Variable 1", className="mt-3"),
                        dcc.Dropdown(id="bivariate-var1", clearable=False),
                        html.H5("Variable 2", className="mt-2"),
                        dcc.Dropdown(id="bivariate-var2", clearable=False),
                        html.Div(id="bivariate-pair-type", className="mt-2"),
                        html.Hr(className="mt-3 mb-2"),
                        html.Div(id="bivariate-tests",
                                 style={"overflowY": "auto", "maxHeight": "100vh"}),
                        html.Hr(className="mt-2 mb-2"),
                        html.Div(id="correlation-insights",
                                 style={"overflowY": "auto", "maxHeight": "100vh",
                                        "fontSize": "12px"}),
                    ], width=3),
                    dbc.Col([
                        dcc.Graph(id="bivariate-plot"),
                        html.Div(id="bivariate-stats", style=_CARD),
                        html.Div([
                            html.H5("Pairwise Correlation Matrix",
                                    className="mt-4 mb-2"),
                            dcc.Graph(id="pairwise-correlation"),
                        ]),
                    ], width=9),
                ], className="mt-3"),
            ]),

            # ── PCA ───────────────────────────────────────────────────────────
            dbc.Tab(label="PCA", children=[
                dbc.Row([

                    # Left sidebar
                    dbc.Col([
                        html.Div(id="pca-summary", style=_SIDEBAR),
                        html.Hr(style={"margin": "10px 0"}),

                        _pc_dd("pca-pc-x", "Axis X (horizontal)"),
                        _pc_dd("pca-pc-y", "Axis Y (vertical)"),

                        html.Hr(style={"margin": "8px 0"}),

                        html.Label("Axis insight",
                                   style={"fontSize": "12px", "fontWeight": "600",
                                          "color": "#2c3e50", "display": "block"}),
                        dcc.Dropdown(id="pca-pc-insight", clearable=False,
                                     style={"fontSize": "12px", "marginTop": "4px",
                                            "marginBottom": "8px"}),

                        html.Hr(style={"margin": "8px 0"}),

                        html.Label("Circle: min cos² filter",
                                   style={"fontSize": "11px", "color": "#6b7280",
                                          "display": "block"}),
                        dcc.Slider(id="pca-cos2-filter", min=0, max=0.9, step=0.1,
                                   value=0,
                                   marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
                                   tooltip={"placement": "bottom"}),

                        html.Hr(style={"margin": "8px 0"}),

                        html.Label("Individuals: annotate top-n",
                                   style={"fontSize": "11px", "color": "#6b7280",
                                          "display": "block"}),
                        dcc.Slider(id="pca-top-ind", min=0, max=20, step=5, value=0,
                                   marks={i: str(i) for i in [0, 5, 10, 15, 20]},
                                   tooltip={"placement": "bottom"}),

                        html.Hr(style={"margin": "8px 0"}),

                        html.Label("Individuals: colour by",
                                   style={"fontSize": "11px", "color": "#6b7280",
                                          "display": "block"}),
                        dcc.RadioItems(
                            id="pca-color-by",
                            options=[
                                {"label": " cos² (quality)",  "value": "cos2"},
                                {"label": " Contribution",    "value": "contrib"},
                                {"label": " None",            "value": "none"},
                            ],
                            value="cos2",
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"display": "block", "fontSize": "11px",
                                        "color": "#374151", "marginBottom": "2px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.Div(id="pca-axis-insight-panel",
                                 style={"overflowY": "auto", "maxHeight": "45vh",
                                        "fontSize": "11px"}),
                    ], width=3),

                    # Right main area — sub-tabs
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(label="Scree plot", children=[
                                dcc.Graph(id="pca-scree"),
                                html.Div(id="pca-eigenvalue-table-wrap",
                                         style=_CARD),
                            ]),
                            dbc.Tab(label="Correlation circle", children=[
                                dcc.Graph(id="pca-circle"),
                                html.Div(id="pca-circle-interp", style=_CARD),
                            ]),
                            dbc.Tab(label="Biplot", children=[
                                dcc.Graph(id="pca-biplot"),
                                html.Div(id="pca-biplot-interp", style=_CARD),
                            ]),
                            dbc.Tab(label="Individuals", children=[
                                dcc.Graph(id="pca-individuals"),
                                html.Div(id="pca-ind-insight-panel", style=_CARD),
                                html.Div(id="pca-ind-interp", style=_CARD),
                            ]),
                            dbc.Tab(label="Contributions", children=[
                                dcc.Graph(id="pca-contrib-bar"),
                                dcc.Graph(id="pca-contrib-heatmap"),
                                html.Div(id="pca-contrib-interp", style=_CARD),
                            ]),
                            dbc.Tab(label="cos² quality", children=[
                                dcc.Graph(id="pca-cos2-heatmap"),
                                html.Div(id="pca-cos2-interp", style=_CARD),
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
