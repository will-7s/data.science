import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import callbacks, ui
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

_ = app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return dash_clientside.no_update;
        var gd = document.querySelector('#univariate-plot .js-plotly-plot');
        if (!gd) return dash_clientside.no_update;
        var ts = new Date().toISOString().slice(0,19).replace(/[:-]/g, '');
        Plotly.downloadImage(gd, {format:'png', width:1200, height:600, scale:2, filename:'univariate_'+ts});
        return dash_clientside.no_update;
    }
    """,
    dash.Output("uni-png-trigger", "children"),
    dash.Input("uni-export-png", "n_clicks"),
    prevent_initial_call=True,
)

_ = app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return dash_clientside.no_update;
        var gd = document.querySelector('#bivariate-plot .js-plotly-plot');
        if (!gd) return dash_clientside.no_update;
        var ts = new Date().toISOString().slice(0,19).replace(/[:-]/g, '');
        Plotly.downloadImage(gd, {format:'png', width:1200, height:600, scale:2, filename:'bivariate_'+ts});
        return dash_clientside.no_update;
    }
    """,
    dash.Output("bi-png-trigger", "children"),
    dash.Input("bi-export-png", "n_clicks"),
    prevent_initial_call=True,
)

_ = app.clientside_callback(
    """
    function(enter) {
        if (!window._fsHandler) {
            window._fsHandler = function() {
                if (!document.fullscreenElement) {
                    document.body.classList.remove('presentation-mode');
                    dash_clientside.set_props('fullscreen-mode', {data: false});
                }
            };
            document.addEventListener('fullscreenchange', window._fsHandler);
        }
        if (enter) {
            document.documentElement.requestFullscreen();
            document.body.classList.add('presentation-mode');
        } else {
            if (document.fullscreenElement) {
                document.exitFullscreen();
            }
            document.body.classList.remove('presentation-mode');
        }
        return dash_clientside.no_update;
    }
    """,
    dash.Output("dummy-output", "children"),
    dash.Input("fullscreen-mode", "data"),
    prevent_initial_call=True,
)

_ACCEPTED = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTENSIONS))

_GRAPH_CFG = {"responsive": False, "displaylogo": False,
              "modeBarButtonsToRemove": ["lasso2d", "select2d"],
              "toImageButtonOptions": {"format": "png", "scale": 2}}


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
            html.Div([
                html.Button("Present", id="present-btn",
                            style={"fontSize": "0.75rem", "padding": "4px 12px",
                                   "borderRadius": "var(--radius-sm)",
                                   "border": "1px solid var(--border)",
                                   "background": "transparent",
                                   "color": "var(--text-muted)", "cursor": "pointer",
                                   "marginRight": 8}),
            ]),
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
        html.Div(id="upload-status", className="upload-status"),
    ], style={"background": "var(--bg-card)", "borderRadius": "var(--radius-lg)",
              "padding": 24, "boxShadow": "var(--shadow-sm)", "border": "1px solid var(--border)"}))),

    html.Div(id="overview-content", style={"display": "none"}),

    html.Div(id="main-content", style={"display": "none"}, children=[
        dcc.Store(id="computation-state", data="idle"),
        dcc.Store(id="panel-collapsed", data={"univariate": False, "bivariate": False}),
        dcc.Store(id="fullscreen-mode", data=False),
        dcc.Download(id="download-csv"),
        html.Div(id="dummy-output", style={"display": "none"}),
        html.Div(id="uni-png-trigger", style={"display": "none"}),
        html.Div(id="bi-png-trigger", style={"display": "none"}),
        html.Div(id="cancel-status", style={"display": "none"}),
        ui.build_cancel_button(),
        dbc.Tabs([
            dbc.Tab(label="Univariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("◀", id="uni-collapse-btn", size="sm",
                                       style={"float": "right", "marginTop": -4,
                                              "padding": "2px 8px", "fontSize": "0.75rem",
                                              "background": "transparent",
                                              "color": "var(--text-muted)",
                                              "border": "1px solid var(--border)"}),
                        ], style={"display": "flex", "justifyContent": "flex-end",
                                  "marginBottom": 4}),
                        html.H5("Variable", className="mt-0",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.Dropdown(id="univariate-variable", clearable=False,
                                     style={"fontSize": "0.8125rem"}),
                        html.H5("Chart type", className="mt-3",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.RadioItems(id="plot-type", options=[], value=None,
                                       className="mt-1",
                                       inputStyle={"marginRight": 6}),
                        html.Hr(className="section-divider"),
                        html.Div(id="univariate-normality",
                                 className="sidebar-scroll"),
                        html.Div(id="uni-norm-badge"),
                    ], id="uni-sidebar", width=3),
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="univariate-plot", config=_GRAPH_CFG, style={"flex": 1}),
                            html.Div([
                                html.Button("PNG", id="uni-export-png", className="export-btn",
                                            title="Export graph as PNG"),
                                html.Button("CSV", id="uni-export-csv", className="export-btn",
                                            title="Export stats as CSV",
                                            style={"marginLeft": 6}),
                            ], style={"display": "flex", "justifyContent": "flex-end",
                                      "marginTop": 6}),
                        ], style={"display": "flex", "flexDirection": "column"}),
                        html.Div(id="univariate-stats",
                                 style={"background": "var(--bg-card)",
                                        "border": "1px solid var(--border)",
                                        "borderRadius": "var(--radius-lg)",
                                        "padding": 20, "marginTop": 16,
                                        "boxShadow": "var(--shadow-sm)"}),
                    ], id="uni-graph", width=9),
                ], className="mt-3"),
            ]),
            dbc.Tab(label="Bivariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("◀", id="bi-collapse-btn", size="sm",
                                       style={"float": "right", "marginTop": -4,
                                              "padding": "2px 8px", "fontSize": "0.75rem",
                                              "background": "transparent",
                                              "color": "var(--text-muted)",
                                              "border": "1px solid var(--border)"}),
                        ], style={"display": "flex", "justifyContent": "flex-end",
                                  "marginBottom": 4}),
                        html.H5("Variable 1", className="mt-0",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.Dropdown(id="bivariate-var1", clearable=True,
                                     style={"fontSize": "0.8125rem"}),
                        html.H5("Variable 2", className="mt-2",
                                style={"fontSize": "0.875rem", "fontWeight": 600}),
                        dcc.Dropdown(id="bivariate-var2", clearable=True,
                                     style={"fontSize": "0.8125rem"}),
                        html.Div(id="bivariate-pair-type"),
                        html.Hr(className="section-divider"),
                        html.Div(id="bivariate-tests",
                                 style={"overflowY": "auto",
                                        "maxHeight": "100vh"}),
                        html.Hr(className="section-divider"),
                        html.Div(id="correlation-insights",
                                 style={"overflowY": "auto",
                                        "maxHeight": "100vh",
                                        "fontSize": "0.75rem"}),
                    ], id="bi-sidebar", width=3),
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="bivariate-plot", config=_GRAPH_CFG, style={"flex": 1}),
                            html.Div([
                                html.Button("PNG", id="bi-export-png", className="export-btn",
                                            title="Export graph as PNG"),
                                html.Button("CSV", id="bi-export-csv", className="export-btn",
                                            title="Export stats as CSV",
                                            style={"marginLeft": 6}),
                            ], style={"display": "flex", "justifyContent": "flex-end",
                                      "marginTop": 6}),
                        ], style={"display": "flex", "flexDirection": "column"}),
                        html.Div(id="bivariate-stats",
                                 style={"background": "var(--bg-card)",
                                        "border": "1px solid var(--border)",
                                        "borderRadius": "var(--radius-lg)",
                                        "padding": 20, "marginTop": 16,
                                        "boxShadow": "var(--shadow-sm)"}),
                        html.Div(id="lightweight-badge"),
                        html.Div([
                            html.H5("Pairwise Correlation Matrix",
                                    className="mt-4 mb-2",
                                    style={"fontSize": "0.875rem", "fontWeight": 600}),
                            dcc.Graph(id="pairwise-correlation", config=_GRAPH_CFG),
                        ]),
                    ], id="bi-graph", width=9),
                ], className="mt-3"),
            ]),
        ])
    ]),

], fluid=True, className="py-4")

# Pre-warm lazy imports so the first callback isn't slowed by scipy/plotly load
import stats, charts
stats._sp()
charts._go()
charts._px()

callbacks.register(app)

if __name__ == "__main__":
    app.run(debug=False)
