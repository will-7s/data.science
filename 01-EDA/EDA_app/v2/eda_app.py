"""
eda_app.py
──────
Application entry point.

Responsibilities
----------------
- Create the Dash app and declare the layout.
- Call callbacks.register() to attach all interactivity.
- Expose `server` for WSGI deployment (gunicorn, uWSGI, …).

Nothing else belongs here.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import callbacks
from parsers import SUPPORTED_EXTENSIONS

# ── app setup ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server   # for WSGI / deployment

_ACCEPTED = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTENSIONS))

# ── layout ────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([

    # ── header ───────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col([
        html.H1("Exploratory Data Analysis", className="text-center text-primary mb-2"),
        html.Hr(),
    ])),

    # ── upload card ───────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(
        html.Div([
            html.H4("Upload a dataset", className="text-center"),
            html.P(_ACCEPTED, className="text-center text-muted small"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag & drop or ', html.A('browse a file')]),
                style={
                    'border': '2px dashed #3498db',
                    'borderRadius': '10px',
                    'padding': '28px',
                    'textAlign': 'center',
                    'cursor': 'pointer',
                },
                accept=_ACCEPTED,
                multiple=False,
            ),
            html.Div(id='upload-status', className="text-center mt-3 fw-semibold"),
        ], className="stats-card p-4"),
    )),

    # ── main content (hidden until a file is loaded) ──────────────────────────
    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Tabs([

            # ── Univariate ───────────────────────────────────────────────────
            dbc.Tab(label="Univariate Analysis", children=[
                dbc.Row([

                    # controls
                    dbc.Col([
                        html.H5("Variable", className="mt-3"),
                        dcc.Dropdown(id='univariate-variable', clearable=False),
                        html.H5("Chart type", className="mt-3"),
                        dcc.RadioItems(
                            id='plot-type',
                            options=[
                                {'label': ' Histogram', 'value': 'histogram'},
                                {'label': ' Box plot',  'value': 'box'},
                                {'label': ' Bar chart', 'value': 'bar'},
                            ],
                            value='histogram',
                            className="mt-1",
                        ),
                    ], width=3),

                    # plot + stats
                    dbc.Col([
                        dcc.Graph(id='univariate-plot'),
                        html.Div(id='univariate-stats', className="stats-card mt-3 p-3"),
                    ], width=9),

                ], className="mt-3"),
            ]),

            # ── Bivariate ────────────────────────────────────────────────────
            dbc.Tab(label="Bivariate Analysis", children=[
                dbc.Row([

                    # controls + side panels
                    dbc.Col([
                        html.H5("Variable 1", className="mt-3"),
                        dcc.Dropdown(id='bivariate-var1', clearable=False),
                        html.H5("Variable 2", className="mt-2"),
                        dcc.Dropdown(id='bivariate-var2', clearable=False),
                        html.Hr(),
                        html.Div(id='bivariate-tests',
                                 className="stats-card p-2 mt-2"),
                        html.Hr(),
                        html.Div(id='normality-tests',
                                 className="stats-card p-2",
                                 style={'overflowY': 'auto', 'height': '100vh'}),
                    ], width=3),

                    # main chart + extras
                    dbc.Col([
                        dcc.Graph(id='bivariate-plot'),
                        html.Div(id='bivariate-stats', className="stats-card mt-3 p-3"),
                        html.H5("Correlation Matrix", className="mt-4"),
                        dcc.Graph(id='pairwise-correlation'),
                        html.Div(id='correlation-insights', className="stats-card mt-2 p-3"),
                    ], width=9),

                ], className="mt-3"),
            ]),

        ])
    ]),

], fluid=True, className="py-4")

# ── wire up interactivity ─────────────────────────────────────────────────────

callbacks.register(app)

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=False)
