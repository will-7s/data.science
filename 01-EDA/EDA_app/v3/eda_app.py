import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import callbacks
from parsers import SUPPORTED_EXTENSIONS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
_ACCEPTED = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTENSIONS))

_CARD = {
    'border': '1px solid #dee2e6',
    'borderRadius': '8px',
    'padding': '16px',
    'marginTop': '12px',
    'background': '#ffffff',
}

app.layout = dbc.Container([
    dbc.Row(dbc.Col([
        html.H1("Exploratory Data Analysis", className="text-center text-primary mb-2"),
        html.Hr(),
    ])),
    dbc.Row(dbc.Col(html.Div([
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
    ], style=_CARD))),
    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Tabs([
            dbc.Tab(label="Univariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Variable", className="mt-3"),
                        dcc.Dropdown(id='univariate-variable', clearable=False),
                        html.H5("Chart type", className="mt-3"),
                        dcc.RadioItems(
                            id='plot-type',
                            options=[],
                            value=None,
                            className="mt-1",
                            inputStyle={"marginRight": "6px"},
                        ),
                        html.Hr(className="mt-3 mb-2"),
                        html.Div(id='univariate-normality', style={'overflowY': 'auto', 'maxHeight': '60vh'}),
                    ], width=3),
                    dbc.Col([
                        dcc.Graph(id='univariate-plot'),
                        html.Div(id='univariate-stats', style=_CARD),
                    ], width=9),
                ], className="mt-3"),
            ]),
            dbc.Tab(label="Bivariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Variable 1", className="mt-3"),
                        dcc.Dropdown(id='bivariate-var1', clearable=False),
                        html.H5("Variable 2", className="mt-2"),
                        dcc.Dropdown(id='bivariate-var2', clearable=False),
                        html.Div(id='bivariate-pair-type', className="mt-2"),
                        html.Hr(className="mt-3 mb-2"),
                        html.Div(id='bivariate-tests', style={'overflowY': 'auto', 'maxHeight': '55vh'}),
                        html.Hr(className="mt-2 mb-2"),
                        html.Div(id='normality-tests', style={'overflowY': 'auto', 'maxHeight': '60vh', 'fontSize': '12px'}),
                    ], width=3),
                    dbc.Col([
                        dcc.Graph(id='bivariate-plot'),
                        html.Div(id='bivariate-stats', style=_CARD),
                        html.Div([
                            html.H5("Pairwise Correlation Matrix", className="mt-4 mb-2"),
                            dcc.Graph(id='pairwise-correlation'),
                            html.Div(id='correlation-insights', style=_CARD),
                        ], id='corr-section'),
                    ], width=9),
                ], className="mt-3"),
            ]),
        ])
    ]),
], fluid=True, className="py-4")

callbacks.register(app)

if __name__ == '__main__':
    app.run(debug=False)