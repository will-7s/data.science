import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Exploratory Data Analysis", className="text-center text-primary mb-4"),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Upload Data File", className="text-center"),
                html.P("CSV", className="text-center text-muted"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and drop or ', html.A('select file')]),
                    style={'border': '2px dashed #3498db', 'borderRadius': '10px', 
                           'padding': '30px', 'textAlign': 'center', 'marginBottom': '20px'},
                    multiple=False
                ),
                html.Div(id='upload-status', className="text-center mt-2")
            ], className="stats-card")
        ], width=12)
    ]),
    
    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Tabs([
            dbc.Tab(label="Univariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Select variable", className="mt-3"),
                        dcc.Dropdown(id='univariate-variable', clearable=False),
                        html.H4("Chart type", className="mt-3"),
                        dcc.RadioItems(id='plot-type', options=[
                            {'label': 'Histogram', 'value': 'histogram'},
                            {'label': 'Box Plot', 'value': 'box'},
                            {'label': 'Bar Chart', 'value': 'bar'}
                        ], value='histogram', inline=True),
                    ], width=3),
                    dbc.Col([
                        dcc.Graph(id='univariate-plot'),
                        html.Div(id='univariate-stats', className="stats-card mt-3")
                    ], width=9)
                ])
            ]),
            
            dbc.Tab(label="Bivariate Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Select Variables", className="mt-3"),
                        html.Label("Variable 1:"),
                        dcc.Dropdown(id='bivariate-var1', clearable=False),
                        html.Label("Variable 2:", className="mt-2"),
                        dcc.Dropdown(id='bivariate-var2', clearable=False),
                        html.Hr(),
                        html.Div(id='bivariate-tests', className="stats-card mt-2"),
                        html.Hr(),
                        html.Div(id='normality-tests', className="stats-card", style={
                            'overflowY': 'auto', 'height': '100vh'
                        })
                    ], width=3),
                    dbc.Col([
                        dcc.Graph(id='bivariate-plot'),
                        html.Div(id='bivariate-stats', className="stats-card mt-3"),
                        html.H4("Correlation Matrix", className="mt-3"),
                        dcc.Graph(id='pairwise-correlation'),
                        html.Div(id='correlation-insights', className="stats-card mt-2")
                    ], width=9)
                ])
            ])
        ])
    ])
], fluid=True)

register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=False)