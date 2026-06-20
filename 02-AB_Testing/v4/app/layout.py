from dash import dcc, html

from app.theme import get_theme_store, get_theme_toggle

STORAGE_KEY = "ab-testing-theme"

TAB_IDS = {
    "overview": "tab-overview",
    "tests": "tab-tests",
    "effects": "tab-effects",
    "bayesian": "tab-bayesian",
    "segments": "tab-segments",
    "power": "tab-power",
    "export": "tab-export",
}

TAB_LABELS = {
    "overview": "Overview",
    "tests": "Statistical Tests",
    "effects": "Effect Sizes",
    "bayesian": "Bayesian",
    "segments": "Segments",
    "power": "Power & Robustness",
    "export": "Export",
}


def _build_header() -> html.Header:
    return html.Header(
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
            "gap": "12px",
            "padding": "12px 0",
        },
        children=[
            html.H1("AB Testing Analysis", className="text-heading", style={"margin": 0}),
            get_theme_toggle(),
        ],
    )


def _build_sidebar() -> html.Div:
    return html.Div(
        id="sidebar",
        className="sidebar",
        children=[
            html.H3("Data & Configuration", style={"marginTop": 0, "fontSize": "1rem"}),

            dcc.Upload(
                id="file-upload",
                children=html.Div([
                    html.Div("Drop your dataset here", style={"fontWeight": 600, "fontSize": "0.9rem"}),
                    html.Div("or click to browse", style={"fontSize": "0.8rem", "color": "var(--text-secondary)", "marginTop": "4px"}),
                    html.Div("CSV, Excel, Parquet or JSON", style={"fontSize": "0.75rem", "color": "var(--text-tertiary)", "marginTop": "2px"}),
                ]),
                className="upload-zone",
                multiple=False,
                accept=".csv,.xlsx,.json,.parquet",
            ),
            html.Div(id="load-feedback", style={"marginTop": "10px", "fontSize": "0.8rem"}),

            html.Hr(style={"margin": "16px 0", "borderColor": "var(--border-1)"}),

            html.Div(id="column-mapping-area", children=[
                html.Label("Target (conversion)", style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block"}),
                dcc.Dropdown(id="target-col-dropdown", placeholder="Select column...", style={"marginTop": "4px", "fontSize": "0.85rem"}),
                html.Label("Group column", style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block", "marginTop": "10px"}),
                dcc.Dropdown(id="group-col-dropdown", placeholder="Select column...", style={"marginTop": "4px", "fontSize": "0.85rem"}),
                html.Label("Control value", style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block", "marginTop": "10px"}),
                dcc.Dropdown(id="control-value-input", placeholder="Select control value...", style={"marginTop": "4px", "fontSize": "0.85rem"}),

                html.Details([
                    html.Summary("Advanced options", style={"cursor": "pointer", "fontSize": "0.85rem", "color": "var(--color-primary)", "marginTop": "12px"}),
                    html.Div(style={"marginTop": "8px"}, children=[
                        html.Label("Covariates", style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block", "marginTop": "10px"}),
                        dcc.Dropdown(id="covariate-multi", multi=True, placeholder="Optional", style={"marginTop": "4px", "fontSize": "0.85rem"}),
                        html.Label("Time column", style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block", "marginTop": "10px"}),
                        dcc.Dropdown(id="time-col-dropdown", placeholder="(None)", style={"marginTop": "4px", "fontSize": "0.85rem"}),
                        html.Label("ID column", style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block", "marginTop": "10px"}),
                        dcc.Dropdown(id="id-col-dropdown", placeholder="(None)", style={"marginTop": "4px", "fontSize": "0.85rem"}),
                    ]),
                ], style={"marginTop": "8px"}),
            ]),

            html.Hr(style={"margin": "16px 0", "borderColor": "var(--border-1)"}),
            html.Label("Significance level \u03b1",
                       style={"fontSize": "0.85rem", "fontWeight": 600, "display": "block"}),
            dcc.Slider(
                id="alpha-slider",
                min=0.01, max=0.20, step=0.01, value=0.05,
                marks={0.01: "0.01", 0.05: "0.05", 0.10: "0.10", 0.20: "0.20"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Button(
                "Run Analysis", id="run-analysis-btn",
                className="run-btn",
            ),
            dcc.Loading(
                id="analysis-loading",
                type="circle",
                parent_className="loading-wrapper",
                children=html.Div(id="run-progress", style={"marginTop": "8px", "minHeight": "40px"}),
            ),
        ],
    )


def _build_pre_run_area() -> html.Div:
    return html.Div(
        id="pre-run-area",
        children=[
            html.Div(id="data-preview", style={"display": "none"}),
        ],
    )


def _build_tab(label: str, tab_id: str) -> dcc.Tab:
    return dcc.Tab(
        label=label, value=tab_id,
        selected_style={"fontWeight": 600, "borderTop": "2px solid var(--color-primary)"},
    )


def _build_results_area() -> html.Div:
    tabs = [_build_tab(TAB_LABELS[k], v) for k, v in TAB_IDS.items()]
    return html.Div(
        id="results-area",
        style={"display": "none"},
        children=[
            html.Div(id="schema-banner"),
            html.Div(id="verdict-banner"),
            html.Div(id="kpi-row", className="kpi-row"),
            dcc.Tabs(id="result-tabs", value=TAB_IDS["overview"], children=tabs),
            html.Div(id="tab-content-container"),
            html.Div(id="loading-overlay"),
            dcc.Download(id="export-download"),
            html.Div(id="export-hidden", style={"display": "none"}),
        ],
    )


def create_layout() -> html.Div:
    return html.Div(
        id="app-root",
        style={
            "display": "flex", "gap": "24px",
            "maxWidth": "1600px", "margin": "0 auto",
            "padding": "16px 24px",
        },
        children=[
            get_theme_store(),
            dcc.Store(id="store-analysis", storage_type="memory"),

            html.Div(
                style={"flex": "1"},
                children=[
                    _build_header(),
                    html.Div(
                        style={"display": "flex", "gap": "24px", "marginTop": "16px"},
                        children=[
                            _build_sidebar(),
                            html.Div(
                                style={"flex": "1", "minWidth": 0},
                                children=[
                                    _build_pre_run_area(),
                                    _build_results_area(),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
