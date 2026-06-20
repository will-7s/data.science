from dash import Input, Output, State, clientside_callback, dcc, html

THEME_STORE_ID = "theme-store"
THEME_TOGGLE_ID = "theme-toggle"
STORAGE_KEY = "ab-testing-theme"

_THEME_CLIENTSIDE_CALLBACK = """
function(_, current) {
    var newTheme = "light";
    try {
        var currentTheme = document.documentElement.getAttribute("data-theme");
        if (currentTheme !== "light" && currentTheme !== "dark") {
            currentTheme = "light";
        }
        newTheme = currentTheme === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("ab-testing-theme", newTheme);
    } catch(e) {}
    return newTheme;
}
"""


def register_theme_clientside_callback() -> None:
    clientside_callback(
        _THEME_CLIENTSIDE_CALLBACK,
        Output(THEME_STORE_ID, "data"),
        Input(THEME_TOGGLE_ID, "n_clicks"),
        State(THEME_STORE_ID, "data"),
        prevent_initial_call=True,
    )


def get_theme_toggle() -> html.Button:
    return html.Button(
        id=THEME_TOGGLE_ID,
        className="theme-toggle",
        title="Toggle light/dark mode",
        n_clicks=0,
        **{"aria-label": "Toggle light/dark mode"},
    )


def get_theme_store() -> dcc.Store:
    return dcc.Store(id=THEME_STORE_ID, storage_type="memory")
