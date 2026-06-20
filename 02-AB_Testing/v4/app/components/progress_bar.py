from dash import html


def build_progress_area(phase: str, pct: int, status: str) -> html.Div:
    if status == "done":
        return html.Div(
            **{"aria-live": "polite"},
            children="Analysis complete.",
            style={"position": "absolute", "width": "1px", "height": "1px", "overflow": "hidden", "clip": "rect(0,0,0,0)", "border": "0"},
        )

    if status == "idle":
        return html.Div(style={"display": "none"})

    if status == "running":
        return html.Div(
            className="progress-area",
            style={"marginTop": "12px"},
            **{"aria-live": "polite"},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    children=[
                        html.Div(
                            className="progress-bar",
                            style={"flex": "1"},
                            children=[
                                html.Div(className="progress-bar-fill", style={"width": f"{max(0, min(100, pct))}%"}),
                            ],
                        ),
                        html.Button("Cancel", id="cancel-analysis-btn", className="cancel-btn"),
                    ],
                ),
                html.Div(
                    f"Computing {phase} ({pct}%)...",
                    className="text-sm",
                    style={"marginTop": "4px", "color": "var(--text-secondary)"},
                ),
            ],
        )

    return html.Div(
        className="progress-area",
        style={"marginTop": "12px"},
        role="alert",
        children=[
            html.Div(status, className="text-sm", style={"color": "var(--color-danger)"}),
        ],
    )


def build_cancelled_message() -> html.Div:
    return html.Div(
        "Cancelled — adjust your selection and re-run",
        className="text-lg",
        style={"textAlign": "center", "color": "var(--color-warning)", "padding": "24px"},
    )
