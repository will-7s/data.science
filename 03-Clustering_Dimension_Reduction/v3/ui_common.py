"""
ui_common.py  —  Shared Dash/HTML component helpers.

FIX (v6.1): extracted from ui_pca.py and ui_clustering.py where ~120 lines
of identical helper functions (_table, _row, _section, _interp_header,
_tip_box, _reading_grid) were duplicated verbatim.  Both modules now
import from here.

All functions are pure — they take data and return Dash component trees.
No store access, no callbacks.
"""
from __future__ import annotations
from dash import html


# ── Table helpers ─────────────────────────────────────────────────────────────

def _row(cells, highlight_col: int | None = None) -> html.Tr:
    tds = []
    for i, c in enumerate(cells):
        style = {"fontWeight": 600} if i == highlight_col else {}
        tds.append(html.Td(c, style=style))
    return html.Tr(tds)


def _table(headers: list, rows: list, highlight_col: int | None = None) -> html.Table:
    thead = html.Thead(html.Tr([html.Th(h) for h in headers]))
    tbody = html.Tbody([_row(r, highlight_col) for r in rows])
    return html.Table(
        [thead, tbody],
        className="interp-table",
    )


def _section(title: str, children) -> html.Div:
    return html.Div([
        html.Div(title, className="interp-section-title"),
        *children,
    ], style={"marginBottom": 12})


# ── Interpretation layout components ─────────────────────────────────────────

def _interp_header(title: str, subtitle: str = "") -> html.Div:
    children = [html.Div(title, className="interp-title")]
    if subtitle:
        children.append(html.Div(subtitle, className="interp-subtitle"))
    return html.Div(children, className="interp-header")


def _tip_box(text: str, kind: str = "info") -> html.Div:
    """
    kind: "info" | "warn" | "expert" | "success"
    """
    icons = {
        "info":    "💡",
        "warn":    "⚠️",
        "expert":  "🎓",
        "success": "✅",
    }
    classes = {
        "info":    "tip-box tip-info",
        "warn":    "tip-box tip-warn",
        "expert":  "tip-box tip-expert",
        "success": "tip-box tip-success",
    }
    icon = icons.get(kind, "💡")
    cls  = classes.get(kind, "tip-box tip-info")
    return html.Div(
        [html.Span(icon, className="tip-icon"), html.Span(text)],
        className=cls,
    )


def _reading_grid(items: list[tuple[str, str, str]]) -> html.Div:
    """
    items: list of (icon, title, description)
    """
    cards = []
    for icon, title, desc in items:
        cards.append(html.Div([
            html.Div([
                html.Span(icon, className="grid-icon"),
                html.Span(title, className="grid-title"),
            ], className="grid-card-header"),
            html.Div(desc, className="grid-card-desc"),
        ], className="grid-card"))
    return html.Div(cards, className="reading-grid")
