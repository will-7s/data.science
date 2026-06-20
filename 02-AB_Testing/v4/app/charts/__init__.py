from __future__ import annotations

# Police système optimisée pour le rendu Dash/Plotly — graisse linéaire
# pour éviter les sauts de layout au chargement des polices web.
_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"

# Palette « Control » (bleu) vs « Treatment » (orange) — choisie pour
# un contraste maximal même en daltonisme (bleu/orange = safe pair).
_LIGHT = {
    "ctrl": "#3B82F6",
    "trt": "#F97316",
    "success": "#10B981",
    "danger": "#EF4444",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "text": "#374151",
    "bg_plot": "rgba(0,0,0,0)",
    "bg_paper": "rgba(0,0,0,0)",
    "ctrl_fill": "rgba(59,130,246,0.10)",
    "trt_fill": "rgba(249,115,22,0.10)",
}

_DARK = {
    "ctrl": "#60A5FA",
    "trt": "#FB923C",
    "success": "#34D399",
    "danger": "#F87171",
    "muted": "#9CA3AF",
    "grid": "#374151",
    "text": "#D1D5DB",
    "bg_plot": "rgba(0,0,0,0)",
    "bg_paper": "rgba(0,0,0,0)",
    "ctrl_fill": "rgba(96,165,250,0.12)",
    "trt_fill": "rgba(251,146,60,0.12)",
}


def theme(dark: bool = False) -> dict:
    return _DARK if dark else _LIGHT


def _base_layout(**overrides) -> dict:
    # Layout commun à tous les graphiques Plotly.
    # - hovermode "x unified" : compare les deux groupes au même point.
    # - itemclick=False : évite de cacher une trace involontairement.
    # - Légende horizontale au-dessus du graphique pour ne pas rogner la zone de données.
    c = theme(overrides.pop("dark", False))
    base = dict(
        font=dict(family=_FONT, size=12, color=c["text"]),
        plot_bgcolor=c["bg_plot"],
        paper_bgcolor=c["bg_paper"],
        margin=dict(l=48, r=24, t=56, b=48),
        hovermode="x unified",
        hoverlabel=dict(
            font=dict(family=_FONT, size=12),
            bordercolor=c["grid"],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family=_FONT, size=11),
            itemclick=False, itemdoubleclick=False,
        ),
    )
    base.update(overrides)
    return base


def _xaxis(**overrides) -> dict:
    # Axe X standard — grille subtile, pas de ligne d'axe, titre configurable.
    c = theme(overrides.pop("dark", False))
    axis = dict(
        gridcolor=c["grid"], zeroline=False, showline=False,
        tickfont=dict(family=_FONT, size=11, color=c["muted"]),
        title=dict(font=dict(family=_FONT, size=12, color=c["text"])),
    )
    axis.update(overrides)
    return axis


def _yaxis(**overrides) -> dict:
    # Axe Y standard — mêmes conventions que X.
    c = theme(overrides.pop("dark", False))
    axis = dict(
        gridcolor=c["grid"], zeroline=False, showline=False,
        tickfont=dict(family=_FONT, size=11, color=c["muted"]),
        title=dict(font=dict(family=_FONT, size=12, color=c["text"])),
    )
    axis.update(overrides)
    return axis


def fallback_figure(msg: str = "No data available", dark: bool = False) -> dict:
    # Figure de repli quand les données sont absentes — évite un graphique vide
    # qui pourrait être interprété à tort comme « aucun effet ».
    c = theme(dark)
    return dict(
        data=[],
        layout=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text=msg, showarrow=False,
                font=dict(family=_FONT, size=13, color=c["muted"]),
                x=0.5, y=0.5, xref="paper", yref="paper",
            )],
            plot_bgcolor=c["bg_plot"],
            paper_bgcolor=c["bg_paper"],
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
        ),
    )
