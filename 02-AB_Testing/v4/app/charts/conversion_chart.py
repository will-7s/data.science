from __future__ import annotations

import plotly.graph_objects as go

from app.charts import _base_layout, _xaxis, _yaxis, theme


def build_conversion_bar_chart(stats: dict, dark: bool = False) -> go.Figure:
    # Barres comparatives des taux de conversion bruts.
    #
    # Interprétation :
    #   - L'écart entre les deux barres est l'effet brut en points de pourcentage (Δ).
    #   - Un Δ > 0 suggère que le Treatment est meilleur, mais ce n'est pas une preuve
    #     statistique — il faut croiser avec le p-value du test z et l'intervalle de
    #     crédibilité bayésien.
    #   - La hauteur absolue dépend du taux de base : un Δ de 1 pp est énorme si le
    #     taux de base est 1 %, mais négligeable s'il est 50 %.
    #   - La significativité statistique est traitée dans l'onglet "Statistical Tests".
    #
    # Usage métier :
    #   - Utiliser la « Difference » dans la KPI row pour l'impact business absolu.
    #   - Utiliser l'« Effect Size » (Cohen's h) pour l'impact relatif normalisé.
    c = theme(dark)
    c_rate = stats.get("control_rate", 0) * 100
    t_rate = stats.get("treatment_rate", 0) * 100
    diff = t_rate - c_rate

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Treatment"],
        y=[c_rate, t_rate],
        marker_color=[c["ctrl"], c["trt"]],
        marker_line=dict(width=0),
        text=[f"{c_rate:.1f}%", f"{t_rate:.1f}%"],
        textposition="outside",
        textfont=dict(color=c["text"], size=13, family="-apple-system, sans-serif"),
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        width=0.5,
    ))

    # Le Δ (delta) dans le titre montre la différence brute en points de %
    # entre Treatment et Control. Il s'agit ici de la métrique la plus lisible
    # pour un public non technique.
    fig.update_layout(
        title=dict(
            text=f"Conversion Rates  —  Δ = {diff:+.1f} pp",
            font=dict(size=14, color=c["text"], family="-apple-system, sans-serif"),
            x=0.02,
        ),
        **_base_layout(dark=dark, hovermode="x", showlegend=False),
        yaxis=dict(
            **_yaxis(dark=dark, title="Conversion rate (%)"),
            tickformat=".1f",
            range=[0, max(8, max(c_rate, t_rate) * 1.35)],
        ),
        xaxis=dict(**_xaxis(dark=dark, title="")),
    )
    return fig
