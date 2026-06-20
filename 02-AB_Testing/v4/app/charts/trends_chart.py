from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app.charts import _base_layout, _xaxis, _yaxis, fallback_figure, theme


def build_trends_line_chart(
    temporal: dict,
    labels: tuple[str, str] | None = None,
    dark: bool = False,
) -> go.Figure:
    # Évolution quotidienne des taux de conversion pour chaque groupe.
    #
    # Interprétation :
    #   - Séparation stable et parallèle → effet fiable et constant dans le temps.
    #   - Convergence des courbes → possible « novelty effect » (l'effet
    #     s'estompe une fois que la nouveauté du traitement passe).
    #   - Divergence → l'effet du traitement se renforce avec le temps
    #     (apprentissage, effet réseau, etc.).
    #   - Motifs saisonniers (pics le week-end, creux en semaine) → intégrer
    #     des effets temporels dans le modèle (enriched logistic regression).
    #
    # Tests de tendance (Pearson r pondéré) :
    #   - r > 0 et p < 0.05 → tendance à la hausse significative.
    #   - La pondération par √n par jour donne plus de poids aux jours avec
    #     beaucoup d'observations.
    #
    # Limites :
    #   - Avec < 4 points temporels, le test de tendance n'est pas calculé.
    #   - Les premiers jours peuvent être instables (ramp-up).
    #   - Les jours fériés / événements ponctuels peuvent créer des artefacts.
    c = theme(dark)
    ctrl_label, trt_label = labels or ("Control", "Treatment")

    daily = temporal.get("daily_data")
    trends = temporal.get("trends", {})
    if daily is None or daily.empty:
        return go.Figure(fallback_figure("Temporal data not available", dark))

    fig = go.Figure()
    colors = {"0": c["ctrl"], "1": c["trt"]}
    names = {"0": ctrl_label, "1": trt_label}

    for grp_val in sorted(daily["group"].unique()):
        d = daily[daily["group"] == grp_val].sort_values("date")
        grp_str = str(grp_val)
        color = colors.get(grp_str, c["ctrl"])
        name = names.get(grp_str, grp_str)

        fig.add_trace(go.Scatter(
            x=d["date"],
            y=d["rate"] * 100,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color, line=dict(width=1, color="white")),
            hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
        ))

        trend = trends.get(grp_str, {})
        r_val = trend.get("pearson_r")
        if r_val is not None and not np.isnan(r_val):
            p_val = trend.get("p_value")
            label = f"r = {r_val:.3f}"
            if p_val is not None and not np.isnan(p_val):
                label += f", p = {p_val:.4f}"
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.02, y=0.95 - 0.07 * int(grp_str or 0),
                text=label, showarrow=False,
                font=dict(size=11, color=color, family="-apple-system, sans-serif"),
                align="left",
            )

    fig.update_layout(
        title=dict(
            text="Daily Conversion Trends",
            font=dict(size=14, color=c["text"], family="-apple-system, sans-serif"),
            x=0.02,
        ),
        **_base_layout(dark=dark, margin=dict(l=48, r=96, t=56, b=48)),
        xaxis=dict(_xaxis(dark=dark, title="Date")),
        yaxis=dict(_yaxis(dark=dark, title="Conversion rate (%)"), tickformat=".1f"),
    )
    return fig
