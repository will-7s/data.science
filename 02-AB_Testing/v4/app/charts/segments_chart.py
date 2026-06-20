from __future__ import annotations

import plotly.graph_objects as go

from app.charts import _base_layout, _xaxis, _yaxis, fallback_figure, theme


def build_segment_bar_chart(segmentation: dict, dark: bool = False) -> go.Figure:
    # Graphique en barres groupées montrant les taux de conversion par segment.
    #
    # Interprétation :
    #   - Permet de détecter des effets de traitement hétérogènes :
    #     * Si le traitement fonctionne uniquement dans certains segments,
    #       envisagez un déploiement ciblé (targeting).
    #     * Si les signes des différences varient selon les segments, cela
    #       peut indiquer un paradoxe de Simpson.
    #   - Les astérisques (*) marquent les segments où la différence est
    #       significative après correction de Benjamini-Hochberg (FDR).
    #   - L'annotation "+X.Xpp" en haut de chaque paire de barres donne
    #       la différence brute en points de pourcentage.
    #
    # Pièges :
    #   - Comparer plusieurs segments augmente le risque de faux positifs.
    #     La correction FDR réduit ce risque mais diminue la puissance.
    #   - Les segments avec peu d'observations peuvent avoir des estimations
    #     instables — regardez la taille d'échantillon (n) de chaque segment.
    #   - N'interprétez jamais un sous-groupe isolément : croisez avec
    #     l'analyse globale et les tests d'interaction.
    c = theme(dark)
    fig = go.Figure()
    has_any = False

    for col_name, segments in segmentation.items():
        seg_items = sorted(segments.items())
        if not seg_items:
            continue

        labels = [str(s) for s, d in seg_items]
        c_rates = [d.get("control_rate", 0) * 100 for _, d in seg_items]
        t_rates = [d.get("treatment_rate", 0) * 100 for _, d in seg_items]

        has_any = True
        fig.add_trace(go.Bar(
            x=labels, y=c_rates,
            name=f"{col_name} / Control",
            marker_color=c["ctrl"],
            marker_line=dict(width=0),
            opacity=0.85,
            legendgroup=col_name,
            hovertemplate=f"{col_name}<br>Control: %{{y:.1f}}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=labels, y=t_rates,
            name=f"{col_name} / Treatment",
            marker_color=c["trt"],
            marker_line=dict(width=0),
            opacity=0.85,
            legendgroup=col_name,
            hovertemplate=f"{col_name}<br>Treatment: %{{y:.1f}}%<extra></extra>",
        ))

        for i, (_, d) in enumerate(seg_items):
            diff = d.get("treatment_rate", 0) * 100 - d.get("control_rate", 0) * 100
            if abs(diff) > 0.1:
                sig = d.get("significant", False)
                color = c["success"] if sig else c["muted"]
                fig.add_annotation(
                    x=labels[i],
                    y=max(c_rates[i], t_rates[i]) + 2,
                    text=f"{diff:+.1f}pp{'*' if sig else ''}",
                    showarrow=False,
                    font=dict(size=10, color=color, family="-apple-system, sans-serif"),
                )

    if not has_any:
        return go.Figure(fallback_figure("Segmentation data not available", dark))

    fig.update_layout(
        title=dict(
            text="Segmentation — Conversion Rates by Segment",
            font=dict(size=14, color=c["text"], family="-apple-system, sans-serif"),
            x=0.02,
        ),
        **_base_layout(dark=dark, barmode="group", margin=dict(l=48, r=24, t=56, b=72)),
        xaxis=dict(_xaxis(dark=dark, title="")),
        yaxis=dict(_yaxis(dark=dark, title="Conversion rate (%)"), tickformat=".1f"),
    )
    return fig
