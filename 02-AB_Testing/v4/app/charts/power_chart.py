from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.special import ndtr
from scipy.stats import norm

from app.charts import _base_layout, _xaxis, _yaxis, fallback_figure, theme


def build_power_curve(power: dict, n_current: int, dark: bool = False) -> go.Figure:
    # Courbe de puissance statistique en fonction de la taille d'échantillon.
    #
    # Interprétation :
    #   - La puissance = probabilité de détecter un effet réel (vrai positif).
    #   - La ligne verte pointillée à 80 % est le seuil conventionnel minimal
    #     pour un test bien dimensionné.
    #   - La ligne rouge pointillée marque l'effectif actuel : sa hauteur sur
    #     la courbe est la « puissance observée ».
    #   - Si la puissance observée < 80 %, le test risque de manquer un effet
    #     réel (faux négatif / erreur de type II).
    #   - La ligne verte verticale (n*) = taille d'échantillon nécessaire par
    #     groupe pour atteindre 80 % de puissance.
    #
    # Usage décisionnel :
    #   - Si n* est proche de l'effectif actuel → envisagez de prolonger le test.
    #   - Si n* est très grand → l'effet est trop faible pour être détecté
    #     avec les moyens actuels (effet n'est peut-être pas pertinent
    #     business).
    #   - Si la puissance observée est déjà > 80 % et que le test n'est pas
    #     significatif → l'effet est probablement négligeable.
    c = theme(dark)

    if power.get("skipped", True):
        return go.Figure(fallback_figure("Power analysis skipped — effect size too small", dark))

    alpha = power.get("alpha", 0.05)
    es = power.get("_effect_size")
    if es is None:
        mde_h = power.get("mde_cohens_h", 0.3)
        es = mde_h

    # Grille d'échantillons pour tracer la courbe : de n/2 à min(n×3, 5000)
    # avec un pas adaptatif (200 points max) pour un rendu fluide.
    n_max = max(n_current * 3, 5000)
    n_vals = np.arange(max(10, n_current // 2), n_max + 1, step=max(1, n_max // 200)).astype(int)

    z_crit = float(norm.ppf(1 - alpha / 2))
    ncp = es * np.sqrt(n_vals / 2)
    pwr_curve = np.clip(1 - ndtr(z_crit - ncp) + ndtr(-z_crit - ncp), 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_vals, y=pwr_curve * 100,
        mode="lines", name="Power",
        line=dict(color=c["ctrl"], width=2.5),
        fill="tozeroy", fillcolor=c["ctrl_fill"],
        hovertemplate="n = %{x}<br>Power: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=80, line_dash="dash", line_color=c["success"],
        opacity=0.7, annotation_text="80% target",
        annotation_position="left",
        annotation_font=dict(color=c["success"], size=11, family="-apple-system, sans-serif"),
    )

    fig.add_vline(
        x=n_current, line_dash="dot", line_color=c["danger"],
        opacity=0.7, annotation_text=f"n = {n_current}",
        annotation_position="top",
        annotation_font=dict(color=c["danger"], size=11, family="-apple-system, sans-serif"),
    )

    n_80 = power.get("n_needed_80pct", float("inf"))
    if np.isfinite(n_80) and n_80 > n_current:
        fig.add_vline(
            x=n_80, line_dash="dot", line_color=c["success"],
            opacity=0.5, annotation_text=f"n* = {int(n_80)}",
            annotation_position="bottom",
            annotation_font=dict(color=c["success"], size=11, family="-apple-system, sans-serif"),
        )

    fig.update_layout(
        title=dict(
            text="Power Curve — Power vs Sample Size",
            font=dict(size=14, color=c["text"], family="-apple-system, sans-serif"),
            x=0.02,
        ),
        **_base_layout(dark=dark, hovermode="x", showlegend=False),
        xaxis=dict(_xaxis(dark=dark, title="Sample size (per group)"), tickformat=",d"),
        yaxis=dict(_yaxis(dark=dark, title="Power (%)"), range=[0, 105]),
    )
    return fig
