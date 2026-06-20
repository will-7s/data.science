from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

from app.charts import _base_layout, _xaxis, _yaxis, fallback_figure, theme


def build_posterior_plot(
    bayesian: dict,
    labels: tuple[str, str] | None = None,
    dark: bool = False,
) -> go.Figure:
    # Distributions a posteriori issues du modèle Beta-Binomial bayésien.
    #
    # Interprétation :
    #   - Chaque courbe représente l'éventail des valeurs plausibles du taux de
    #     conversion pour un groupe, compte tenu des données observées.
    #   - Le chevauchement entre les deux courbes mesure l'incertitude : plus il
    #     est faible, plus la preuve est forte qu'un groupe surpasse l'autre.
    #   - La métrique P(T > C) (= Probabilité que le Treatment soit meilleur que
    #     le Control) est directement dérivée de ces distributions. Un P(T > C)
    #     > 95 % est considéré comme une preuve forte.
    #   - La ROPE (Region Of Practical Equivalence) est une bande [-0.005, +0.005]
    #     autour de zéro : si la majeure partie de la distribution de la différence
    #     tombe dans la ROPE, les deux groupes sont pratiquement équivalents.
    #
    # Limites :
    #   - Ce graphique n'est disponible que pour les données binaires (conversion
    #     oui/non). Pour les données continues, utilisez le modèle Normal-Normal
    #     (cf. onglet Bayesian).
    #   - La forme des courbes dépend du prior (Beta(1,1) par défaut — uniforme).
    #     Avec peu de données, le prior domine ; avec beaucoup de données, il
    #     s'efface.
    c = theme(dark)
    ctrl_label, trt_label = labels or ("Control", "Treatment")

    alpha_a = bayesian.get("posterior_alpha_control")
    beta_a = bayesian.get("posterior_beta_control")
    alpha_b = bayesian.get("posterior_alpha_treatment")
    beta_b = bayesian.get("posterior_beta_treatment")

    if not all(isinstance(v, (int, float)) and v > 0 for v in [alpha_a, beta_a, alpha_b, beta_b]):
        return go.Figure(fallback_figure("Posterior distributions not available", dark))

    # Grille de 300 points sur [0, 1] — résolution suffisante pour une courbe
    # lisible sans alourdir le rendu Plotly.
    x = np.linspace(0, 1, 300)
    pdf_c = beta.pdf(x, alpha_a, beta_a)
    pdf_t = beta.pdf(x, alpha_b, beta_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pdf_c, mode="lines", name=ctrl_label,
        line=dict(color=c["ctrl"], width=2.5),
        fill="tozeroy", fillcolor=c["ctrl_fill"],
        hovertemplate=f"{ctrl_label}<br>Rate: %{{x:.3f}}<br>Density: %{{y:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pdf_t, mode="lines", name=trt_label,
        line=dict(color=c["trt"], width=2.5),
        fill="tozeroy", fillcolor=c["trt_fill"],
        hovertemplate=f"{trt_label}<br>Rate: %{{x:.3f}}<br>Density: %{{y:.2f}}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Posterior Distributions",
            font=dict(size=14, color=c["text"], family="-apple-system, sans-serif"),
            x=0.02,
        ),
        **_base_layout(dark=dark, height=280),
        xaxis=dict(_xaxis(dark=dark, title="Conversion rate"), tickformat=".0%"),
        yaxis=dict(_yaxis(dark=dark, title="Density")),
    )
    return fig
