from __future__ import annotations

import plotly.graph_objects as go

from app.charts import _base_layout, _xaxis, theme


def _wilson_ci(n: int, p: float, z: float = 1.96) -> tuple[float, float]:
    # Intervalle de confiance de Wilson pour une proportion.
    # Préféré à l'intervalle de Wald (normal) car il reste dans [0, 1] et
    # fonctionne même avec des proportions extrêmes ou de petits échantillons.
    if n == 0:
        return (0.0, 0.0)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return (centre - margin, centre + margin)


def build_ci_error_plot(stats: dict, dark: bool = False) -> go.Figure:
    # Intervalles de confiance à 95 % (Wilson) pour chaque groupe.
    #
    # Interprétation :
    #   - Des intervalles qui ne se chevauchent PAS suggèrent une différence
    #     statistiquement significative au seuil α = 5 %.
    #   - ATTENTION : des intervalles qui se chevauchent ne garantissent PAS
    #     l'absence de significativité — le test z ou le test t est plus puissant.
    #   - L'intervalle de Wilson est asymétrique : il se resserre du côté où la
    #     proportion s'éloigne de 50 %.
    #
    # Usage avancé :
    #   - Comparer la largeur des intervalles : un intervalle large indique une
    #     grande incertitude (petit échantillon ou forte variance).
    #   - Pour une vision bayésienne de l'incertitude, voir les distributions
    #     a posteriori dans l'onglet Bayesian.
    c = theme(dark)
    c_rate = stats.get("control_rate", 0)
    t_rate = stats.get("treatment_rate", 0)
    c_n = stats.get("n_control", 0)
    t_n = stats.get("n_treatment", 0)

    c_lo, c_hi = _wilson_ci(c_n, c_rate)
    t_lo, t_hi = _wilson_ci(t_n, t_rate)

    fig = go.Figure()
    for rate, lo, hi, label, color in [
        (c_rate, c_lo, c_hi, "Control", c["ctrl"]),
        (t_rate, t_lo, t_hi, "Treatment", c["trt"]),
    ]:
        fig.add_trace(go.Scatter(
            x=[rate],
            y=[label],
            error_x=dict(
                type="data", symmetric=False,
                array=[hi - rate], arrayminus=[rate - lo],
                color=color, thickness=2.5, width=14,
            ),
            marker=dict(color=color, size=14, line=dict(color="white", width=2)),
            mode="markers",
            name=label,
            hovertemplate=f"{label}<br>%{{x:.4f}}<br>CI: [{lo:.4f}, {hi:.4f}]<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Conversion Rate — 95% Wilson CI",
            font=dict(size=14, color=c["text"], family="-apple-system, sans-serif"),
            x=0.02,
        ),
        **_base_layout(dark=dark, height=200, hovermode="y unified"),
        xaxis=dict(
            **_xaxis(dark=dark, title="Conversion rate"),
            tickformat=".0%",
            range=[0, 1],
        ),
        yaxis=dict(title="", autorange="reversed"),
    )
    return fig
