from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from app.export_utils import build_summary_df, generate_text_report

CTRL_COLOR = "#3B82F6"
TRT_COLOR = "#F97316"
GRID_COLOR = "#E5E7EB"
TEXT_COLOR = "#374151"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _chart_layout(**overrides) -> dict:
    base = dict(
        font=dict(family=FONT, size=12, color=TEXT_COLOR),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=56, b=48),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT, size=12), bordercolor=GRID_COLOR),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family=FONT, size=11),
            itemclick=False, itemdoubleclick=False,
        ),
    )
    base.update(overrides)
    return base


def _xaxis(**overrides) -> dict:
    axis = dict(
        gridcolor=GRID_COLOR, zeroline=False, showline=False,
        tickfont=dict(family=FONT, size=11, color="#6B7280"),
        title=dict(font=dict(family=FONT, size=12, color=TEXT_COLOR)),
    )
    axis.update(overrides)
    return axis


def _yaxis(**overrides) -> dict:
    axis = dict(
        gridcolor=GRID_COLOR, zeroline=False, showline=False,
        tickfont=dict(family=FONT, size=11, color="#6B7280"),
        title=dict(font=dict(family=FONT, size=12, color=TEXT_COLOR)),
    )
    axis.update(overrides)
    return axis


def _metric_card(label: str, value: str, subtitle: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Div(value, className="metric-value"),
            html.Div(label, className="metric-label"),
            html.Div(subtitle or "", className="metric-subtitle"),
        ],
        className="metric-card",
    )


def _info_box(text: str) -> html.Div:
    return html.Div(text, className="info-box")


def _section_title(title: str) -> html.Div:
    return html.Div(title, className="section-title")


def _chart(fig: go.Figure) -> html.Div:
    return html.Div(
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        style={"marginTop": "6px", "marginBottom": "14px"},
    )


# Figure cache with LRU eviction
_fig_cache: collections.OrderedDict[str, go.Figure] = collections.OrderedDict()
_MAX_CACHE = 30

def _cached_fig(key: str, builder_fn) -> go.Figure:
    if key in _fig_cache:
        _fig_cache.move_to_end(key)
        return _fig_cache[key]
    if len(_fig_cache) >= _MAX_CACHE:
        _fig_cache.popitem(last=False)
    fig = builder_fn()
    _fig_cache[key] = fig
    return fig


# ---------------------------------------------------------------------------
# Tab 0 — Overview (Vue d'ensemble)
# ---------------------------------------------------------------------------
# Ce premier onglet donne un résumé visuel et textuel de l'expérience :
#   - Barres de conversion : écart brut entre Control et Treatment.
#   - Intervalles de confiance Wilson : incertitude autour de chaque taux.
#   - Tendances quotidiennes : stabilité temporelle de l'effet.
#   - Récapitulatif des colonnes utilisées : traçabilité des choix.
#
# Interprétation rapide :
#   - Le verdict (bannière colorée) résume la décision : DEPLOY, REJECT,
#     INCONCLUSIVE ou NO EVIDENCE.
#   - La KPI row donne les chiffres clés : taux, différence, p(T > C), p-value.
#   - Ce n'est PAS un diagnostic final — croisez toujours avec les onglets
#     spécialisés (Statistical Tests, Bayesian, Effect Sizes).
# ---------------------------------------------------------------------------

def _build_overview(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    if not R.get("stats"):
        return html.Div("No results yet.", style={"color": "var(--text-secondary)"})

    s = R["stats"]
    ctrl = s["ctrl_label"]
    trt = s["trt_label"]

    def _bar_chart() -> go.Figure:
        # Side-by-side comparison of group conversion rates.
        # The gap between bars is the raw treatment effect in percentage points.
        # Statistical significance and Bayesian probability are shown in the KPI row above.
        fig = go.Figure()
        rates_pct = [s["control_rate_pct"], s["treatment_rate_pct"]]
        fig.add_trace(go.Bar(
            x=[ctrl, trt], y=rates_pct,
            marker_color=[CTRL_COLOR, TRT_COLOR],
            marker_line=dict(width=0),
            text=[f"{r:.2f}%" for r in rates_pct],
            textposition="outside",
            textfont={"size": 12, "color": TEXT_COLOR, "family": FONT},
            width=0.5,
        ))
        fig.update_layout(
            title=dict(text="Conversion Rates by Group", font=dict(size=14, color=TEXT_COLOR, family=FONT), x=0.02),
            yaxis=dict(**_yaxis(title="Conversion Rate (%)"), range=[0, max(rates_pct) * 1.35]),
            **_chart_layout(height=340),
        )
        return fig

    def _ci_chart() -> go.Figure:
        # Wilson 95% confidence intervals around each group's conversion rate.
        # Interpret with caution: overlapping CIs do NOT mean "no difference".
        # Formal hypothesis testing and Bayesian analysis provide more reliable inference.
        z = 1.96
        rates = [s["control_rate"], s["treatment_rate"]]
        ns = [s["control_n"], s["treatment_n"]]
        ci_lo, ci_hi = [], []
        for r, n in zip(rates, ns):
            if n == 0:
                ci_lo.append(r)
                ci_hi.append(r)
            else:
                denom = 1 + z * z / n
                centre = (r + z * z / (2 * n)) / denom
                margin = z * math.sqrt((r * (1 - r) + z * z / (4 * n)) / n) / denom
                ci_lo.append(centre - margin)
                ci_hi.append(centre + margin)

        fig = go.Figure()
        for i, (r, lo, hi) in enumerate(zip(rates, ci_lo, ci_hi)):
            color = CTRL_COLOR if i == 0 else TRT_COLOR
            label = ctrl if i == 0 else trt
            fig.add_trace(go.Scatter(
                x=[r * 100], y=[i],
                error_x=dict(type="data", symmetric=False,
                             array=[(hi - r) * 100], arrayminus=[(r - lo) * 100],
                             thickness=2.5, width=14, color=color),
                mode="markers",
                marker=dict(size=14, color=color, line=dict(color="white", width=2)),
                showlegend=False,
                hovertemplate=f"{label}: %{{x:.2f}}%<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text="95% Confidence Intervals (Wilson)", font=dict(size=14, color=TEXT_COLOR, family=FONT), x=0.02),
            xaxis=dict(**_xaxis(title="Conversion Rate (%)")),
            yaxis=dict(tickvals=[0, 1], ticktext=[ctrl, trt], range=[-0.5, 1.5]),
            **_chart_layout(height=260),
        )
        return fig

    fig_bar = _cached_fig(f"overview_bar_{ctrl}_{trt}_{s['control_rate']}_{s['treatment_rate']}", _bar_chart)
    fig_ci = _cached_fig(f"overview_ci_{ctrl}_{trt}_{s['control_rate']}_{s['treatment_rate']}", _ci_chart)

    temporal = R.get("temporal") or {}
    daily_raw = temporal.get("daily_data")
    daily = pd.DataFrame(daily_raw) if daily_raw else None
    trends_el = html.Div()
    if daily is not None and not daily.empty:
        def _trend_chart() -> go.Figure:
            # Daily conversion rates over the experiment duration.
            # Helpful diagnostics:
            #   - A widening gap = increasing treatment effect.
            #   - Narrowing gap = novelty effect or control catching up.
            #   - Seasonal patterns may explain variance.
            # Weighted Pearson r (annotated below) tests for linear trend per group.
            fig = go.Figure()
            for grp_val in daily["group"].unique():
                d = daily[daily["group"] == grp_val]
                lbl = ctrl if str(grp_val) == str(ctrl) else trt
                col = CTRL_COLOR if lbl == ctrl else TRT_COLOR
                fig.add_trace(go.Scatter(
                    x=d["date"], y=d["rate"] * 100,
                    mode="lines+markers", name=lbl,
                    line=dict(color=col, width=2),
                    marker=dict(size=5, color=col, line=dict(width=1, color="white")),
                ))
            fig.update_layout(
                title=dict(text=f"Daily Conversion Rates (from '{R.get('time_col', '')}')",
                           font=dict(size=14, color=TEXT_COLOR, family=FONT), x=0.02),
                yaxis=dict(**_yaxis(title="Conversion Rate (%)")),
                **_chart_layout(height=340),
            )
            return fig

        trends_el = _chart(_cached_fig(f"trends_{R.get('time_col', '')}", _trend_chart))

        trends = temporal.get("trends", {})
        if trends:
            lines = []
            for g, t in trends.items():
                if not math.isnan(t.get("pearson_r", float("nan"))):
                    sig_str = "significant trend" if t.get("significant_trend") else "no trend"
                    lines.append(f"**{g}**: r = {t['pearson_r']:.3f}, p = {t['p_value']:.4f} ({sig_str})")
            if lines:
                trends_el = html.Div([
                    trends_el,
                    _info_box("Weighted trend test (Pearson r weighted by \u221an per day): " + " | ".join(lines)),
                ])

    has_cov = bool(R.get("covariate_cols"))
    has_time = bool(R.get("time_col"))
    has_id = bool(R.get("id_col"))
    usage = [
        ("Target", R.get("target_col", "\u2014"), "Binary outcome. Used in all tests."),
        ("Group", R.get("group_col", "\u2014"), "Defines control vs. treatment."),
        ("Control", R.get("control_value", "\u2014"), "Baseline group label (explicit selection)."),
    ]
    if has_cov:
        for c in R["covariate_cols"]:
            usage.append(("Covariate", c, "Added to enriched logistic regression."))
    else:
        usage.append(("Covariates", "\u2014", "None selected — simple model only."))
    usage.append(("Time", R.get("time_col") or "\u2014",
                  "Daily trends + time features in enriched model." if has_time else "Not provided."))
    usage.append(("ID", R.get("id_col") or "\u2014",
                  "Used for deduplication in data cleaning." if has_id else "Not provided."))

    rows = [html.Tr([html.Td(r[0], style={"fontWeight": 600}), html.Td(r[1]), html.Td(r[2])])
            for r in usage]
    usage_table = html.Table(
        [html.Thead(html.Tr([
            html.Th("Role"), html.Th("Column"), html.Th("Impact"),
        ])), html.Tbody(rows)],
        className="data-table",
    )

    return html.Div([
        html.Div([_chart(fig_bar), _chart(fig_ci)],
                 style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        _section_title("Daily Trends"),
        trends_el,
        _section_title("Column Usage Summary"),
        usage_table,
    ])


# ---------------------------------------------------------------------------
# Tab 1 — Statistical Tests (Tests d'hypothèses fréquentistes)
# ---------------------------------------------------------------------------
# Cinq tests sont rapportés :
#   - Z-test (bilatéral) : test principal pour données binaires. Compare
#     les proportions via l'approximation normale. Valide si n×p ≥ 5 dans
#     chaque groupe.
#   - Z-test unilatéral (T > C) : même statistique, p-value / 2. À utiliser
#     seulement si l'hypothèse directionnelle était pré-enregistrée.
#   - Test du χ² : équivalent au Z-test bilatéral, basé sur la table de
#     contingence 2×2. Donne la même p-value à un facteur près.
#   - Test t de Welch : ne suppose PAS l'égalité des variances
#     (contrairement au t de Student). Utile pour données non-binaires.
#   - Mann-Whitney U : test non-paramétrique sans hypothèse de normalité.
#     Moins puissant que le t-test si les données sont normales.
#
# Règle pratique :
#   - Données binaires (conversion 0/1) → Z-test ou χ².
#   - Données continues (temps, revenu) → t-test ou Mann-Whitney.
#   - Si les résultats divergent entre tests, suspectez une violation
#     des hypothèses (hétéroscédasticité, outliers, non-normalité).
# ---------------------------------------------------------------------------

def _build_test_results(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    tests = R.get("tests") or {}
    s = R.get("stats") or {}
    alpha = s.get("alpha", 0.05)

    rows = []
    test_names = [
        ("Chi-squared", "chi_squared"),
        ("Z-test (two-sided)", "ztest"),
        ("Z-test (one-sided T>C)", None),
        ("T-test (Welch)", "ttest"),
        ("Mann-Whitney U", "mann_whitney"),
    ]
    ztest = tests.get("ztest") or {}
    for label, key in test_names:
        if key is None:
            p_one = ztest.get("p_value")
            if p_one is not None:
                p_one_v = p_one / 2
                sig_one = p_one_v < alpha
                rows.append(html.Tr([
                    html.Td(label, style={"fontWeight": 600}),
                    html.Td("\u2014"),
                    html.Td(f"{p_one_v:.4f}"),
                    html.Td(f"{'Significant' if sig_one else 'Not significant'}"),
                ]))
            continue
        t = tests.get(key) or {}
        stat = t.get("statistic")
        p = t.get("p_value")
        if stat is None or p is None:
            continue
        sig = p < alpha
        rows.append(html.Tr([
            html.Td(label, style={"fontWeight": 600}),
            html.Td(f"{stat:.4f}"),
            html.Td(f"{p:.4f}"),
            html.Td(f"{'Significant' if sig else 'Not significant'}"),
        ]))

    p_two = ztest.get("p_value")
    sig_two = p_two is not None and p_two < alpha
    p_one = p_two / 2 if p_two is not None else None
    sig_one = p_one is not None and p_one < alpha

    interp_two = f"{'Statistically significant' if sig_two else 'Not statistically significant'} at \u03b1 = {alpha}"
    interp_one = f"{'Treatment > Control' if sig_one else 'No directional evidence'} at \u03b1 = {alpha}"

    note = _info_box(
        "One-sided test correctly tests H\u2081: p_treatment > p_control. "
        "Reject H\u2080 only if you are willing to commit to this direction before seeing the data."
    )

    return html.Div([
        _section_title("Hypothesis Tests"),
        html.Table(
            [html.Thead(html.Tr([
                html.Th("Test"), html.Th("Statistic"), html.Th("p-value"), html.Th("Sig. at \u03b1"),
            ])), html.Tbody(rows)],
            className="data-table",
        ),
        html.Div([
            html.Div([
                html.P("Z-Test (two-sided)", style={"fontSize": "0.85rem", "fontWeight": 600, "color": "var(--color-success)", "margin": 0}),
                html.P(f"p = {p_two:.4f} — {interp_two}" if p_two is not None else "N/A",
                       style={"fontSize": "0.95rem", "color": "var(--color-success)", "margin": "0.3rem 0 0"}),
            ], style={
                "background": "var(--color-success-bg)", "border": "1px solid var(--color-success-light)",
                "borderRadius": "10px", "padding": "1rem 1.5rem", "flex": "1",
            }),
            html.Div([
                html.P("Z-Test one-sided (H\u2081: T > C)", style={"fontSize": "0.85rem", "fontWeight": 600, "color": "var(--color-info)", "margin": 0}),
                html.P(f"p = {p_one:.4f} — {interp_one}" if p_one is not None else "N/A",
                       style={"fontSize": "0.95rem", "color": "var(--color-info)", "margin": "0.3rem 0 0"}),
            ], style={
                "background": "var(--color-info-bg)", "border": "1px solid var(--color-info)",
                "borderRadius": "10px", "padding": "1rem 1.5rem", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginTop": "1rem"}),
        note,
    ])


# ---------------------------------------------------------------------------
# Tab 2 — Effect Sizes (Taille d'effet et régression logistique)
# ---------------------------------------------------------------------------
# La significativité statistique (p < α) ne dit rien sur l'AMPLEUR de l'effet.
# Cette section fournit des métriques d'effet standardisées et interprétables :
#   - Cohen's h (d) : différence standardisée entre les deux proportions.
#     Seuils : 0.2 (petit), 0.5 (moyen), 0.8 (grand).
#   - Risk Ratio : rapport des taux de conversion (T/C). RR = 1.10 signifie
#     une hausse de 10 % relative.
#   - NNT (Number Needed to Treat) : combien de personnes doivent recevoir
#     le traitement pour observer un succès additionnel. NNT = 50 est bon
#     si le coût est faible, mauvais si le traitement est coûteux.
#
# Régression logistique :
#   - Modèle simple : seulement l'effet du traitement (sans ajustement).
#   - Modèle enrichi : ajoute les covariates et/ou features temporelles
#     choisies dans la sidebar. L'odds ratio ajusté peut différer du brut
#     si les covariables sont déséquilibrées entre groupes.
#   - Le test du ratio de vraisemblance (LR test) compare les deux modèles :
#     un p < 0.05 signifie que les covariables améliorent significativement
#     la prédiction.
# ---------------------------------------------------------------------------

def _build_effects(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    es = R.get("effect_sizes") or {}
    lr_simple = R.get("log_simple") or {}
    lr_enriched = R.get("log_enriched") or {}
    lr_test = R.get("lr_test") or {}
    s = R.get("stats") or {}
    has_cov = bool(R.get("covariate_cols"))
    has_time = bool(R.get("time_col"))

    def fmt_nnt(v):
        if v is None or not math.isfinite(v):
            return "\u221e"
        return f"{v:.1f}"

    def fmt_ci(val):
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            return "\u2014"
        return f"{val:.4f}"

    return html.Div([
        _section_title("Effect Size Estimates"),
        html.Div([
            _metric_card("Cohen's h", f"{es.get('cohens_d', 0):.4f}", es.get("cohens_d_interpretation", "")),
            _metric_card("Odds Ratio", f"{es.get('odds_ratio', 0):.4f}" if es.get("odds_ratio") is not None else "\u2014",
                         f"CI 95% [{es.get('odds_ratio_ci_95', [0, 0])[0]:.3f}, {es.get('odds_ratio_ci_95', [0, 0])[1]:.3f}]"
                         if es.get("odds_ratio_ci_95") else ""),
            _metric_card("Risk Ratio", f"{es.get('risk_ratio', 0):.4f}", "Relative risk T/C"),
            _metric_card("NNT", fmt_nnt(es.get("nnt")),
                         f"Number Needed to Treat ({'better' if (s.get('diff_pp') or 0) > 0 else 'worse'})"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        _section_title("Logistic Regression"),
        html.Div([
            html.Div([
                html.H4("Simple Model", style={"margin": "0 0 0.75rem", "color": "var(--text-primary)"}),
                html.P("Treatment only, no covariates", style={"fontSize": "0.85rem", "color": "var(--text-secondary)", "margin": "0.2rem 0"}),
                html.P(f"OR: {lr_simple.get('odds_ratio', 0):.4f}", style={"margin": "0.2rem 0"}),
                html.P(f"CI 95%: [{fmt_ci(lr_simple.get('or_ci_95', [0, 0])[0])}, {fmt_ci(lr_simple.get('or_ci_95', [0, 0])[1])}]", style={"margin": "0.2rem 0"}),
                html.P(f"p-value: {lr_simple.get('p_value', 0):.4f}", style={"margin": "0.2rem 0"}),
            ], className="model-card"),
            html.Div([
                html.H4("Enriched Model", style={"margin": "0 0 0.75rem", "color": "var(--text-primary)"}),
                html.P(f"Covariates: {', '.join(R.get('covariate_cols', [])) if has_cov else 'none'}{' + hour/weekend' if has_time else ''}",
                       style={"fontSize": "0.85rem", "color": "var(--text-secondary)", "margin": "0.2rem 0"}),
                html.P(f"OR: {lr_enriched.get('odds_ratio', 0):.4f}", style={"margin": "0.2rem 0"}),
                html.P(f"CI 95%: [{fmt_ci(lr_enriched.get('or_ci_95', [0, 0])[0])}, {fmt_ci(lr_enriched.get('or_ci_95', [0, 0])[1])}]", style={"margin": "0.2rem 0"}),
                html.P(f"p-value: {lr_enriched.get('p_value', 0):.4f}", style={"margin": "0.2rem 0"}),
                html.P(f"LR test vs. simple: LR = {lr_test.get('lr_statistic', 0):.2f}, p = {lr_test.get('p_value', 0):.4f}", style={"margin": "0.2rem 0"}),
            ], className="model-card"),
        ], style={"display": "flex", "gap": "16px", "marginTop": "8px"}),
    ])


# ---------------------------------------------------------------------------
# Tab 3 — Bayesian (Analyse bayésienne)
# ---------------------------------------------------------------------------
# L'approche bayésienne complète l'inférence fréquentiste :
#   - P(T > C) : probabilité directe que le traitement surpasse le contrôle.
#     Interprétation intuitive : « il y a X % de chances que T soit meilleur ».
#     Pas de seuil magique, mais > 95 % est une preuve forte.
#   - ROPE (Region Of Practical Equivalence) : intervalle [-0.005, +0.005]
#     autour de zéro. Si la majorité de la distribution postérieure tombe dans
#     la ROPE, les groupes sont pratiquement équivalents.
#   - Expected Loss : perte attendue si on adopte le traitement alors qu'il
#     est en réalité moins bon (en unités de taux de conversion).
#   - Distribution postérieure (graphique) : pour données binaires uniquement
#     (modèle Beta-Binomial). Visualise l'incertitude résiduelle.
#
# Décision bayésienne (automate) :
#   - adopt_treatment : P(diff > ROPE_upper) > 95 %
#   - keep_control : P(diff < ROPE_lower) > 95 %
#   - practical_equivalence : P(diff dans ROPE) > 95 %
#   - insufficient_evidence : aucun des cas ci-desssus
# ---------------------------------------------------------------------------

def _build_bayesian(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    bayes = R.get("bayesian") or {}
    gs = R.get("gs") or {}
    s = R.get("stats") or {}
    ctrl = s.get("ctrl_label", "Control")
    trt = s.get("trt_label", "Treatment")

    dec_map = {
        "adopt_treatment": "Adopt Treatment",
        "keep_control": "Keep Control",
        "practical_equivalence": "Practical Equiv.",
        "insufficient_evidence": "Need More Data",
    }

    is_beta = bayes.get("n_simulations", 0) > 0
    fig_posterior = None
    if is_beta:
        def _posterior_chart() -> go.Figure:
            from scipy.stats import beta as beta_dist
            a_c = 1 + gs.get("conv_control", 0)
            b_c = 1 + (gs.get("n_control", 0) - gs.get("conv_control", 0))
            a_t = 1 + gs.get("conv_treatment", 0)
            b_t = 1 + (gs.get("n_treatment", 0) - gs.get("conv_treatment", 0))
            max_rate = max(s.get("control_rate", 0), s.get("treatment_rate", 0))
            x_grid = np.linspace(0, max(max_rate * 2, 0.02), 400)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_grid, y=beta_dist.pdf(x_grid, a_c, b_c),
                mode="lines", name=f"Control ({ctrl})",
                line=dict(color=CTRL_COLOR, width=2.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
            ))
            fig.add_trace(go.Scatter(
                x=x_grid, y=beta_dist.pdf(x_grid, a_t, b_t),
                mode="lines", name=f"Treatment ({trt})",
                line=dict(color=TRT_COLOR, width=2.5),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.10)",
            ))
            fig.update_layout(
                title=dict(text="Posterior Distributions (Beta-Binomial)",
                           font=dict(size=14, color=TEXT_COLOR, family=FONT), x=0.02),
                xaxis=dict(**_xaxis(title="Conversion Rate")),
                yaxis=dict(**_yaxis(title="Posterior Density")),
                **_chart_layout(height=320),
            )
            return fig

        fig_posterior = _cached_fig(f"posterior_{ctrl}_{trt}_{gs.get('n_control',0)}_{gs.get('n_treatment',0)}", _posterior_chart)

    rope_lo = bayes.get("rope_lower", -0.002)
    rope_hi = bayes.get("rope_upper", 0.002)
    n_sims = bayes.get("n_simulations", 50_000)
    ci_95 = bayes.get("ci_95", (0, 0))

    title_suffix = "Beta-Binomial" if is_beta else "Normal-Normal"
    return html.Div([
        _section_title(f"Bayesian Analysis ({title_suffix})"),
        html.Div([
            _metric_card("P(Treatment > Control)", f"{bayes.get('p_treatment_better_pct', bayes.get('p_treatment_better', 0) * 100):.1f}%"),
            _metric_card("Expected Loss", f"{bayes.get('expected_loss', 0):.5f}"),
            _metric_card("P(diff in ROPE)",
                         f"{bayes.get('p_rope_region', 0) * 100:.1f}%",
                         f"ROPE [{rope_lo:+.3f}, {rope_hi:+.3f}]"),
            _metric_card("Decision", dec_map.get(bayes.get("decision", ""), bayes.get("decision", "\u2014"))),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        _chart(fig_posterior) if fig_posterior else _info_box("Posterior chart available only for binary data (Beta-Binomial)."),
        _info_box(
            f"Prior: Beta(1, 1) — uniform. "
            f"Simulations: {n_sims:,}. "
            f"ROPE: [{rope_lo:+.3f}, {rope_hi:+.3f}] (configurable in config.py). "
            f"CI 95%: [{ci_95[0] * 100:.2f}%, {ci_95[1] * 100:.2f}%]."
        ),
    ])


# ---------------------------------------------------------------------------
# Tab 4 — Segments (Analyse segmentée)
# ---------------------------------------------------------------------------
# Explore l'hétérogénéité de l'effet traitement à travers les sous-groupes
# définis par les colonnes catégorielles du dataset.
#
# Interprétation :
#   - Une différence de signe entre segments peut révéler un paradoxe de
#     Simpson (l'effet global est positif mais négatif dans tous les
#     sous-groupes, ou l'inverse).
#   - Un segment avec un effet très fort mais peu d'observations est
#     probablement un faux positif — d'où la correction FDR.
#   - Les segments Weekday/Weekend sont générés automatiquement si une
#     colonne de date est fournie.
#
# Correction FDR (Benjamini-Hochberg) :
#   - Sans correction, 5 % des segments seront significatifs par hasard.
#   - La colonne « p (FDR adj.) » donne la p-value ajustée.
#   - Seuls les segments marqués ✅ sont fiables après correction.
# ---------------------------------------------------------------------------

def _build_segments(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    seg = R.get("segmentation") or {}
    if not seg:
        return html.Div("No categorical segments found with sufficient data.",
                        style={"color": "var(--text-secondary)", "marginTop": "16px"})

    seg_rows = []
    for col_name, segments in seg.items():
        for seg_name, data in segments.items():
            if not (isinstance(data, dict) and "control_rate" in data):
                continue
            diff_seg = (data["treatment_rate"] - data["control_rate"]) * 100
            seg_rows.append({
                "Segment": f"{col_name}: {seg_name}",
                "Control %": f"{data['control_rate'] * 100:.2f}%",
                "_ctrl_pct": data['control_rate'] * 100,
                "n Control": data.get("control_n", 0),
                "Treatment %": f"{data['treatment_rate'] * 100:.2f}%",
                "_trt_pct": data['treatment_rate'] * 100,
                "n Treatment": data.get("treatment_n", 0),
                "Diff (pp)": f"{diff_seg:+.2f}",
                "p (raw)": f"{data.get('p_value_raw', data.get('p_value', 1.0)):.4f}",
                "p (FDR adj.)": f"{data.get('p_value', 1.0):.4f}",
                "Sig.*": "\u2705" if data.get("significant") else "\u274c",
            })

    chart_el = html.Div()
    if len(seg_rows) > 1:
        def _seg_chart() -> go.Figure:
            labels = [r["Segment"] for r in seg_rows]
            ctrl_vals = [r["_ctrl_pct"] for r in seg_rows]
            trt_vals = [r["_trt_pct"] for r in seg_rows]
            diffs = [float(r["Diff (pp)"]) for r in seg_rows]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Control", x=labels, y=ctrl_vals,
                marker_color=CTRL_COLOR, marker_line=dict(width=0),
                width=0.35, offset=-0.175, opacity=0.85,
            ))
            fig.add_trace(go.Bar(
                name="Treatment", x=labels, y=trt_vals,
                marker_color=TRT_COLOR, marker_line=dict(width=0),
                width=0.35, offset=0.175, opacity=0.85,
            ))
            annotations = []
            for i, (d, mx) in enumerate(zip(diffs, [max(c, t) for c, t in zip(ctrl_vals, trt_vals)])):
                annotations.append(dict(
                    x=i, y=mx + 0.5,
                    text=f"{d:+.2f}pp",
                    showarrow=False, font=dict(
                        size=10, color="#10B981" if d > 0 else "#EF4444", family=FONT,
                    ),
                ))
            fig.update_layout(
                title=dict(text="Conversion Rate by Segment",
                           font=dict(size=14, color=TEXT_COLOR, family=FONT), x=0.02),
                yaxis=dict(**_yaxis(title="Conversion Rate (%)")),
                xaxis=dict(**_xaxis(title=""), tickangle=30),
                barmode="group",
                annotations=annotations,
                **_chart_layout(height=380, margin=dict(l=48, r=24, t=56, b=80)),
            )
            return fig

        cache_key = f"seg_{len(seg_rows)}_{seg_rows[0]['Segment'] if seg_rows else 'none'}"
        chart_el = _chart(_cached_fig(cache_key, _seg_chart))

    table_rows = []
    for r in seg_rows:
        table_rows.append(html.Tr([
            html.Td(r["Segment"], style={"fontWeight": 600}),
            html.Td(r["Control %"]), html.Td(str(r["n Control"])),
            html.Td(r["Treatment %"]), html.Td(str(r["n Treatment"])),
            html.Td(r["Diff (pp)"]), html.Td(r["p (raw)"]),
            html.Td(r["p (FDR adj.)"]), html.Td(r["Sig.*"]),
        ]))

    return html.Div([
        _section_title("Segmentation Analysis"),
        _info_box("p-values corrected for multiple comparisons using Benjamini-Hochberg FDR. "
                  " = significant after correction."),
        chart_el,
        html.Table(
            [html.Thead(html.Tr([
                html.Th("Segment"), html.Th("Control %"), html.Th("n Control"),
                html.Th("Treatment %"), html.Th("n Treatment"),
                html.Th("Diff (pp)"), html.Th("p (raw)"),
                html.Th("p (FDR adj.)"), html.Th("Sig.*"),
            ])), html.Tbody(table_rows)],
            className="segments-table",
        ),
    ])


# ---------------------------------------------------------------------------
# Tab 5 — Power & Robustness (Puissance et robustesse)
# ---------------------------------------------------------------------------
# Évalue la fiabilité des résultats :
#   - Puissance observée : probabilité de détecter l'effet mesuré avec
#     l'effectif actuel. Si < 80 %, le test manque de puissance.
#   - N nécessaire pour 80 % : combien d'observations par groupe seraient
#     nécessaires pour atteindre la puissance conventionnelle.
#   - MDE (Minimum Detectable Effect) : le plus petit effet détectable avec
#     l'effectif actuel et α = 0.05. Si le MDE est plus grand que l'effet
#     mesuré, l'effet est « dans le bruit ».
#
# Bootstrap (non-paramétrique) :
#   - Rééchantillonnage avec remise pour estimer la distribution de la
#     différence. Les intervalles bootstrap sont valides sans hypothèse
#     de distribution.
#   - % d'échantillons T > C : interprétation fréquentiste directe.
#
# Test de permutation :
#   - Mélange aléatoire des étiquettes de groupe pour générer la
#     distribution sous H0. Purement non-paramétrique.
#   - Si p(permutation) est très différent du p(Z-test), suspectez une
#     violation des hypothèses du test paramétrique.
# ---------------------------------------------------------------------------

def _build_power(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    power = R.get("power") or {}
    es = R.get("effect_sizes") or {}
    gs = R.get("gs") or {}
    boot = R.get("bootstrap") or {}
    perm = R.get("permutation") or {}
    s = R.get("stats") or {}

    if not power or power.get("skipped"):
        return html.Div([
            _section_title("Power Analysis"),
            _info_box("Power analysis skipped — effect size is effectively zero."),
        ])

    def _power_curve() -> go.Figure:
        fig = go.Figure()
        cohens_h = es.get("cohens_d", 0.5)
        n_control = gs.get("n_control", 1000)
        n_vals = np.linspace(1000, max(int(power.get("n_needed_80pct", 1000) * 1.5), n_control * 2), 50).astype(int)

        from src.power_analysis import compute_power_vectorized
        alpha_val = s.get("alpha", 0.05)
        pw_vals = compute_power_vectorized(cohens_h, n_vals, alpha_val) * 100

        fig.add_trace(go.Scatter(
            x=n_vals, y=pw_vals, mode="lines",
            line=dict(color=CTRL_COLOR, width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            name="Power",
        ))
        fig.add_hline(y=80, line=dict(color="#10B981", dash="dash", width=1.5),
                      annotation_text="80% target",
                      annotation_font=dict(color="#10B981", size=11, family=FONT))
        fig.add_vline(x=n_control, line=dict(color="#EF4444", dash="dot", width=1.5),
                      annotation_text=f"n = {n_control:,}",
                      annotation_font=dict(color="#EF4444", size=11, family=FONT))
        fig.update_layout(
            title=dict(text="Power vs. Sample Size",
                       font=dict(size=14, color=TEXT_COLOR, family=FONT), x=0.02),
            xaxis=dict(**_xaxis(title="Sample size per group")),
            yaxis=dict(**_yaxis(title="Statistical Power (%)"), range=[0, 105]),
            **_chart_layout(height=340, showlegend=False),
        )
        return fig

    pw_cache_key = f"power_{es.get('cohens_d', 0)}_{gs.get('n_control', 0)}_{s.get('alpha', 0.05)}"
    fig_pw = _cached_fig(pw_cache_key, _power_curve)

    return html.Div([
        _section_title("Power Analysis"),
        html.Div([
            _metric_card("Observed Power", f"{power.get('power_observed', 0) * 100:.1f}%"),
            _metric_card("N/group for 80%", f"{power.get('n_needed_80pct', 0):,.0f}"),
            _metric_card("MDE (Cohen's h)", f"{power.get('mde_cohens_h', 0):.4f}"),
            _metric_card("MDE (approx.)", f"\u00b1{power.get('mde_pp', 0) * 100:.2f} pp"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        _chart(fig_pw),
        _section_title("Robustness Checks"),
        html.Div([
            html.Div([
                html.H4("Bootstrap (non-parametric)", style={"margin": "0 0 0.75rem", "color": "var(--text-primary)"}),
                html.P(f"CI 95%: [{boot.get('ci_95', (0, 0))[0] * 100:.3f}%, {boot.get('ci_95', (0, 0))[1] * 100:.3f}%]", style={"margin": "0.2rem 0"}),
                html.P(f"CI 90%: [{boot.get('ci_90', (0, 0))[0] * 100:.3f}%, {boot.get('ci_90', (0, 0))[1] * 100:.3f}%]", style={"margin": "0.2rem 0"}),
                html.P(f"% samples T > C: {boot.get('pct_positive', 0):.1f}%", style={"margin": "0.2rem 0"}),
                html.P(f"Mean diff: {boot.get('mean_diff', 0) * 100:.4f} pp", style={"margin": "0.2rem 0"}),
            ], className="model-card"),
            html.Div([
                html.H4("Permutation Test", style={"margin": "0 0 0.75rem", "color": "var(--text-primary)"}),
                html.P(f"p (two-sided): {perm.get('p_value_two_sided', 0):.4f}", style={"margin": "0.2rem 0"}),
                html.P(f"p (one-sided): {perm.get('p_value_one_sided', 0):.4f}", style={"margin": "0.2rem 0"}),
                html.P(f"Observed diff: {perm.get('observed_diff', 0) * 100:.4f} pp", style={"margin": "0.2rem 0"}),
            ], className="model-card"),
        ], style={"display": "flex", "gap": "16px", "marginTop": "8px"}),
    ])


# ---------------------------------------------------------------------------
# Tab 6 — Export
# ---------------------------------------------------------------------------

def _build_export(R: dict) -> html.Div:
    if R.get("error"):
        return html.Div(f"Error: {R['error']}", style={"color": "var(--color-danger)"})
    report_text = generate_text_report(R)
    df_summary = build_summary_df(R)
    return html.Div([
        html.H3("Export Results", className="section-title", style={"marginTop": "0"}),
        html.Button("Download Full Report (.txt)", id="download-txt-btn",
                    className="btn btn-secondary"),
        html.Details([
            html.Summary("Preview report",
                         style={"cursor": "pointer", "fontSize": "0.9rem", "color": "var(--color-primary)", "marginTop": "12px"}),
            html.Pre(report_text, style={"background": "var(--surface-1)", "padding": "16px", "borderRadius": "8px",
                                         "fontSize": "0.8rem", "lineHeight": "1.5", "overflowX": "auto",
                                         "maxHeight": "400px", "overflowY": "auto", "color": "var(--text-primary)"}),
        ], style={"marginTop": "8px"}),
        html.Hr(style={"margin": "1.5rem 0", "borderColor": "var(--border-1)"}),
        html.Button("Download Summary Metrics (.csv)", id="download-csv-btn",
                    className="btn btn-secondary"),
        html.Div([
            html.H4("Summary Preview", className="metric-label", style={"fontSize": "0.95rem", "fontWeight": 600, "margin": "12px 0 8px"}),
            html.Table([html.Tbody([
                html.Tr([html.Td(r[0], style={"fontWeight": 600}), html.Td(str(r[1]))])
                for r in df_summary.values
            ])], style={"fontSize": "0.85rem", "borderCollapse": "collapse"}),
        ], style={"marginTop": "16px"}),
    ])


# ---------------------------------------------------------------------------
# Tab builder registry
# ---------------------------------------------------------------------------

_TAB_BUILDERS = {
    "tab-overview": _build_overview,
    "tab-tests": _build_test_results,
    "tab-effects": _build_effects,
    "tab-bayesian": _build_bayesian,
    "tab-segments": _build_segments,
    "tab-power": _build_power,
    "tab-export": _build_export,
}


def register_tab_callbacks(app):
    @app.callback(
        Output("tab-content-container", "children"),
        Input("result-tabs", "value"),
        Input("store-analysis", "data"),
        prevent_initial_call=False,
    )
    def render_tab(tab_value, results):
        if not results or "verdict" not in results:
            return html.Div("Run an analysis to see results.",
                            style={"color": "var(--text-secondary)", "marginTop": "16px"})
        builder = _TAB_BUILDERS.get(tab_value)
        if builder is None:
            return html.Div()
        return builder(results)
