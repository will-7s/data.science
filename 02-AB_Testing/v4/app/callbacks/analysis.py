from __future__ import annotations

import base64
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dash import Input, Output, State, html, no_update

from src.bayesian_analysis import beta_binomial_analysis, normal_normal_analysis
from src.data_loader import get_group_stats, prepare_data
from src.effect_size import compute_all_effect_sizes
from src.hypothesis_testing import run_all_tests
from src.logistic_regression import (
    enriched_logistic_regression,
    likelihood_ratio_test,
    simple_logistic_regression,
)
from src.power_analysis import compute_power
from src.robustness import bootstrap_ci, permutation_test
from src.segmentation import run_segmentation_analysis
from src.temporal_analysis import run_temporal_analysis

RESULTS_STORE_ID = "store-analysis"


def _wilson_ci(n: int, successes: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return (centre - margin, centre + margin)


def _derive_verdict(bayesian: dict, tests: dict, diff_pp: float, alpha: float) -> str:
    # Algorithme de décision combinant fréquentiste et bayésien :
    # 1. Si p < α et diff > 0 → DEPLOY (preuve fréquentiste forte).
    # 2. Si p < α et diff < 0 → REJECT (le contrôle est meilleur).
    # 3. Si P(T > C) > 95 % → DEPLOY (preuve bayésienne forte).
    # 4. Si P(T > C) > 80 % ou p < 0.10 → INCONCLUSIVE (tendance).
    # 5. Sinon → NO EVIDENCE (pas assez de données).
    ztest = tests.get("ztest") or {}
    p_two = ztest.get("p_value")
    sig_two = p_two is not None and p_two < alpha
    p_bayes = (bayesian.get("p_treatment_better") or 0)
    if sig_two and diff_pp > 0:
        return "DEPLOY — Statistically significant lift detected"
    if sig_two and diff_pp < 0:
        return "REJECT — Control is significantly better"
    if p_bayes > 0.95:
        return "DEPLOY — Bayesian probability > 95%"
    if p_bayes > 0.80 or (p_two is not None and p_two < 0.10):
        return "INCONCLUSIVE — Trend detected, extend the test"
    return "No evidence of improvement — Keep control"


def _verdict_class(verdict: str) -> str:
    if verdict.startswith("DEPLOY"):
        return "verdict-banner deploy"
    if verdict.startswith("REJECT"):
        return "verdict-banner reject"
    if verdict.startswith("INCONCLUSIVE"):
        return "verdict-banner inconclusive"
    return "verdict-banner no-evidence"


def _schema_banner(
    target_col: str, group_col: str, control_value: str,
    covariate_cols: list[str], time_col: str | None, id_col: str | None,
    alpha: float, cleaning: dict | None = None,
) -> html.Div:
    cleaning_badge = ""
    if cleaning:
        mm = f"{cleaning.get('n_mismatch_removed', 0):,}"
        dp = f"{cleaning.get('n_dupes_removed', 0):,}"
        cl = f"{cleaning.get('n_clean', '?'):,}"
        pc = f"{cleaning.get('pct_removed', 0):.2f}"
        cleaning_badge = html.Span(
            f"Cleaning: {mm} mismatches + {dp} duplicates removed — {cl} clean rows ({pc}% dropped)",
            className="schema-tag",
            style={"background": "var(--color-warning-bg)", "borderColor": "var(--color-warning)"},
        )

    def tag(label: str, value: str, control: bool = False) -> html.Span:
        cls = "schema-tag control-tag" if control else "schema-tag"
        return html.Span([
            html.Strong(f"{label}: "), value
        ], className=cls)

    tags = [
        tag("Target", target_col),
        tag("Group", group_col),
        tag("Control", f'"{control_value}"', control=True),
        tag("Covariates", ", ".join(covariate_cols) if covariate_cols else "\u2014"),
        tag("Time", time_col or "\u2014"),
        tag("ID", id_col or "\u2014"),
        tag("\u03b1", f"{alpha}"),
    ]
    return html.Div(
        className="schema-banner",
        children=[
            html.H3("Analysis Configuration",
                    style={"margin": "0 0 0.75rem 0", "fontSize": "1.1rem", "fontWeight": 600}),
            html.Div(tags, style={"marginBottom": "0.6rem"}),
            cleaning_badge,
        ],
    )


def _verdict_banner(verdict: str) -> html.Div:
    cls = _verdict_class(verdict)
    return html.Div(
        className=cls,
        children=[
            html.P(verdict, style={"color": "white", "fontSize": "1.15rem", "fontWeight": 700, "margin": 0}),
        ],
    )


def _kpi_row(
    control_rate_pct: float, treatment_rate_pct: float,
    ctrl_label: str, trt_label: str,
    n_control: int, n_treatment: int,
    diff_pp: float, rel_pct: float,
    p_bayes: float, p_value: float | None, sig_two: bool, alpha: float,
) -> list[html.Div]:
    def _metric(label: str, value: str, delta: str | None = None) -> html.Div:
        children = [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
        ]
        if delta:
            children.append(html.Div(delta, className="kpi-delta"))
        return html.Div(children, className="kpi-card")

    return [
        _metric(f"Control ({ctrl_label})", f"{control_rate_pct:.2f}%", f"n = {n_control:,}"),
        _metric(f"Treatment ({trt_label})", f"{treatment_rate_pct:.2f}%", f"n = {n_treatment:,}"),
        _metric("Difference", f"{diff_pp:+.3f} pp", f"{rel_pct:+.2f}% rel."),
        _metric("P(T > C)", f"{p_bayes * 100:.1f}%", "Bayesian"),
        _metric("Z-test p-value", f"{p_value:.4f}" if p_value is not None else "\u2014",
                "Significant" if sig_two else "Not significant"),
    ]


def register_analysis_callbacks(app):
    @app.callback(
        Output("results-area", "style"),
        Output("schema-banner", "children"),
        Output("verdict-banner", "children"),
        Output("kpi-row", "children"),
        Output("run-progress", "children"),
        Output(RESULTS_STORE_ID, "data", allow_duplicate=True),
        Input("run-analysis-btn", "n_clicks"),
        State("store-analysis", "data"),
        State("target-col-dropdown", "value"),
        State("group-col-dropdown", "value"),
        State("control-value-input", "value"),
        State("covariate-multi", "value"),
        State("time-col-dropdown", "value"),
        State("id-col-dropdown", "value"),
        State("alpha-slider", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(
        _n_clicks, store,
        target_col, group_col, control_value,
        covariate_cols, time_col, id_col,
        alpha,
    ):
        missing = []
        if not store:
            missing.append("data not loaded")
        if not target_col:
            missing.append("target column")
        if not group_col:
            missing.append("group column")
        if not control_value:
            missing.append("control value")
        if missing:
            msg = f"Cannot run analysis: missing {', '.join(missing)}."
            return (
                {"display": "block"},
                html.Div(),
                html.Div(msg, className="info-box", style={"color": "var(--color-warning)"}),
                [],
                html.Div(),
                no_update,
            )

        data_key = store.get("data_key")
        if not data_key:
            msg = "No data found. Please load data first."
            return (
                {"display": "block"},
                html.Div(),
                html.Div(msg, className="info-box", style={"color": "var(--color-warning)"}),
                [],
                html.Div(),
                no_update,
            )

        from src.data_cache import get as cache_get
        df = cache_get(data_key)
        if df is None:
            msg = "Data expired or missing. Please re-load data."
            return (
                {"display": "block"},
                html.Div(),
                html.Div(msg, className="info-box", style={"color": "var(--color-danger)"}),
                [],
                html.Div(),
                no_update,
            )

        try:
            time_col = time_col if time_col and time_col != "(None)" else None
            id_col = id_col if id_col and id_col != "(None)" else None
            cov_cols = covariate_cols if covariate_cols else []

            prepared = prepare_data(
                df, target_col, group_col, control_value,
                time_col=time_col, covariate_cols=cov_cols,
            )

            gs = get_group_stats(prepared)

            n_workers = min(2, os.cpu_count() or 2)
            with ThreadPoolExecutor(max_workers=n_workers) as exe:
                futures = {
                    exe.submit(run_all_tests, prepared): "tests",
                    exe.submit(compute_all_effect_sizes, prepared): "effects",
                    exe.submit(bootstrap_ci, prepared): "bootstrap",
                    exe.submit(permutation_test, prepared, gs): "permutation",
                    exe.submit(simple_logistic_regression, prepared, control_value): "log_simple",
                    exe.submit(enriched_logistic_regression, prepared, control_value): "log_enriched",
                    exe.submit(run_segmentation_analysis, prepared, control_value, target_col, group_col): "segmentation",
                    exe.submit(run_temporal_analysis, prepared): "temporal",
                }
                if prepared.get("is_binary", False):
                    futures[exe.submit(beta_binomial_analysis, prepared, gs)] = "bayesian"
                else:
                    futures[exe.submit(normal_normal_analysis, prepared, gs)] = "bayesian"

                results_map = {}
                for future in as_completed(futures):
                    key = futures[future]
                    results_map[key] = future.result()

            tests = results_map["tests"]
            effects = results_map["effects"]
            bayesian = results_map["bayesian"]
            log_simple = results_map["log_simple"]
            log_enriched = results_map["log_enriched"]
            segmentation = results_map["segmentation"]
            temporal = results_map["temporal"]
            bootstrap = results_map["bootstrap"]
            permutation = results_map["permutation"]
            robustness = {"bootstrap": bootstrap, "permutation": permutation}

            # Conversion pandas → dict pour la sérialisation JSON du Store Dash
            if temporal.get("daily_data") is not None:
                temporal["daily_data"] = temporal["daily_data"].to_dict("records")

            pwr = None
            if effects.get("cohens_d") is not None and abs(effects["cohens_d"]) > 0:
                pwr = compute_power(
                    effects["cohens_d"],
                    gs["n_control"], gs["n_treatment"],
                    alpha=alpha,
                )

            lr_test = likelihood_ratio_test(
                log_simple, log_enriched,
                n_covariates=len(cov_cols), has_time=bool(time_col),
            )

            ctrl_label = str(control_value)
            group_vals = df[group_col].astype(str).unique()
            trt_label = [v for v in group_vals if v != str(control_value)]
            trt_label = trt_label[0] if trt_label else "Treatment"

            control_rate = gs["control_rate"]
            treatment_rate = gs["treatment_rate"]
            diff_pp = (treatment_rate - control_rate) * 100
            rel_pct = ((treatment_rate - control_rate) / control_rate * 100) if control_rate != 0 else 0.0

            ztest = tests.get("ztest") or {}
            p_value = ztest.get("p_value")
            sig_two = p_value is not None and p_value < alpha

            verdict = _derive_verdict(bayesian, tests, treatment_rate - control_rate, alpha)

            results = {
                "data_key": data_key,
                "stats": {
                    "control_rate": control_rate,
                    "treatment_rate": treatment_rate,
                    "control_rate_pct": control_rate * 100,
                    "treatment_rate_pct": treatment_rate * 100,
                    "control_n": gs["n_control"],
                    "treatment_n": gs["n_treatment"],
                    "ctrl_label": ctrl_label,
                    "trt_label": trt_label,
                    "diff_pp": diff_pp,
                    "rel_pct": rel_pct,
                    "p_value": p_value,
                    "sig_two": sig_two,
                    "alpha": alpha,
                },
                "tests": tests,
                "effect_sizes": effects,
                "bayesian": bayesian,
                "power": pwr,
                "robustness": robustness,
                "bootstrap": robustness.get("bootstrap", {}),
                "permutation": robustness.get("permutation", {}),
                "log_simple": log_simple,
                "log_enriched": log_enriched,
                "lr_test": lr_test,
                "segmentation": segmentation,
                "temporal": temporal,
                "gs": gs,
                "verdict": verdict,
                "ctrl_label": ctrl_label,
                "trt_label": trt_label,
                "target_col": target_col,
                "group_col": group_col,
                "control_value": control_value,
                "covariate_cols": cov_cols,
                "time_col": time_col,
                "id_col": id_col,
            }

            schema = _schema_banner(
                target_col, group_col, control_value, cov_cols, time_col, id_col, alpha,
            )
            verdict_el = _verdict_banner(verdict)
            kpi = _kpi_row(
                control_rate * 100, treatment_rate * 100,
                ctrl_label, trt_label,
                gs["n_control"], gs["n_treatment"],
                diff_pp, rel_pct,
                bayesian.get("p_treatment_better", 0),
                p_value, sig_two, alpha,
            )

            return {"display": "block"}, schema, verdict_el, kpi, html.Div(), results

        except Exception as e:
            results = {"error": str(e)}
            return (
                {"display": "block"},
                html.Div("Error during analysis"),
                html.Div(f"Analysis failed: {e}",
                         className="info-box", style={"color": "var(--color-danger)"}),
                [],
                html.Div(),
                results,
            )


def register_data_callbacks(app):
    @app.callback(
        Output("target-col-dropdown", "options"),
        Output("group-col-dropdown", "options"),
        Output("control-value-input", "options"),
        Output("covariate-multi", "options"),
        Output("time-col-dropdown", "options"),
        Output("id-col-dropdown", "options"),
        Output("data-preview", "children"),
        Output("load-feedback", "children"),
        Output("store-analysis", "data"),
        Input("file-upload", "contents"),
        State("file-upload", "filename"),
        State("store-analysis", "data"),
        prevent_initial_call=True,
    )
    def load_data(contents, filename, store):
        from src.data_cache import store as cache_store
        try:
            if not contents or not filename:
                return [no_update] * 7 + [
                    html.Div("No data source selected.",
                             style={"color": "var(--color-danger)", "fontSize": "0.8rem"}),
                    store,
                ]

            content_type, content_string = contents.split(",")
            decoded = io.BytesIO(base64.b64decode(content_string))
            if filename.endswith(".csv"):
                df = pd.read_csv(decoded)
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(decoded)
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(decoded, engine="openpyxl")
            elif filename.endswith(".json"):
                df = pd.read_json(decoded)
            else:
                raise ValueError(f"Unsupported format: {filename}")
            file_label = filename

            data_key = cache_store(df)
            cols = df.columns.tolist()

            def opts(items):
                return [{"label": c, "value": c} for c in items]

            all_opts = opts(cols)
            control_opts = []
            cov_opts = [{"label": c, "value": c} for c in cols]
            time_opts = [{"label": "(None)", "value": "(None)"}] + opts(cols)
            id_opts = [{"label": "(None)", "value": "(None)"}] + opts(cols)

            preview = html.Div([
                html.Div(f"Loaded: {file_label} — {len(df):,} rows \u00d7 {len(df.columns)} columns",
                         style={"fontWeight": 600, "marginBottom": "8px"}),
                html.Div(
                    f"Target candidates: {', '.join(df.select_dtypes(include='number').columns[:5])}",
                    style={"color": "var(--text-secondary)", "fontSize": "0.85rem"},
                ),
            ])
            feedback = html.Div(f"Loaded {len(df):,} rows \u00d7 {len(df.columns)} columns",
                                style={"color": "var(--color-success)", "fontSize": "0.8rem"})

            store = store or {}
            store["data_key"] = data_key
            store["columns"] = cols

            return all_opts, all_opts, control_opts, cov_opts, time_opts, id_opts, preview, feedback, store

        except Exception as e:
            return [no_update] * 7 + [
                html.Div(f"Error: {e}", style={"color": "var(--color-danger)", "fontSize": "0.8rem"}),
                store,
            ]

    @app.callback(
        Output("control-value-input", "options", allow_duplicate=True),
        Input("group-col-dropdown", "value"),
        State("store-analysis", "data"),
        prevent_initial_call=True,
    )
    def update_control_options(group_col, store):
        if not group_col or not store:
            return []
        from src.data_cache import get as cache_get
        df = cache_get(store.get("data_key", ""))
        if df is None or group_col not in df.columns:
            return []
        values = sorted(str(v) for v in df[group_col].unique())
        return [{"label": v, "value": v} for v in values]
