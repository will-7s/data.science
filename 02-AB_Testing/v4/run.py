from __future__ import annotations

import argparse
import json
from pathlib import Path

from dash import Dash
from flask_compress import Compress

from app.callbacks.analysis import register_analysis_callbacks, register_data_callbacks
from app.callbacks.export import register_export_callbacks
from app.callbacks.tabs import register_tab_callbacks
from app.layout import create_layout
from app.theme import STORAGE_KEY, register_theme_clientside_callback


def create_app() -> Dash:
    app = Dash(
        __name__,
        title="AB Testing App",
        suppress_callback_exceptions=True,
        update_title=None,
        serve_locally=False,
    )

    Compress().init_app(app.server)

    @app.server.after_request
    def _add_cache_headers(response):
        if response.content_type == "text/css":
            response.headers["Cache-Control"] = "public, max-age=86400, immutable"
        elif response.content_type.startswith("image/"):
            response.headers["Cache-Control"] = "public, max-age=604800, immutable"
        return response

    init_script = f"""
    (function() {{
        try {{
            var key = "{STORAGE_KEY}";
            var t = localStorage.getItem(key);
            if (t !== "light" && t !== "dark") {{
                t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
            }}
            document.documentElement.setAttribute("data-theme", t);
        }} catch(e) {{}}
    }})();
    """

    app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script>{init_script}</script>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
</head>
<body>
    {{%app_entry%}}
    <footer>
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
    </footer>
</body>
</html>
"""

    app.layout = create_layout()
    app.validation_layout = create_layout()
    register_theme_clientside_callback()
    register_data_callbacks(app)
    register_analysis_callbacks(app)
    register_tab_callbacks(app)
    register_export_callbacks(app)
    return app


def run_dash(debug: bool = False) -> None:
    app = create_app()
    app.run(debug=debug)


def export_json(data_path: str | None, output_path: str) -> None:
    if not data_path:
        msg = (
            "CLI export requires a data file. "
            "Usage: python run.py data.csv --export-json result.json"
        )
        raise NotImplementedError(msg)

    from config import config as cfg
    from src.bayesian_analysis import beta_binomial_analysis
    from src.confidence_score import compute_confidence_score
    from src.data_loader import get_group_stats, load_data, prepare_data
    from src.effect_size import compute_all_effect_sizes
    from src.hypothesis_testing import run_all_tests
    from src.power_analysis import compute_power as power_analysis
    from src.report_generator import generate_json_report
    from src.robustness import run_robustness_checks

    load_result = load_data(data_path)
    if load_result["error"]:
        raise RuntimeError(f"Failed to load data: {load_result['error']}")

    df = load_result["result"]["df"]
    target_col = load_result["result"]["overview"].get("target_column")
    if not target_col:
        raise RuntimeError("No numeric target column found")

    cols = df.columns.tolist()
    group_col = None
    for c in cols:
        if c != target_col and df[c].nunique() <= 10:
            group_col = c
            break
    if not group_col:
        group_col = [c for c in cols if c != target_col][0]

    control_value = str(df[group_col].iloc[0])

    prepared = prepare_data(df, target_col, group_col, control_value)
    tests = run_all_tests(prepared)
    bayesian = beta_binomial_analysis(prepared)
    effects = compute_all_effect_sizes(prepared)
    robustness = run_robustness_checks(prepared)

    gs = get_group_stats(prepared)
    n_avg = (gs["n_control"] + gs["n_treatment"]) // 2
    pwr = None
    if effects.get("cohens_d") is not None and abs(effects["cohens_d"]) > 0:
        pwr = power_analysis(effects["cohens_d"], n_avg, n_avg, alpha=1 - cfg.analysis.confidence_level)

    ci = robustness.get("bootstrap", {}).get("ci_95", (0, 0))
    ci_width = abs(ci[1] - ci[0]) if ci[0] is not None else 0
    pwr_power = pwr["power_observed"] if pwr and not pwr["skipped"] else 0
    confidence = compute_confidence_score(pwr_power, bayesian.get("p_treatment_better", 0), ci_width, cfg.sprt.effect_size)

    diff_pp = (gs["treatment_rate"] - gs["control_rate"]) * 100
    ztest = tests.get("ztest") or {}
    p_two = ztest.get("p_value")
    alpha_cli = 1 - cfg.analysis.confidence_level
    sig_two = p_two is not None and p_two < alpha_cli
    p_bayes = (bayesian.get("p_treatment_better") or 0)
    if sig_two and diff_pp > 0:
        verdict = "✅ DEPLOY — Statistically significant lift detected"
    elif sig_two and diff_pp < 0:
        verdict = "❌ REJECT — Control is significantly better"
    elif p_bayes > 0.95:
        verdict = "✅ DEPLOY — Bayesian probability > 95%"
    elif p_bayes > 0.80 or (p_two is not None and p_two < 0.10):
        verdict = "⚠️ INCONCLUSIVE — Trend detected, extend the test"
    else:
        verdict = "❌ No evidence of improvement — Keep control"

    results = {
        "verdict": verdict,
        "tests": tests,
        "bayesian": bayesian,
        "effect_sizes": effects,
        "robustness": robustness,
        "diff_pct": effects.get("absolute_difference"),
        "p_value": (tests.get("ztest") or tests.get("ttest") or {}).get("p_value"),
        "p_t_greater_c": bayesian.get("p_treatment_better"),
        "nnt": effects.get("nnt"),
        "overall_confidence": confidence.get("score"),
    }

    params = {"target_col": target_col, "group_col": group_col, "control_value": control_value}

    report = generate_json_report(results, params)
    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AB Testing App v4")
    parser.add_argument("--export-json", type=str, metavar="OUTPUT", help="Export analysis as JSON to path")
    parser.add_argument("--dev", action="store_true", help="Enable Dash debug mode (hot reload)")
    parser.add_argument("data", nargs="?", help="Data file for CLI export mode")
    args = parser.parse_args()

    if args.export_json:
        export_json(args.data, args.export_json)
    else:
        run_dash(debug=args.dev)
