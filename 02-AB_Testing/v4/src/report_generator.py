from __future__ import annotations

from datetime import datetime, timezone


def generate_json_report(results: dict, params: dict | None = None) -> dict:
    gs = results.get("descriptive_stats") or {}
    if isinstance(gs, dict) and "result" in gs:
        gs = gs["result"] or {}

    diff_pct = results.get("diff_pct")
    relative_diff = None
    n_groups = len(gs.get("n_per_group", {}))
    if diff_pct is not None and n_groups == 2:
        rates = list(gs.get("conversion_rate_per_group", {}).values())
        if len(rates) == 2 and rates[0] and rates[0] != 0:
            relative_diff = (rates[1] - rates[0]) / rates[0]

    return {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": results.get("verdict"),
        "summary_metrics": {
            "difference_pct": diff_pct,
            "relative_difference_pct": relative_diff,
            "p_value": results.get("p_value"),
            "P_T_greater_C": results.get("p_t_greater_c"),
            "nnt": results.get("nnt"),
        },
        "confidence_badges": {
            "overall": results.get("overall_confidence"),
        },
        "effect_sizes": {
            "cohens_d": (results.get("effect_sizes") or {}).get("cohens_d"),
            "risk_ratio": (results.get("effect_sizes") or {}).get("risk_ratio"),
            "nnt": (results.get("effect_sizes") or {}).get("nnt"),
        },
        "bayesian": {
            "P_T_greater_C": (results.get("bayesian") or {}).get("p_treatment_better"),
            "credible_interval": (results.get("bayesian") or {}).get("ci_95"),
            "decision": (results.get("bayesian") or {}).get("decision"),
            "expected_loss": (results.get("bayesian") or {}).get("expected_loss"),
        },
        "params": {
            "target_col": (params or {}).get("target_col"),
            "group_col": (params or {}).get("group_col"),
            "control_value": (params or {}).get("control_value"),
        },
    }
