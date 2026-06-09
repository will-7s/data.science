from __future__ import annotations
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from src.data_loader import get_group_stats


def _safe_ci(conv: int, n: int) -> tuple:
    if n < 1 or conv < 0:
        return (0.0, 0.0)
    try:
        return proportion_confint(conv, n, alpha=0.05, method="wilson")
    except Exception:
        rate = conv / n if n > 0 else 0.0
        return (rate, rate)


def compute_descriptive_stats(prepared: dict) -> dict:
    gs = get_group_stats(prepared)

    n_c = gs["n_control"]
    n_t = gs["n_treatment"]
    conv_c = gs["conv_control"]
    conv_t = gs["conv_treatment"]
    rate_c = gs["rate_control"]
    rate_t = gs["rate_treatment"]

    ci_c = _safe_ci(conv_c, n_c)
    ci_t = _safe_ci(conv_t, n_t)

    z = norm.ppf(0.975)
    se_diff = np.sqrt(
        rate_c * (1 - rate_c) / max(n_c, 1) +
        rate_t * (1 - rate_t) / max(n_t, 1)
    )
    diff = rate_t - rate_c
    ci_diff = (diff - z * se_diff, diff + z * se_diff)

    uniq = np.unique(prepared["group"])
    ctrl_label = str(uniq[0]) if len(uniq) > 0 else "control"
    trt_label = str(uniq[1]) if len(uniq) > 1 else "treatment"

    return {
        "control": {
            "n": n_c,
            "conversions": conv_c,
            "rate": rate_c,
            "rate_pct": rate_c * 100,
            "ci_95": (float(ci_c[0]), float(ci_c[1])),
            "label": ctrl_label,
        },
        "treatment": {
            "n": n_t,
            "conversions": conv_t,
            "rate": rate_t,
            "rate_pct": rate_t * 100,
            "ci_95": (float(ci_t[0]), float(ci_t[1])),
            "label": trt_label,
        },
        "difference": {
            "absolute": diff,
            "absolute_pp": diff * 100,
            "relative": diff / rate_c if rate_c != 0 else 0,
            "relative_pct": diff / rate_c * 100 if rate_c != 0 else 0,
            "ratio": rate_t / rate_c if rate_c != 0 else 0,
            "ci_95": (float(ci_diff[0]), float(ci_diff[1])),
        },
    }
