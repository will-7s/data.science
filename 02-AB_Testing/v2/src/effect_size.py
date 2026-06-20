from __future__ import annotations
import numpy as np
from src.data_loader import get_group_stats


def cohens_h(rate_control: float, rate_treatment: float) -> float:
    return float(
        2 * np.arcsin(np.sqrt(rate_treatment)) -
        2 * np.arcsin(np.sqrt(rate_control))
    )


def interpret_cohens_h(h: float) -> str:
    ah = abs(h)
    if ah < 0.2:   return "trivial"
    if ah < 0.5:   return "small"
    if ah < 0.8:   return "medium"
    return "large"


def odds_ratio(rate_control, rate_treatment,
               conv_control, n_control,
               conv_treatment, n_treatment) -> dict:
    if rate_control >= 1 or rate_treatment >= 1 or rate_control == 0:
        return {"or": float("nan"), "ci_95": (float("nan"), float("nan"))}
    odds_c = rate_control  / (1 - rate_control)
    odds_t = rate_treatment / (1 - rate_treatment)
    if odds_c == 0:
        return {"or": float("nan"), "ci_95": (float("nan"), float("nan"))}
    or_val = odds_t / odds_c

    # Woolf SE: sqrt(1/a + 1/b + 1/c + 1/d)
    a, b = conv_control,   n_control   - conv_control
    c, d = conv_treatment, n_treatment - conv_treatment
    if 0 in (a, b, c, d):
        return {"or": float(or_val), "ci_95": (float("nan"), float("nan"))}
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    log_or = np.log(or_val)
    return {
        "or":     float(or_val),
        "ci_95":  (float(np.exp(log_or - 1.96 * se)),
                   float(np.exp(log_or + 1.96 * se))),
    }


def risk_ratio(rate_control: float, rate_treatment: float) -> float:
    return float(rate_treatment / rate_control) if rate_control != 0 else float("nan")


def nnt(absolute_diff: float) -> dict:
    """FIX: distinguish NNT (positive diff) from NNH (negative diff).

    Returns a dict with the value and its label so callers can display
    the correct terminology instead of always calling it 'NNT'.
    """
    if absolute_diff == 0:
        return {"value": float("inf"), "label": "NNT", "direction": "neutral"}
    value = abs(1.0 / absolute_diff)
    if absolute_diff > 0:
        return {"value": float(value), "label": "NNT", "direction": "beneficial"}
    return {"value": float(value), "label": "NNH", "direction": "harmful"}


def phi_coefficient(chi2: float, n: int) -> float:
    return float(np.sqrt(chi2 / n)) if n > 0 else 0.0


def compute_all_effect_sizes(prepared: dict) -> dict:
    from src.hypothesis_testing import chi_squared_test

    gs  = get_group_stats(prepared)
    h   = cohens_h(gs["rate_control"], gs["rate_treatment"])
    or_ = odds_ratio(
        gs["rate_control"], gs["rate_treatment"],
        gs["conv_control"], gs["n_control"],
        gs["conv_treatment"], gs["n_treatment"],
    )
    rr    = risk_ratio(gs["rate_control"], gs["rate_treatment"])
    nnt_  = nnt(gs["absolute_diff"])
    chi2  = chi_squared_test(prepared)["statistic"]
    phi   = phi_coefficient(chi2, gs["n_control"] + gs["n_treatment"])

    return {
        "cohens_h":                h,
        "cohens_h_interpretation": interpret_cohens_h(h),
        "odds_ratio":              or_["or"],
        "odds_ratio_ci_95":        or_["ci_95"],
        "risk_ratio":              rr,
        "nnt":                     nnt_["value"],
        "nnt_label":               nnt_["label"],    # "NNT" or "NNH"
        "nnt_direction":           nnt_["direction"],
        "phi_coefficient":         phi,
    }
