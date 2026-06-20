from __future__ import annotations
import numpy as np
from src.data_loader import get_group_stats


def cohens_h(rate_control: float, rate_treatment: float) -> float:
    h = 2 * np.arcsin(np.sqrt(rate_treatment)) - 2 * np.arcsin(np.sqrt(rate_control))
    return float(h)


def interpret_cohens_h(h: float) -> str:
    if abs(h) < 0.2:
        return "trivial"
    elif abs(h) < 0.5:
        return "small"
    elif abs(h) < 0.8:
        return "medium"
    else:
        return "large"


def odds_ratio(rate_control: float, rate_treatment: float,
               conv_control: int, n_control: int,
               conv_treatment: int, n_treatment: int) -> dict:
    odds_c = rate_control / (1 - rate_control) if rate_control < 1 else float("inf")
    odds_t = rate_treatment / (1 - rate_treatment) if rate_treatment < 1 else float("inf")

    if odds_c == 0 or odds_t == 0 or odds_c == float("inf") or odds_t == float("inf"):
        return {"or": float("nan"), "ci_95": (float("nan"), float("nan"))}

    or_val = odds_t / odds_c

    err1 = 1 / conv_control if conv_control > 0 else 0
    err2 = 1 / (n_control - conv_control) if (n_control - conv_control) > 0 else 0
    err3 = 1 / conv_treatment if conv_treatment > 0 else 0
    err4 = 1 / (n_treatment - conv_treatment) if (n_treatment - conv_treatment) > 0 else 0
    se_val = err1 + err2 + err3 + err4
    if se_val <= 0:
        return {"or": float(or_val), "ci_95": (float(or_val), float(or_val))}
    se = np.sqrt(se_val)
    ci_low = np.exp(np.log(or_val) - 1.96 * se)
    ci_high = np.exp(np.log(or_val) + 1.96 * se)
    return {"or": float(or_val), "ci_95": (float(ci_low), float(ci_high))}


def risk_ratio(rate_control: float, rate_treatment: float) -> float:
    return float(rate_treatment / rate_control) if rate_control != 0 else float("nan")


def nnt(absolute_diff: float) -> float:
    return float(abs(1 / absolute_diff)) if absolute_diff != 0 else float("inf")


def phi_coefficient(chi2: float, n: int) -> float:
    return float(np.sqrt(chi2 / n)) if n > 0 else 0.0


def compute_all_effect_sizes(prepared: dict) -> dict:
    from src.hypothesis_testing import chi_squared_test

    gs = get_group_stats(prepared)
    h = cohens_h(gs["rate_control"], gs["rate_treatment"])
    or_res = odds_ratio(
        gs["rate_control"], gs["rate_treatment"],
        gs["conv_control"], gs["n_control"],
        gs["conv_treatment"], gs["n_treatment"],
    )
    rr = risk_ratio(gs["rate_control"], gs["rate_treatment"])
    nnt_v = nnt(gs["absolute_diff"])
    chi2 = chi_squared_test(prepared)["statistic"]
    phi = phi_coefficient(chi2, gs["n_control"] + gs["n_treatment"])

    return {
        "cohens_h": h,
        "cohens_h_interpretation": interpret_cohens_h(h),
        "odds_ratio": or_res["or"],
        "odds_ratio_ci_95": or_res["ci_95"],
        "risk_ratio": rr,
        "nnt": nnt_v,
        "phi_coefficient": phi,
    }
