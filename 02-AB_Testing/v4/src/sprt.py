from __future__ import annotations

import math


def _bounds(alpha: float, beta: float) -> tuple[float, float]:
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    return upper, lower


def _log(x: float) -> float:
    if x <= 0:
        return -float("inf")
    return math.log(x)


def sprt(
    treatment_successes: int,
    treatment_total: int,
    control_successes: int,
    control_total: int,
    alpha: float = 0.05,
    beta: float = 0.20,
    effect_size: float = 0.05,
) -> dict:
    tr_s = int(treatment_successes)
    tr_n = int(treatment_total)
    ct_s = int(control_successes)
    ct_n = int(control_total)

    if tr_s < 0 or ct_s < 0:
        return {"result": None, "error": "successes must be non-negative"}
    if tr_n <= 0 or ct_n <= 0:
        return {"result": None, "error": "totals must be positive"}
    if tr_s > tr_n or ct_s > ct_n:
        return {"result": None, "error": "successes cannot exceed totals"}
    if not (0 < alpha < 1):
        return {"result": None, "error": "alpha must be between 0 and 1"}
    if not (0 < beta < 1):
        return {"result": None, "error": "beta must be between 0 and 1"}
    if effect_size <= 0:
        return {"result": None, "error": "effect_size must be positive"}

    p_pool = (tr_s + ct_s) / (tr_n + ct_n)
    if effect_size >= 2 * min(p_pool, 1 - p_pool):
        return {"result": None, "error": "effect_size too large for observed proportions"}

    p_tr_h0 = p_pool
    p_ct_h0 = p_pool
    p_tr_h1 = p_pool + effect_size / 2
    p_ct_h1 = p_pool - effect_size / 2

    def _safe_bernoulli_llr(successes: int, total: int, p_h1: float, p_h0: float) -> float:
        if total == 0:
            return 0.0
        if successes == 0:
            return total * (_log(1 - p_h1) - _log(1 - p_h0))
        if successes == total:
            return total * (_log(p_h1) - _log(p_h0))
        return successes * (_log(p_h1) - _log(p_h0)) + (total - successes) * (
            _log(1 - p_h1) - _log(1 - p_h0)
        )

    llr = _safe_bernoulli_llr(tr_s, tr_n, p_tr_h1, p_tr_h0) + _safe_bernoulli_llr(
        ct_s, ct_n, p_ct_h1, p_ct_h0
    )

    upper, lower = _bounds(alpha, beta)

    if llr >= upper:
        decision = "accept_treatment"
    elif llr <= lower:
        decision = "accept_control"
    else:
        decision = "continue"

    return {
        "result": {
            "boundary_crossed": decision != "continue",
            "log_likelihood_ratio": float(llr),
            "upper_bound": float(upper),
            "lower_bound": float(lower),
            "decision": decision,
            "error": None,
        },
        "error": None,
    }
