def compute_confidence_score(
    power: float | None,
    p_t_greater_c: float | None,
    bootstrap_ci_width: float | None,
    mde: float,
) -> dict:
    if any(x is None for x in [power, p_t_greater_c, bootstrap_ci_width]):
        return {
            "score": "fragile",
            "conditions": {},
            "failed_conditions": ["invalid_input"],
        }
    if mde <= 0:
        return {
            "score": "fragile",
            "conditions": {},
            "failed_conditions": ["invalid_mde"],
        }

    conditions = {
        "power": bool(power >= 0.80),
        "p_T_greater_C": bool(p_t_greater_c >= 0.95),
        "bootstrap_stable": bool(bootstrap_ci_width <= 2 * mde),
    }
    failed = [k for k, v in conditions.items() if not v]

    if all(conditions.values()):
        score = "robust"
    elif power >= 0.60 and p_t_greater_c >= 0.80:
        score = "moderate"
    else:
        score = "fragile"

    return {
        "score": score,
        "conditions": conditions,
        "failed_conditions": failed,
    }
