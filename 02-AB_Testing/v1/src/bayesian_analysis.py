from __future__ import annotations
import numpy as np
from src.data_loader import get_group_stats

_ALPHA = 1
_BETA = 1
_N_SIMS = 20000


def beta_binomial_analysis(prepared: dict) -> dict:
    gs = get_group_stats(prepared)

    alpha_post_c = _ALPHA + gs["conv_control"]
    beta_post_c = _BETA + (gs["n_control"] - gs["conv_control"])
    alpha_post_t = _ALPHA + gs["conv_treatment"]
    beta_post_t = _BETA + (gs["n_treatment"] - gs["conv_treatment"])

    np.random.seed(42)
    control_sim = np.random.beta(alpha_post_c, beta_post_c, _N_SIMS)
    treatment_sim = np.random.beta(alpha_post_t, beta_post_t, _N_SIMS)

    diff_sim = treatment_sim - control_sim

    p_better = float(np.mean(treatment_sim > control_sim))
    p_better_rope = float(np.mean(diff_sim > 0.001))
    p_worse_rope = float(np.mean(diff_sim < -0.001))
    p_rope = float(np.mean((diff_sim >= -0.001) & (diff_sim <= 0.001)))

    expected_loss = float(np.mean(np.maximum(0, control_sim - treatment_sim)))

    ci_95 = (float(np.percentile(diff_sim, 2.5)), float(np.percentile(diff_sim, 97.5)))
    ci_90 = (float(np.percentile(diff_sim, 5.0)), float(np.percentile(diff_sim, 95.0)))

    if p_better_rope > 0.95:
        decision = "adopt_treatment"
    elif p_worse_rope > 0.95:
        decision = "keep_control"
    elif p_rope > 0.95:
        decision = "practical_equivalence"
    else:
        decision = "insufficient_evidence"

    return {
        "p_treatment_better": p_better,
        "p_treatment_better_pct": p_better * 100,
        "p_practical_significance": p_better_rope,
        "p_rope_region": p_rope,
        "p_worse": p_worse_rope,
        "expected_loss": expected_loss,
        "median_difference": float(np.median(diff_sim)),
        "mean_difference": float(np.mean(diff_sim)),
        "std_difference": float(np.std(diff_sim)),
        "ci_95": ci_95,
        "ci_90": ci_90,
        "decision": decision,
    }
