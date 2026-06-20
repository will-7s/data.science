from __future__ import annotations
import numpy as np
from config import (
    BAYESIAN_PRIOR_ALPHA, BAYESIAN_PRIOR_BETA,
    BAYESIAN_SIMULATIONS, ROPE_LOWER, ROPE_UPPER, RANDOM_SEED,
)
from src.data_loader import get_group_stats


def beta_binomial_analysis(prepared: dict) -> dict:
    """Beta-Binomial A/B test with ROPE decision framework.

    FIX: now reads all constants (prior, n_sims, ROPE bounds) from config.py
    instead of using private module-level constants that ignored config.
    ROPE bounds widened from ±0.001 to ±0.002 (configurable) to reflect
    realistic practical equivalence thresholds for conversion rate contexts.
    """
    gs = get_group_stats(prepared)

    alpha_c = BAYESIAN_PRIOR_ALPHA + gs["conv_control"]
    beta_c  = BAYESIAN_PRIOR_BETA  + (gs["n_control"]   - gs["conv_control"])
    alpha_t = BAYESIAN_PRIOR_ALPHA + gs["conv_treatment"]
    beta_t  = BAYESIAN_PRIOR_BETA  + (gs["n_treatment"]  - gs["conv_treatment"])

    rng = np.random.default_rng(RANDOM_SEED)
    ctrl_sim = rng.beta(alpha_c, beta_c, BAYESIAN_SIMULATIONS)
    trt_sim  = rng.beta(alpha_t, beta_t, BAYESIAN_SIMULATIONS)
    diff_sim = trt_sim - ctrl_sim

    p_better       = float(np.mean(trt_sim  > ctrl_sim))
    p_better_rope  = float(np.mean(diff_sim > ROPE_UPPER))
    p_worse_rope   = float(np.mean(diff_sim < ROPE_LOWER))
    p_rope         = float(np.mean((diff_sim >= ROPE_LOWER) & (diff_sim <= ROPE_UPPER)))
    expected_loss  = float(np.mean(np.maximum(0.0, ctrl_sim - trt_sim)))

    ci_95 = (float(np.percentile(diff_sim, 2.5)),  float(np.percentile(diff_sim, 97.5)))
    ci_90 = (float(np.percentile(diff_sim, 5.0)),  float(np.percentile(diff_sim, 95.0)))

    if p_better_rope > 0.95:
        decision = "adopt_treatment"
    elif p_worse_rope > 0.95:
        decision = "keep_control"
    elif p_rope > 0.95:
        decision = "practical_equivalence"
    else:
        decision = "insufficient_evidence"

    return {
        "p_treatment_better":      p_better,
        "p_treatment_better_pct":  p_better * 100,
        "p_practical_significance": p_better_rope,
        "p_rope_region":           p_rope,
        "p_worse":                 p_worse_rope,
        "expected_loss":           expected_loss,
        "median_difference":       float(np.median(diff_sim)),
        "mean_difference":         float(np.mean(diff_sim)),
        "std_difference":          float(np.std(diff_sim)),
        "ci_95":                   ci_95,
        "ci_90":                   ci_90,
        "decision":                decision,
        "rope_lower":              ROPE_LOWER,
        "rope_upper":              ROPE_UPPER,
        "n_simulations":           BAYESIAN_SIMULATIONS,
    }
