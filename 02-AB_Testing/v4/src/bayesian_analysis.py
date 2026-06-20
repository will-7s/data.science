from __future__ import annotations

import numpy as np
import scipy.stats

from config import (
    BAYESIAN_PRIOR_ALPHA,
    BAYESIAN_PRIOR_BETA,
    BAYESIAN_SIMULATIONS,
    RANDOM_SEED,
    ROPE_LOWER,
    ROPE_UPPER,
)


def normal_normal_analysis(prepared: dict, gs: dict | None = None) -> dict:
    # Modèle bayésien Normal-Normal pour données continues (non-binaires).
    # Prior : N(μ₀, σ₀²) où μ₀ = moyenne globale, σ₀ = 3×σ_global (prior faiblement informatif).
    # Posterior : N(μ_post, σ_post²) — formule conjuguée fermée (pas de simulation MCMC).
    # Retourne : P(T > C), intervalles de crédibilité, ROPE, expected loss.
    # Avantage : solution analytique instantanée (pas d'itérations).
    if gs is None:
        from src.data_loader import get_group_stats
        gs = get_group_stats(prepared)
    all_vals = prepared["target"][~np.isnan(prepared["target"])]
    mu_0 = float(np.mean(all_vals)) if len(all_vals) > 0 else 0.0
    sigma_0 = float(np.std(all_vals, ddof=1)) * 3 if len(all_vals) > 1 else 10.0

    ctrl_vals = prepared["target"][prepared["control_mask"]]
    trt_vals = prepared["target"][prepared["treatment_mask"]]
    n_c = len(ctrl_vals)
    n_t = len(trt_vals)
    mean_c = float(np.mean(ctrl_vals)) if n_c > 0 else 0.0
    mean_t = float(np.mean(trt_vals)) if n_t > 0 else 0.0
    var_c = float(np.var(ctrl_vals, ddof=1)) if n_c > 1 else 1.0
    var_t = float(np.var(trt_vals, ddof=1)) if n_t > 1 else 1.0
    se_c = var_c / max(n_c, 1)
    se_t = var_t / max(n_t, 1)

    post_var_c = 1.0 / (1.0 / sigma_0**2 + 1.0 / se_c) if se_c > 0 else sigma_0**2
    post_var_t = 1.0 / (1.0 / sigma_0**2 + 1.0 / se_t) if se_t > 0 else sigma_0**2
    post_mean_c = post_var_c * (mu_0 / sigma_0**2 + mean_c / se_c) if se_c > 0 else mu_0
    post_mean_t = post_var_t * (mu_0 / sigma_0**2 + mean_t / se_t) if se_t > 0 else mu_0

    diff_mean = post_mean_t - post_mean_c
    diff_var = post_var_t + post_var_c
    diff_std = np.sqrt(diff_var)

    if diff_std > 0:
        p_better = float(1.0 - scipy.stats.norm.cdf(0.0, loc=diff_mean, scale=diff_std))
        p_better_rope = float(
            1.0 - scipy.stats.norm.cdf(ROPE_UPPER, loc=diff_mean, scale=diff_std)
        )
        p_worse_rope = float(scipy.stats.norm.cdf(ROPE_LOWER, loc=diff_mean, scale=diff_std))
        p_rope = float(
            scipy.stats.norm.cdf(ROPE_UPPER, loc=diff_mean, scale=diff_std)
            - scipy.stats.norm.cdf(ROPE_LOWER, loc=diff_mean, scale=diff_std)
        )
        ci_95 = (
            float(diff_mean - 1.96 * diff_std),
            float(diff_mean + 1.96 * diff_std),
        )
        ci_90 = (
            float(diff_mean - 1.645 * diff_std),
            float(diff_mean + 1.645 * diff_std),
        )
    else:
        p_better = 0.5
        p_better_rope = 0.0
        p_worse_rope = 0.0
        p_rope = 1.0
        ci_95 = (diff_mean, diff_mean)
        ci_90 = (diff_mean, diff_mean)

    expected_loss = 0.0
    if diff_std > 0:
        z = -diff_mean / diff_std
        expected_loss = float(
            diff_std * (scipy.stats.norm.pdf(z) - z * (1.0 - scipy.stats.norm.cdf(z)))
        )

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
        "median_difference": diff_mean,
        "mean_difference": diff_mean,
        "std_difference": diff_std,
        "ci_95": ci_95,
        "ci_90": ci_90,
        "decision": decision,
        "rope_lower": ROPE_LOWER,
        "rope_upper": ROPE_UPPER,
        "n_simulations": 0,
        "posterior_control": {"mean": post_mean_c, "var": post_var_c},
        "posterior_treatment": {"mean": post_mean_t, "var": post_var_t},
    }


def beta_binomial_analysis(prepared: dict, gs: dict | None = None) -> dict:
    # Modèle bayésien Beta-Binomial pour données binaires (conversion 0/1).
    # Prior : Beta(α₀, β₀) avec α₀ = β₀ = 1 (uniforme = non-informatif).
    # Posterior : Beta(α₀ + succès, β₀ + échecs) — conjuguaison fermée.
    # La distribution de la différence est simulée (n = BAYESIAN_SIMULATIONS
    # tirages Monte Carlo) pour calculer P(T > C), ROPE, intervalles, etc.
    # Avantage : interprétation intuitive et quantification de l'incertitude.
    if gs is None:
        from src.data_loader import get_group_stats
        gs = get_group_stats(prepared)

    if not prepared.get("is_binary", False):
        return {
            "p_treatment_better": 0.5,
            "p_treatment_better_pct": 50.0,
            "p_practical_significance": 0.0,
            "p_rope_region": 0.0,
            "p_worse": 0.0,
            "expected_loss": 0.0,
            "median_difference": float(gs["absolute_diff"]),
            "mean_difference": float(gs["absolute_diff"]),
            "std_difference": 0.0,
            "ci_95": (None, None),
            "ci_90": (None, None),
            "decision": "insufficient_evidence",
            "rope_lower": ROPE_LOWER,
            "rope_upper": ROPE_UPPER,
            "n_simulations": 0,
            "posterior_control": {"alpha": 0, "beta": 0},
            "posterior_treatment": {"alpha": 0, "beta": 0},
        }

    alpha_c = BAYESIAN_PRIOR_ALPHA + gs["conv_control"]
    beta_c = BAYESIAN_PRIOR_BETA + (gs["n_control"] - gs["conv_control"])
    alpha_t = BAYESIAN_PRIOR_ALPHA + gs["conv_treatment"]
    beta_t = BAYESIAN_PRIOR_BETA + (gs["n_treatment"] - gs["conv_treatment"])

    rng = np.random.default_rng(RANDOM_SEED)
    ctrl_sim = rng.beta(alpha_c, beta_c, BAYESIAN_SIMULATIONS)
    trt_sim = rng.beta(alpha_t, beta_t, BAYESIAN_SIMULATIONS)
    diff_sim = trt_sim - ctrl_sim

    p_better = float(np.mean(trt_sim > ctrl_sim))
    p_better_rope = float(np.mean(diff_sim > ROPE_UPPER))
    p_worse_rope = float(np.mean(diff_sim < ROPE_LOWER))
    p_rope = float(np.mean((diff_sim >= ROPE_LOWER) & (diff_sim <= ROPE_UPPER)))
    expected_loss = float(np.mean(np.maximum(0.0, ctrl_sim - trt_sim)))

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
        "rope_lower": ROPE_LOWER,
        "rope_upper": ROPE_UPPER,
        "n_simulations": BAYESIAN_SIMULATIONS,
        "posterior_control": {"alpha": float(alpha_c), "beta": float(beta_c)},
        "posterior_treatment": {"alpha": float(alpha_t), "beta": float(beta_t)},
    }
