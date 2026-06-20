from __future__ import annotations
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from config import ALPHA
from src.data_loader import get_group_stats


def chi_squared_test(prepared: dict) -> dict:
    gs = get_group_stats(prepared)
    n_c, n_t = gs["n_control"], gs["n_treatment"]
    conv_c, conv_t = gs["conv_control"], gs["conv_treatment"]
    contingency = np.array([
        [n_c - conv_c, conv_c],
        [n_t - conv_t, conv_t],
    ])
    if (contingency < 0).any() or n_c == 0 or n_t == 0:
        return {"statistic": 0.0, "p_value": 1.0, "dof": 1, "significant": False}
    try:
        chi2, p, dof, _ = chi2_contingency(contingency)
        return {
            "statistic": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "significant": bool(p < ALPHA),
        }
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0, "dof": 1, "significant": False}


def z_test_proportions(prepared: dict) -> dict:
    """FIX: one-sided test now correctly tests H₁: treatment > control.

    Previous code passed [conv_control, conv_treatment] with alternative='larger',
    which tested H₁: p_control > p_treatment — the wrong direction.

    Fix: pass [conv_treatment, conv_control] with alternative='larger' so that
    the first proportion (treatment) is the one tested as being larger.
    The two-sided statistic is identical; only the one-sided p-value changes.
    """
    gs = get_group_stats(prepared)
    n_c, n_t = gs["n_control"], gs["n_treatment"]
    if n_c == 0 or n_t == 0:
        return {
            "z_statistic_two_sided": 0.0, "p_value_two_sided": 1.0,
            "significant_two_sided": False,
            "z_statistic_one_sided": 0.0, "p_value_one_sided": 0.5,
            "significant_one_sided": False,
        }

    # Two-sided: order does not matter for the statistic magnitude
    counts_two = np.array([gs["conv_control"], gs["conv_treatment"]])
    nobs_two   = np.array([n_c, n_t])

    # One-sided H₁: treatment > control → treatment is first element
    counts_one = np.array([gs["conv_treatment"], gs["conv_control"]])
    nobs_one   = np.array([n_t, n_c])

    try:
        stat_two, p_two = proportions_ztest(counts_two, nobs_two, alternative="two-sided")
        stat_one, p_one = proportions_ztest(counts_one, nobs_one, alternative="larger")
    except Exception:
        return {
            "z_statistic_two_sided": 0.0, "p_value_two_sided": 1.0,
            "significant_two_sided": False,
            "z_statistic_one_sided": 0.0, "p_value_one_sided": 0.5,
            "significant_one_sided": False,
        }

    return {
        "z_statistic_two_sided": float(stat_two),
        "p_value_two_sided":     float(p_two),
        "significant_two_sided": bool(p_two < ALPHA),
        "z_statistic_one_sided": float(stat_one),
        "p_value_one_sided":     float(p_one),
        "significant_one_sided": bool(p_one < ALPHA),
    }


def t_test_independent(prepared: dict) -> dict:
    target = prepared["target"]
    c_vals = target[prepared["control_mask"]]
    t_vals = target[prepared["treatment_mask"]]
    if len(c_vals) < 2 or len(t_vals) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        t_stat, p = ttest_ind(t_vals, c_vals, alternative="two-sided")
    except Exception:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
    return {
        "t_statistic": float(t_stat),
        "p_value":     float(p),
        "significant": bool(p < ALPHA),
    }


def mann_whitney_u_test(prepared: dict) -> dict:
    target = prepared["target"]
    c_vals = target[prepared["control_mask"]]
    t_vals = target[prepared["treatment_mask"]]
    if len(c_vals) < 2 or len(t_vals) < 2:
        return {"u_statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        u_stat, p = mannwhitneyu(t_vals, c_vals, alternative="two-sided")
    except Exception:
        return {"u_statistic": 0.0, "p_value": 1.0, "significant": False}
    return {
        "u_statistic": float(u_stat),
        "p_value":     float(p),
        "significant": bool(p < ALPHA),
    }


def run_all_tests(prepared: dict) -> dict:
    return {
        "chi_squared":  chi_squared_test(prepared),
        "z_test":       z_test_proportions(prepared),
        "t_test":       t_test_independent(prepared),
        "mann_whitney": mann_whitney_u_test(prepared),
    }
