from __future__ import annotations
import numpy as np
from src.data_loader import get_group_stats


def _bootstrap_n(n: int) -> int:
    if n > 10000:
        return 500
    return 2000


def _bootstrap_sample_size(n: int) -> int:
    return min(n, 5000)


def bootstrap_ci(prepared: dict) -> dict:
    target = prepared["target"]
    c_mask = prepared["control_mask"]
    t_mask = prepared["treatment_mask"]

    control_vals = target[c_mask]
    treatment_vals = target[t_mask]
    n_c = len(control_vals)
    n_t = len(treatment_vals)

    if n_c == 0 or n_t == 0:
        return {
            "ci_95": (float("nan"), float("nan")),
            "ci_90": (float("nan"), float("nan")),
            "pct_positive": 0.0,
            "pct_negative": 0.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }

    n_iter = _bootstrap_n(max(n_c, n_t))
    sample_c = _bootstrap_sample_size(n_c)
    sample_t = _bootstrap_sample_size(n_t)

    np.random.seed(42)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        boot_c = np.random.choice(control_vals, size=sample_c, replace=True)
        boot_t = np.random.choice(treatment_vals, size=sample_t, replace=True)
        diffs[i] = float(np.nanmean(boot_t) - np.nanmean(boot_c))

    return {
        "ci_95": (
            float(np.nanpercentile(diffs, 2.5)),
            float(np.nanpercentile(diffs, 97.5)),
        ),
        "ci_90": (
            float(np.nanpercentile(diffs, 5.0)),
            float(np.nanpercentile(diffs, 95.0)),
        ),
        "pct_positive": float(np.nanmean(diffs > 0) * 100),
        "pct_negative": float(np.nanmean(diffs < 0) * 100),
        "mean_diff": float(np.nanmean(diffs)),
        "std_diff": float(np.nanstd(diffs)),
    }


def permutation_test(prepared: dict) -> dict:
    gs = get_group_stats(prepared)
    observed_diff = gs["absolute_diff"]
    target = prepared["target"]
    groups_code = np.where(prepared["control_mask"], 0, 1)

    n_c = int((groups_code == 0).sum())
    n_t = int((groups_code == 1).sum())
    if n_c == 0 or n_t == 0:
        return {
            "p_value_two_sided": 1.0, "p_value_one_sided": 0.5,
            "significant_two_sided": False, "significant_one_sided": False,
            "observed_diff": float(observed_diff),
        }

    n_total = n_c + n_t
    n_iter = _bootstrap_n(n_total)

    np.random.seed(42)
    perm_diffs = np.empty(n_iter)
    for i in range(n_iter):
        perm = np.random.permutation(n_total)
        perm_c = np.nanmean(target[perm[:n_c]])
        perm_t = np.nanmean(target[perm[n_c:]])
        perm_diffs[i] = float(perm_t - perm_c)

    p_two = float((np.abs(perm_diffs) >= np.abs(observed_diff)).mean())
    p_one = float((perm_diffs >= observed_diff).mean())

    return {
        "p_value_two_sided": p_two,
        "p_value_one_sided": p_one,
        "significant_two_sided": p_two < 0.05,
        "significant_one_sided": p_one < 0.05,
        "observed_diff": float(observed_diff),
    }


def run_robustness_checks(prepared: dict) -> dict:
    return {
        "bootstrap": bootstrap_ci(prepared),
        "permutation": permutation_test(prepared),
    }
