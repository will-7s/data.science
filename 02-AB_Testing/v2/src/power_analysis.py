"""power_analysis.py — centralised power & MDE calculations.

FIX / NEW MODULE: previously, power analysis was duplicated between run.py
and app/dashboard.py with subtle differences (ratio handling, error handling).
Both now import from here.
"""
from __future__ import annotations
import numpy as np
from statsmodels.stats.power import NormalIndPower


def compute_power(
    cohens_h: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> dict:
    """Compute observed power, n needed for 80%, and MDE.

    Parameters
    ----------
    cohens_h : effect size (signed — abs taken internally)
    n_control, n_treatment : observed group sizes
    alpha : significance level
    """
    pa = NormalIndPower()
    es = abs(cohens_h)

    if es < 1e-9 or n_control < 2 or n_treatment < 2:
        return {
            "power_observed":  0.0,
            "n_needed_80pct":  float("inf"),
            "mde_cohens_h":    float("nan"),
            "mde_pp":          float("nan"),
            "alpha":           alpha,
            "skipped":         True,
        }

    ratio = n_treatment / max(n_control, 1)

    try:
        power_obs = float(pa.solve_power(
            effect_size=es, nobs1=n_control,
            ratio=ratio, alpha=alpha, alternative="two-sided",
        ))
        n_needed = float(pa.solve_power(
            effect_size=es, power=0.80,
            alpha=alpha, ratio=1.0, alternative="two-sided",
        ))
        mde_h = float(pa.solve_power(
            nobs1=n_control, ratio=ratio,
            power=0.80, alpha=alpha, alternative="two-sided",
        ))
        # Approximate MDE in percentage points at current base rate
        # (uses sin² transform inversion)
        mde_pp = float(abs(2 * np.sin(mde_h / 2)))
    except Exception:
        return {
            "power_observed":  0.0,
            "n_needed_80pct":  float("inf"),
            "mde_cohens_h":    float("nan"),
            "mde_pp":          float("nan"),
            "alpha":           alpha,
            "skipped":         True,
        }

    return {
        "power_observed":  power_obs,
        "n_needed_80pct":  n_needed,
        "mde_cohens_h":    mde_h,
        "mde_pp":          mde_pp,
        "alpha":           alpha,
        "skipped":         False,
    }
