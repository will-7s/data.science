from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm


def compute_power_vectorized(
    effect_size: float,
    n_values: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    # Calcule la puissance statistique pour un vecteur de tailles d'échantillon.
    # Formule : puissance = Φ(Z_α/2 − δ×√(n/2)) + Φ(−Z_α/2 − δ×√(n/2))
    # où δ = taille d'effet (Cohen's h), Φ = fonction de répartition normale.
    # Utilisé pour tracer la courbe de puissance (power curve).
    es = abs(effect_size)
    z_crit = float(norm.ppf(1 - alpha / 2))
    ncp = es * np.sqrt(n_values / 2)
    power = np.clip(1 - ndtr(z_crit - ncp) + ndtr(-z_crit - ncp), 0, 1)
    return power


def compute_power(
    effect_size: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> dict:
    es = abs(effect_size)
    if es < 1e-9 or n_control < 2 or n_treatment < 2:
        return {
            "power_observed": 0.0,
            "n_needed_80pct": float("inf"),
            "mde_cohens_h": float("nan"),
            "mde_pp": float("nan"),
            "alpha": alpha,
            "skipped": True,
        }

    n_avg = (n_control + n_treatment) // 2
    if n_avg < 1:
        return {
            "power_observed": 0.0,
            "n_needed_80pct": float("inf"),
            "mde_cohens_h": float("nan"),
            "mde_pp": float("nan"),
            "alpha": alpha,
            "skipped": True,
        }

    try:
        z_crit = float(norm.ppf(1 - alpha / 2))
        ncp = es * np.sqrt(n_avg / 2)
        power_obs = float(np.clip(1 - ndtr(z_crit - ncp) + ndtr(-z_crit - ncp), 0, 1))

        z_80 = float(norm.ppf(0.80))
        z_alpha = float(norm.ppf(1 - alpha / 2))
        n_needed_80 = max(2, int(np.ceil(2 * ((z_alpha + z_80) / es) ** 2)))

        mde_h = (z_alpha + z_80) / np.sqrt(n_avg / 2)
        mde_pp = float(abs(2 * np.sin(mde_h / 2)))
    except Exception:
        return {
            "power_observed": 0.0,
            "n_needed_80pct": float("inf"),
            "mde_cohens_h": float("nan"),
            "mde_pp": float("nan"),
            "alpha": alpha,
            "skipped": True,
        }

    return {
        "power_observed": power_obs,
        "n_needed_80pct": n_needed_80,
        "mde_cohens_h": float(mde_h),
        "mde_pp": mde_pp,
        "alpha": alpha,
        "skipped": False,
    }
