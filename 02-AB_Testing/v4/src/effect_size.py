from __future__ import annotations

from math import ceil

import numpy as np


def cohens_d(treatment: np.ndarray, control: np.ndarray) -> dict:
    # d de Cohen standardisé : (moyenne_t - moyenne_c) / écart-type mutualisé.
    # Interprétation (convention) :
    #   |d| < 0.2 → négligeable
    #   |d| < 0.5 → petit
    #   |d| < 0.8 → moyen
    #   |d| ≥ 0.8 → grand
    # Pour les proportions binaires, utiliser Cohen's h à la place :
    #   h = 2 × arcsin(√p_t) − 2 × arcsin(√p_c)
    #   (non implémenté ici — le d de Cohen reste une bonne approximation).
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if treatment.size == 0 or control.size == 0:
        return {"result": None, "error": "arrays must be non-empty"}
    if treatment.size < 2 or control.size < 2:
        return {"result": None, "error": "each group must have at least 2 observations"}

    m1, m2 = treatment.mean(), control.mean()
    v1, v2 = treatment.var(ddof=1), control.var(ddof=1)
    n1, n2 = treatment.size, control.size
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))

    if pooled == 0:
        return {"result": None, "error": "zero variance"}

    d = (m1 - m2) / pooled
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {"result": {"cohens_d": float(d), "interpretation": interp}, "error": None}


def nnt(treatment_rate: float, control_rate: float) -> dict:
    diff = abs(float(treatment_rate) - float(control_rate))
    if diff == 0:
        return {"result": None, "error": "rates are identical"}
    return {"result": {"nnt": ceil(1 / diff)}, "error": None}


def risk_ratio(treatment_rate: float, control_rate: float) -> dict:
    tr = float(treatment_rate)
    ct = float(control_rate)
    if ct == 0:
        return {"result": None, "error": "control rate is zero"}
    if tr == 0:
        return {"result": {"risk_ratio": 0.0}, "error": None}
    return {"result": {"risk_ratio": tr / ct}, "error": None}


def compute_all_effect_sizes(prepared: dict) -> dict:
    c_mask = prepared["control_mask"]
    t_mask = prepared["treatment_mask"]
    target = prepared["target"]
    control = target[c_mask]
    treatment = target[t_mask]

    c_rate = float(np.nanmean(control)) if len(control) > 0 else 0.0
    t_rate = float(np.nanmean(treatment)) if len(treatment) > 0 else 0.0

    d_result = cohens_d(treatment, control)
    nnt_result = nnt(t_rate, c_rate)
    rr_result = risk_ratio(t_rate, c_rate)

    return {
        "cohens_d": d_result["result"]["cohens_d"] if d_result["error"] is None else None,
        "cohens_d_interpretation": d_result["result"]["interpretation"]
        if d_result["error"] is None
        else None,
        "nnt": nnt_result["result"]["nnt"] if nnt_result["error"] is None else None,
        "risk_ratio": rr_result["result"]["risk_ratio"] if rr_result["error"] is None else None,
        "control_rate": c_rate,
        "treatment_rate": t_rate,
        "absolute_difference": t_rate - c_rate,
    }



