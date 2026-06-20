from __future__ import annotations

import warnings
from math import sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr
from statsmodels.stats.multitest import multipletests

from config import config


def _fast_prop_ztest(counts: list[int], nobs: list[int]) -> float:
    p1 = counts[0] / nobs[0]
    p2 = counts[1] / nobs[1]
    p_pool = (counts[0] + counts[1]) / (nobs[0] + nobs[1])
    if p_pool <= 0 or p_pool >= 1:
        return 1.0
    se = sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2 * (1 - ndtr(abs(z))))


def _segment_stats(
    target: np.ndarray,
    group: np.ndarray,
    mask: np.ndarray,
    ctrl_val: str,
    alpha: float,
) -> dict | None:
    t_seg = target[mask]
    g_seg = group[mask]
    c_mask = g_seg == ctrl_val
    t_mask = ~c_mask
    n_c, n_t = int(c_mask.sum()), int(t_mask.sum())
    conv_c = int(t_seg[c_mask].sum())
    conv_t = int(t_seg[t_mask].sum())
    if n_c < 1 or n_t < 1 or (conv_c + conv_t) < 1:
        return None
    p_val = _fast_prop_ztest([conv_c, conv_t], [n_c, n_t])
    return {
        "control_rate": float(conv_c / n_c) if n_c > 0 else 0.0,
        "control_n": n_c,
        "treatment_rate": float(conv_t / n_t) if n_t > 0 else 0.0,
        "treatment_n": n_t,
        "p_value_raw": float(p_val),
        "significant_raw": p_val < alpha,
    }


def _apply_fdr(results_flat: list[dict], alpha: float) -> None:
    p_raw = np.array([r["p_value_raw"] for r in results_flat])
    if len(p_raw) < 2:
        for r in results_flat:
            r["p_value"] = r["p_value_raw"]
            r["significant"] = r["significant_raw"]
        return

    _, p_corrected, _, _ = multipletests(p_raw, method="fdr_bh")
    for r, pc in zip(results_flat, p_corrected):
        r["p_value"] = float(pc)
        r["significant"] = bool(pc < alpha)


def run_segmentation_analysis(
    prepared: dict,
    control_value: str,
    target_col: str,
    group_col: str,
) -> dict:
    # Analyse segmentée : pour chaque colonne catégorielle, calcule les taux
    # de conversion par sous-groupe et teste la significativité de la différence.
    # Ajoute automatiquement un segment Weekday/Weekend si une colonne temporelle
    # est disponible. Les p-values sont corrigées du FDR (Benjamini-Hochberg)
    # pour contrôler le taux de faux positifs lié aux comparaisons multiples.
    alpha = 1.0 - config.analysis.confidence_level
    ctrl_val = str(control_value)
    target = prepared["target"]
    group = prepared["group"]
    all_data = prepared["all"]
    meta = prepared["meta"]

    skip_cols = {target_col, group_col}

    raw_results: dict[str, dict[str, dict]] = {}
    flat: list[dict] = []

    for col_name, arr in all_data.items():
        if col_name in skip_cols:
            continue
        if meta.get(col_name) != "categorical":
            continue
        uniq = np.unique(arr)
        if not (2 <= len(uniq) <= 10):
            continue

        raw_results[col_name] = {}
        for seg_val in uniq:
            mask = arr == seg_val
            res = _segment_stats(target, group, mask, ctrl_val, alpha)
            if res is not None:
                raw_results[col_name][str(seg_val)] = res
                flat.append(res)

    if prepared.get("time") is not None:
        try:
            ts = pd.to_datetime(prepared["time"], errors="coerce")
            if isinstance(ts, pd.DatetimeIndex):
                ts = ts.to_series().reset_index(drop=True)
            if pd.notna(ts).sum() >= 10:
                dow = ts.dt.dayofweek
                wknd_mask = np.isin(dow, [5, 6])
                for label, mask in [("Weekday", ~wknd_mask), ("Weekend", wknd_mask)]:
                    res = _segment_stats(target, group, mask, ctrl_val, alpha)
                    if res is not None:
                        raw_results.setdefault("Weekday/Weekend", {})[label] = res
                        flat.append(res)
        except Exception as exc:
            warnings.warn(
                f"Weekday/Weekend segmentation failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    _apply_fdr(flat, alpha)

    return raw_results
