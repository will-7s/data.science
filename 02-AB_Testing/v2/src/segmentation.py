from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from src.data_loader import get_group_stats, get_control_value


def _segment_stats(target: np.ndarray, group: np.ndarray,
                   mask: np.ndarray, ctrl_val: str) -> dict | None:
    """Compute per-segment group stats and raw p-value."""
    t_seg = target[mask]
    g_seg = group[mask]
    c_mask = g_seg == ctrl_val
    t_mask = ~c_mask
    n_c, n_t = int(c_mask.sum()), int(t_mask.sum())
    conv_c = int(t_seg[c_mask].sum())
    conv_t = int(t_seg[t_mask].sum())
    if n_c < 1 or n_t < 1 or (conv_c + conv_t) < 1:
        return None
    try:
        _, p_val = proportions_ztest([conv_c, conv_t], [n_c, n_t], alternative="two-sided")
    except Exception:
        p_val = 1.0
    return {
        "control_rate":   float(conv_c / n_c),
        "control_n":      n_c,
        "treatment_rate": float(conv_t / n_t),
        "treatment_n":    n_t,
        "p_value_raw":    float(p_val),
        "significant_raw": p_val < 0.05,
        # p_value and significant will be added after FDR correction
    }


def _apply_fdr(results_flat: list[dict]) -> None:
    """FIX: apply Benjamini-Hochberg FDR correction across all segments.

    The original code imported multipletests but never called it, leaving
    all significance flags uncorrected.  With 5–10 simultaneous segment
    tests, uncorrected α = 5% leads to ~23–40% family-wise error rate.
    """
    p_raw = np.array([r["p_value_raw"] for r in results_flat])
    if len(p_raw) < 2:
        for r in results_flat:
            r["p_value"]    = r["p_value_raw"]
            r["significant"] = r["significant_raw"]
        return

    _, p_corrected, _, _ = multipletests(p_raw, method="fdr_bh")
    for r, pc in zip(results_flat, p_corrected):
        r["p_value"]    = float(pc)
        r["significant"] = bool(pc < 0.05)


def run_segmentation_analysis(prepared: dict) -> dict:
    """Segment by all suitable categorical columns + weekday/weekend.

    Returns a nested dict:  { col_name: { segment_value: stats_dict } }
    All p-values are FDR-corrected (BH) across all segments globally.
    """
    ctrl_val = get_control_value()
    target   = prepared["target"]
    group    = prepared["group"]
    all_data = prepared["all"]
    meta     = prepared["meta"]

    from src.store import schema as store_schema
    skip_cols = {store_schema.target_col, store_schema.group_col}

    raw_results: dict[str, dict[str, dict]] = {}
    flat: list[dict] = []          # all segment dicts in insertion order

    # ── Categorical columns ────────────────────────────────────────────────
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
            res  = _segment_stats(target, group, mask, ctrl_val)
            if res is not None:
                raw_results[col_name][str(seg_val)] = res
                flat.append(res)

    # ── Weekday / Weekend ─────────────────────────────────────────────────
    if prepared.get("time") is not None:
        try:
            ts = pd.to_datetime(prepared["time"], errors="coerce")
            dow = ts.dt.dayofweek if not isinstance(ts, pd.DatetimeIndex) else ts.dayofweek
            if pd.notna(ts).sum() >= 10:
                wknd_mask = np.isin(dow, [5, 6])
                for label, mask in [("Weekday", ~wknd_mask), ("Weekend", wknd_mask)]:
                    res = _segment_stats(target, group, mask, ctrl_val)
                    if res is not None:
                        raw_results.setdefault("Weekday/Weekend", {})[label] = res
                        flat.append(res)
        except Exception:
            pass

    # ── FDR correction across all segments ────────────────────────────────
    _apply_fdr(flat)

    return raw_results
