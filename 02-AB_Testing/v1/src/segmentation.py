from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from src.data_loader import get_group_stats, get_control_value


def segment_by_group(prepared: dict, segment_arr: np.ndarray, segment_name: str) -> dict:
    target = prepared["target"]
    group = prepared["group"]
    uniq_segments = np.unique(segment_arr)

    results = {}
    for seg_val in uniq_segments:
        mask = segment_arr == seg_val
        seg_target = target[mask]
        seg_group = group[mask]
        control_val = get_control_value()

        c_mask = seg_group == control_val
        t_mask = ~c_mask

        n_c = int(c_mask.sum())
        n_t = int(t_mask.sum())
        conv_c = int(seg_target[c_mask].sum())
        conv_t = int(seg_target[t_mask].sum())

        if n_c > 0 and n_t > 0 and (conv_c + conv_t) > 0:
            _, p_val = proportions_ztest(
                [conv_c, conv_t], [n_c, n_t], alternative="two-sided"
            )
            results[str(seg_val)] = {
                "control_rate": float(conv_c / n_c),
                "control_n": n_c,
                "treatment_rate": float(conv_t / n_t),
                "treatment_n": n_t,
                "p_value": float(p_val),
                "significant_raw": p_val < 0.05,
            }

    return results


def run_segmentation_analysis(prepared: dict) -> dict:
    results = {}

    all_data = prepared["all"]
    meta = prepared["meta"]

    from src.store import schema as store_schema
    skip_cols = {store_schema.target_col, store_schema.group_col}
    for col_name, arr in all_data.items():
        if col_name in skip_cols:
            continue
        m = meta.get(col_name, "")
        if m == "categorical":
            uniq = np.unique(arr)
            if 2 <= len(uniq) <= 10:
                results[col_name] = segment_by_group(prepared, arr, col_name)

    if prepared.get("time") is not None:
        try:
            ts = pd.to_datetime(prepared["time"], errors="coerce")
            if ts.notna().sum() < 10:
                pass
            else:
                dayofweek = ts.dayofweek if isinstance(ts, pd.DatetimeIndex) else ts.dt.dayofweek
                weekday_mask = np.isin(dayofweek, [5, 6]).to_numpy() if hasattr(dayofweek, 'to_numpy') else np.isin(dayofweek, [5, 6])
                results["Weekday/Weekend"] = {
                    "Weekday": _segment_result_from_mask(prepared, ~weekday_mask),
                    "Weekend": _segment_result_from_mask(prepared, weekday_mask),
                }
        except Exception:
            pass

    return results


def _segment_result_from_mask(prepared: dict, mask: np.ndarray) -> dict:
    target = prepared["target"][mask]
    group = prepared["group"][mask]
    control_val = get_control_value()
    c_mask = group == control_val
    t_mask = ~c_mask
    n_c = int(c_mask.sum())
    n_t = int(t_mask.sum())
    conv_c = int(target[c_mask].sum())
    conv_t = int(target[t_mask].sum())

    if n_c > 0 and n_t > 0 and (conv_c + conv_t) > 0:
        try:
            _, p_val = proportions_ztest(
                [conv_c, conv_t], [n_c, n_t], alternative="two-sided"
            )
        except Exception:
            p_val = 1.0
        return {
            "control_rate": float(conv_c / n_c),
            "control_n": n_c,
            "treatment_rate": float(conv_t / n_t),
            "treatment_n": n_t,
            "p_value": float(p_val),
            "significant_raw": p_val < 0.05,
        }
    return {}
