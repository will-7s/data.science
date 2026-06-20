from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _safe_to_datetime(arr) -> pd.DatetimeIndex | pd.Series | None:
    try:
        ts = pd.to_datetime(arr, errors="coerce")
        if isinstance(ts, pd.DatetimeIndex):
            valid = ts.notna()
        else:
            valid = ts.notna()
        if valid.sum() == 0:
            return None
        return ts
    except Exception:
        return None


def run_temporal_analysis(prepared: dict) -> dict:
    if prepared.get("time") is None:
        return {"daily_data": None, "trends": {}, "error": "No time column provided"}

    ts = _safe_to_datetime(prepared["time"])
    if ts is None:
        return {"daily_data": None, "trends": {}, "error": "Time column could not be parsed"}

    target = prepared["target"]
    group = prepared["group"]

    dates = ts.date if isinstance(ts, pd.DatetimeIndex) else ts.dt.date

    df = pd.DataFrame({
        "date": dates,
        "group": group,
        "converted": target,
    }).dropna(subset=["date"])

    if df.empty:
        return {"daily_data": None, "trends": {}, "error": "No valid dates after parsing"}

    daily = df.groupby(["date", "group"])["converted"].agg(["count", "mean"]).reset_index()
    daily.columns = ["date", "group", "n", "rate"]

    trends = {}
    for grp_val in np.unique(group):
        d = daily[daily["group"] == grp_val].copy()
        d["day_num"] = range(len(d))
        if len(d) > 3 and d["rate"].nunique() > 1:
            try:
                r, p_val = pearsonr(d["day_num"], d["rate"])
                trends[str(grp_val)] = {
                    "pearson_r": float(r),
                    "p_value": float(p_val),
                    "significant_trend": bool(p_val < 0.05),
                }
            except Exception:
                trends[str(grp_val)] = {
                    "pearson_r": float("nan"),
                    "p_value": float("nan"),
                    "significant_trend": False,
                }
        else:
            trends[str(grp_val)] = {
                "pearson_r": float("nan"),
                "p_value": float("nan"),
                "significant_trend": False,
            }

    return {"daily_data": daily, "trends": trends}
