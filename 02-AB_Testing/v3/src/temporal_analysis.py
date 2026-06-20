from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from config import ALPHA


def _safe_to_datetime(arr) -> pd.Series | None:
    try:
        ts = pd.to_datetime(arr, errors="coerce")
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts.to_series().reset_index(drop=True)
        return ts if ts.notna().sum() > 0 else None
    except Exception:
        return None


def run_temporal_analysis(prepared: dict) -> dict:
    if prepared.get("time") is None:
        return {"daily_data": None, "trends": {}, "error": "No time column provided"}

    ts = _safe_to_datetime(prepared["time"])
    if ts is None:
        return {"daily_data": None, "trends": {}, "error": "Time column could not be parsed"}

    df = pd.DataFrame({
        "date":      ts.dt.date.values,
        "group":     prepared["group"],
        "converted": prepared["target"],
    }).dropna(subset=["date"])

    if df.empty:
        return {"daily_data": None, "trends": {}, "error": "No valid dates after parsing"}

    daily = (
        df.groupby(["date", "group"])["converted"]
        .agg(n="count", rate="mean")
        .reset_index()
    )

    trends: dict[str, dict] = {}
    for grp_val in np.unique(prepared["group"]):
        d = daily[daily["group"] == grp_val].copy().reset_index(drop=True)
        if len(d) < 4 or d["rate"].nunique() < 2:
            trends[str(grp_val)] = {"pearson_r": float("nan"), "p_value": float("nan"),
                                     "significant_trend": False}
            continue
        try:
            # FIX: weight by sqrt(n) to partially account for heteroscedasticity
            # (days with more users have lower variance in their rate estimate)
            weights = np.sqrt(d["n"].values.astype(float))
            w = weights / weights.sum()
            x = np.arange(len(d), dtype=float)
            x_w = np.average(x, weights=w)
            y_w = np.average(d["rate"].values, weights=w)
            r_num = np.sum(w * (x - x_w) * (d["rate"].values - y_w))
            r_den = np.sqrt(
                np.sum(w * (x - x_w) ** 2) *
                np.sum(w * (d["rate"].values - y_w) ** 2)
            )
            r = float(r_num / r_den) if r_den > 1e-12 else 0.0
            # Approx p-value via t-distribution with n-2 df
            n_eff = len(d)
            if abs(r) < 1.0 and n_eff > 2:
                from scipy.stats import t as t_dist
                t_stat = r * np.sqrt((n_eff - 2) / (1 - r ** 2))
                p_val  = float(2 * t_dist.sf(abs(t_stat), df=n_eff - 2))
            else:
                p_val = float("nan")

            trends[str(grp_val)] = {
                "pearson_r":        r,
                "p_value":          p_val,
                "significant_trend": (not np.isnan(p_val)) and (p_val < ALPHA),
            }
        except Exception as exc:
            warnings.warn(f"Temporal trend computation failed for {grp_val}: {exc}", RuntimeWarning, stacklevel=2)
            trends[str(grp_val)] = {"pearson_r": float("nan"), "p_value": float("nan"),
                                     "significant_trend": False}

    return {"daily_data": daily, "trends": trends}
