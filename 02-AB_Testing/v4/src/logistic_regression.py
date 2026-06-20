from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as glm_module
from scipy import stats

from config import config

glm_module.SET_USE_BIC_LLF(True)


def _make_group_dummy(group_arr: np.ndarray, control_value: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"group_treatment": (group_arr != control_value).astype(float)}
    )


def _null_result(default_or: float = 1.0) -> dict:
    return {
        "odds_ratio": default_or,
        "or_ci_95": (default_or, default_or),
        "p_value": 1.0,
        "significant": False,
        "log_likelihood": 0.0,
        "aic": 0.0,
        "bic": 0.0,
        "pseudo_r2": 0.0,
        "coefficient": 0.0,
    }


def _safe_glm(y: np.ndarray, x: pd.DataFrame) -> dict:
    y_clean = np.nan_to_num(y, nan=0.0)
    x_clean = x.fillna(0.0)

    if np.var(y_clean) == 0 or np.ptp(y_clean) == 0:
        return _null_result()

    alpha = 1.0 - config.analysis.confidence_level

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Perfect separation")
            model = sm.GLM(
                y_clean, x_clean.astype(float),
                family=sm.families.Binomial(),
            ).fit(disp=False)

        non_const = [c for c in x_clean.columns if c != "const"]
        if not non_const:
            return {
                "odds_ratio": 1.0,
                "or_ci_95": (1.0, 1.0),
                "p_value": 1.0,
                "significant": False,
                "log_likelihood": float(model.llf),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "pseudo_r2": float(model.pseudo_rsquared()),
                "coefficient": 0.0,
            }

        treatment_col = non_const[0]
        coef = float(model.params[treatment_col])
        or_val = float(np.exp(coef))
        try:
            ci = model.conf_int().loc[treatment_col]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow")
                ci_lo = float(np.exp(ci.iloc[0]))
                ci_hi = float(np.exp(ci.iloc[1]))
            if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
                ci_exp = (0.0, float("inf"))
            else:
                ci_exp = (ci_lo, ci_hi)
        except Exception:
            ci_exp = (0.0, float("inf"))
        p_val = float(model.pvalues[treatment_col])

        return {
            "odds_ratio": or_val,
            "or_ci_95": ci_exp,
            "p_value": p_val,
            "significant": bool(p_val < alpha),
            "log_likelihood": float(model.llf),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "pseudo_r2": float(model.pseudo_rsquared()),
            "coefficient": coef,
        }

    except Exception as exc:
        warnings.warn(f"GLM fitting failed: {exc}", RuntimeWarning, stacklevel=2)
        return _null_result()


def simple_logistic_regression(prepared: dict, control_value: str) -> dict:
    x = _make_group_dummy(prepared["group"], control_value)
    x = sm.add_constant(x.astype(float))
    return _safe_glm(prepared["target"].astype(float), x)


def enriched_logistic_regression(prepared: dict, control_value: str) -> dict:
    x = _make_group_dummy(prepared["group"], control_value)

    for c_name, cov in (prepared.get("covariates") or {}).items():
        if np.asarray(cov).dtype.kind == "f":
            x[c_name] = np.nan_to_num(cov.astype(float))

    if prepared.get("time") is not None:
        try:
            ts = pd.to_datetime(prepared["time"], errors="coerce")
            if isinstance(ts, pd.DatetimeIndex):
                ts = ts.to_series().reset_index(drop=True)
            x["hour"] = np.nan_to_num(ts.dt.hour.astype(float))
            x["weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(float).values
        except Exception as exc:
            warnings.warn(
                f"Temporal feature extraction failed: {exc}", RuntimeWarning, stacklevel=2
            )

    x = sm.add_constant(x.astype(float))
    return _safe_glm(prepared["target"].astype(float), x)


def likelihood_ratio_test(
    model_simple: dict | None,
    model_enriched: dict | None,
    n_covariates: int = 0,
    has_time: bool = False,
) -> dict:
    if model_simple is None or model_enriched is None:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}

    ll_simple = model_simple.get("log_likelihood", 0.0)
    ll_enriched = model_enriched.get("log_likelihood", 0.0)

    lr_stat = 2 * (ll_enriched - ll_simple)
    if lr_stat < 0:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}

    df_diff = n_covariates + (2 if has_time else 0)
    if df_diff <= 0:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}

    alpha = 1.0 - config.analysis.confidence_level
    p_value = float(1 - stats.chi2.cdf(lr_stat, df_diff))

    return {
        "lr_statistic": float(lr_stat),
        "df_difference": df_diff,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
    }
