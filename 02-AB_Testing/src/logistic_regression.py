from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from src.data_loader import get_group_stats, get_control_value


def make_group_dummy(group_arr: np.ndarray, control_value: str = None) -> pd.DataFrame:
    if control_value is None:
        control_value = str(np.unique(group_arr)[0])
    dummies = pd.get_dummies(pd.Series(group_arr), prefix="group", drop_first=False).astype(float)
    drop_cols = [c for c in dummies.columns if control_value in c or c.endswith(f"_{control_value}")]
    if drop_cols:
        dummies = dummies.drop(columns=drop_cols[:1])
    return dummies


def _safe_glm(y: np.ndarray, X: pd.DataFrame, default_or: float = 1.0) -> dict:
    y_clean = np.nan_to_num(y, nan=0.0)
    X_clean = X.fillna(0.0).values if hasattr(X, 'fillna') else X

    if np.var(y_clean) == 0 or np.ptp(y_clean) == 0:
        return {
            "model": None, "coefficient": 0.0, "odds_ratio": default_or,
            "or_ci_95": (default_or, default_or), "p_value": 1.0,
            "significant": False, "log_likelihood": 0.0, "aic": 0.0,
            "bic": 0.0, "pseudo_r2": 0.0, "summary": None,
        }
    try:
        model = sm.GLM(y_clean, X_clean.astype(float), family=sm.families.Binomial()).fit(disp=False)
        non_const = [c for c in X.columns if c != "const"]
        if not non_const:
            return {
                "model": model, "coefficient": 0.0, "odds_ratio": default_or,
                "or_ci_95": (default_or, default_or), "p_value": 1.0,
                "significant": False, "log_likelihood": float(model.llf),
                "aic": float(model.aic), "bic": float(model.bic),
                "pseudo_r2": float(model.pseudo_rsquared()), "summary": model.summary(),
            }
        treatment_col = non_const[0]
        coef = float(model.params[treatment_col])
        or_val = float(np.exp(coef))
        ci = model.conf_int().loc[treatment_col]
        ci_exp = (float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1])))
        p_val = float(model.pvalues[treatment_col])
        return {
            "model": model, "coefficient": coef, "odds_ratio": or_val,
            "or_ci_95": ci_exp, "p_value": p_val,
            "significant": bool(p_val < 0.05),
            "log_likelihood": float(model.llf), "aic": float(model.aic),
            "bic": float(model.bic), "pseudo_r2": float(model.pseudo_rsquared()),
            "summary": model.summary(),
        }
    except Exception:
        return {
            "model": None, "coefficient": 0.0, "odds_ratio": default_or,
            "or_ci_95": (default_or, default_or), "p_value": 1.0,
            "significant": False, "log_likelihood": 0.0, "aic": 0.0,
            "bic": 0.0, "pseudo_r2": 0.0, "summary": None,
        }


def simple_logistic_regression(prepared: dict) -> dict:
    group = prepared["group"]
    target = prepared["target"]
    control_val = get_control_value()

    X = make_group_dummy(group, control_val)
    X = sm.add_constant(X.astype(float))
    y = target.astype(float)

    return _safe_glm(y, X)


def enriched_logistic_regression(prepared: dict) -> dict:
    group = prepared["group"]
    target = prepared["target"]
    control_val = get_control_value()

    X = make_group_dummy(group, control_val)
    all_data = prepared["all"]

    for c_name in prepared.get("covariates", {}):
        cov = all_data[c_name]
        if cov.dtype.kind == "f":
            X[c_name] = np.nan_to_num(cov.astype(float))

    if prepared.get("time") is not None:
        try:
            ts = pd.to_datetime(prepared["time"], errors="coerce")
            if isinstance(ts, pd.DatetimeIndex):
                X["hour"] = np.nan_to_num(ts.hour.astype(float))
                X["weekend"] = np.isin(ts.dayofweek, [5, 6]).astype(float)
            else:
                X["hour"] = np.nan_to_num(ts.dt.hour.astype(float))
                X["weekend"] = np.isin(ts.dt.dayofweek, [5, 6]).astype(float)
        except Exception:
            pass

    X = sm.add_constant(X.astype(float))
    y = target.astype(float)

    return _safe_glm(y, X)


def likelihood_ratio_test(model_simple, model_enriched) -> dict:
    if model_simple is None or model_enriched is None:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}
    lr_stat = 2 * (model_enriched.llf - model_simple.llf)
    df_diff = int(model_enriched.df_model - model_simple.df_model)
    if df_diff <= 0:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    return {
        "lr_statistic": float(lr_stat),
        "df_difference": df_diff,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
