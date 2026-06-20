from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as glm_module
from scipy import stats
from config import ALPHA
from src.data_loader import get_group_stats, get_control_value

glm_module.SET_USE_BIC_LLF(True)


def make_group_dummy(group_arr: np.ndarray, control_value: str) -> pd.DataFrame:
    """Return a one-column DataFrame: 1 = treatment, 0 = control."""
    return pd.DataFrame(
        {"group_treatment": (group_arr != control_value).astype(float)}
    )


def _safe_glm(y: np.ndarray, X: pd.DataFrame, default_or: float = 1.0) -> dict:
    """Fit a binomial GLM and extract the treatment coefficient.

    FIX: pass the DataFrame X directly to sm.GLM (not X.values).
    Previously, converting to a numpy array caused sm.GLM to return
    model.params as a plain ndarray without named indices.  The subsequent
    access model.params[treatment_col] then raised an IndexError that was
    silently caught, always returning OR = 1.0, p = 1.0.
    """
    y_clean = np.nan_to_num(y, nan=0.0)
    X_clean = X.fillna(0.0)  # keep as DataFrame — preserves column names

    if np.var(y_clean) == 0 or np.ptp(y_clean) == 0:
        return _null_result(default_or)

    try:
        model = sm.GLM(
            y_clean, X_clean.astype(float),  # DataFrame, NOT .values
            family=sm.families.Binomial()
        ).fit(disp=False)

        non_const = [c for c in X_clean.columns if c != "const"]
        if not non_const:
            return _base_result(model, default_or)

        treatment_col = non_const[0]
        coef   = float(model.params[treatment_col])
        or_val = float(np.exp(coef))
        ci     = model.conf_int().loc[treatment_col]
        ci_exp = (float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1])))
        p_val  = float(model.pvalues[treatment_col])

        return {
            "model":         model,
            "coefficient":   coef,
            "odds_ratio":    or_val,
            "or_ci_95":      ci_exp,
            "p_value":       p_val,
            "significant":   bool(p_val < ALPHA),
            "log_likelihood": float(model.llf),
            "aic":           float(model.aic),
            "bic":           float(model.bic),
            "pseudo_r2":     float(model.pseudo_rsquared()),
            "summary":       model.summary(),
        }

    except Exception as exc:
        warnings.warn(f"GLM fitting failed: {exc}", RuntimeWarning, stacklevel=2)
        return _null_result(default_or)


def _null_result(default_or: float) -> dict:
    return {
        "model": None, "coefficient": 0.0, "odds_ratio": default_or,
        "or_ci_95": (default_or, default_or), "p_value": 1.0,
        "significant": False, "log_likelihood": 0.0,
        "aic": 0.0, "bic": 0.0, "pseudo_r2": 0.0, "summary": None,
    }


def _base_result(model, default_or: float) -> dict:
    return {
        "model": model, "coefficient": 0.0, "odds_ratio": default_or,
        "or_ci_95": (default_or, default_or), "p_value": 1.0,
        "significant": False,
        "log_likelihood": float(model.llf), "aic": float(model.aic),
        "bic": float(model.bic), "pseudo_r2": float(model.pseudo_rsquared()),
        "summary": model.summary(),
    }


def simple_logistic_regression(prepared: dict) -> dict:
    ctrl = get_control_value()
    X = make_group_dummy(prepared["group"], ctrl)
    X = sm.add_constant(X.astype(float))
    return _safe_glm(prepared["target"].astype(float), X)


def enriched_logistic_regression(prepared: dict) -> dict:
    ctrl = get_control_value()
    X = make_group_dummy(prepared["group"], ctrl)

    # Add numeric covariates
    for c_name, cov in prepared.get("covariates", {}).items():
        if np.asarray(cov).dtype.kind == "f":
            X[c_name] = np.nan_to_num(cov.astype(float))

    # Add temporal features
    if prepared.get("time") is not None:
        try:
            ts = pd.to_datetime(prepared["time"], errors="coerce")
            if isinstance(ts, pd.DatetimeIndex):
                ts = ts.to_series().reset_index(drop=True)
            X["hour"]    = np.nan_to_num(ts.dt.hour.astype(float))
            X["weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(float).values
        except Exception as exc:
            warnings.warn(f"Temporal feature extraction failed: {exc}", RuntimeWarning, stacklevel=2)

    X = sm.add_constant(X.astype(float))
    return _safe_glm(prepared["target"].astype(float), X)


def likelihood_ratio_test(model_simple, model_enriched) -> dict:
    if model_simple is None or model_enriched is None:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}
    lr_stat  = 2 * (model_enriched.llf - model_simple.llf)
    df_diff  = int(model_enriched.df_model - model_simple.df_model)
    if df_diff <= 0:
        return {"lr_statistic": 0.0, "df_difference": 0, "p_value": 1.0, "significant": False}
    p_value = float(1 - stats.chi2.cdf(lr_stat, df_diff))
    return {
        "lr_statistic":  float(lr_stat),
        "df_difference": df_diff,
        "p_value":       p_value,
        "significant":   bool(p_value < ALPHA),
    }
