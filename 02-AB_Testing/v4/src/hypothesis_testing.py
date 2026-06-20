from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.special import betainc, gammainc, ndtr
from scipy.stats import rankdata


def _t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    prob = betainc(df / 2, 0.5, x)
    if t >= 0:
        return float(1 - 0.5 * prob)
    return float(0.5 * prob)


def _chi2_cdf(x: float, df: int) -> float:
    return float(gammainc(df / 2, x / 2))


def _check_arrays(*arrays) -> bool:
    for a in arrays:
        if a is None:
            return False
        arr = np.asarray(a)
        if arr.ndim != 1 or arr.size == 0:
            return False
    return True


def _result(statistic, p_value, test_name, error=None):
    return {
        "result": None
        if error
        else {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "test_name": test_name,
            "error": None,
        },
        "error": error,
    }


def ztest(treatment: np.ndarray, control: np.ndarray) -> dict:
    # Test Z de comparaison de deux proportions (échantillons indépendants).
    # H0 : p_treatment = p_control. Hypothèse : distribution normale approx.
    # Valide si n×p ≥ 5 pour chaque groupe (règle de Cochran).
    # Retourne la statistique Z et la p-value bilatérale.
    if not _check_arrays(treatment, control):
        return _result(None, None, "ztest", "treatment and control must be non-empty 1D arrays")
    n1, n2 = treatment.size, control.size
    p1 = treatment.mean()
    p2 = control.mean()
    p_pool = (treatment.sum() + control.sum()) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return _result(None, None, "ztest", "zero variance (all outcomes identical)")
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return _result(None, None, "ztest", "zero standard error")
    z = (p1 - p2) / se
    p_value = 2 * (1 - ndtr(abs(z)))
    return _result(z, p_value, "ztest")


def chi_squared(contingency_table) -> dict:
    # Test du χ² d'indépendance sur une table de contingence 2×2.
    # H0 : les deux variables sont indépendantes (pas d'association groupe/conversion).
    # Équivalent au Z-test bilatéral (χ² = Z² pour 1 ddl).
    table = np.asarray(contingency_table, dtype=float)
    if table.shape != (2, 2):
        return _result(None, None, "chi_squared", "contingency_table must be 2x2")
    if np.any(np.isnan(table)):
        return _result(None, None, "chi_squared", "table contains NaN")
    if np.any(table < 0):
        return _result(None, None, "chi_squared", "table values must be non-negative")

    total = table.sum()
    if total == 0:
        return _result(None, None, "chi_squared", "table sum is zero")
    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    if np.any(expected == 0):
        return _result(None, None, "chi_squared", "expected cell count is zero")

    chi2 = np.nansum((table - expected) ** 2 / expected)
    df = 1
    p_value = 1 - _chi2_cdf(chi2, df)
    return _result(chi2, p_value, "chi_squared")


def ttest_ind(treatment: np.ndarray, control: np.ndarray) -> dict:
    if not _check_arrays(treatment, control):
        return _result(None, None, "ttest_ind", "treatment and control must be non-empty 1D arrays")
    if treatment.size < 2 or control.size < 2:
        return _result(None, None, "ttest_ind", "each group must have at least 2 observations")

    n1, n2 = treatment.size, control.size
    m1, m2 = treatment.mean(), control.mean()
    v1, v2 = treatment.var(ddof=1), control.var(ddof=1)
    se = sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return _result(None, None, "ttest_ind", "zero standard error")

    t = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if df_den == 0:
        return _result(None, None, "ttest_ind", "zero degrees of freedom denominator")

    df = df_num / df_den
    p_value = 2 * (1 - _t_cdf(abs(t), df))
    return _result(t, p_value, "ttest_ind")


def mann_whitney(treatment: np.ndarray, control: np.ndarray) -> dict:
    if not _check_arrays(treatment, control):
        return _result(
            None, None, "mann_whitney", "treatment and control must be non-empty 1D arrays"
        )

    n1, n2 = treatment.size, control.size
    combined = np.concatenate([treatment, control])
    rank = rankdata(combined)
    r1 = rank[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    _, counts = np.unique(combined, return_counts=True)
    tie_correction = sum(c**3 - c for c in counts[counts > 1])
    n = n1 + n2
    sigma = sqrt((n1 * n2 / 12) * ((n + 1) - tie_correction / (n * (n - 1))))
    if sigma == 0:
        return _result(None, None, "mann_whitney", "zero standard deviation")

    z = (u - mu) / sigma
    p_value = 2 * (1 - ndtr(abs(z)))
    return _result(u, p_value, "mann_whitney")


def logistic_regression(
    target: np.ndarray, group: np.ndarray, covariates: np.ndarray | None = None
) -> dict:
    target = np.asarray(target, dtype=float)
    group = np.asarray(group, dtype=float)

    if target.ndim != 1 or group.ndim != 1 or target.size == 0 or group.size != target.size:
        return _result(
            None, None, "logistic_regression", "target and group must be equal-length 1D arrays"
        )

    n = target.size

    cov = np.empty((n, 0))
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        if cov.shape[0] != n:
            return _result(None, None, "logistic_regression", "covariates must match target length")

    design = np.column_stack([np.ones(n), group] + [cov[:, i] for i in range(cov.shape[1])])
    k = design.shape[1]
    beta = np.zeros(k)

    for _ in range(50):
        eta = design @ beta
        pi = 1 / (1 + np.exp(-np.clip(eta, -100, 100)))
        w_diag = pi * (1 - pi)
        xtwx = design.T @ (w_diag[:, None] * design)
        xtwy = design.T @ (w_diag * eta + (target - pi))
        try:
            beta_new = np.linalg.solve(xtwx, xtwy)
        except np.linalg.LinAlgError:
            return _result(None, None, "logistic_regression", "singular matrix in IRLS iteration")
        if np.all(np.abs(beta_new - beta) < 1e-8):
            beta = beta_new
            break
        beta = beta_new

    eta = design @ beta
    pi = 1 / (1 + np.exp(-np.clip(eta, -100, 100)))
    w_diag = pi * (1 - pi)
    try:
        cov_mat = np.linalg.inv(design.T @ (w_diag[:, None] * design))
    except np.linalg.LinAlgError:
        return _result(None, None, "logistic_regression", "singular covariance matrix")

    se_beta = np.sqrt(np.diag(cov_mat))
    z = beta[1] / se_beta[1]
    p_value = 2 * (1 - ndtr(abs(z)))
    return _result(float(beta[1]), p_value, "logistic_regression")


def _build_contingency_table(control: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    c_success = int(np.nansum(control))
    c_fail = int(np.sum(~np.isnan(control))) - c_success
    t_success = int(np.nansum(treatment))
    t_fail = int(np.sum(~np.isnan(treatment))) - t_success
    return np.array([[c_fail, c_success], [t_fail, t_success]])


def run_all_tests(prepared: dict) -> dict:
    # Point d'entrée unique pour tous les tests fréquentistes.
    # Les données binaires déclenchent automatiquement le χ² et la
    # régression logistique en plus des tests génériques (Z, t, MW).
    target = prepared["target"]
    c_mask = prepared["control_mask"]
    t_mask = prepared["treatment_mask"]
    control = target[c_mask]
    treatment = target[t_mask]
    group_code = np.where(c_mask, 0, 1)
    is_binary = prepared.get("is_binary", False)

    def _unpack(r):
        if r.get("error") is not None:
            return None
        result = r.get("result")
        if result is None or result.get("error") is not None:
            return None
        return {"statistic": result.get("statistic"), "p_value": result.get("p_value")}

    result = {
        "ztest": _unpack(ztest(treatment, control)),
        "ttest": _unpack(ttest_ind(treatment, control)),
        "mann_whitney": _unpack(mann_whitney(treatment, control)),
        "chi_squared": None,
        "logistic_regression": None,
    }

    if is_binary:
        contingency = _build_contingency_table(control, treatment)
        result["chi_squared"] = _unpack(chi_squared(contingency))
        result["logistic_regression"] = _unpack(logistic_regression(target, group_code))

    return result
