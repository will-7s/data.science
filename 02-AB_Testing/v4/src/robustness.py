from __future__ import annotations

import numpy as np

from config import (
    ALPHA,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_MAX_SAMPLE,
    PERMUTATION_ITERATIONS,
    RANDOM_SEED,
)

# Taille de lot réduite pour éviter les pics mémoire avec de gros jeux de données.
# Le bootstrap utilise des batches de 1000 itérations (au lieu de 5000).
# Le test de permutation passe de 2000 à 200 pour ne plus allouer de matrices
# (batch × n_total) en mémoire vive, ce qui causait du swap disque.
_BOOTSTRAP_BATCH = 1000
_PERMUTATION_BATCH = 200


def _clean_nan(a: np.ndarray) -> np.ndarray:
    return a[~np.isnan(a)]


def bootstrap_ci(prepared: dict, gs: dict | None = None) -> dict:
    # Bootstrap non-paramétrique : on rééchantillonne avec remise dans chaque groupe
    # pour estimer la distribution de la différence des taux de conversion.
    # L'intervalle de confiance bootstrap (95%) est plus robuste que l'intervalle
    # de Wilson lorsque les données ne suivent pas une distribution normale.
    # Si le percentile 2.5% est > 0, on a une preuve forte que Treatment > Control.
    target = prepared["target"]
    c_mask, t_mask = prepared["control_mask"], prepared["treatment_mask"]

    ctrl_vals = _clean_nan(target[c_mask])
    trt_vals = _clean_nan(target[t_mask])
    n_c, n_t = len(ctrl_vals), len(trt_vals)

    if n_c == 0 or n_t == 0:
        return {
            "ci_95": (float("nan"), float("nan")),
            "ci_90": (float("nan"), float("nan")),
            "pct_positive": 0.0,
            "pct_negative": 0.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }

    n_iter = BOOTSTRAP_ITERATIONS
    samp_c = min(n_c, BOOTSTRAP_MAX_SAMPLE)
    samp_t = min(n_t, BOOTSTRAP_MAX_SAMPLE)

    rng = np.random.default_rng(RANDOM_SEED)
    diffs = np.empty(n_iter)
    for start in range(0, n_iter, _BOOTSTRAP_BATCH):
        end = min(start + _BOOTSTRAP_BATCH, n_iter)
        b = end - start
        c_idx = rng.integers(0, n_c, size=(b, samp_c))
        t_idx = rng.integers(0, n_t, size=(b, samp_t))
        diffs[start:end] = trt_vals[t_idx].mean(axis=1) - ctrl_vals[c_idx].mean(axis=1)

    return {
        "ci_95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "ci_90": (float(np.percentile(diffs, 5.0)), float(np.percentile(diffs, 95.0))),
        "pct_positive": float(np.mean(diffs > 0) * 100),
        "pct_negative": float(np.mean(diffs < 0) * 100),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=0)),
    }


def permutation_test(prepared: dict, gs: dict | None = None) -> dict:
    # Test de permutation : on mélange aléatoirement l'étiquette de groupe
    # pour simuler la distribution sous H0 (aucun effet). La p-value est la
    # proportion des permutations où la différence observée est dépassée.
    # Avantage : aucune hypothèse de normalité (test non-paramétrique).
    # Désavantage : coût calculatoire — on réduit le batch à 200 pour rester
    # sous la barre des 100 Mo en RAM même avec 100k lignes.
    if gs is None:
        from src.data_loader import get_group_stats
        gs = get_group_stats(prepared)
    observed_diff = gs["absolute_diff"]
    target = prepared["target"]
    groups_code = np.where(prepared["control_mask"], 0, 1)

    n_c = int((groups_code == 0).sum())
    n_t = int((groups_code == 1).sum())
    if n_c == 0 or n_t == 0:
        return {
            "p_value_two_sided": 1.0,
            "p_value_one_sided": 0.5,
            "significant_two_sided": False,
            "significant_one_sided": False,
            "observed_diff": float(observed_diff),
        }

    n_total = n_c + n_t
    n_iter = PERMUTATION_ITERATIONS
    rng = np.random.default_rng(RANDOM_SEED)
    target_vals = np.where(np.isnan(target), 0.0, target)

    perm_diffs = np.empty(n_iter)
    base_idx = np.arange(n_total)
    for start in range(0, n_iter, _PERMUTATION_BATCH):
        end = min(start + _PERMUTATION_BATCH, n_iter)
        b = end - start
        idx = rng.permuted(np.tile(base_idx, (b, 1)), axis=1)
        permuted = target_vals[idx]
        perm_diffs[start:end] = permuted[:, :n_c].mean(axis=1) - permuted[:, n_c:].mean(axis=1)

    p_two = float((np.abs(perm_diffs) >= np.abs(observed_diff)).mean())
    p_one = float((perm_diffs >= observed_diff).mean())

    return {
        "p_value_two_sided": p_two,
        "p_value_one_sided": p_one,
        "significant_two_sided": p_two < ALPHA,
        "significant_one_sided": p_one < ALPHA,
        "observed_diff": float(observed_diff),
    }


def run_robustness_checks(prepared: dict, gs: dict | None = None) -> dict:
    return {
        "bootstrap": bootstrap_ci(prepared, gs=gs),
        "permutation": permutation_test(prepared, gs=gs),
    }
