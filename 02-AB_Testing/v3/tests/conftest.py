from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


@pytest.fixture
def sample_prepared() -> dict:
    n_c, n_t = 500, 500
    rng = np.random.default_rng(42)
    target_c = rng.binomial(1, 0.10, n_c)
    target_t = rng.binomial(1, 0.12, n_t)
    target = np.concatenate([target_c, target_t])
    group = np.array(["control"] * n_c + ["treatment"] * n_t)
    control_mask = group == "control"
    treatment_mask = ~control_mask
    return {
        "target": target.astype(float),
        "group": group,
        "control_mask": control_mask,
        "treatment_mask": treatment_mask,
        "group_categories": np.array(["control", "treatment"]),
        "all": {"converted": target, "group": group},
        "meta": {"converted": "numeric", "group": "categorical"},
        "covariates": {},
        "cleaning_report": {"n_raw": 1000, "n_clean": 1000, "pct_removed": 0},
    }


@pytest.fixture
def zero_effect_prepared() -> dict:
    n_c, n_t = 500, 500
    rng = np.random.default_rng(1)
    target = rng.binomial(1, 0.10, n_c + n_t)
    group = np.array(["control"] * n_c + ["treatment"] * n_t)
    control_mask = group == "control"
    treatment_mask = ~control_mask
    return {
        "target": target.astype(float),
        "group": group,
        "control_mask": control_mask,
        "treatment_mask": treatment_mask,
        "group_categories": np.array(["control", "treatment"]),
        "all": {"converted": target, "group": group},
        "meta": {"converted": "numeric", "group": "categorical"},
        "covariates": {},
        "cleaning_report": {"n_raw": 1000, "n_clean": 1000, "pct_removed": 0},
    }
