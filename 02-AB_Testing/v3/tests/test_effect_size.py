from __future__ import annotations
import numpy as np
import pytest
from src.effect_size import (
    cohens_h,
    interpret_cohens_h,
    odds_ratio,
    risk_ratio,
    phi_coefficient,
    nnt,
    compute_all_effect_sizes,
)


class TestCohensH:
    def test_equal_rates(self):
        assert cohens_h(0.10, 0.10) == pytest.approx(0.0, abs=1e-10)

    def test_positive_difference(self):
        h = cohens_h(0.10, 0.15)
        assert h > 0

    def test_negative_difference(self):
        h = cohens_h(0.15, 0.10)
        assert h < 0

    def test_extreme_rates(self):
        h = cohens_h(0.0, 1.0)
        assert np.isfinite(h)
        assert abs(h) > 0


class TestInterpretCohensH:
    def test_trivial(self):
        assert interpret_cohens_h(0.05) == "trivial"

    def test_small(self):
        assert interpret_cohens_h(0.20) == "small"

    def test_medium(self):
        assert interpret_cohens_h(0.50) == "medium"

    def test_large(self):
        assert interpret_cohens_h(0.80) == "large"

    def test_boundary_small(self):
        assert interpret_cohens_h(0.10) == "trivial"
        assert interpret_cohens_h(0.20) == "small"
        assert interpret_cohens_h(0.50) == "medium"
        assert interpret_cohens_h(0.80) == "large"


class TestOddsRatio:
    def test_no_effect(self):
        result = odds_ratio(0.10, 0.10, 50, 500, 50, 500)
        assert result["or"] == pytest.approx(1.0, abs=0.1)

    def test_protective(self):
        result = odds_ratio(0.15, 0.10, 75, 500, 50, 500)
        assert result["or"] < 1.0

    def test_risk_factor(self):
        result = odds_ratio(0.10, 0.15, 50, 500, 75, 500)
        assert result["or"] > 1.0

    def test_zero_cell(self):
        result = odds_ratio(0.0, 0.10, 0, 500, 50, 500)
        assert np.isnan(result["or"])


class TestRiskRatio:
    def test_no_effect(self):
        result = risk_ratio(0.10, 0.10)
        assert result == pytest.approx(1.0)

    def test_risk_factor(self):
        result = risk_ratio(0.10, 0.15)
        assert result == pytest.approx(1.5)

    def test_protective(self):
        result = risk_ratio(0.15, 0.10)
        assert result == pytest.approx(0.6667, abs=0.01)


class TestPhiCoefficient:
    def test_no_association(self):
        assert phi_coefficient(0.0, 1000) == 0.0

    def test_perfect_association(self):
        n = 1000
        chi2 = n
        assert phi_coefficient(chi2, n) == pytest.approx(1.0)

    def test_scaling(self):
        assert phi_coefficient(100, 10000) == pytest.approx(0.10)


class TestNNT:
    def test_positive_nnt(self):
        result = nnt(0.05)
        assert result["value"] == 20
        assert result["label"] == "NNT"

    def test_negative_nnt(self):
        result = nnt(-0.05)
        assert result["value"] == 20
        assert result["label"] == "NNH"

    def test_zero_diff(self):
        result = nnt(0.0)
        assert result["value"] == float("inf")

    def test_very_small_diff(self):
        result = nnt(0.001)
        assert result["value"] == 1000


class TestComputeAllEffectSizes:
    def test_returns_dict(self, sample_prepared):
        result = compute_all_effect_sizes(sample_prepared)
        assert isinstance(result, dict)
        assert "cohens_h" in result
        assert "odds_ratio" in result
        assert "risk_ratio" in result
        assert "phi_coefficient" in result
        assert "nnt" in result

    def test_cohens_h_nonzero(self, sample_prepared):
        result = compute_all_effect_sizes(sample_prepared)
        assert result["cohens_h"] != 0.0

    def test_zero_effect(self, zero_effect_prepared):
        result = compute_all_effect_sizes(zero_effect_prepared)
        assert abs(result["cohens_h"]) < 0.10
