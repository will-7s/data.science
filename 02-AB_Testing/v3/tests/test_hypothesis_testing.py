from __future__ import annotations
import numpy as np
import pytest
from src.hypothesis_testing import (
    chi_squared_test,
    z_test_proportions,
    t_test_independent,
    mann_whitney_u_test,
    run_all_tests,
)


class TestChiSquared:
    def test_returns_dict(self, sample_prepared):
        result = chi_squared_test(sample_prepared)
        assert isinstance(result, dict)
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result

    def test_p_value_range(self, sample_prepared):
        result = chi_squared_test(sample_prepared)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_positive_statistic(self, sample_prepared):
        result = chi_squared_test(sample_prepared)
        assert result["statistic"] >= 0.0


class TestZTest:
    def test_returns_dict(self, sample_prepared):
        result = z_test_proportions(sample_prepared)
        assert isinstance(result, dict)
        assert "z_statistic_two_sided" in result
        assert "p_value_two_sided" in result

    def test_p_values_range(self, sample_prepared):
        result = z_test_proportions(sample_prepared)
        assert 0.0 <= result["p_value_two_sided"] <= 1.0
        assert 0.0 <= result["p_value_one_sided"] <= 1.0

    def test_small_samples(self):
        prepared = {
            "target": np.array([1.0, 0.0]),
            "group": np.array(["control", "treatment"]),
            "control_mask": np.array([True, False]),
            "treatment_mask": np.array([False, True]),
        }
        result = z_test_proportions(prepared)
        assert isinstance(result, dict)
        assert result["significant_two_sided"] is False


class TestTTest:
    def test_returns_dict(self, sample_prepared):
        result = t_test_independent(sample_prepared)
        assert isinstance(result, dict)
        assert "t_statistic" in result
        assert "p_value" in result

    def test_p_value_range(self, sample_prepared):
        result = t_test_independent(sample_prepared)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_insufficient_samples(self):
        prepared = {
            "target": np.array([1.0]),
            "group": np.array(["control"]),
            "control_mask": np.array([True]),
            "treatment_mask": np.array([False]),
        }
        result = t_test_independent(prepared)
        assert result["significant"] is False


class TestMannWhitney:
    def test_returns_dict(self, sample_prepared):
        result = mann_whitney_u_test(sample_prepared)
        assert isinstance(result, dict)
        assert "u_statistic" in result
        assert "p_value" in result

    def test_p_value_range(self, sample_prepared):
        result = mann_whitney_u_test(sample_prepared)
        assert 0.0 <= result["p_value"] <= 1.0


class TestRunAllTests:
    def test_returns_all_tests(self, sample_prepared):
        result = run_all_tests(sample_prepared)
        assert "chi_squared" in result
        assert "z_test" in result
        assert "t_test" in result
        assert "mann_whitney" in result

    def test_each_has_p_value(self, sample_prepared):
        result = run_all_tests(sample_prepared)
        for name, res in result.items():
            assert "p_value" in res or "p_value_two_sided" in res
