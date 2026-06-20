from __future__ import annotations
import numpy as np
import pytest
from src.power_analysis import compute_power


class TestComputePower:
    def test_returns_dict(self):
        result = compute_power(0.2, 500, 500)
        assert isinstance(result, dict)
        assert "power_observed" in result
        assert "n_needed_80pct" in result
        assert "mde_cohens_h" in result
        assert "skipped" in result

    def test_power_increases_with_effect(self):
        small = compute_power(0.1, 500, 500)
        large = compute_power(0.5, 500, 500)
        assert large["power_observed"] > small["power_observed"]

    def test_power_increases_with_sample(self):
        small = compute_power(0.2, 100, 100)
        large = compute_power(0.2, 1000, 1000)
        assert large["power_observed"] > small["power_observed"]

    def test_skipped_for_zero_effect(self):
        result = compute_power(0.0, 500, 500)
        assert result["skipped"] is True

    def test_skipped_for_tiny_effect(self):
        result = compute_power(1e-10, 500, 500)
        assert result["skipped"] is True

    def test_n_needed_finite(self):
        result = compute_power(0.2, 500, 500)
        assert np.isfinite(result["n_needed_80pct"])

    def test_mde_pp_reasonable(self):
        result = compute_power(0.2, 500, 500)
        assert result["mde_pp"] >= 0.0
        assert result["mde_pp"] <= 1.0

    def test_small_groups(self):
        result = compute_power(0.5, 1, 1)
        assert result["skipped"] is True

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_alpha_respected(self, alpha):
        result = compute_power(0.3, 500, 500, alpha)
        assert result["alpha"] == alpha
