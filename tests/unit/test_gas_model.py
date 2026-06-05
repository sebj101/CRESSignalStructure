"""
Unit tests for GasModel and BaseCrossSection
"""

import numpy as np
import pytest
from CRESSignalStructure.scattering.BaseCrossSection import BaseCrossSection
from CRESSignalStructure.scattering.GasModel import GasModel


class ConstantCrossSection(BaseCrossSection):
    """Test stub with fixed cross section and identity scatter."""

    def __init__(self, sigma=1e-20):
        self.__sigma = sigma

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy, pitch_angle


class EnergyLossCrossSection(BaseCrossSection):
    """Test stub that removes a fixed fraction of energy."""

    def __init__(self, sigma=1e-20, loss_fraction=0.01):
        self.__sigma = sigma
        self.__loss_fraction = loss_fraction

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy * (1 - self.__loss_fraction), pitch_angle


class TestGasModelConstruction:

    def test_valid_construction(self):
        xs = ConstantCrossSection()
        gas = GasModel([(xs, 1e16)])
        assert gas is not None

    def test_empty_species_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            GasModel([])

    def test_negative_density_raises(self):
        xs = ConstantCrossSection()
        with pytest.raises(ValueError, match="non-negative"):
            GasModel([(xs, -1.0)])

    def test_non_cross_section_raises(self):
        with pytest.raises(TypeError, match="BaseCrossSection"):
            GasModel([("not_a_model", 1e16)])

    def test_multiple_species(self):
        xs1 = ConstantCrossSection(1e-20)
        xs2 = ConstantCrossSection(2e-20)
        gas = GasModel([(xs1, 1e16), (xs2, 1e15)])
        assert gas is not None


class TestScatterRate:

    def test_single_species_rate(self):
        sigma = 1e-20
        n = 1e16
        speed = 1e7
        xs = ConstantCrossSection(sigma)
        gas = GasModel([(xs, n)])
        expected = n * sigma * speed
        assert gas.total_scatter_rate(18600.0, speed) == pytest.approx(
            expected)

    def test_multiple_species_rate_is_sum(self):
        sigma1, n1 = 1e-20, 1e16
        sigma2, n2 = 2e-20, 1e15
        speed = 1e7
        xs1 = ConstantCrossSection(sigma1)
        xs2 = ConstantCrossSection(sigma2)
        gas = GasModel([(xs1, n1), (xs2, n2)])
        expected = n1 * sigma1 * speed + n2 * sigma2 * speed
        assert gas.total_scatter_rate(18600.0, speed) == pytest.approx(
            expected)

    def test_zero_density_gives_zero_rate(self):
        xs = ConstantCrossSection(1e-20)
        gas = GasModel([(xs, 0.0)])
        assert gas.total_scatter_rate(18600.0, 1e7) == 0.0


class TestScatterTimeSampling:

    def test_returns_positive_time(self):
        xs = ConstantCrossSection(1e-20)
        gas = GasModel([(xs, 1e16)])
        rng = np.random.default_rng(42)
        t = gas.sample_time_to_scatter(18600.0, 1e7, rng)
        assert t > 0

    def test_zero_rate_returns_infinity(self):
        xs = ConstantCrossSection(1e-20)
        gas = GasModel([(xs, 0.0)])
        rng = np.random.default_rng(42)
        t = gas.sample_time_to_scatter(18600.0, 1e7, rng)
        assert t == np.inf

    def test_mean_scatter_time_matches_rate(self):
        sigma = 1e-20
        n = 1e16
        speed = 1e7
        xs = ConstantCrossSection(sigma)
        gas = GasModel([(xs, n)])
        rng = np.random.default_rng(42)
        expected_mean = 1.0 / (n * sigma * speed)
        times = [gas.sample_time_to_scatter(18600.0, speed, rng)
                 for _ in range(10000)]
        assert np.mean(times) == pytest.approx(expected_mean, rel=0.05)


class TestScatterSampling:

    def test_elastic_scatter_preserves_energy(self):
        xs = ConstantCrossSection()
        gas = GasModel([(xs, 1e16)])
        rng = np.random.default_rng(42)
        new_e, new_pa = gas.sample_scatter(18600.0, 1.5, 1e7, rng)
        assert new_e == 18600.0
        assert new_pa == 1.5

    def test_inelastic_scatter_reduces_energy(self):
        xs = EnergyLossCrossSection(loss_fraction=0.01)
        gas = GasModel([(xs, 1e16)])
        rng = np.random.default_rng(42)
        new_e, _ = gas.sample_scatter(18600.0, 1.5, 1e7, rng)
        assert new_e == pytest.approx(18600.0 * 0.99)

    def test_species_selection_proportional_to_rate(self):
        """Higher cross-section species should be selected more often."""
        xs_small = ConstantCrossSection(1e-22)
        xs_large = EnergyLossCrossSection(1e-20, 0.05)
        gas = GasModel([(xs_small, 1e16), (xs_large, 1e16)])
        rng = np.random.default_rng(42)

        energy_changed = 0
        n_trials = 5000
        for _ in range(n_trials):
            new_e, _ = gas.sample_scatter(18600.0, 1.5, 1e7, rng)
            if new_e != 18600.0:
                energy_changed += 1

        # xs_large has 100x the cross section so should dominate
        fraction = energy_changed / n_trials
        assert fraction > 0.95
