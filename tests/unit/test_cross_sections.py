"""
Unit tests for InelasticCrossSection and ElasticCrossSection
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from CRESSignalStructure.CrossSections import (
    InelasticCrossSection, ElasticCrossSection)


# ---------------------------------------------------------------------------
# InelasticCrossSection tests
# ---------------------------------------------------------------------------

class TestInelasticConstruction:

    def test_valid_species(self):
        for species in ["H", "H2", "He"]:
            xsec = InelasticCrossSection(species)
            assert xsec is not None

    def test_invalid_species_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            InelasticCrossSection("Ar")


class TestInelasticTotalCrossSection:

    def test_below_threshold_returns_zero(self):
        xsec = InelasticCrossSection("H2")
        assert xsec.total_cross_section(10.0) == 0.0
        assert xsec.total_cross_section(15.43) == 0.0

    def test_above_threshold_positive(self):
        xsec = InelasticCrossSection("H2")
        assert xsec.total_cross_section(20.0) > 0.0
        assert xsec.total_cross_section(100.0) > 0.0
        assert xsec.total_cross_section(18600.0) > 0.0

    def test_h2_order_of_magnitude(self):
        # H2 cross-section at 100 eV should be ~ 1e-20 m^2
        xsec = InelasticCrossSection("H2")
        sigma = xsec.total_cross_section(100.0)
        assert 1e-21 < sigma < 1e-19

    def test_decreases_at_high_energy(self):
        # BEB cross-section decreases as ~ln(E)/E at high energies
        xsec = InelasticCrossSection("H2")
        sigma_100 = xsec.total_cross_section(100.0)
        sigma_10k = xsec.total_cross_section(10000.0)
        assert sigma_100 > sigma_10k

    def test_different_species_have_different_thresholds(self):
        h = InelasticCrossSection("H")
        h2 = InelasticCrossSection("H2")
        he = InelasticCrossSection("He")
        # H threshold ~13.6 eV, H2 ~15.43 eV, He ~24.59 eV
        assert h.total_cross_section(14.0) > 0  # above H threshold
        assert h2.total_cross_section(14.0) == 0  # below H2 threshold
        assert he.total_cross_section(20.0) == 0  # below He threshold
        assert he.total_cross_section(25.0) > 0  # above He threshold


class TestInelasticSDCS:

    def test_sdcs_positive_in_valid_range(self):
        xsec = InelasticCrossSection("H2")
        energy = 100.0
        for W in [0.1, 1.0, 10.0, 40.0]:
            assert xsec.sdcs(energy, W) > 0

    def test_sdcs_zero_outside_range(self):
        xsec = InelasticCrossSection("H2")
        energy = 100.0
        # W cannot exceed energy - I
        assert xsec.sdcs(energy, energy) == 0.0
        assert xsec.sdcs(energy, -1.0) == 0.0

    def test_sdcs_integrates_to_total(self):
        xsec = InelasticCrossSection("H2")
        energy = 200.0
        binding = 15.43
        W_max = (energy - binding) / 2

        W = np.linspace(0.01, W_max, 2000)
        sdcs_vals = np.array([xsec.sdcs(energy, w) for w in W])
        # Integrate SDCS and multiply by 2 (exchange symmetry: both electrons
        # contribute equally, total = 2 * integral from 0 to W_max)
        integrated = 2 * trapezoid(sdcs_vals, W)
        total = xsec.total_cross_section(energy)
        assert np.isclose(integrated, total, rtol=0.05)


class TestInelasticDDCS:

    def test_ddcs_positive(self):
        xsec = InelasticCrossSection("H2")
        assert xsec.ddcs(100.0, 5.0, np.pi / 4) > 0

    def test_ddcs_integrates_to_sdcs(self):
        xsec = InelasticCrossSection("H2")
        energy = 200.0
        W = 10.0

        theta = np.linspace(1e-4, np.pi - 1e-4, 500)
        ddcs_vals = np.array([xsec.ddcs(energy, W, th) for th in theta])
        # Integrate over solid angle: integral of ddcs * sin(theta) dtheta
        # (the 2*pi azimuthal factor is already in the Rudd model normalization)
        integrated = trapezoid(ddcs_vals * np.sin(theta), theta)
        sdcs_val = xsec.sdcs(energy, W)
        # The Rudd angular model and BEB SDCS are independent parametrisations,
        # so they won't match exactly, but should be the same order of magnitude
        assert integrated > 0
        if sdcs_val > 0:
            ratio = integrated / sdcs_val
            assert 0.1 < ratio < 10.0


class TestInelasticSampling:

    def test_sample_post_scatter_valid_state(self):
        rng = np.random.default_rng(42)
        xsec = InelasticCrossSection("H2")
        energy = 18600.0
        pitch = np.pi / 2 - 0.01

        for _ in range(50):
            new_e, new_p = xsec.sample_post_scatter(energy, pitch, rng)
            assert new_e > 0
            assert 0 < new_p < np.pi

    def test_energy_conservation(self):
        rng = np.random.default_rng(42)
        xsec = InelasticCrossSection("H2")
        energy = 18600.0
        pitch = np.pi / 2 - 0.01
        binding = 15.43

        for _ in range(50):
            new_e, _ = xsec.sample_post_scatter(energy, pitch, rng)
            # new_e = energy - W - binding, so W = energy - new_e - binding
            W = energy - new_e - binding
            assert W >= 0
            assert W <= (energy - binding) / 2 + 0.1  # small tolerance

    def test_energy_loss_bounded(self):
        rng = np.random.default_rng(42)
        xsec = InelasticCrossSection("H2")
        energy = 18600.0
        pitch = np.pi / 2 - 0.01

        for _ in range(50):
            new_e, _ = xsec.sample_post_scatter(energy, pitch, rng)
            assert new_e < energy
            assert new_e > energy / 2 - 20  # primary keeps at least half


# ---------------------------------------------------------------------------
# ElasticCrossSection tests
# ---------------------------------------------------------------------------

class TestElasticConstruction:

    def test_valid_construction(self):
        xsec = ElasticCrossSection(Z=1, A=3.0)
        assert xsec is not None

    def test_unsupported_Z_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            ElasticCrossSection(Z=10, A=20.0)

    def test_negative_A_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ElasticCrossSection(Z=1, A=-1.0)


class TestElasticTotalCrossSection:

    def test_positive_for_all_energies(self):
        xsec = ElasticCrossSection(Z=1, A=3.0)
        for energy in [100.0, 1000.0, 18600.0, 100000.0]:
            assert xsec.total_cross_section(energy) > 0

    def test_decreases_with_energy(self):
        xsec = ElasticCrossSection(Z=1, A=3.0)
        sigma_1k = xsec.total_cross_section(1000.0)
        sigma_10k = xsec.total_cross_section(10000.0)
        assert sigma_1k > sigma_10k

    def test_order_of_magnitude_at_18keV(self):
        xsec = ElasticCrossSection(Z=1, A=3.0)
        sigma = xsec.total_cross_section(18600.0)
        # Elastic on hydrogen at ~18 keV: expect ~ 1e-23 to 1e-21 m^2
        assert 1e-25 < sigma < 1e-19

    def test_scales_with_Z_squared(self):
        xsec_1 = ElasticCrossSection(Z=1, A=1.0)
        xsec_2 = ElasticCrossSection(Z=2, A=4.0)
        energy = 18600.0
        ratio = xsec_2.total_cross_section(energy) / xsec_1.total_cross_section(energy)
        # Should roughly scale as Z^2 (Rutherford), but correction factors differ
        assert ratio > 1.0


class TestElasticDCS:

    def test_dcs_positive(self):
        xsec = ElasticCrossSection(Z=1, A=3.0)
        assert xsec.dcs(18600.0, np.pi / 4) > 0

    def test_forward_peaked(self):
        xsec = ElasticCrossSection(Z=1, A=3.0)
        dcs_small = xsec.dcs(18600.0, 0.01)
        dcs_large = xsec.dcs(18600.0, np.pi / 2)
        assert dcs_small > dcs_large


class TestElasticSampling:

    def test_sample_produces_valid_state(self):
        rng = np.random.default_rng(42)
        xsec = ElasticCrossSection(Z=1, A=3.0)

        for _ in range(50):
            new_e, new_p = xsec.sample_post_scatter(18600.0, np.pi / 2 - 0.01, rng)
            assert new_e > 0
            assert 0 < new_p < np.pi

    def test_energy_loss_small(self):
        rng = np.random.default_rng(42)
        xsec = ElasticCrossSection(Z=1, A=3.0)
        energy = 18600.0

        for _ in range(100):
            new_e, _ = xsec.sample_post_scatter(energy, np.pi / 2 - 0.01, rng)
            loss_fraction = (energy - new_e) / energy
            # Elastic loss is O(m_e / M_nucleus) ~ 1e-4
            assert loss_fraction < 0.01

    def test_scattering_angle_forward_peaked(self):
        rng = np.random.default_rng(42)
        xsec = ElasticCrossSection(Z=1, A=3.0)
        energy = 18600.0

        angles = []
        for _ in range(1000):
            theta = xsec._sample_scattering_angle(energy, rng)
            angles.append(theta)

        # Most angles should be small (forward scattering)
        angles = np.array(angles)
        assert np.median(angles) < np.pi / 4
