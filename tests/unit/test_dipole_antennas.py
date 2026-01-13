"""
Unit tests for dipole antenna implementations

Tests for ShortDipoleAntenna and HalfWaveDipoleAntenna classes
"""

import pytest
import numpy as np
import scipy.constants as sc
from CRESSignalStructure.antennas import ShortDipoleAntenna, HalfWaveDipoleAntenna


class TestShortDipoleAntenna:
    """Tests for ShortDipoleAntenna class"""

    def test_construction_valid_parameters(self):
        """Test that short dipole can be constructed with valid parameters"""
        position = np.array([0.01, 0.0, 0.0])
        orientation = np.array([0.0, 0.0, 1.0])
        length = 0.005
        resistance = 1.0

        antenna = ShortDipoleAntenna(position, orientation, length, resistance)

        np.testing.assert_array_almost_equal(antenna.GetPosition(), position)
        np.testing.assert_array_almost_equal(antenna.GetOrientation(), orientation)

    def test_construction_normalizes_orientation(self):
        """Test that orientation vector is automatically normalized"""
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.array([0.0, 0.0, 2.0])  # Non-unit vector
        length = 0.005

        antenna = ShortDipoleAntenna(position, orientation, length)

        # Should be normalized to unit vector
        result_orientation = antenna.GetOrientation()
        np.testing.assert_almost_equal(np.linalg.norm(result_orientation), 1.0)
        np.testing.assert_array_almost_equal(result_orientation, [0.0, 0.0, 1.0])

    def test_invalid_position_raises_error(self):
        """Test that invalid position raises appropriate error"""
        with pytest.raises(ValueError, match="Position must be a 3-vector"):
            ShortDipoleAntenna(
                position=np.array([1.0, 2.0]),  # Only 2 elements
                orientation=np.array([0.0, 0.0, 1.0]),
                length=0.005
            )

    def test_invalid_length_raises_error(self):
        """Test that invalid length raises appropriate error"""
        with pytest.raises(ValueError, match="Length must be positive"):
            ShortDipoleAntenna(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 1.0]),
                length=-0.005  # Negative length
            )

    def test_zero_length_orientation_raises_error(self):
        """Test that zero-length orientation vector raises error"""
        with pytest.raises(ValueError, match="non-zero length"):
            ShortDipoleAntenna(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0]),  # Zero vector
                length=0.005
            )

    def test_effective_length_perpendicular_incidence(self):
        """Test effective length for perpendicular wave incidence"""
        # Vertical dipole, horizontal wave incidence
        antenna = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),  # Vertical
            length=0.005
        )

        frequency = 26e9  # 26 GHz
        theta = np.pi / 2  # Horizontal
        phi = 0.0

        l_eff = antenna.GetEffectiveLength(frequency, theta, phi)

        # For perpendicular incidence, effective length should be full length
        assert np.linalg.norm(l_eff) == pytest.approx(0.005, rel=1e-6)
        # Should be in z-direction
        assert l_eff[2] == pytest.approx(0.005, rel=1e-6)

    def test_effective_length_parallel_incidence(self):
        """Test effective length for parallel wave incidence (along dipole)"""
        # Vertical dipole, vertical wave incidence
        antenna = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        frequency = 26e9
        theta = 0.0  # Along z-axis
        phi = 0.0

        l_eff = antenna.GetEffectiveLength(frequency, theta, phi)

        # For parallel incidence, effective length should be zero
        assert np.linalg.norm(l_eff) == pytest.approx(0.0, abs=1e-10)

    def test_impedance_is_capacitive(self):
        """Test that short dipole impedance is capacitive (negative reactance)"""
        antenna = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005,
            resistance=1.0
        )

        frequency = 26e9
        Z = antenna.GetImpedance(frequency)

        # Should have positive resistance
        assert Z.real > 0
        # Should have negative reactance (capacitive)
        assert Z.imag < 0

    def test_gain_pattern_maximum_perpendicular(self):
        """Test that gain is maximum perpendicular to dipole axis"""
        antenna = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        # Maximum gain should be at theta = pi/2 (perpendicular to dipole)
        gain_perp = antenna.GetGain(np.pi / 2, 0.0)

        # For short dipole, maximum gain is 1.5
        assert gain_perp == pytest.approx(1.5, rel=1e-6)

    def test_gain_pattern_null_along_axis(self):
        """Test that gain is zero along dipole axis"""
        antenna = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        # Gain should be zero along dipole axis (theta = 0)
        gain_along = antenna.GetGain(0.0, 0.0)

        assert gain_along == pytest.approx(0.0, abs=1e-10)


class TestHalfWaveDipoleAntenna:
    """Tests for HalfWaveDipoleAntenna class"""

    def test_construction_valid_parameters(self):
        """Test that half-wave dipole can be constructed with valid parameters"""
        position = np.array([0.01, 0.0, 0.0])
        orientation = np.array([0.0, 0.0, 1.0])
        resonant_frequency = 26e9

        antenna = HalfWaveDipoleAntenna(position, orientation, resonant_frequency)

        np.testing.assert_array_almost_equal(antenna.GetPosition(), position)
        np.testing.assert_array_almost_equal(antenna.GetOrientation(), orientation)
        assert antenna.GetResonantFrequency() == resonant_frequency

    def test_length_is_half_wavelength(self):
        """Test that antenna length is λ/2 at resonant frequency"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        wavelength = sc.c / resonant_frequency
        expected_length = wavelength / 2

        assert antenna.GetLength() == pytest.approx(expected_length, rel=1e-10)

    def test_invalid_resonant_frequency_raises_error(self):
        """Test that invalid resonant frequency raises error"""
        with pytest.raises(ValueError, match="Resonant frequency must be positive"):
            HalfWaveDipoleAntenna(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 1.0]),
                resonant_frequency=-26e9  # Negative frequency
            )

    def test_impedance_at_resonance(self):
        """Test that impedance at resonance is approximately 73 + j42.5 Ω"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        Z = antenna.GetImpedance(resonant_frequency)

        # At resonance, impedance should be approximately 73 + j42.5 Ω
        assert Z.real == pytest.approx(73.0, rel=1e-3)
        assert Z.imag == pytest.approx(42.5, rel=1e-3)

    def test_impedance_below_resonance_is_capacitive(self):
        """Test that impedance below resonance has negative reactance"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        # Test at 95% of resonant frequency
        Z = antenna.GetImpedance(0.95 * resonant_frequency)

        # Below resonance, should be capacitive (negative reactance)
        assert Z.imag < 0

    def test_impedance_above_resonance_is_inductive(self):
        """Test that impedance above resonance has positive reactance"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        # Test at 105% of resonant frequency
        Z = antenna.GetImpedance(1.05 * resonant_frequency)

        # Above resonance, should be inductive (positive reactance)
        assert Z.imag > 0

    def test_effective_length_perpendicular_incidence(self):
        """Test effective length for perpendicular incidence"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        theta = np.pi / 2  # Perpendicular
        phi = 0.0

        l_eff = antenna.GetEffectiveLength(resonant_frequency, theta, phi)

        wavelength = sc.c / resonant_frequency
        expected_magnitude = wavelength / np.pi  # ~0.318 λ

        assert np.linalg.norm(l_eff) == pytest.approx(expected_magnitude, rel=1e-3)

    def test_effective_length_parallel_incidence(self):
        """Test effective length for parallel incidence (should be zero)"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        theta = 0.0  # Parallel to dipole
        phi = 0.0

        l_eff = antenna.GetEffectiveLength(resonant_frequency, theta, phi)

        # Should be zero for parallel incidence
        assert np.linalg.norm(l_eff) == pytest.approx(0.0, abs=1e-10)

    def test_gain_pattern_maximum_perpendicular(self):
        """Test that gain is maximum perpendicular to dipole"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        # Maximum gain at theta = pi/2
        gain_perp = antenna.GetGain(np.pi / 2, 0.0)

        # Half-wave dipole has maximum gain of ~1.643
        assert gain_perp == pytest.approx(1.643, rel=1e-2)

    def test_gain_pattern_null_along_axis(self):
        """Test that gain is zero along dipole axis"""
        resonant_frequency = 26e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        # Gain should be zero along axis
        gain_along = antenna.GetGain(0.0, 0.0)

        assert gain_along == pytest.approx(0.0, abs=1e-10)

    def test_gain_greater_than_short_dipole(self):
        """Test that half-wave dipole has higher gain than short dipole"""
        resonant_frequency = 26e9
        half_wave = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=resonant_frequency
        )

        short_dipole = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        # Compare gains perpendicular to dipole
        theta = np.pi / 2
        phi = 0.0

        gain_hw = half_wave.GetGain(theta, phi)
        gain_sd = short_dipole.GetGain(theta, phi)

        # Half-wave should have higher gain (1.643 vs 1.5)
        assert gain_hw > gain_sd


class TestAntennaComparison:
    """Tests comparing short dipole and half-wave dipole behaviors"""

    def test_short_dipole_more_capacitive(self):
        """Test that short dipole has more capacitive reactance"""
        frequency = 26e9

        short = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        half_wave = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=frequency
        )

        Z_short = short.GetImpedance(frequency)
        Z_hw = half_wave.GetImpedance(frequency)

        # Short dipole should be much more capacitive (more negative reactance)
        assert Z_short.imag < Z_hw.imag

    def test_position_getter_returns_copy(self):
        """Test that GetPosition returns a copy, not a reference"""
        original_position = np.array([1.0, 2.0, 3.0])
        antenna = ShortDipoleAntenna(
            position=original_position,
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        position = antenna.GetPosition()
        position[0] = 999.0  # Modify returned array

        # Original should be unchanged
        assert antenna.GetPosition()[0] == pytest.approx(1.0)

    def test_orientation_getter_returns_copy(self):
        """Test that GetOrientation returns a copy, not a reference"""
        antenna = ShortDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            length=0.005
        )

        orientation = antenna.GetOrientation()
        orientation[0] = 999.0  # Modify returned array

        # Original should be unchanged
        assert antenna.GetOrientation()[0] == pytest.approx(0.0)
