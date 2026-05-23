"""
Unit tests for IsotropicAntenna class
"""

import numpy as np
import pytest
from CRESSignalStructure.antennas.IsotropicAntenna import IsotropicAntenna


class TestIsotropicAntennaConstruction:
    """Tests for IsotropicAntenna constructor"""

    def test_valid_antenna_creation(self):
        """Test creating a valid isotropic antenna"""
        position = np.array([0.0, 0.0, 0.1])
        impedance = 50.0 + 0j
        effective_length = 0.01

        antenna = IsotropicAntenna(position, impedance, effective_length)

        assert np.array_equal(antenna.GetPosition(), position)
        assert antenna.GetImpedance(1e9) == impedance
        assert antenna.GetEffectiveLengthMagnitude() == effective_length

    def test_antenna_with_default_parameters(self):
        """Test creating antenna with default impedance and effective length"""
        position = np.array([1.0, 2.0, 3.0])
        antenna = IsotropicAntenna(position)

        assert antenna.GetImpedance(1e9) == 50.0 + 0j
        assert antenna.GetEffectiveLengthMagnitude() == 0.01

    def test_antenna_with_complex_impedance(self):
        """Test antenna with reactive impedance component"""
        position = np.zeros(3)
        impedance = 50.0 + 10.0j  # Resistance + reactance

        antenna = IsotropicAntenna(position, impedance)
        assert antenna.GetImpedance(1e9) == impedance

    def test_negative_resistance_raises_error(self):
        """Test that negative resistance raises ValueError"""
        with pytest.raises(ValueError, match="resistance.*non-negative"):
            IsotropicAntenna(np.zeros(3), impedance=-50.0)

    def test_complex_impedance_with_negative_resistance_raises_error(self):
        """Test that complex impedance with negative resistance raises error"""
        with pytest.raises(ValueError, match="resistance.*non-negative"):
            IsotropicAntenna(np.zeros(3), impedance=-50.0 + 10.0j)

    def test_infinite_impedance_raises_error(self):
        """Test that infinite impedance raises ValueError"""
        with pytest.raises(ValueError, match="Impedance must be finite"):
            IsotropicAntenna(np.zeros(3), impedance=np.inf)

    def test_negative_effective_length_raises_error(self):
        """Test that negative effective length raises ValueError"""
        with pytest.raises(ValueError, match="Effective length must be positive"):
            IsotropicAntenna(np.zeros(3), effective_length=-0.01)

    def test_zero_effective_length_raises_error(self):
        """Test that zero effective length raises ValueError"""
        with pytest.raises(ValueError, match="Effective length must be positive"):
            IsotropicAntenna(np.zeros(3), effective_length=0.0)

    def test_infinite_effective_length_raises_error(self):
        """Test that infinite effective length raises ValueError"""
        with pytest.raises(ValueError, match="Effective length must be finite"):
            IsotropicAntenna(np.zeros(3), effective_length=np.inf)


class TestIsotropicAntennaGain:
    """Tests for gain pattern (should be unity everywhere)"""

    def test_unity_gain_at_various_angles(self):
        """Test that gain is 1.0 at all angles"""
        antenna = IsotropicAntenna(np.zeros(3))

        angles = [
            (0.0, 0.0),           # North pole
            (np.pi, 0.0),         # South pole
            (np.pi/2, 0.0),       # Equator, x-direction
            (np.pi/2, np.pi/2),   # Equator, y-direction
            (np.pi/4, np.pi/4),   # General angle
        ]

        for theta, phi in angles:
            gain = antenna.GetGain(theta, phi)
            assert gain == 1.0

    def test_gain_in_dbi(self):
        """Test that gain in dBi is 0 (reference)"""
        antenna = IsotropicAntenna(np.zeros(3))
        gain_linear = antenna.GetGain(np.pi/4, 0.0)
        gain_dbi = 10 * np.log10(gain_linear)
        assert np.isclose(gain_dbi, 0.0)


class TestRadiationPattern:
    """Tests for GetETheta and GetEPhi methods"""

    def test_etheta_single_position(self):
        """Test GetETheta with single position"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([1.0, 0.0, 0.0])  # Point on x-axis

        e_theta = antenna.GetETheta(pos)

        # Should return (1, 3) array
        assert e_theta.shape == (1, 3)
        assert np.all(np.isfinite(e_theta))

    def test_etheta_multiple_positions(self):
        """Test GetETheta with multiple positions"""
        antenna = IsotropicAntenna(np.zeros(3))
        positions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        e_theta = antenna.GetETheta(positions)

        assert e_theta.shape == (3, 3)
        assert np.all(np.isfinite(e_theta))

    def test_etheta_is_unit_vector(self):
        """Test that ETheta vectors have unit magnitude (where defined)"""
        antenna = IsotropicAntenna(np.zeros(3))

        # Test at equator (not at poles where theta-hat is undefined)
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        e_theta = antenna.GetETheta(pos)
        magnitudes = np.linalg.norm(e_theta, axis=1)

        # Should be approximately unit vectors
        assert np.all(np.abs(magnitudes - 1.0) < 1e-10)

    def test_etheta_perpendicular_to_radial(self):
        """Test that ETheta is perpendicular to radial direction"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.5, 0.3]])

        e_theta = antenna.GetETheta(pos)
        r_hat = pos / np.linalg.norm(pos)

        # Dot product should be zero (perpendicular)
        dot_product = np.dot(e_theta[0], r_hat[0])
        assert np.abs(dot_product) < 1e-10

    def test_etheta_zero_at_poles(self):
        """Test that ETheta is zero at poles (where undefined)"""
        antenna = IsotropicAntenna(np.zeros(3))

        # North and south poles
        north_pole = np.array([[0.0, 0.0, 1.0]])
        south_pole = np.array([[0.0, 0.0, -1.0]])

        e_theta_north = antenna.GetETheta(north_pole)
        e_theta_south = antenna.GetETheta(south_pole)

        assert np.allclose(e_theta_north, 0.0)
        assert np.allclose(e_theta_south, 0.0)

    def test_ephi_single_position(self):
        """Test GetEPhi with single position"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([1.0, 0.0, 0.0])

        e_phi = antenna.GetEPhi(pos)

        assert e_phi.shape == (1, 3)
        assert np.all(np.isfinite(e_phi))

    def test_ephi_multiple_positions(self):
        """Test GetEPhi with multiple positions"""
        antenna = IsotropicAntenna(np.zeros(3))
        positions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        e_phi = antenna.GetEPhi(positions)

        assert e_phi.shape == (3, 3)
        assert np.all(np.isfinite(e_phi))

    def test_ephi_is_unit_vector(self):
        """Test that EPhi vectors have unit magnitude (where defined)"""
        antenna = IsotropicAntenna(np.zeros(3))

        # Test at equator (not at poles)
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        e_phi = antenna.GetEPhi(pos)
        magnitudes = np.linalg.norm(e_phi, axis=1)

        # Should be approximately unit vectors
        assert np.all(np.abs(magnitudes - 1.0) < 1e-10)

    def test_ephi_perpendicular_to_radial(self):
        """Test that EPhi is perpendicular to radial direction"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.5, 0.3]])

        e_phi = antenna.GetEPhi(pos)
        r_hat = pos / np.linalg.norm(pos)

        # Dot product should be zero (perpendicular)
        dot_product = np.dot(e_phi[0], r_hat[0])
        assert np.abs(dot_product) < 1e-10

    def test_ephi_zero_at_poles(self):
        """Test that EPhi is zero at poles (where undefined)"""
        antenna = IsotropicAntenna(np.zeros(3))

        # North and south poles
        north_pole = np.array([[0.0, 0.0, 1.0]])
        south_pole = np.array([[0.0, 0.0, -1.0]])

        e_phi_north = antenna.GetEPhi(north_pole)
        e_phi_south = antenna.GetEPhi(south_pole)

        assert np.allclose(e_phi_north, 0.0)
        assert np.allclose(e_phi_south, 0.0)

    def test_etheta_ephi_orthogonal(self):
        """Test that ETheta and EPhi are orthogonal"""
        antenna = IsotropicAntenna(np.zeros(3))

        # Test at several non-pole positions
        positions = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5]
        ])

        e_theta = antenna.GetETheta(positions)
        e_phi = antenna.GetEPhi(positions)

        # Dot products should be zero (orthogonal)
        for i in range(len(positions)):
            dot_product = np.dot(e_theta[i], e_phi[i])
            assert np.abs(dot_product) < 1e-10


class TestEffectiveLength:
    """Tests for effective length calculations"""

    def test_effective_length_magnitude(self):
        """Test that effective length has correct magnitude"""
        eff_len = 0.02  # 2 cm
        antenna = IsotropicAntenna(np.zeros(3), effective_length=eff_len)
        pos = np.array([[1.0, 0.0, 0.0]])
        frequency = 26e9  # 26 GHz

        l_eff = antenna.GetEffectiveLength(frequency, pos)

        magnitude = np.linalg.norm(l_eff[0])
        assert np.isclose(magnitude, eff_len, rtol=1e-10)

    def test_effective_length_perpendicular_to_propagation(self):
        """Test that effective length is perpendicular to propagation direction"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.5, 0.3]])
        frequency = 26e9

        l_eff = antenna.GetEffectiveLength(frequency, pos)
        k_hat = pos / np.linalg.norm(pos)

        # Effective length should be perpendicular to k
        dot_product = np.dot(l_eff[0], k_hat[0])
        assert np.abs(dot_product) < 1e-10

    def test_effective_length_frequency_independent(self):
        """Test that effective length is independent of frequency"""
        antenna = IsotropicAntenna(np.zeros(3), effective_length=0.01)
        pos = np.array([[1.0, 0.0, 0.0]])

        l_eff_1ghz = antenna.GetEffectiveLength(1e9, pos)
        l_eff_26ghz = antenna.GetEffectiveLength(26e9, pos)

        # Should be equal (frequency independent for isotropic)
        assert np.allclose(l_eff_1ghz, l_eff_26ghz)

    def test_effective_length_multiple_positions(self):
        """Test effective length with multiple positions"""
        antenna = IsotropicAntenna(np.zeros(3))
        positions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.7]
        ])
        frequency = 26e9

        l_eff = antenna.GetEffectiveLength(frequency, positions)

        assert l_eff.shape == (3, 3)
        assert np.all(np.isfinite(l_eff))

        # Check magnitudes
        magnitudes = np.linalg.norm(l_eff, axis=1)
        assert np.allclose(magnitudes, 0.01, rtol=1e-10)

    def test_effective_length_negative_frequency_raises_error(self):
        """Test that negative frequency raises error"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="Frequency must be positive"):
            antenna.GetEffectiveLength(-1e9, pos)

    def test_effective_length_zero_frequency_raises_error(self):
        """Test that zero frequency raises error"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="Frequency must be positive"):
            antenna.GetEffectiveLength(0.0, pos)


class TestImpedance:
    """Tests for antenna impedance"""

    def test_impedance_frequency_independent(self):
        """Test that impedance is independent of frequency"""
        impedance = 75.0 + 5.0j
        antenna = IsotropicAntenna(np.zeros(3), impedance=impedance)

        z1 = antenna.GetImpedance(1e9)
        z2 = antenna.GetImpedance(26e9)
        z3 = antenna.GetImpedance(100e9)

        assert z1 == impedance
        assert z2 == impedance
        assert z3 == impedance

    def test_impedance_purely_real(self):
        """Test impedance with zero reactance"""
        resistance = 50.0
        antenna = IsotropicAntenna(np.zeros(3), impedance=resistance)

        z = antenna.GetImpedance(26e9)

        assert np.real(z) == resistance
        assert np.imag(z) == 0.0

    def test_impedance_with_reactance(self):
        """Test impedance with reactive component"""
        impedance = 50.0 + 20.0j
        antenna = IsotropicAntenna(np.zeros(3), impedance=impedance)

        z = antenna.GetImpedance(26e9)

        assert np.real(z) == 50.0
        assert np.imag(z) == 20.0


class TestGettersAndSetters:
    """Tests for getter and setter methods"""

    def test_get_position(self):
        """Test GetPosition returns correct position"""
        position = np.array([1.0, 2.0, 3.0])
        antenna = IsotropicAntenna(position)

        pos = antenna.GetPosition()
        assert np.array_equal(pos, position)

    def test_get_position_returns_copy(self):
        """Test that GetPosition returns a copy, not reference"""
        position = np.array([1.0, 2.0, 3.0])
        antenna = IsotropicAntenna(position)

        pos = antenna.GetPosition()
        pos[0] = 999.0  # Modify returned array

        # Original should be unchanged
        assert antenna.GetPosition()[0] == 1.0

    def test_get_orientation(self):
        """Test GetOrientation returns z-axis"""
        antenna = IsotropicAntenna(np.zeros(3))

        orientation = antenna.GetOrientation()

        # For isotropic antenna, orientation is [0, 0, 1]
        assert np.array_equal(orientation, np.array([0.0, 0.0, 1.0]))

    def test_get_effective_length_magnitude(self):
        """Test GetEffectiveLengthMagnitude"""
        eff_len = 0.015
        antenna = IsotropicAntenna(np.zeros(3), effective_length=eff_len)

        magnitude = antenna.GetEffectiveLengthMagnitude()
        assert magnitude == eff_len

    def test_set_effective_length(self):
        """Test SetEffectiveLength updates the value"""
        antenna = IsotropicAntenna(np.zeros(3), effective_length=0.01)

        new_length = 0.02
        antenna.SetEffectiveLength(new_length)

        assert antenna.GetEffectiveLengthMagnitude() == new_length

    def test_set_effective_length_affects_calculations(self):
        """Test that setting effective length affects GetEffectiveLength"""
        antenna = IsotropicAntenna(np.zeros(3), effective_length=0.01)
        pos = np.array([[1.0, 0.0, 0.0]])

        # Original length
        l_eff1 = antenna.GetEffectiveLength(26e9, pos)
        mag1 = np.linalg.norm(l_eff1[0])

        # Update length
        antenna.SetEffectiveLength(0.02)
        l_eff2 = antenna.GetEffectiveLength(26e9, pos)
        mag2 = np.linalg.norm(l_eff2[0])

        assert np.isclose(mag1, 0.01)
        assert np.isclose(mag2, 0.02)

    def test_set_negative_effective_length_raises_error(self):
        """Test that setting negative effective length raises error"""
        antenna = IsotropicAntenna(np.zeros(3))

        with pytest.raises(ValueError, match="Effective length must be positive"):
            antenna.SetEffectiveLength(-0.01)

    def test_set_zero_effective_length_raises_error(self):
        """Test that setting zero effective length raises error"""
        antenna = IsotropicAntenna(np.zeros(3))

        with pytest.raises(ValueError, match="Effective length must be positive"):
            antenna.SetEffectiveLength(0.0)


class TestAngleCalculations:
    """Tests for GetTheta and GetPhi methods (inherited from BaseAntenna)"""

    def test_get_theta_on_axis(self):
        """Test GetTheta for point on z-axis"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[0.0, 0.0, 1.0]])  # On z-axis

        theta = antenna.GetTheta(pos)

        # Should be 0 (north pole) or π (south pole depending on direction)
        assert np.isclose(theta, 0.0) or np.isclose(theta, np.pi)

    def test_get_theta_equatorial(self):
        """Test GetTheta for point on equator"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.0, 0.0]])  # On equator

        theta = antenna.GetTheta(pos)

        # Should be π/2
        assert np.isclose(theta, np.pi/2)

    def test_get_phi_on_x_axis(self):
        """Test GetPhi for point on x-axis"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[1.0, 0.0, 0.0]])

        phi = antenna.GetPhi(pos)

        # Should be 0 or ±π
        assert np.isclose(np.abs(phi), 0.0) or np.isclose(np.abs(phi), np.pi)

    def test_get_phi_on_y_axis(self):
        """Test GetPhi for point on y-axis"""
        antenna = IsotropicAntenna(np.zeros(3))
        pos = np.array([[0.0, 1.0, 0.0]])

        phi = antenna.GetPhi(pos)

        # Should be ±π/2
        assert np.isclose(np.abs(phi), np.pi/2)


class TestTypicalCRESParameters:
    """Tests using typical CRES experimental parameters"""

    def test_typical_cres_antenna(self):
        """Test with typical CRES antenna parameters"""
        # Typical setup: antenna at trap center, 1 cm effective length, 50Ω
        position = np.array([0.0, 0.0, 0.0])
        impedance = 50.0 + 0j
        effective_length = 0.01  # 1 cm

        antenna = IsotropicAntenna(position, impedance, effective_length)

        # Test at typical CRES frequency (26 GHz for 18.6 keV electron in 1 T)
        frequency = 26e9

        # Check basic properties
        assert antenna.GetGain(np.pi/4, 0.0) == 1.0
        assert antenna.GetImpedance(frequency) == impedance

        # Check effective length at typical observation point
        obs_point = np.array([[0.05, 0.0, 0.1]])  # 5 cm radially, 10 cm axially
        l_eff = antenna.GetEffectiveLength(frequency, obs_point)

        assert np.isclose(np.linalg.norm(l_eff[0]), effective_length)

    def test_antenna_off_axis_position(self):
        """Test antenna positioned off-axis (typical experimental setup)"""
        # Antenna positioned at waveguide wall
        position = np.array([0.01, 0.0, 0.0])  # 1 cm from axis

        antenna = IsotropicAntenna(position)

        # Should still have unity gain everywhere
        assert antenna.GetGain(0.0, 0.0) == 1.0
        assert antenna.GetGain(np.pi/2, np.pi/4) == 1.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_very_small_effective_length(self):
        """Test with very small effective length"""
        antenna = IsotropicAntenna(np.zeros(3), effective_length=1e-6)

        assert antenna.GetEffectiveLengthMagnitude() == 1e-6

    def test_large_effective_length(self):
        """Test with large effective length"""
        antenna = IsotropicAntenna(np.zeros(3), effective_length=1.0)

        assert antenna.GetEffectiveLengthMagnitude() == 1.0

    def test_purely_reactive_impedance(self):
        """Test antenna with zero resistance (purely reactive)"""
        impedance = 0.0 + 50.0j
        antenna = IsotropicAntenna(np.zeros(3), impedance=impedance)

        z = antenna.GetImpedance(26e9)
        assert np.real(z) == 0.0
        assert np.imag(z) == 50.0

    def test_position_far_from_origin(self):
        """Test antenna positioned far from origin"""
        position = np.array([100.0, 200.0, 300.0])
        antenna = IsotropicAntenna(position)

        assert np.array_equal(antenna.GetPosition(), position)

        # Should still work correctly
        pos = np.array([[101.0, 200.0, 300.0]])
        l_eff = antenna.GetEffectiveLength(26e9, pos)

        assert np.allclose(np.linalg.norm(l_eff[0]), 0.01)

    def test_multiple_antennas_independent(self):
        """Test that multiple antenna instances are independent"""
        antenna1 = IsotropicAntenna(np.zeros(3), effective_length=0.01)
        antenna2 = IsotropicAntenna(np.ones(3), effective_length=0.02)

        # Modifying one shouldn't affect the other
        antenna1.SetEffectiveLength(0.03)

        assert antenna1.GetEffectiveLengthMagnitude() == 0.03
        assert antenna2.GetEffectiveLengthMagnitude() == 0.02


class TestPhysicsConsistency:
    """Tests for overall physics consistency"""

    def test_isotropic_means_no_preferred_direction(self):
        """Test that antenna response is truly isotropic"""
        antenna = IsotropicAntenna(np.zeros(3))

        # Sample many random directions
        np.random.seed(42)
        n_samples = 100
        gains = []

        for _ in range(n_samples):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            gains.append(antenna.GetGain(theta, phi))

        # All gains should be exactly 1.0
        assert np.all(np.array(gains) == 1.0)

    def test_effective_length_scales_linearly(self):
        """Test that doubling effective length doubles magnitude"""
        antenna = IsotropicAntenna(np.zeros(3), effective_length=0.01)
        pos = np.array([[1.0, 0.0, 0.0]])

        l_eff1 = antenna.GetEffectiveLength(26e9, pos)
        mag1 = np.linalg.norm(l_eff1[0])

        antenna.SetEffectiveLength(0.02)
        l_eff2 = antenna.GetEffectiveLength(26e9, pos)
        mag2 = np.linalg.norm(l_eff2[0])

        assert np.isclose(mag2 / mag1, 2.0)

    def test_radiation_pattern_completeness(self):
        """Test that radiation pattern vectors form orthonormal basis with r_hat"""
        antenna = IsotropicAntenna(np.zeros(3))

        # Test at a non-special point
        pos = np.array([[0.5, 0.7, 0.3]])

        e_theta = antenna.GetETheta(pos)[0]
        e_phi = antenna.GetEPhi(pos)[0]
        r_hat = pos[0] / np.linalg.norm(pos[0])

        # Check magnitudes (unit vectors)
        assert np.isclose(np.linalg.norm(e_theta), 1.0)
        assert np.isclose(np.linalg.norm(e_phi), 1.0)
        assert np.isclose(np.linalg.norm(r_hat), 1.0)

        # Check orthogonality
        assert np.abs(np.dot(e_theta, e_phi)) < 1e-10
        assert np.abs(np.dot(e_theta, r_hat)) < 1e-10
        assert np.abs(np.dot(e_phi, r_hat)) < 1e-10

        # Check they form a right-handed system
        cross_product = np.cross(e_theta, e_phi)
        # Should be parallel to r_hat (or -r_hat)
        assert np.abs(np.abs(np.dot(cross_product, r_hat)) - 1.0) < 1e-10
