"""
Unit tests for CircularWaveguide class
"""

import numpy as np
import pytest
import scipy.constants as sc
from scipy.special import j1
from CRESSignalStructure.CircularWaveguide import CircularWaveguide


class TestCircularWaveguideConstruction:
    """Tests for CircularWaveguide constructor and validation"""

    def test_valid_waveguide_creation(self):
        """Test creating a valid waveguide"""
        radius = 0.01  # 1 cm radius
        wg = CircularWaveguide(radius)
        assert wg.wgR == radius

    def test_waveguide_string_representation(self):
        """Test string representation"""
        radius = 0.01
        wg = CircularWaveguide(radius)
        assert str(wg) == f"Waveguide with radius {radius} metres"

    def test_negative_radius_raises_error(self):
        """Test that negative radius raises ValueError"""
        with pytest.raises(ValueError, match="Radius must be positive"):
            CircularWaveguide(-0.01)

    def test_zero_radius_raises_error(self):
        """Test that zero radius raises ValueError"""
        with pytest.raises(ValueError, match="Radius must be positive"):
            CircularWaveguide(0.0)

    def test_non_numeric_radius_raises_error(self):
        """Test that non-numeric radius raises TypeError"""
        with pytest.raises(TypeError, match="Radius must be a number"):
            CircularWaveguide("0.01")

    def test_infinite_radius_raises_error(self):
        """Test that infinite radius raises ValueError"""
        with pytest.raises(ValueError, match="Radius must be finite"):
            CircularWaveguide(np.inf)

    def test_nan_radius_raises_error(self):
        """Test that NaN radius raises ValueError"""
        with pytest.raises(ValueError, match="Radius must be finite"):
            CircularWaveguide(np.nan)


class TestFrequencyValidation:
    """Tests for frequency validation"""

    def test_valid_frequency(self):
        """Test that valid frequency above cutoff passes validation"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c  # Well above cutoff
        wg._ValidateFrequency(omega)

    def test_frequency_below_cutoff_raises_error(self):
        """Test that frequency below cutoff raises ValueError"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 0.5 * omega_c  # Below cutoff
        with pytest.raises(ValueError, match="below cutoff frequency"):
            wg._ValidateFrequency(omega)

    def test_negative_frequency_raises_error(self):
        """Test that negative frequency raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Frequency must be positive"):
            wg._ValidateFrequency(-1e11)

    def test_zero_frequency_raises_error(self):
        """Test that zero frequency raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Frequency must be positive"):
            wg._ValidateFrequency(0.0)

    def test_non_numeric_frequency_raises_error(self):
        """Test that non-numeric frequency raises TypeError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(TypeError, match="Frequency must be a number"):
            wg._ValidateFrequency("1e11")

    def test_infinite_frequency_raises_error(self):
        """Test that infinite frequency raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Frequency must be finite"):
            wg._ValidateFrequency(np.inf)

    def test_nan_frequency_raises_error(self):
        """Test that NaN frequency raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Frequency must be finite"):
            wg._ValidateFrequency(np.nan)


class TestPositionValidation:
    """Tests for position validation"""

    def test_valid_position(self):
        """Test that valid position passes validation"""
        wg = CircularWaveguide(0.01)
        rho, phi = wg._ValidatePosition(0.005, np.pi / 4)
        assert np.isclose(rho, 0.005)
        assert np.isclose(phi, np.pi / 4)

    def test_negative_rho_raises_error(self):
        """Test that negative radial position raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Radial position must be non-negative"):
            wg._ValidatePosition(-0.001, 0.0)

    def test_non_numeric_rho_raises_error(self):
        """Test that non-numeric radial position raises TypeError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(TypeError, match="Radial position must be numeric"):
            wg._ValidatePosition("0.005", 0.0)

    def test_non_numeric_phi_raises_error(self):
        """Test that non-numeric azimuthal angle raises TypeError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(TypeError, match="Azimuthal angle must be numeric"):
            wg._ValidatePosition(0.005, "0.0")

    def test_infinite_rho_raises_error(self):
        """Test that infinite radial position raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Radial position must be finite"):
            wg._ValidatePosition(np.inf, 0.0)

    def test_infinite_phi_raises_error(self):
        """Test that infinite azimuthal angle raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Azimuthal angle must be finite"):
            wg._ValidatePosition(0.005, np.inf)


class TestAmplitudeValidation:
    """Tests for amplitude validation"""

    def test_valid_amplitude(self):
        """Test that valid amplitude passes validation"""
        wg = CircularWaveguide(0.01)
        A = wg._ValidateAmplitude(1.0)
        assert np.isclose(A, 1.0)

    def test_non_numeric_amplitude_raises_error(self):
        """Test that non-numeric amplitude raises TypeError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(TypeError, match="Amplitude must be numeric"):
            wg._ValidateAmplitude("1.0")

    def test_infinite_amplitude_raises_error(self):
        """Test that infinite amplitude raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Amplitude must be finite"):
            wg._ValidateAmplitude(np.inf)

    def test_nan_amplitude_raises_error(self):
        """Test that NaN amplitude raises ValueError"""
        wg = CircularWaveguide(0.01)
        with pytest.raises(ValueError, match="Amplitude must be finite"):
            wg._ValidateAmplitude(np.nan)


class TestElectricFieldMode1:
    """Tests for TE11 mode 1 electric field calculations"""

    def test_efield_outside_waveguide(self):
        """Test that electric field is zero outside waveguide"""
        wg = CircularWaveguide(0.01)
        rho = 0.02  # Outside waveguide
        phi = 0.0
        A = 1.0

        E_rho = wg.EFieldTE11Rho_1(rho, phi, A)
        E_phi = wg.EFieldTE11Phi_1(rho, phi, A)
        E_z = wg.EFieldTE11Z(rho, phi, A)

        assert E_rho == 0.0
        assert E_phi == 0.0
        assert E_z == 0.0

    def test_efield_on_axis(self):
        """Test electric field at rho=0 (on axis)"""
        wg = CircularWaveguide(0.01)
        rho = 0.0
        phi = 0.0
        A = 1.0
        kc = 1.841 / 0.01

        E_rho = wg.EFieldTE11Rho_1(rho, phi, A)
        expected = A * np.cos(phi) / kc
        assert np.isclose(E_rho, expected)

    def test_efield_inside_waveguide(self):
        """Test electric field inside waveguide"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        A = 1.0
        kc = 1.841 / 0.01

        E_rho = wg.EFieldTE11Rho_1(rho, phi, A)
        expected = A * j1(kc * rho) / (kc * rho) * np.cos(phi)
        assert np.isclose(E_rho, expected)

    def test_efield_z_always_zero(self):
        """Test that z-component of electric field is always zero for TE mode"""
        wg = CircularWaveguide(0.01)
        rho_values = np.array([0.0, 0.005, 0.01, 0.02])
        phi = 0.0
        A = 1.0

        for rho in rho_values:
            E_z = wg.EFieldTE11Z(rho, phi, A)
            assert E_z == 0.0

    def test_efield_cartesian_conversion(self):
        """Test electric field in Cartesian coordinates"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        A = 1.0

        E_vec = wg.EFieldTE11_1(rho, phi, A)
        assert E_vec.shape == (3,)
        assert E_vec[2] == 0.0  # z-component should be zero

    def test_efield_from_position_vector(self):
        """Test electric field calculation from position vector"""
        wg = CircularWaveguide(0.01)
        x, y, z = 0.005, 0.005, 0.0
        pos = np.array([x, y, z])
        A = 1.0

        E_vec = wg.EFieldTE11Pos_1(pos, A)
        assert E_vec.shape == (3,)

        # Compare with direct calculation
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        E_vec_direct = wg.EFieldTE11_1(rho, phi, A)
        assert np.allclose(E_vec, E_vec_direct)


class TestMagneticFieldMode1:
    """Tests for TE11 mode 1 magnetic field calculations"""

    def test_hfield_outside_waveguide(self):
        """Test that magnetic field is zero outside waveguide"""
        wg = CircularWaveguide(0.01)
        rho = 0.02  # Outside waveguide
        phi = 0.0
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_rho = wg.HFieldTE11Rho_1(rho, phi, omega, A)
        H_phi = wg.HFieldTE11Phi_1(rho, phi, omega, A)

        assert H_rho == 0.0
        assert H_phi == 0.0

    def test_hfield_inside_waveguide(self):
        """Test magnetic field inside waveguide"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_rho = wg.HFieldTE11Rho_1(rho, phi, omega, A)
        H_phi = wg.HFieldTE11Phi_1(rho, phi, omega, A)

        # Should be non-zero inside waveguide
        assert H_rho != 0.0
        assert H_phi != 0.0

    def test_hfield_cartesian_conversion(self):
        """Test magnetic field in Cartesian coordinates"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_vec = wg.HFieldTE11_1(rho, phi, omega, A)
        assert H_vec.shape == (3,)
        assert H_vec[2] == 0.0  # z-component should be zero for TE11

    def test_hfield_from_position_vector(self):
        """Test magnetic field calculation from position vector"""
        wg = CircularWaveguide(0.01)
        x, y, z = 0.005, 0.005, 0.0
        pos = np.array([x, y, z])
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_vec = wg.HFieldTE11Pos_1(pos, omega, A)
        assert H_vec.shape == (3,)

        # Compare with direct calculation
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        H_vec_direct = wg.HFieldTE11_1(rho, phi, omega, A)
        assert np.allclose(H_vec, H_vec_direct)


class TestElectricFieldMode2:
    """Tests for TE11 mode 2 electric field calculations"""

    def test_efield_mode2_outside_waveguide(self):
        """Test that electric field is zero outside waveguide for mode 2"""
        wg = CircularWaveguide(0.01)
        rho = 0.02
        phi = 0.0
        A = 1.0

        E_rho = wg.EFieldTE11Rho_2(rho, phi, A)
        E_phi = wg.EFieldTE11Phi_2(rho, phi, A)

        assert E_rho == 0.0
        assert E_phi == 0.0

    def test_efield_mode2_on_axis(self):
        """Test electric field at rho=0 for mode 2"""
        wg = CircularWaveguide(0.01)
        rho = 0.0
        phi = 0.0
        A = 1.0
        kc = 1.841 / 0.01

        E_rho = wg.EFieldTE11Rho_2(rho, phi, A)
        expected = -A * np.sin(phi) / kc
        assert np.isclose(E_rho, expected)

    def test_efield_mode2_cartesian_conversion(self):
        """Test mode 2 electric field in Cartesian coordinates"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        A = 1.0

        E_vec = wg.EFieldTE11_2(rho, phi, A)
        assert E_vec.shape == (3,)
        assert E_vec[2] == 0.0

    def test_efield_mode2_from_position_vector(self):
        """Test mode 2 electric field from position vector"""
        wg = CircularWaveguide(0.01)
        x, y, z = 0.005, 0.005, 0.0
        pos = np.array([x, y, z])
        A = 1.0

        E_vec = wg.EFieldTE11Pos_2(pos, A)
        assert E_vec.shape == (3,)


class TestMagneticFieldMode2:
    """Tests for TE11 mode 2 magnetic field calculations"""

    def test_hfield_mode2_outside_waveguide(self):
        """Test that magnetic field is zero outside waveguide for mode 2"""
        wg = CircularWaveguide(0.01)
        rho = 0.02
        phi = 0.0
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_rho = wg.HFieldTE11Rho_2(rho, phi, omega, A)
        H_phi = wg.HFieldTE11Phi_2(rho, phi, omega, A)

        assert H_rho == 0.0
        assert H_phi == 0.0

    def test_hfield_mode2_inside_waveguide(self):
        """Test magnetic field inside waveguide for mode 2"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_rho = wg.HFieldTE11Rho_2(rho, phi, omega, A)
        H_phi = wg.HFieldTE11Phi_2(rho, phi, omega, A)

        # Should be non-zero inside waveguide
        assert H_rho != 0.0
        assert H_phi != 0.0

    def test_hfield_mode2_cartesian_conversion(self):
        """Test mode 2 magnetic field in Cartesian coordinates"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi / 4
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_vec = wg.HFieldTE11_2(rho, phi, omega, A)
        assert H_vec.shape == (3,)
        assert H_vec[2] == 0.0

    def test_hfield_mode2_from_position_vector(self):
        """Test mode 2 magnetic field from position vector"""
        wg = CircularWaveguide(0.01)
        x, y, z = 0.005, 0.005, 0.0
        pos = np.array([x, y, z])
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_vec = wg.HFieldTE11Pos_2(pos, omega, A)
        assert H_vec.shape == (3,)


class TestVelocityCalculations:
    """Tests for phase and group velocity calculations"""

    def test_phase_velocity_above_cutoff(self):
        """Test phase velocity is greater than c above cutoff"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c

        v_phase = wg.GetPhaseVelocity(omega)
        assert v_phase > sc.c

    def test_group_velocity_below_c(self):
        """Test group velocity is less than c"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c

        v_group = wg.GetGroupVelocity(omega)
        assert v_group < sc.c

    def test_velocity_product_equals_c_squared(self):
        """Test that v_phase * v_group = c^2 for waveguides"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c

        v_phase = wg.GetPhaseVelocity(omega)
        v_group = wg.GetGroupVelocity(omega)

        assert np.isclose(v_phase * v_group, sc.c**2, rtol=1e-10)

    def test_velocity_at_high_frequency(self):
        """Test that both velocities approach c at high frequency"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 100 * omega_c  # Very high frequency

        v_phase = wg.GetPhaseVelocity(omega)
        v_group = wg.GetGroupVelocity(omega)

        # At high frequency, both should approach c
        assert np.isclose(v_phase, sc.c, rtol=0.01)
        assert np.isclose(v_group, sc.c, rtol=0.01)

    def test_velocity_frequency_below_cutoff_raises_error(self):
        """Test that velocity calculations fail below cutoff"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 0.5 * omega_c

        with pytest.raises(ValueError, match="below cutoff frequency"):
            wg.GetPhaseVelocity(omega)

        with pytest.raises(ValueError, match="below cutoff frequency"):
            wg.GetGroupVelocity(omega)


class TestImpedanceCalculations:
    """Tests for impedance calculations"""

    def test_impedance_calculation(self):
        """Test impedance calculation for TE11 mode"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c

        Z = wg.CalcTE11Impedance(omega)

        # Impedance should be positive and real
        assert Z > 0
        assert np.isreal(Z)

    def test_impedance_at_high_frequency(self):
        """Test that impedance approaches free space impedance at high frequency"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 100 * omega_c

        Z = wg.CalcTE11Impedance(omega)
        # Free space impedance (~377 Ohms)
        Z0 = np.sqrt(sc.mu_0 / sc.epsilon_0)

        # At high frequency, waveguide impedance should approach Z0
        assert np.isclose(Z, Z0, rtol=0.01)

    def test_impedance_below_cutoff_raises_error(self):
        """Test that impedance calculation fails below cutoff"""
        wg = CircularWaveguide(0.01)
        omega_c = 1.841 * sc.c / 0.01
        omega = 0.5 * omega_c

        with pytest.raises(ValueError, match="below cutoff frequency"):
            wg.CalcTE11Impedance(omega)


class TestNormalisationFactor:
    """Tests for normalisation factor calculation"""

    def test_normalisation_factor_positive(self):
        """Test that normalisation factor is positive"""
        wg = CircularWaveguide(0.01)
        norm = wg.CalcNormalisationFactor()
        assert norm > 0

    def test_normalisation_factor_finite(self):
        """Test that normalisation factor is finite"""
        wg = CircularWaveguide(0.01)
        norm = wg.CalcNormalisationFactor()
        assert np.isfinite(norm)


class TestArrayInputs:
    """Tests for array inputs to field calculations"""

    def test_efield_scalar_inputs(self):
        """Test electric field with scalar inputs"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = np.pi/4
        A = 1.0

        E_rho = wg.EFieldTE11Rho_1(rho, phi, A)
        assert np.isfinite(E_rho)
        assert isinstance(E_rho, (float, np.floating, np.ndarray))

    def test_efield_phi_array(self):
        """Test electric field Phi component with array for positions outside rho singularity"""
        wg = CircularWaveguide(0.01)
        rho = np.array([0.003, 0.005, 0.007])
        phi = np.array([0.0, np.pi/4, np.pi/2])
        A = 1.0

        E_phi = wg.EFieldTE11Phi_1(rho, phi, A)
        assert E_phi.shape == rho.shape
        assert np.all(np.isfinite(E_phi))

    def test_hfield_array_inputs(self):
        """Test magnetic field with array inputs"""
        wg = CircularWaveguide(0.01)
        rho = np.array([0.003, 0.005, 0.007])
        phi = np.array([0.0, np.pi/4, np.pi/2])
        omega_c = 1.841 * sc.c / 0.01
        omega = 2 * omega_c
        A = 1.0

        H_rho = wg.HFieldTE11Rho_1(rho, phi, omega, A)
        assert H_rho.shape == rho.shape
        assert np.all(np.isfinite(H_rho))


class TestPhysicsConsistency:
    """Tests for overall physics consistency"""

    def test_cutoff_frequency_correct(self):
        """Test that cutoff frequency is calculated correctly"""
        radius = 0.01
        wg = CircularWaveguide(radius)
        omega_c_expected = 1.841 * sc.c / radius

        # Frequency just below cutoff should fail
        with pytest.raises(ValueError, match="below cutoff frequency"):
            wg._ValidateFrequency(0.999 * omega_c_expected)

        # Frequency just above cutoff should pass
        wg._ValidateFrequency(1.001 * omega_c_expected)

    def test_mode_orthogonality(self):
        """Test that mode 1 and mode 2 are orthogonal"""
        wg = CircularWaveguide(0.01)
        rho = 0.005
        phi = 0.0
        A = 1.0

        E1 = wg.EFieldTE11_1(rho, phi, A)
        E2 = wg.EFieldTE11_2(rho, phi, A)

        # At phi=0, mode 2 should have zero E_x and maximum E_y
        # while mode 1 should have maximum E_x and zero E_y (approximately)
        # This tests the 90-degree phase relationship
        assert not np.allclose(E1, E2)

    def test_field_continuity_at_boundaries(self):
        """Test field behavior at waveguide boundary"""
        wg = CircularWaveguide(0.01)
        rho_inside = 0.99 * 0.01
        rho_outside = 1.01 * 0.01
        phi = 0.0
        A = 1.0

        E_inside = wg.EFieldTE11Rho_1(rho_inside, phi, A)
        E_outside = wg.EFieldTE11Rho_1(rho_outside, phi, A)

        # Field inside should be non-zero, outside should be zero
        assert E_inside != 0.0
        assert E_outside == 0.0
