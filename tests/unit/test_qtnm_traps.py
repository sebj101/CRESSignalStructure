"""
Unit tests for QTNM analytical traps
"""

import pytest
import numpy as np
import scipy.constants as sc
from CRESSignalStructure.QTNMTraps import HarmonicTrap, BathtubTrap


class TestHarmonicTrap:
    """Tests for HarmonicTrap"""

    # ==================== Constructor Tests ====================
    def test_valid_trap_creation(self):
        """Test creating a valid harmonic trap"""
        B0 = 1.0
        L0 = 0.2
        gradB = 4e-3
        trap = HarmonicTrap(B0=B0, L0=L0, gradB=gradB)

        assert trap.GetB0() == B0
        assert trap.GetL0() == L0
        assert trap.GetGradB() == gradB

    def test_valid_trap_creation_without_gradB(self):
        """Test creating a trap without specifying gradB (should default to 0)"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        assert trap.GetGradB() == 0.0

    # ==================== B0 Validation Tests ====================
    def test_negative_B0_raises_error(self):
        """Tests that negative values of B0 raise a ValueError"""
        with pytest.raises(ValueError, match="B0 must be positive"):
            HarmonicTrap(B0=-1.0, L0=0.2)

    def test_zero_B0_raises_error(self):
        """Tests that zero value of B0 raises a ValueError"""
        with pytest.raises(ValueError, match="B0 must be positive"):
            HarmonicTrap(B0=0.0, L0=0.2)

    def test_non_numeric_B0_raises_error(self):
        """Tests that non-numeric B0 raises a TypeError"""
        with pytest.raises(TypeError, match="B0 must be a number"):
            HarmonicTrap(B0="1.0", L0=0.2)

    def test_non_finite_B0_raises_error(self):
        """Tests that non-finite B0 (inf, nan) raises a ValueError"""
        with pytest.raises(ValueError, match="B0 must be finite"):
            HarmonicTrap(B0=np.inf, L0=0.2)
        with pytest.raises(ValueError, match="B0 must be finite"):
            HarmonicTrap(B0=np.nan, L0=0.2)

    # ==================== L0 Validation Tests ====================
    def test_negative_L0_raises_error(self):
        """Tests that negative values of L0 raise a ValueError"""
        with pytest.raises(ValueError, match="L0 must be positive"):
            HarmonicTrap(B0=1.0, L0=-0.2)

    def test_zero_L0_raises_error(self):
        """Tests that zero value of L0 raises a ValueError"""
        with pytest.raises(ValueError, match="L0 must be positive"):
            HarmonicTrap(B0=1.0, L0=0.0)

    def test_non_numeric_L0_raises_error(self):
        """Tests that non-numeric L0 raises a TypeError"""
        with pytest.raises(TypeError, match="L0 must be a number"):
            HarmonicTrap(B0=1.0, L0="0.2")

    def test_non_finite_L0_raises_error(self):
        """Tests that non-finite L0 (inf, nan) raises a ValueError"""
        with pytest.raises(ValueError, match="L0 must be finite"):
            HarmonicTrap(B0=1.0, L0=np.inf)
        with pytest.raises(ValueError, match="L0 must be finite"):
            HarmonicTrap(B0=1.0, L0=np.nan)

    # ==================== gradB Validation Tests ====================
    def test_non_numeric_gradB_raises_error(self):
        """Tests that non-numeric gradB raises a TypeError"""
        with pytest.raises(TypeError, match="Gradient must be a number"):
            HarmonicTrap(B0=1.0, L0=0.2, gradB="0.004")

    def test_non_finite_gradB_raises_error(self):
        """Tests that non-finite gradB raises a ValueError"""
        with pytest.raises(ValueError, match="Gradient must be finite"):
            HarmonicTrap(B0=1.0, L0=0.2, gradB=np.inf)

    def test_set_gradB(self):
        """Tests SetGradB method"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        trap.SetGradB(0.005)
        assert trap.GetGradB() == 0.005

    def test_set_negative_gradB(self):
        """Tests that negative gradB values are allowed (field can decrease)"""
        trap = HarmonicTrap(B0=1.0, L0=0.2, gradB=-0.004)
        assert trap.GetGradB() == -0.004

    # ==================== CalcZMax Tests ====================
    def test_calc_zmax_single_value(self):
        """Tests CalcZMax with a single pitch angle"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        pitch_angle = np.pi / 4  # 45 degrees
        zmax = trap.CalcZMax(pitch_angle)
        expected = 0.2 / np.tan(np.pi / 4)  # Should be 0.2 m
        assert np.isclose(zmax, expected)

    def test_calc_zmax_array(self):
        """Tests CalcZMax with an array of pitch angles"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        pitch_angles = np.array([np.pi/6, np.pi/4, np.pi/3])
        zmax = trap.CalcZMax(pitch_angles)
        expected = 0.2 / np.tan(pitch_angles)
        assert np.allclose(zmax, expected)

    def test_calc_zmax_near_zero_pitch_angle(self):
        """Tests CalcZMax with pitch angle near zero (should return inf)"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        zmax = trap.CalcZMax(1e-11)
        assert np.isinf(zmax)

    def test_calc_zmax_ninety_degrees(self):
        """Tests CalcZMax with pitch angle of π/2 (should return 0)"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        zmax = trap.CalcZMax(np.pi/2)
        assert np.isclose(zmax, 0.0)

    def test_calc_zmax_invalid_pitch_angle(self):
        """Tests that invalid pitch angles raise errors"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            trap.CalcZMax(0.0)
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            trap.CalcZMax(np.pi)
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            trap.CalcZMax(-0.1)

    # ==================== CalcOmegaAxial Tests ====================
    def test_calc_omega_axial_single_value(self):
        """Tests CalcOmegaAxial with single values"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = 1e8  # m/s
        pitch_angle = np.pi / 4
        omega_axial = trap.CalcOmegaAxial(v, pitch_angle)
        expected = v * np.sin(pitch_angle) / 0.2
        assert np.isclose(omega_axial, expected)

    def test_calc_omega_axial_array(self):
        """Tests CalcOmegaAxial with arrays"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = np.array([1e8, 1.5e8, 2e8])
        pitch_angle = np.array([np.pi/6, np.pi/4, np.pi/3])
        omega_axial = trap.CalcOmegaAxial(v, pitch_angle)
        expected = v * np.sin(pitch_angle) / 0.2
        assert np.allclose(omega_axial, expected)

    def test_calc_omega_axial_invalid_velocity(self):
        """Tests that invalid velocities raise errors"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        with pytest.raises(ValueError, match="Velocity must be positive"):
            trap.CalcOmegaAxial(-1e8, np.pi/4)
        with pytest.raises(ValueError, match="Velocity exceeds speed of light"):
            trap.CalcOmegaAxial(sc.c * 1.1, np.pi/4)

    # ==================== CalcOmega0 Tests ====================
    def test_calc_omega0_single_value(self):
        """Tests CalcOmega0 with single values"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = 1e8  # m/s
        pitch_angle = np.pi / 4
        omega0 = trap.CalcOmega0(v, pitch_angle)

        # Calculate expected value
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        zmax = trap.CalcZMax(pitch_angle)
        expected = sc.e * 1.0 / (sc.m_e * gamma) * (1 + zmax**2 / (2 * 0.2**2))

        assert np.isclose(omega0, expected)

    def test_calc_omega0_array(self):
        """Tests CalcOmega0 with arrays"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = np.array([1e8, 1.5e8])
        pitch_angle = np.array([np.pi/6, np.pi/3])
        omega0 = trap.CalcOmega0(v, pitch_angle)

        assert omega0.shape == (2,)
        assert np.all(omega0 > 0)

    def test_calc_omega0_relativistic(self):
        """Tests CalcOmega0 with relativistic velocity"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = 0.9 * sc.c  # 90% speed of light
        pitch_angle = np.pi / 4
        omega0 = trap.CalcOmega0(v, pitch_angle)

        # Should be positive and smaller than non-relativistic case due to gamma
        omega0_slow = trap.CalcOmega0(1e7, pitch_angle)
        assert omega0 > 0
        assert omega0 < omega0_slow

    # ==================== Calcq Tests ====================
    def test_calcq_single_value(self):
        """Tests Calcq with single values"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = 1e8
        pitch_angle = np.pi / 4
        q = trap.Calcq(v, pitch_angle)

        # Calculate expected value
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        zmax = trap.CalcZMax(pitch_angle)
        omega_axial = trap.CalcOmegaAxial(v, pitch_angle)
        expected = -sc.e * 1.0 * zmax**2 / (gamma * sc.m_e * 4 * 0.2**2 * omega_axial)

        assert np.isclose(q, expected)

    def test_calcq_array(self):
        """Tests Calcq with arrays"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = np.array([1e8, 1.5e8])
        pitch_angle = np.array([np.pi/6, np.pi/3])
        q = trap.Calcq(v, pitch_angle)

        assert q.shape == (2,)

    def test_calcq_ninety_degrees(self):
        """Tests Calcq at pitch angle of π/2"""
        trap = HarmonicTrap(B0=1.0, L0=0.2)
        v = 1e8
        pitch_angle = np.pi / 2
        q = trap.Calcq(v, pitch_angle)

        # At 90 degrees, zmax = 0, so q should be 0
        assert np.isclose(q, 0.0)


class TestBathtubTrap:
    """Tests for BathtubTrap"""

    # ==================== Constructor Tests ====================
    def test_valid_trap_creation(self):
        """Test creating a valid bathtub trap"""
        B0 = 1.0
        L0 = 0.2
        L1 = 0.5
        gradB = 4e-3
        trap = BathtubTrap(B0=B0, L0=L0, L1=L1, gradB=gradB)

        assert trap.GetB0() == B0
        assert trap.GetL0() == L0
        assert trap.GetL1() == L1
        assert trap.GetGradB() == gradB

    def test_valid_trap_creation_without_gradB(self):
        """Test creating a trap without specifying gradB (should default to 0)"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        assert trap.GetGradB() == 0.0

    # ==================== B0 Validation Tests ====================
    def test_negative_B0_raises_error(self):
        """Tests that negative values of B0 raise a ValueError"""
        with pytest.raises(ValueError, match="B0 must be positive"):
            BathtubTrap(B0=-1.0, L0=0.2, L1=0.5)

    def test_zero_B0_raises_error(self):
        """Tests that zero value of B0 raises a ValueError"""
        with pytest.raises(ValueError, match="B0 must be positive"):
            BathtubTrap(B0=0.0, L0=0.2, L1=0.5)

    def test_non_numeric_B0_raises_error(self):
        """Tests that non-numeric B0 raises a TypeError"""
        with pytest.raises(TypeError, match="B0 must be a number"):
            BathtubTrap(B0="1.0", L0=0.2, L1=0.5)

    def test_non_finite_B0_raises_error(self):
        """Tests that non-finite B0 raises a ValueError"""
        with pytest.raises(ValueError, match="B0 must be finite"):
            BathtubTrap(B0=np.inf, L0=0.2, L1=0.5)

    # ==================== L0 Validation Tests ====================
    def test_negative_L0_raises_error(self):
        """Tests that negative values of L0 raise a ValueError"""
        with pytest.raises(ValueError, match="L0 must be positive"):
            BathtubTrap(B0=1.0, L0=-0.2, L1=0.5)

    def test_zero_L0_raises_error(self):
        """Tests that zero value of L0 raises a ValueError"""
        with pytest.raises(ValueError, match="L0 must be positive"):
            BathtubTrap(B0=1.0, L0=0.0, L1=0.5)

    def test_non_numeric_L0_raises_error(self):
        """Tests that non-numeric L0 raises a TypeError"""
        with pytest.raises(TypeError, match="L0 must be a number"):
            BathtubTrap(B0=1.0, L0="0.2", L1=0.5)

    def test_non_finite_L0_raises_error(self):
        """Tests that non-finite L0 raises a ValueError"""
        with pytest.raises(ValueError, match="L0 must be finite"):
            BathtubTrap(B0=1.0, L0=np.inf, L1=0.5)

    # ==================== L1 Validation Tests ====================
    def test_negative_L1_raises_error(self):
        """Tests that negative values of L1 raise a ValueError"""
        with pytest.raises(ValueError, match="L1 must be positive"):
            BathtubTrap(B0=1.0, L0=0.2, L1=-0.5)

    def test_zero_L1_raises_error(self):
        """Tests that zero value of L1 raises a ValueError"""
        with pytest.raises(ValueError, match="L1 must be positive"):
            BathtubTrap(B0=1.0, L0=0.2, L1=0.0)

    def test_non_numeric_L1_raises_error(self):
        """Tests that non-numeric L1 raises a TypeError"""
        with pytest.raises(TypeError, match="L1 must be a number"):
            BathtubTrap(B0=1.0, L0=0.2, L1="0.5")

    def test_non_finite_L1_raises_error(self):
        """Tests that non-finite L1 raises a ValueError"""
        with pytest.raises(ValueError, match="L1 must be finite"):
            BathtubTrap(B0=1.0, L0=0.2, L1=np.inf)

    # ==================== gradB Validation Tests ====================
    def test_non_numeric_gradB_raises_error(self):
        """Tests that non-numeric gradB raises a TypeError"""
        with pytest.raises(TypeError, match="Gradient must be a number"):
            BathtubTrap(B0=1.0, L0=0.2, L1=0.5, gradB="0.004")

    def test_non_finite_gradB_raises_error(self):
        """Tests that non-finite gradB raises a ValueError"""
        with pytest.raises(ValueError, match="Gradient must be finite"):
            BathtubTrap(B0=1.0, L0=0.2, L1=0.5, gradB=np.inf)

    def test_set_gradB(self):
        """Tests SetGradB method"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        trap.SetGradB(0.005)
        assert trap.GetGradB() == 0.005

    def test_set_negative_gradB(self):
        """Tests that negative gradB values are allowed"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5, gradB=-0.004)
        assert trap.GetGradB() == -0.004

    # ==================== CalcZMax Tests ====================
    def test_calc_zmax_single_value(self):
        """Tests CalcZMax with a single pitch angle"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        pitch_angle = np.pi / 4
        zmax = trap.CalcZMax(pitch_angle)
        expected = 0.2 / np.tan(np.pi / 4)
        assert np.isclose(zmax, expected)

    def test_calc_zmax_array(self):
        """Tests CalcZMax with an array of pitch angles"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        pitch_angles = np.array([np.pi/6, np.pi/4, np.pi/3])
        zmax = trap.CalcZMax(pitch_angles)
        expected = 0.2 / np.tan(pitch_angles)
        assert np.allclose(zmax, expected)

    def test_calc_zmax_near_zero_pitch_angle(self):
        """Tests CalcZMax with pitch angle near zero (should return inf)"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        zmax = trap.CalcZMax(1e-11)
        assert np.isinf(zmax)

    def test_calc_zmax_ninety_degrees(self):
        """Tests CalcZMax with pitch angle of π/2 (should return 0)"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        zmax = trap.CalcZMax(np.pi/2)
        assert np.isclose(zmax, 0.0)

    # ==================== CalcOmegaAxial Tests ====================
    def test_calc_omega_axial_single_value(self):
        """Tests CalcOmegaAxial with single values"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = 1e8
        pitch_angle = np.pi / 4
        omega_axial = trap.CalcOmegaAxial(v, pitch_angle)

        wa = v * np.sin(pitch_angle) / 0.2
        expected = wa / (1 + 0.5 * np.tan(pitch_angle) / (0.2 * np.pi))
        assert np.isclose(omega_axial, expected)

    def test_calc_omega_axial_array(self):
        """Tests CalcOmegaAxial with arrays"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = np.array([1e8, 1.5e8, 2e8])
        pitch_angle = np.array([np.pi/6, np.pi/4, np.pi/3])
        omega_axial = trap.CalcOmegaAxial(v, pitch_angle)

        assert omega_axial.shape == (3,)
        assert np.all(omega_axial > 0)

    def test_calc_omega_axial_invalid_velocity(self):
        """Tests that invalid velocities raise errors"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        with pytest.raises(ValueError, match="Velocity must be positive"):
            trap.CalcOmegaAxial(-1e8, np.pi/4)
        with pytest.raises(ValueError, match="Velocity exceeds speed of light"):
            trap.CalcOmegaAxial(sc.c * 1.1, np.pi/4)

    # ==================== CalcOmega0 Tests ====================
    def test_calc_omega0_single_value(self):
        """Tests CalcOmega0 with single values"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = 1e8
        pitch_angle = np.pi / 4
        omega0 = trap.CalcOmega0(v, pitch_angle)

        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        prefac = sc.e * 1.0 / (sc.m_e * gamma)
        zmax = trap.CalcZMax(pitch_angle)
        expected = prefac * (1 + (zmax**2 / (2 * 0.2**2)) * (1 + 0.5 * np.tan(pitch_angle) / (np.pi * 0.2))**(-1))

        assert np.isclose(omega0, expected)

    def test_calc_omega0_array(self):
        """Tests CalcOmega0 with arrays"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = np.array([1e8, 1.5e8])
        pitch_angle = np.array([np.pi/6, np.pi/3])
        omega0 = trap.CalcOmega0(v, pitch_angle)

        assert omega0.shape == (2,)
        assert np.all(omega0 > 0)

    # ==================== CalcT1 Tests ====================
    def test_calc_t1_single_value(self):
        """Tests CalcT1 with single values"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = 1e8
        pitch_angle = np.pi / 4
        t1 = trap.CalcT1(v, pitch_angle)

        expected = 0.5 / (v * np.cos(pitch_angle))
        assert np.isclose(t1, expected)

    def test_calc_t1_array(self):
        """Tests CalcT1 with arrays"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = np.array([1e8, 1.5e8, 2e8])
        pitch_angle = np.array([np.pi/6, np.pi/4, np.pi/3])
        t1 = trap.CalcT1(v, pitch_angle)

        expected = 0.5 / (v * np.cos(pitch_angle))
        assert np.allclose(t1, expected)

    def test_calc_t1_ninety_degrees(self):
        """Tests CalcT1 at pitch angle of π/2 (should give very large value)"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        v = 1e8
        pitch_angle = np.pi / 2
        t1 = trap.CalcT1(v, pitch_angle)

        # At 90 degrees, cos(π/2) ≈ 0, so t1 should be very large
        # Due to floating point precision, cos(π/2) is not exactly 0
        assert t1 > 1e7  # Should be a very large value

    def test_calc_t1_invalid_velocity(self):
        """Tests that invalid velocities raise errors"""
        trap = BathtubTrap(B0=1.0, L0=0.2, L1=0.5)
        with pytest.raises(ValueError, match="Velocity must be positive"):
            trap.CalcT1(-1e8, np.pi/4)
        with pytest.raises(ValueError, match="Velocity exceeds speed of light"):
            trap.CalcT1(sc.c * 1.1, np.pi/4)

    # ==================== Comparison Tests ====================
    def test_bathtub_vs_harmonic_omega_axial(self):
        """Tests that BathtubTrap reduces axial frequency compared to HarmonicTrap"""
        # For same B0 and L0, bathtub should have lower axial frequency due to flat region
        B0, L0 = 1.0, 0.2
        harmonic = HarmonicTrap(B0=B0, L0=L0)
        bathtub = BathtubTrap(B0=B0, L0=L0, L1=0.5)

        v = 1e8
        pitch_angle = np.pi / 4

        omega_harmonic = harmonic.CalcOmegaAxial(v, pitch_angle)
        omega_bathtub = bathtub.CalcOmegaAxial(v, pitch_angle)

        # Bathtub should have lower frequency due to longer path
        assert omega_bathtub < omega_harmonic
