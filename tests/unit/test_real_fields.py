"""
Unit tests for RealFields classes (CoilField, BathtubField, HarmonicField)
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure.RealFields import CoilField, BathtubField, HarmonicField
from CRESSignalStructure.Particle import Particle


class TestCoilFieldConstruction:
    """Tests for CoilField constructor"""

    def test_valid_coil_creation(self):
        """Test creating a valid coil field"""
        radius = 0.1  # 10 cm
        current = 100.0  # 100 A
        z_pos = 0.5  # 50 cm

        coil = CoilField(radius, current, z_pos)

        assert coil.radius == radius
        assert coil.current == current
        assert coil.z == z_pos

    def test_coil_with_default_position(self):
        """Test creating coil with default Z=0"""
        coil = CoilField(0.1, 100.0)
        assert coil.z == 0.0


class TestCoilFieldOnAxis:
    """Tests for on-axis magnetic field"""

    def test_field_at_coil_center(self):
        """Test field at center of coil (z=0, rho=0)"""
        radius = 0.1
        current = 100.0
        coil = CoilField(radius, current, Z=0.0)

        # At center, field should be B = μ₀*I/(2*R)
        expected_field = sc.mu_0 * current / (2 * radius)

        bx, by, bz = coil.evaluate_field(0.0, 0.0, 0.0)

        assert bx == 0.0
        assert by == 0.0
        assert np.isclose(bz, expected_field, rtol=1e-10)

    def test_on_axis_field_formula(self):
        """Test on-axis field matches analytical formula"""
        radius = 0.1
        current = 100.0
        z_position = 0.05  # 5 cm from coil
        coil = CoilField(radius, current, Z=0.0)

        # On-axis formula: B = (μ₀*I*R²)/(2*(R²+z²)^(3/2))
        expected_bz = (sc.mu_0 * current * radius**2 /
                       (2 * (radius**2 + z_position**2)**1.5))

        bx, by, bz = coil.evaluate_field(0.0, 0.0, z_position)

        assert bx == 0.0
        assert by == 0.0
        assert np.isclose(bz, expected_bz, rtol=1e-10)

    def test_field_decay_with_distance(self):
        """Test that field decays with distance from coil"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # Measure field at increasing distances
        z_positions = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        fields = []

        for z in z_positions:
            _, _, bz = coil.evaluate_field(0.0, 0.0, z)
            fields.append(np.abs(bz))

        fields = np.array(fields)

        # Field should monotonically decrease with distance
        assert np.all(np.diff(fields) < 0)

    def test_field_symmetry_about_z_axis(self):
        """Test that on-axis field has no x or y components"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        z_positions = np.linspace(-0.5, 0.5, 10)

        for z in z_positions:
            bx, by, bz = coil.evaluate_field(0.0, 0.0, z)
            assert bx == 0.0
            assert by == 0.0
            assert bz != 0.0  # Should have z-component


class TestCoilFieldOffAxis:
    """Tests for off-axis magnetic field"""

    def test_off_axis_field_nonzero(self):
        """Test that off-axis field has non-zero components"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # Off-axis point
        x, y, z = 0.05, 0.0, 0.1

        bx, by, bz = coil.evaluate_field(x, y, z)

        # Should have both radial and axial components
        assert bx != 0.0
        assert bz != 0.0

    def test_azimuthal_symmetry(self):
        """Test azimuthal symmetry of field"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # Points at same radial distance but different angles
        rho = 0.05
        z = 0.1

        # Point at phi=0
        x1, y1 = rho, 0.0
        bx1, by1, bz1 = coil.evaluate_field(x1, y1, z)
        b_rho1 = np.sqrt(bx1**2 + by1**2)

        # Point at phi=π/2
        x2, y2 = 0.0, rho
        bx2, by2, bz2 = coil.evaluate_field(x2, y2, z)
        b_rho2 = np.sqrt(bx2**2 + by2**2)

        # Radial and z components should be equal
        assert np.isclose(b_rho1, b_rho2, rtol=1e-10)
        assert np.isclose(bz1, bz2, rtol=1e-10)

    def test_field_direction_off_axis(self):
        """Test that radial field points in correct direction"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # Point off-axis
        x, y, z = 0.05, 0.0, 0.1

        bx, by, bz = coil.evaluate_field(x, y, z)

        # Radial component should point away from axis (positive x direction)
        # when x > 0 and z > 0 (above the coil)
        assert bx * x > 0  # Same sign as x

    def test_field_in_plane_of_coil(self):
        """Test field in the plane of the coil (z=0)"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # Point in plane of coil
        x, y, z = 0.05, 0.0, 0.0

        bx, by, bz = coil.evaluate_field(x, y, z)

        # In plane of coil, field is purely axial at any radial position
        # by symmetry (no radial component in the z=0 plane)
        assert bx == 0.0
        assert by == 0.0
        # z-component should be non-zero
        assert bz != 0.0


class TestCoilFieldArrayInputs:
    """Tests for array inputs to field evaluation"""

    def test_array_positions(self):
        """Test field evaluation with array inputs"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        x = np.array([0.0, 0.05, 0.1])
        y = np.array([0.0, 0.0, 0.0])
        z = np.array([0.0, 0.1, 0.2])

        bx, by, bz = coil.evaluate_field(x, y, z)

        assert bx.shape == x.shape
        assert by.shape == y.shape
        assert bz.shape == z.shape
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))

    def test_scalar_returns_scalar(self):
        """Test that scalar inputs return scalar outputs"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        bx, by, bz = coil.evaluate_field(0.05, 0.0, 0.1)

        assert isinstance(bx, (float, np.floating))
        assert isinstance(by, (float, np.floating))
        assert isinstance(bz, (float, np.floating))


class TestCoilFieldMagnitude:
    """Tests for field magnitude calculations"""

    def test_field_magnitude_on_axis(self):
        """Test field magnitude calculation on axis"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # On axis, magnitude should equal |Bz|
        z = 0.05
        _, _, bz = coil.evaluate_field(0.0, 0.0, z)
        magnitude = coil.evaluate_field_magnitude(0.0, 0.0, z)

        assert np.isclose(magnitude, np.abs(bz), rtol=1e-10)

    def test_field_magnitude_off_axis(self):
        """Test field magnitude calculation off axis"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        x, y, z = 0.05, 0.03, 0.1
        bx, by, bz = coil.evaluate_field(x, y, z)
        magnitude = coil.evaluate_field_magnitude(x, y, z)

        expected = np.sqrt(bx**2 + by**2 + bz**2)
        assert np.isclose(magnitude, expected, rtol=1e-10)

    def test_magnitude_always_positive(self):
        """Test that magnitude is always positive"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        positions = [
            (0.0, 0.0, 0.0),
            (0.05, 0.0, 0.1),
            (0.0, 0.08, -0.2),
            (-0.03, -0.04, 0.15)
        ]

        for x, y, z in positions:
            magnitude = coil.evaluate_field_magnitude(x, y, z)
            assert magnitude >= 0


class TestCoilFieldGradient:
    """Tests for field gradient calculations"""

    def test_gradient_on_axis(self):
        """Test gradient calculation on axis"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # On axis at z=0, gradient should be zero by symmetry
        gradient = coil.evaluate_field_gradient(0.0, 0.0)
        assert np.isclose(gradient, 0.0, atol=1e-6)

    def test_gradient_off_axis(self):
        """Test gradient calculation off axis"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        rho = 0.05
        z = 0.1
        gradient = coil.evaluate_field_gradient(rho, z)

        # Gradient should be finite
        assert np.isfinite(gradient)

    def test_gradient_sign_away_from_coil(self):
        """Test gradient sign makes physical sense away from coil"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        # Far from coil (z >> R), field should decrease radially
        rho = 0.05
        z = 0.5  # Far from coil
        gradient = coil.evaluate_field_gradient(rho, z)

        # Field decreases moving away from axis when far from coil
        assert gradient < 0


class TestBathtubFieldConstruction:
    """Tests for BathtubField constructor"""

    def test_valid_bathtub_creation(self):
        """Test creating a valid bathtub field"""
        radius = 0.1
        current = 100.0
        z1 = -0.2
        z2 = 0.2
        background = np.array([0.0, 0.0, 0.1])

        field = BathtubField(radius, current, z1, z2, background)

        assert field.coil1.radius == radius
        assert field.coil1.current == current
        assert field.coil1.z == z1
        assert field.coil2.z == z2
        assert np.array_equal(field.background, background)

    def test_bathtub_with_default_background(self):
        """Test creating bathtub with default zero background"""
        field = BathtubField(0.1, 100.0, -0.2, 0.2)
        assert np.array_equal(field.background, np.zeros(3))


class TestBathtubFieldEvaluation:
    """Tests for BathtubField field evaluation"""

    def test_field_at_center(self):
        """Test field at center between coils"""
        radius = 0.1
        current = 100.0
        z1 = -0.2
        z2 = 0.2
        field = BathtubField(radius, current, z1, z2)

        # At center (0, 0, 0), fields from both coils should add
        bx, by, bz = field.evaluate_field(0.0, 0.0, 0.0)

        assert bx == 0.0
        assert by == 0.0
        assert bz > 0  # Both coils contribute

    def test_field_superposition(self):
        """Test that bathtub field is sum of individual coils"""
        radius = 0.1
        current = 100.0
        z1 = -0.2
        z2 = 0.2
        bathtub = BathtubField(radius, current, z1, z2)

        # Test point
        x, y, z = 0.05, 0.0, 0.1

        # Get bathtub field
        bx_total, by_total, bz_total = bathtub.evaluate_field(x, y, z)

        # Get individual coil fields
        bx1, by1, bz1 = bathtub.coil1.evaluate_field(x, y, z)
        bx2, by2, bz2 = bathtub.coil2.evaluate_field(x, y, z)

        # Should be equal to sum
        assert np.isclose(bx_total, bx1 + bx2, rtol=1e-10)
        assert np.isclose(by_total, by1 + by2, rtol=1e-10)
        assert np.isclose(bz_total, bz1 + bz2, rtol=1e-10)

    def test_background_field_addition(self):
        """Test that background field is correctly added"""
        radius = 0.1
        current = 100.0
        z1 = -0.2
        z2 = 0.2
        background = np.array([0.01, 0.02, 0.5])

        field = BathtubField(radius, current, z1, z2, background)

        # Test at a point
        x, y, z = 0.0, 0.0, 0.0

        bx, by, bz = field.evaluate_field(x, y, z)

        # Background should be added
        # Create field without background to compare
        field_no_bg = BathtubField(radius, current, z1, z2, np.zeros(3))
        bx0, by0, bz0 = field_no_bg.evaluate_field(x, y, z)

        assert np.isclose(bx, bx0 + background[0], rtol=1e-10)
        assert np.isclose(by, by0 + background[1], rtol=1e-10)
        assert np.isclose(bz, bz0 + background[2], rtol=1e-10)

    def test_symmetry_about_midplane(self):
        """Test symmetry about z=0 for symmetric coil placement"""
        radius = 0.1
        current = 100.0
        z_offset = 0.2
        field = BathtubField(radius, current, -z_offset, z_offset)

        # Points symmetric about z=0
        z1 = 0.1
        z2 = -0.1
        x, y = 0.05, 0.0

        _, _, bz_pos = field.evaluate_field(x, y, z1)
        _, _, bz_neg = field.evaluate_field(x, y, z2)

        # z-component should be symmetric
        assert np.isclose(bz_pos, bz_neg, rtol=1e-10)


class TestBathtubFieldZMax:
    """Tests for BathtubField CalcZMax"""

    def test_bathtub_zmax(self):
        """Test CalcZMax for bathtub field"""
        field = BathtubField(0.1, 100.0, -0.2, 0.2)

        ke = 18600.0
        pitch_angle = 89.0 * np.pi / 180.0
        particle = Particle(ke=ke, startPos=np.zeros(3), pitchAngle=pitch_angle)

        zmax = field.CalcZMax(particle)

        # Should be positive and less than coil separation
        assert zmax > 0
        assert zmax < 0.2
        assert np.isfinite(zmax)

    def test_zmax_increases_with_smaller_pitch_angle(self):
        """Test that zmax increases as pitch angle decreases from 90°"""
        field = BathtubField(0.1, 100.0, -0.2, 0.2)

        ke = 18600.0
        pitch_angles = [89.5, 89.0, 88.0, 87.0]  # Degrees (decreasing from 90°)

        zmaxs = []
        for angle_deg in pitch_angles:
            angle_rad = angle_deg * np.pi / 180.0
            particle = Particle(ke=ke, startPos=np.zeros(3), pitchAngle=angle_rad)
            zmaxs.append(field.CalcZMax(particle))

        # Smaller pitch angle -> more axial velocity -> larger zmax
        # (less perpendicular velocity means smaller magnetic moment, weaker trapping)
        zmaxs = np.array(zmaxs)
        assert np.all(np.diff(zmaxs) > 0)


class TestHarmonicFieldConstruction:
    """Tests for HarmonicField constructor"""

    def test_valid_harmonic_creation(self):
        """Test creating a valid harmonic field"""
        radius = 0.1
        current = 100.0
        background = 1.0  # 1 Tesla

        field = HarmonicField(radius, current, background)

        assert field.coil.radius == radius
        assert field.coil.current == current
        assert field.coil.z == 0.0
        assert np.isclose(field.background[2], -background)


class TestHarmonicFieldEvaluation:
    """Tests for HarmonicField field evaluation"""

    def test_field_at_center(self):
        """Test field at center of harmonic trap"""
        radius = 0.1
        current = 100.0
        background = 1.0
        field = HarmonicField(radius, current, background)

        bx, by, bz = field.evaluate_field(0.0, 0.0, 0.0)

        # Should be sum of coil field and background
        coil_field = sc.mu_0 * current / (2 * radius)
        expected_bz = coil_field - background

        assert bx == 0.0
        assert by == 0.0
        assert np.isclose(bz, expected_bz, rtol=1e-10)

    def test_field_equals_coil_plus_background(self):
        """Test that field is correctly sum of coil and background"""
        radius = 0.1
        current = 100.0
        background = 0.5
        field = HarmonicField(radius, current, background)

        x, y, z = 0.05, 0.0, 0.1

        bx, by, bz = field.evaluate_field(x, y, z)

        # Get coil field alone
        bx_coil, by_coil, bz_coil = field.coil.evaluate_field(x, y, z)

        # Check superposition
        assert np.isclose(bx, bx_coil, rtol=1e-10)
        assert np.isclose(by, by_coil, rtol=1e-10)
        assert np.isclose(bz, bz_coil - background, rtol=1e-10)

    def test_background_opposite_to_coil(self):
        """Test that background opposes coil field (creates trap)"""
        radius = 0.1
        current = 100.0
        background = 0.5
        field = HarmonicField(radius, current, background)

        # At center, coil points up, background points down
        _, _, bz_center = field.evaluate_field(0.0, 0.0, 0.0)

        # Far from coil, field should be dominated by background (negative)
        _, _, bz_far = field.evaluate_field(0.0, 0.0, 1.0)

        # Background is negative, so far field should be negative
        assert bz_far < 0


class TestHarmonicFieldZMax:
    """Tests for HarmonicField CalcZMax"""

    def test_harmonic_zmax(self):
        """Test CalcZMax for harmonic field"""
        field = HarmonicField(0.1, 100.0, 1.0)

        ke = 18600.0
        pitch_angle = 89.0 * np.pi / 180.0
        particle = Particle(ke=ke, startPos=np.zeros(3), pitchAngle=pitch_angle)

        zmax = field.CalcZMax(particle)

        assert zmax > 0
        assert np.isfinite(zmax)


class TestPhysicsConsistency:
    """Tests for overall physics consistency across field types"""

    def test_field_magnitude_positive(self):
        """Test that field magnitude is always positive"""
        coil = CoilField(0.1, 100.0, Z=0.0)
        bathtub = BathtubField(0.1, 100.0, -0.2, 0.2)
        harmonic = HarmonicField(0.1, 100.0, 1.0)

        positions = [(0.0, 0.0, 0.0), (0.05, 0.03, 0.1), (0.0, 0.08, -0.15)]

        for field in [coil, bathtub, harmonic]:
            for x, y, z in positions:
                mag = field.evaluate_field_magnitude(x, y, z)
                assert mag >= 0

    def test_field_components_consistent(self):
        """Test that field components satisfy magnitude relationship"""
        coil = CoilField(0.1, 100.0, Z=0.0)

        x, y, z = 0.05, 0.03, 0.1
        bx, by, bz = coil.evaluate_field(x, y, z)
        magnitude = coil.evaluate_field_magnitude(x, y, z)

        calculated_mag = np.sqrt(bx**2 + by**2 + bz**2)
        assert np.isclose(magnitude, calculated_mag, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_zero_current_gives_zero_field(self):
        """Test that zero current produces zero field"""
        coil = CoilField(0.1, 0.0, Z=0.0)

        bx, by, bz = coil.evaluate_field(0.05, 0.03, 0.1)

        assert bx == 0.0
        assert by == 0.0
        assert bz == 0.0

    def test_very_small_radius_coil(self):
        """Test coil with very small radius"""
        coil = CoilField(0.001, 100.0, Z=0.0)

        # Should still produce valid field
        mag = coil.evaluate_field_magnitude(0.0, 0.0, 0.0)
        assert np.isfinite(mag)
        assert mag > 0

    def test_large_radius_coil(self):
        """Test coil with large radius"""
        coil = CoilField(1.0, 100.0, Z=0.0)

        # Should produce valid field
        mag = coil.evaluate_field_magnitude(0.0, 0.0, 0.0)
        assert np.isfinite(mag)
        assert mag > 0

    def test_coil_with_negative_current(self):
        """Test that negative current reverses field direction"""
        coil_pos = CoilField(0.1, 100.0, Z=0.0)
        coil_neg = CoilField(0.1, -100.0, Z=0.0)

        bx_pos, by_pos, bz_pos = coil_pos.evaluate_field(0.0, 0.0, 0.1)
        bx_neg, by_neg, bz_neg = coil_neg.evaluate_field(0.0, 0.0, 0.1)

        # Fields should be opposite
        assert np.isclose(bx_pos, -bx_neg)
        assert np.isclose(by_pos, -by_neg)
        assert np.isclose(bz_pos, -bz_neg)