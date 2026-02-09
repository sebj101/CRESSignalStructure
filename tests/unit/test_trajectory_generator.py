"""
Unit tests for TrajectoryGenerator and Trajectory classes

Tests focus on calculation correctness, physical consistency, and edge case handling.
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure.TrajectoryGenerator import Trajectory, TrajectoryGenerator
from CRESSignalStructure.RealFields import HarmonicField, BathtubField
from CRESSignalStructure.Particle import Particle


class TestTrajectoryContainer:
    """Tests for the Trajectory container class"""

    @pytest.fixture
    def simple_trajectory_data(self):
        """Create simple trajectory data for testing"""
        n_points = 100
        time = np.linspace(0, 1e-6, n_points)
        position = np.random.randn(n_points, 3) * 0.01
        velocity = np.random.randn(n_points, 3) * 1e5
        acceleration = np.random.randn(n_points, 3) * 1e10

        # Create simple field and particle
        field = HarmonicField(radius=0.05, current=100.0, background=1.0)
        particle = Particle(ke=18600.0, startPos=np.zeros(3))

        return time, position, velocity, acceleration, field, particle

    def test_trajectory_construction(self, simple_trajectory_data):
        """Test basic trajectory construction"""
        t, pos, vel, acc, field, particle = simple_trajectory_data

        traj = Trajectory(t, pos, vel, acc, field, particle)

        assert np.array_equal(traj.time, t)
        assert np.array_equal(traj.position, pos)
        assert np.array_equal(traj.velocity, vel)
        assert np.array_equal(traj.acceleration, acc)
        assert traj.field is field
        assert traj.particle is particle

    def test_trajectory_shape_validation(self, simple_trajectory_data):
        """Test that trajectory validates array shapes"""
        t, pos, vel, acc, field, particle = simple_trajectory_data

        # Wrong position shape
        with pytest.raises(ValueError, match="Position must have shape"):
            Trajectory(t, pos[:, :2], vel, acc, field, particle)

        # Wrong velocity shape
        with pytest.raises(ValueError, match="Velocity must have shape"):
            Trajectory(t, pos, vel[:-1], acc, field, particle)

        # Wrong acceleration shape
        with pytest.raises(ValueError, match="Acceleration must have shape"):
            Trajectory(t, pos, vel, np.zeros((len(t), 2)), field, particle)

    def test_trajectory_time_must_be_1d(self, simple_trajectory_data):
        """Test that time array must be 1D"""
        t, pos, vel, acc, field, particle = simple_trajectory_data

        t_2d = t.reshape(-1, 1)
        with pytest.raises(ValueError, match="Time must be a 1D array"):
            Trajectory(t_2d, pos, vel, acc, field, particle)

    def test_get_sample_rate(self, simple_trajectory_data):
        """Test sample rate calculation"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        expected_rate = 1.0 / (t[1] - t[0])
        calculated_rate = traj.get_sample_rate()

        np.testing.assert_almost_equal(
            calculated_rate, expected_rate, decimal=3)

    def test_get_sample_rate_requires_multiple_points(self):
        """Test that get_sample_rate needs at least 2 points"""
        field = HarmonicField(radius=0.05, current=100.0, background=1.0)
        particle = Particle(ke=18600.0, startPos=np.zeros(3))

        # Single point trajectory
        t = np.array([0.0])
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        acc = np.zeros((1, 3))

        traj = Trajectory(t, pos, vel, acc, field, particle)

        with pytest.raises(ValueError, match="Need at least 2 time points"):
            traj.get_sample_rate()

    def test_get_relative_position(self, simple_trajectory_data):
        """Test relative position calculation"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        reference = np.array([0.01, 0.02, 0.03])
        rel_pos = traj.get_relative_position(reference)

        expected = pos - reference
        np.testing.assert_array_almost_equal(rel_pos, expected)

    def test_get_relative_position_validates_reference(self, simple_trajectory_data):
        """Test that get_relative_position validates reference shape"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        with pytest.raises(ValueError, match="Reference position must be a 3-vector"):
            traj.get_relative_position(np.array([1.0, 2.0]))

    def test_get_beta(self, simple_trajectory_data):
        """Test normalized velocity calculation"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        beta = traj.get_beta()
        expected = vel / sc.c

        np.testing.assert_array_almost_equal(beta, expected)

    def test_get_beta_dot(self, simple_trajectory_data):
        """Test normalized acceleration calculation"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        beta_dot = traj.get_beta_dot()
        expected = acc / sc.c

        np.testing.assert_array_almost_equal(beta_dot, expected)

    def test_get_duration(self, simple_trajectory_data):
        """Test trajectory duration calculation"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        duration = traj.get_duration()
        expected = t[-1] - t[0]

        assert duration == expected

    def test_get_n_points(self, simple_trajectory_data):
        """Test number of points getter"""
        t, pos, vel, acc, field, particle = simple_trajectory_data
        traj = Trajectory(t, pos, vel, acc, field, particle)

        assert traj.get_n_points() == len(t)


class TestTrajectoryGeneratorBasics:
    """Tests for TrajectoryGenerator basic functionality"""

    @pytest.fixture
    def harmonic_field_setup(self):
        """Create a simple harmonic field setup"""
        # Harmonic field: single coil + background
        coil_radius = 0.05  # 5 cm
        trap_depth = 4e-3  # 4 mT
        coil_current = 2 * trap_depth * coil_radius / sc.mu_0
        background_field = 1.0  # 1 T

        field = HarmonicField(
            radius=coil_radius,
            current=coil_current,
            background=background_field
        )

        # Standard electron parameters
        ke = 18.6e3  # 18.6 keV
        pitch_angle = 88.0 * np.pi / 180.0  # 88 degrees
        initial_pos = np.array([0.001, 0.0, 0.0])  # 1 mm off-axis

        particle = Particle(
            ke=ke,
            startPos=initial_pos,
            pitchAngle=pitch_angle
        )

        return field, particle

    def test_generator_construction(self, harmonic_field_setup):
        """Test basic TrajectoryGenerator construction"""
        field, particle = harmonic_field_setup

        gen = TrajectoryGenerator(field, particle)

        assert gen.field is field
        assert gen.particle is particle

    def test_generate_returns_trajectory(self, harmonic_field_setup):
        """Test that generate returns a Trajectory object"""
        field, particle = harmonic_field_setup
        gen = TrajectoryGenerator(field, particle)

        traj = gen.generate(sample_rate=5e9, t_max=1e-6)

        assert isinstance(traj, Trajectory)
        assert traj.field is field
        assert traj.particle is particle

    def test_generate_correct_number_of_points(self, harmonic_field_setup):
        """Test that generated trajectory has correct number of points"""
        field, particle = harmonic_field_setup
        gen = TrajectoryGenerator(field, particle)

        sample_rate = 1e9  # 1 GHz
        t_max = 5e-6  # 5 microseconds

        traj = gen.generate(sample_rate=sample_rate, t_max=t_max)

        expected_points = int(np.round(t_max * sample_rate))
        assert traj.get_n_points() == expected_points

    def test_generate_correct_duration(self, harmonic_field_setup):
        """Test that generated trajectory has correct duration"""
        field, particle = harmonic_field_setup
        gen = TrajectoryGenerator(field, particle)

        t_max = 2e-6
        traj = gen.generate(sample_rate=1e9, t_max=t_max)

        np.testing.assert_almost_equal(traj.get_duration(), t_max, decimal=10)

    def test_generate_correct_sample_rate(self, harmonic_field_setup):
        """Test that generated trajectory has correct sample rate"""
        field, particle = harmonic_field_setup
        gen = TrajectoryGenerator(field, particle)

        sample_rate = 2.5e9
        traj = gen.generate(sample_rate=sample_rate, t_max=1e-6)

        calculated_rate = traj.get_sample_rate()
        # Allow 1% tolerance due to rounding in number of points
        np.testing.assert_allclose(calculated_rate, sample_rate, rtol=0.01)


class TestTrajectoryGeneratorValidation:
    """Tests for parameter validation in TrajectoryGenerator"""

    @pytest.fixture
    def basic_setup(self):
        """Basic field and particle for validation tests"""
        # Harmonic field: single coil + background
        coil_radius = 0.05  # 5 cm
        trap_depth = 4e-3  # 4 mT
        coil_current = 2 * trap_depth * coil_radius / sc.mu_0
        background_field = 1.0  # 1 T
        field = HarmonicField(
            radius=coil_radius,
            current=coil_current,
            background=background_field
        )
        particle = Particle(ke=18600.0, startPos=np.zeros(3))
        return field, particle

    def test_negative_sample_rate_raises_error(self, basic_setup):
        """Test that negative sample rate raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            gen.generate(sample_rate=-1e9, t_max=1e-6)

    def test_zero_sample_rate_raises_error(self, basic_setup):
        """Test that zero sample rate raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            gen.generate(sample_rate=0.0, t_max=1e-6)

    def test_negative_t_max_raises_error(self, basic_setup):
        """Test that negative t_max raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        with pytest.raises(ValueError, match="Maximum time must be positive"):
            gen.generate(sample_rate=1e9, t_max=-1e-6)

    def test_zero_t_max_raises_error(self, basic_setup):
        """Test that zero t_max raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        with pytest.raises(ValueError, match="Maximum time must be positive"):
            gen.generate(sample_rate=1e9, t_max=0.0)

    def test_infinite_sample_rate_raises_error(self, basic_setup):
        """Test that infinite sample rate raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        with pytest.raises(ValueError, match="Sample rate must be finite"):
            gen.generate(sample_rate=np.inf, t_max=1e-6)

    def test_infinite_t_max_raises_error(self, basic_setup):
        """Test that infinite t_max raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        with pytest.raises(ValueError, match="Maximum time must be finite"):
            gen.generate(sample_rate=1e9, t_max=np.inf)

    def test_too_few_points_raises_error(self, basic_setup):
        """Test that too few points raises ValueError"""
        field, particle = basic_setup
        gen = TrajectoryGenerator(field, particle)

        # Parameters that would give < 10 points
        with pytest.raises(ValueError, match="Duration too short"):
            gen.generate(sample_rate=1e6, t_max=1e-9)


class TestTrajectoryPhysics:
    """Tests for physical correctness of generated trajectories"""

    @pytest.fixture
    def standard_setup(self):
        """Standard trapped electron setup"""
        # Harmonic field: single coil + background
        coil_radius = 0.05  # 5 cm
        trap_depth = 4e-3  # 4 mT
        coil_current = 2 * trap_depth * coil_radius / sc.mu_0
        background_field = 1.0  # 1 T
        field = HarmonicField(
            radius=coil_radius,
            current=coil_current,
            background=background_field
        )
        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.001, 0.0, 0.0]),
            pitchAngle=89.0 * np.pi / 180.0
        )
        return field, particle

    def test_velocity_below_speed_of_light(self, standard_setup):
        """Test that velocity never exceeds speed of light"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        traj = gen.generate(sample_rate=5e9, t_max=1e-6)

        vel_mags = np.linalg.norm(traj.velocity, axis=1)
        assert np.all(vel_mags < sc.c)

    def test_position_starts_at_initial_position(self, standard_setup):
        """Test that trajectory starts at particle's initial position"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        traj = gen.generate(sample_rate=1e9, t_max=1e-6)

        initial_pos = traj.position[0]
        expected_pos = particle.GetPosition()

        # Should match closely (may have small numerical differences)
        np.testing.assert_allclose(initial_pos, expected_pos, atol=1e-6)

    def test_axial_motion_respects_mirror_force(self, standard_setup):
        """Test that particle bounces at expected axial position"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        traj = gen.generate(sample_rate=10e9, t_max=5e-6)

        # Get maximum axial position
        z_max = np.max(np.abs(traj.position[:, 2]))
        z_max_expected = field.CalcZMax(particle)

        # Should be close to the calculated maximum
        np.testing.assert_allclose(z_max, z_max_expected, rtol=0.1)

    def test_cyclotron_motion_frequency(self, standard_setup):
        """Test that cyclotron frequency matches expected value"""
        field, _ = standard_setup  # Get field from fixture

        # Use nearly perpendicular pitch angle to minimize axial motion
        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.001, 0.0, 0.0]),
            pitchAngle=89.999 * np.pi / 180.0
        )
        gen = TrajectoryGenerator(field, particle)

        # Use high sample rate to capture cyclotron motion
        sample_rate = 80e9
        traj = gen.generate(sample_rate=sample_rate, t_max=1e-7)

        # Calculate expected cyclotron frequency at starting position
        pos = particle.GetPosition()
        B0 = field.evaluate_field_magnitude(pos[0], pos[1], pos[2])
        expected_f_c = sc.e * B0 / (2 * np.pi * sc.m_e * particle.GetGamma())

        # FFT of x-velocity to find dominant frequency
        from scipy.fft import fft, fftfreq
        n = len(traj.velocity)
        fft_vals = fft(traj.velocity[:, 0])
        freqs = fftfreq(n, 1.0 / sample_rate)

        # Find peak frequency (positive frequencies only)
        pos_freqs = freqs[:n//2]
        pos_fft = np.abs(fft_vals[:n//2])
        peak_idx = np.argmax(pos_fft[1:]) + 1  # Skip DC component
        measured_f_c = pos_freqs[peak_idx]

        # Should match within 5% (accounting for field variation)
        np.testing.assert_allclose(measured_f_c, expected_f_c, rtol=0.05)

    def test_perpendicular_velocity_squared_conserved(self, standard_setup):
        """Test adiabatic invariant: v_perp^2 / B is approximately constant"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        traj = gen.generate(sample_rate=5e9, t_max=2e-6)

        # Calculate v_perp^2 at each point
        v_perp_squared = traj.velocity[:, 0]**2 + traj.velocity[:, 1]**2

        # Get B field at each point
        pos = traj.position
        B_vals = field.evaluate_field_magnitude(
            pos[:, 0], pos[:, 1], pos[:, 2])

        # Calculate adiabatic invariant
        invariant = v_perp_squared / B_vals

        # Should be approximately constant (allowing ~1% variation)
        mean_invariant = np.mean(invariant)
        std_invariant = np.std(invariant)

        assert std_invariant / mean_invariant < 0.01

    def test_no_radiation_energy_constant(self, standard_setup):
        """Test that energy remains constant when radiation is disabled"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        traj = gen.generate(sample_rate=500e6, t_max=1e-3,
                            include_radiation=False)

        # Calculate speed at different times
        speed_initial = np.linalg.norm(traj.velocity[0])
        speed_final = np.linalg.norm(traj.velocity[-1])

        # Speed should remain constant (within numerical precision)
        np.testing.assert_allclose(speed_initial, speed_final, rtol=1e-5)

    def test_radiation_decreases_speed(self, standard_setup):
        """Test that speed decreases when radiation is enabled"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        # Generate trajectory with radiation
        traj = gen.generate(sample_rate=500e6, t_max=5e-3,
                            include_radiation=True)

        # Speed should decrease
        speed_initial = np.linalg.norm(traj.velocity[0])
        speed_final = np.linalg.norm(traj.velocity[-1])

        assert speed_final < speed_initial

    def test_energy_loss_matches_power_times_time(self, standard_setup):
        """Test that energy loss equals Larmor power times time"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        # Calculate expected energy loss
        P_larmor = gen._calc_larmor_power()
        t_max = 5e-3

        expected_loss_joules = P_larmor * t_max
        expected_loss_eV = expected_loss_joules / sc.e

        # Generate trajectory
        traj = gen.generate(sample_rate=3e9, t_max=t_max,
                            include_radiation=True)

        # Calculate actual energy loss from _calc_energy_vs_time
        E_t = gen._calc_energy_vs_time(traj.time)
        actual_loss_eV = E_t[0] - E_t[-1]

        # Should match exactly (both use same calculation)
        np.testing.assert_allclose(actual_loss_eV, expected_loss_eV, rtol=1e-8)

    def test_radiation_flag_default_false(self, standard_setup):
        """Test that include_radiation defaults to False"""
        field, particle = standard_setup
        gen = TrajectoryGenerator(field, particle)

        # Generate trajectory without specifying radiation flag
        traj = gen.generate(sample_rate=500e6, t_max=1e-3)

        # Should behave as if radiation is off (constant speed)
        speed_initial = np.linalg.norm(traj.velocity[0])
        speed_final = np.linalg.norm(traj.velocity[-1])

        np.testing.assert_allclose(speed_initial, speed_final, rtol=1e-5)


class TestTrajectoryEdgeCases:
    """Tests for edge cases and special configurations"""

    def test_perpendicular_pitch_angle_no_axial_motion(self):
        """Test that 90 degree pitch angle gives no axial motion"""
        field = HarmonicField(radius=0.05, current=100.0, background=1.0)
        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.001, 0.0, 0.0]),
            pitchAngle=np.pi / 2  # Exactly 90 degrees
        )

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=5e9, t_max=1e-6)

        # z position should remain constant
        z_positions = traj.position[:, 2]
        z_variation = np.std(z_positions)

        # Should have negligible variation
        assert z_variation < 1e-10

    def test_on_axis_particle(self):
        """Test trajectory generation for particle starting on axis"""
        field = HarmonicField(radius=0.05, current=100.0, background=1.0)
        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.0, 0.0, 0.0]),  # On axis
            pitchAngle=89.0 * np.pi / 180.0
        )

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=5e9, t_max=1e-6)

        # Should stay on axis (x and y should remain ~0)
        x_positions = traj.position[:, 0]
        y_positions = traj.position[:, 1]

        assert np.max(np.abs(x_positions)) < 1e-10
        assert np.max(np.abs(y_positions)) < 1e-10

    def test_very_short_duration(self):
        """Test trajectory with very short duration (just above minimum)"""
        field = HarmonicField(radius=0.05, current=100.0, background=1.0)
        particle = Particle(ke=18.6e3, startPos=np.array([0.001, 0.0, 0.0]))

        gen = TrajectoryGenerator(field, particle)

        # Generate trajectory with exactly 10 points (minimum allowed)
        sample_rate = 1e9
        t_max = 10.0 / sample_rate

        traj = gen.generate(sample_rate=sample_rate, t_max=t_max)

        assert traj.get_n_points() >= 10

    def test_bathtub_field(self):
        """Test trajectory generation with bathtub field configuration"""
        # Bathtub field parameters
        R_coil = 0.03  # 3 cm
        trap_depth = 4e-3
        I_coil = 2 * trap_depth * R_coil / sc.mu_0
        trap_length = 0.2  # 20 cm

        field = BathtubField(
            R_coil, I_coil,
            -trap_length/2, trap_length/2,
            np.array([0., 0., 1.0])
        )

        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.001, 0.0, 0.0]),
            pitchAngle=89.0 * np.pi / 180.0
        )

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=5e9, t_max=2e-6)

        # Basic sanity checks
        assert traj.get_n_points() > 0
        assert np.all(np.isfinite(traj.position))
        assert np.all(np.isfinite(traj.velocity))
        assert np.all(np.isfinite(traj.acceleration))

    def test_high_energy_particle(self):
        """Test trajectory with relativistic particle"""
        R_coil = 0.03  # 3 cm
        trap_depth = 4e-3
        I_coil = 2 * trap_depth * R_coil / sc.mu_0
        field = HarmonicField(
            radius=R_coil,
            current=I_coil,
            background=1.0
        )

        # Use 100 keV electron (moderately relativistic)
        particle = Particle(
            ke=100e3,
            startPos=np.array([0.001, 0.0, 0.0]),
            pitchAngle=89.0 * np.pi / 180.0
        )

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=10e9, t_max=1e-6)

        # Check that relativistic effects are included
        gamma = particle.GetGamma()
        assert gamma > 1.1  # Should be noticeably relativistic

        # Velocity should still be below c
        vel_mags = np.linalg.norm(traj.velocity, axis=1)
        assert np.all(vel_mags < sc.c)


class TestTrajectoryAcceleration:
    """Tests specifically for acceleration calculations"""

    def test_acceleration_perpendicular_to_velocity_in_uniform_field(self):
        """Test that acceleration is perpendicular to velocity (Lorentz force)"""
        # Use uniform field (no gradients) by setting current to 0
        field = HarmonicField(radius=0.5, current=0.0, background=1.0)

        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.0001, 0.0, 0.0]),  # Very close to axis
            pitchAngle=np.pi / 2
        )

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=50e9, t_max=1e-7)

        # Calculate dot product of velocity and acceleration
        # Should be small (perpendicular)
        # Exclude endpoints where numerical derivatives are less accurate
        vel = traj.velocity[10:-10]
        acc = traj.acceleration[10:-10]

        vel_normalized = vel / np.linalg.norm(vel, axis=1, keepdims=True)
        acc_normalized = acc / np.linalg.norm(acc, axis=1, keepdims=True)

        dot_products = np.sum(vel_normalized * acc_normalized, axis=1)

        # Should be close to zero (allowing for numerical errors)
        assert np.max(np.abs(dot_products)) < 0.1

    def test_acceleration_has_correct_order_of_magnitude(self):
        """Test that acceleration magnitude is physically reasonable"""
        # Use uniform field for simplicity
        field = HarmonicField(radius=0.05, current=0.0, background=1.0)
        particle = Particle(
            ke=18.6e3,
            startPos=np.array([0.001, 0.0, 0.0]),
            pitchAngle=np.pi / 2  # 90 degrees - pure cyclotron motion
        )

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=50e9, t_max=1e-7)

        # Calculate expected centripetal acceleration for cyclotron motion
        # a_c = omega_c * v for perpendicular motion
        B0 = field.evaluate_field_magnitude(0.001, 0.0, 0.0)
        omega_c = sc.e * B0 / (particle.GetGamma() * particle.GetMass())
        v = particle.GetSpeed()
        expected_acc = omega_c * v  # Order of magnitude

        acc_mags = np.linalg.norm(traj.acceleration, axis=1)
        mean_acc = np.mean(acc_mags)

        # Should be within an order of magnitude
        assert mean_acc > expected_acc * 0.1
        assert mean_acc < expected_acc * 10.0

    def test_acceleration_is_finite(self):
        """Test that acceleration values are always finite"""
        field = HarmonicField(radius=0.05, current=100.0, background=1.0)
        particle = Particle(ke=18.6e3, startPos=np.array([0.001, 0.0, 0.0]))

        gen = TrajectoryGenerator(field, particle)
        traj = gen.generate(sample_rate=5e9, t_max=1e-6)

        assert np.all(np.isfinite(traj.acceleration))
