"""
TrajectoryGenerator.py

Classes for generating and storing electron trajectories in magnetic traps.
This module provides tools for calculating position, velocity, and acceleration
of trapped electrons including grad-B drift effects.
"""

import numpy as np
from numpy.typing import NDArray
import scipy.constants as sc
from scipy.integrate import cumulative_simpson
from scipy.interpolate import interp1d

from CRESSignalStructure.BaseField import BaseField
from CRESSignalStructure.Particle import Particle


class Trajectory:
    """
    Container for electron trajectory data

    This class stores the complete trajectory information including position,
    velocity, and acceleration as functions of time, along with references to
    the magnetic field and particle that generated the trajectory.

    Attributes
    ----------
    time : NDArray
        Time array in seconds, shape (N,)
    position : NDArray
        Position array in meters, shape (N, 3) with columns [x, y, z]
    velocity : NDArray
        Velocity array in m/s, shape (N, 3) with columns [vx, vy, vz]
    acceleration : NDArray
        Acceleration array in m/s^2, shape (N, 3) with columns [ax, ay, az]
    field : BaseField
        Magnetic field configuration used to generate trajectory
    particle : Particle
        Particle object with mass, charge, and initial conditions
    """

    def __init__(self, time: NDArray, position: NDArray,
                 velocity: NDArray, acceleration: NDArray,
                 field: BaseField, particle: Particle):
        """
        Initialize Trajectory object

        Parameters
        ----------
        time : NDArray
            Time array in seconds, shape (N,)
        position : NDArray
            Position array in meters, shape (N, 3)
        velocity : NDArray
            Velocity array in m/s, shape (N, 3)
        acceleration : NDArray
            Acceleration array in m/s^2, shape (N, 3)
        field : BaseField
            Magnetic field configuration
        particle : Particle
            Particle object

        Raises
        ------
        ValueError
            If array shapes are inconsistent
        """
        # Validate shapes
        if time.ndim != 1:
            raise ValueError("Time must be a 1D array")

        n_points = len(time)

        if position.shape != (n_points, 3):
            raise ValueError(f"Position must have shape ({n_points}, 3)")

        if velocity.shape != (n_points, 3):
            raise ValueError(f"Velocity must have shape ({n_points}, 3)")

        if acceleration.shape != (n_points, 3):
            raise ValueError(f"Acceleration must have shape ({n_points}, 3)")

        self.time = time
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.field = field
        self.particle = particle

    def get_sample_rate(self) -> float:
        """
        Get the sample rate in Hz

        Returns
        -------
        float
            Sample rate in Hertz
        """
        if len(self.time) < 2:
            raise ValueError(
                "Need at least 2 time points to determine sample rate")

        dt = self.time[1] - self.time[0]
        return 1.0 / dt

    def get_relative_position(self, reference_position: NDArray) -> NDArray:
        """
        Get position relative to a reference point (e.g., antenna position)

        Parameters
        ----------
        reference_position : NDArray
            Reference position as 3-vector [x, y, z] in meters

        Returns
        -------
        NDArray
            Relative position array, shape (N, 3)
        """
        reference_position = np.asarray(reference_position)
        if reference_position.shape != (3,):
            raise ValueError("Reference position must be a 3-vector")

        return self.position - reference_position

    def get_beta(self) -> NDArray:
        """
        Get normalized velocity (v/c)

        Returns
        -------
        NDArray
            Velocity divided by speed of light, shape (N, 3)
        """
        return self.velocity / sc.c

    def get_beta_dot(self) -> NDArray:
        """
        Get normalized acceleration (a/c)

        Returns
        -------
        NDArray
            Acceleration divided by speed of light, shape (N, 3)
        """
        return self.acceleration / sc.c

    def get_duration(self) -> float:
        """
        Get the total duration of the trajectory

        Returns
        -------
        float
            Duration in seconds
        """
        return self.time[-1] - self.time[0]

    def get_n_points(self) -> int:
        """
        Get the number of time points in the trajectory

        Returns
        -------
        int
            Number of points
        """
        return len(self.time)


class TrajectoryGenerator:
    """
    Generate electron trajectories in magnetic traps

    This class computes the full 3D trajectory of an electron in a magnetic
    trap, including:
    - Axial (z) motion due to the magnetic mirror force
    - Azimuthal (φ) motion including grad-B drift
    - Cyclotron (ψ) motion at the local cyclotron frequency
    - Time-dependent velocity and acceleration

    The calculation properly accounts for relativistic effects and the
    time-varying pitch angle as the electron moves through the non-uniform
    magnetic field.
    """

    def __init__(self, field: BaseField, particle: Particle):
        """
        Initialize TrajectoryGenerator

        Parameters
        ----------
        field : BaseField
            Magnetic field configuration
        particle : Particle
            Particle with initial position, energy, and pitch angle
        """
        self.field = field
        self.particle = particle

    def generate(self, sample_rate: float, t_max: float) -> Trajectory:
        """
        Generate complete trajectory with position, velocity, and acceleration

        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz (must satisfy Nyquist criterion for cyclotron frequency)
        t_max : float
            Maximum time in seconds

        Returns
        -------
        Trajectory
            Complete trajectory data object

        Raises
        ------
        ValueError
            If sample rate is too low or parameters are invalid
        """
        # Validate parameters
        self._validate_parameters(sample_rate, t_max)

        # Calculate position components
        t, pos = self._calc_position(sample_rate, t_max)

        # Calculate velocity from field and particle properties
        vel = self._calc_velocity(t, pos, sample_rate)

        # Calculate acceleration via gradient
        acc = self._calc_acceleration(vel, sample_rate)

        return Trajectory(t, pos, vel, acc, self.field, self.particle)

    def _validate_parameters(self, sample_rate: float, t_max: float) -> None:
        """
        Validate trajectory generation parameters

        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz
        t_max : float
            Maximum time in seconds

        Raises
        ------
        TypeError
            If parameters are not numeric
        ValueError
            If parameters are not positive, finite, or physically reasonable
        """
        # Type checks
        if not isinstance(sample_rate, (int, float)):
            raise TypeError("Sample rate must be a number")
        if not isinstance(t_max, (int, float)):
            raise TypeError("Maximum time must be a number")

        # Value checks
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if t_max <= 0:
            raise ValueError("Maximum time must be positive")
        if not np.isfinite(sample_rate):
            raise ValueError("Sample rate must be finite")
        if not np.isfinite(t_max):
            raise ValueError("Maximum time must be finite")

        # Check that we'll have at least a few points
        n_points = int(np.round(t_max * sample_rate))
        if n_points < 10:
            raise ValueError(
                f"Duration too short: only {n_points} points. "
                f"Increase t_max or sample_rate"
            )

    def _calc_position(self, sample_rate: float, t_max: float) -> tuple[NDArray, NDArray]:
        """
        Calculate position [x, y, z] as a function of time

        This combines the axial motion (z), azimuthal angle (φ) including
        grad-B drift, and the initial radial position (ρ) to construct the
        full 3D position trajectory.

        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz
        t_max : float
            Maximum time in seconds

        Returns
        -------
        tuple[NDArray, NDArray]
            Time array (N,) and position array (N, 3)
        """
        # Get axial position as function of time
        t, z = self._calc_axial_motion(sample_rate, t_max)

        # Get azimuthal angle as function of time
        phi = self._calc_azimuthal_motion(t, z)

        # Calculate radial position (constant for now)
        p_start = self.particle.GetPosition()
        rho = np.sqrt(p_start[0]**2 + p_start[1]**2)

        # Convert to Cartesian coordinates
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        # Stack into position array
        pos = np.column_stack([x, y, z])

        return t, pos

    def _calc_axial_motion(self, sample_rate: float, t_max: float) -> tuple[NDArray, NDArray]:
        """
        Calculate axial (z) position as a function of time

        Uses the field's calc_t_vs_z method to get the analytic relationship,
        then interpolates to create uniformly-sampled time points.

        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz
        t_max : float
            Maximum time in seconds

        Returns
        -------
        tuple[NDArray, NDArray]
            Time array (N,) and z-position array (N,)
        """
        # Get analytic t vs z relationship
        t_analytic, z_analytic = self.field.calc_t_vs_z(self.particle)
        Ta = t_analytic[-1]  # Axial period

        # Create interpolator
        t_to_z = interp1d(t_analytic, z_analytic, kind='cubic')

        # Create uniformly-spaced time array
        n_points = int(np.round(t_max * sample_rate))
        t_vals = np.linspace(0, t_max, n_points)

        # Interpolate, accounting for periodicity
        z_pos = t_to_z(np.mod(t_vals, Ta))

        return t_vals, z_pos

    def _calc_azimuthal_motion(self, t: NDArray, z: NDArray) -> NDArray:
        """
        Calculate azimuthal angle φ(t) including grad-B drift

        The grad-B drift causes the electron to drift in the azimuthal direction
        at a rate proportional to the perpendicular velocity squared and the
        radial gradient of the magnetic field.

        Parameters
        ----------
        t : NDArray
            Time array in seconds
        z : NDArray
            Axial position array in meters

        Returns
        -------
        NDArray
            Azimuthal angle in radians
        """
        # Get particle position and calculate cylindrical radius
        p_start = self.particle.GetPosition()
        rho = np.sqrt(p_start[0]**2 + p_start[1]**2)

        # Get magnetic field values along trajectory
        field_mags = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], z)
        field_0 = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], 0.0)

        # Get B-field z-component (dominant component)
        _, _, field_vec_z = self.field.evaluate_field(
            p_start[0], p_start[1], z)

        # Get radial field gradient
        field_grads = self.field.evaluate_field_gradient(rho, z)

        # Calculate cyclotron frequency as function of position
        omega_c = sc.e * field_mags / \
            (self.particle.GetGamma() * self.particle.GetMass())

        # Calculate magnetic moment (adiabatic invariant)
        mu_mag = (self.particle.GetGamma() * self.particle.GetMass() *
                  (np.sin(self.particle.GetPitchAngle()) * self.particle.GetSpeed())**2 /
                  (2 * field_0))

        # Calculate grad-B drift velocity
        v_grad_B = ((mu_mag / (self.particle.GetMass() * omega_c * field_mags)) *
                    field_grads * field_vec_z)

        # Calculate initial azimuthal angle
        phi_i = np.arctan2(p_start[1], p_start[0])

        # Integrate to get azimuthal position
        # If rho is too small, avoid division by zero
        if rho < 1e-10:
            return np.full_like(t, phi_i)

        phi = cumulative_simpson(v_grad_B / rho, x=t, initial=phi_i)

        return phi

    def _calc_velocity(self, t: NDArray, pos: NDArray, sample_rate: float) -> NDArray:
        """
        Calculate velocity as a function of time

        The velocity has three components:
        - Cyclotron motion at frequency ω_c in the x-y plane
        - Axial motion in the z direction
        Both depend on the time-varying pitch angle as the electron moves
        through the non-uniform field.

        Parameters
        ----------
        t : NDArray
            Time array in seconds
        pos : NDArray
            Position array (N, 3)
        sample_rate : float
            Sample rate in Hz

        Returns
        -------
        NDArray
            Velocity array (N, 3)
        """
        # Get z positions
        z_pos = pos[:, 2]

        # Get initial position
        p_start = self.particle.GetPosition()

        # Calculate magnetic field magnitude along trajectory
        B_vals = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], z_pos)

        # Calculate cyclotron phase by integrating ω_c
        omega_c = sc.e * B_vals / \
            (self.particle.GetGamma() * self.particle.GetMass())
        psi = cumulative_simpson(omega_c, x=t, initial=0.0)

        # Calculate time-varying pitch angle from adiabatic invariant
        # sin²θ(t) * B(t) = sin²θ_0 * B_0
        field_0 = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], 0.0)
        sin_theta_squared = np.sin(
            self.particle.GetPitchAngle())**2 * field_0 / B_vals

        # Clamp to [0, 1] to handle numerical issues
        sin_theta_squared = np.clip(sin_theta_squared, 0.0, 1.0)
        sin_theta = np.sqrt(sin_theta_squared)
        cos_theta = np.sqrt(1.0 - sin_theta_squared)

        # Determine direction of axial motion from gradient of z
        z_diff = np.gradient(z_pos, t)
        moving_positive = z_diff > 0.0

        # Velocity components
        speed = self.particle.GetSpeed()
        vel_x = speed * np.cos(psi) * sin_theta
        vel_y = speed * np.sin(psi) * sin_theta
        vel_z = np.where(moving_positive, cos_theta, -cos_theta) * speed

        return np.column_stack([vel_x, vel_y, vel_z])

    def _calc_acceleration(self, vel: NDArray, sample_rate: float) -> NDArray:
        """
        Calculate acceleration via numerical gradient of velocity

        Parameters
        ----------
        vel : NDArray
            Velocity array (N, 3)
        sample_rate : float
            Sample rate in Hz

        Returns
        -------
        NDArray
            Acceleration array (N, 3)
        """
        dt = 1.0 / sample_rate
        return np.gradient(vel, dt, axis=0)
