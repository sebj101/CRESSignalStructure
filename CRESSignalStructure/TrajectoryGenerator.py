"""
TrajectoryGenerator.py

Classes for generating and storing electron trajectories in magnetic traps.
This module provides tools for calculating position, velocity, and acceleration
of trapped electrons including grad-B drift effects.
"""

import numpy as np
from numpy.typing import NDArray
import scipy.constants as sc
from scipy.integrate import cumulative_simpson, trapezoid
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

    def generate(self, sample_rate: float, t_max: float,
                 include_radiation: bool = False) -> Trajectory:
        """
        Generate complete trajectory with position, velocity, and acceleration

        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz (must satisfy Nyquist criterion for cyclotron frequency)
        t_max : float
            Maximum time in seconds
        include_radiation : bool, optional
            Whether to include radiative energy loss (default False)

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

        # Calculate position (with or without radiation)
        if include_radiation:
            t, pos = self._calc_position_with_radiation(sample_rate, t_max)
            E_t = self._calc_energy_vs_time(t)
        else:
            t, pos = self._calc_position(sample_rate, t_max)
            E_t = None

        # Calculate velocity and acceleration (passing energy if applicable)
        vel = self._calc_velocity(t, pos, sample_rate, E_t)
        acc = self._calc_acceleration(pos, vel, sample_rate, E_t)

        return Trajectory(t, pos, vel, acc, self.field, self.particle)

    def _calc_larmor_power(self) -> float:
        """
        Calculate time-averaged relativistic Larmor power over one axial period

        The relativistic Larmor power for an electron in a magnetic field is:
        P = (q^4 * B^2 * v_perp^2) / (6π ε_0 m^2 c^3 γ^2)

        Since B(z) and v_perp(z) vary along the trajectory, we calculate the
        trajectory over one axial period (without energy loss) and compute the
        time-averaged power.

        Returns
        -------
        float
            Time-averaged radiated power in Watts
        """
        PERPENDICULAR_THRESHOLD = 1e-4 * np.pi / 180

        # Get particle's starting position
        p_start = self.particle.GetPosition()

        # Special case: nearly perpendicular pitch angle (no axial motion)
        if abs(np.pi/2 - self.particle.GetPitchAngle()) < PERPENDICULAR_THRESHOLD:
            # Use power at starting position
            B = self.field.evaluate_field_magnitude(
                p_start[0], p_start[1], p_start[2])
            v_perp = self.particle.GetSpeed()  # All velocity is perpendicular

            q = abs(self.particle.GetCharge())
            m = self.particle.GetMass()
            gamma = self.particle.GetGamma()

            prefactor = q**4 / (6 * np.pi * sc.epsilon_0 *
                                m**2 * sc.c**3 * gamma**2)
            return prefactor * B**2 * v_perp**2

        # General case: average over one axial period
        # Get one axial period trajectory (without energy loss)
        t_analytic, z_analytic = self.field.calc_t_vs_z(self.particle)
        T_axial = t_analytic[-1]  # Axial period

        # Evaluate B field along the trajectory
        B_vals = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], z_analytic)

        # Calculate time-dependent perpendicular velocity using adiabatic invariant
        # sin²θ(z) * B(z) = sin²θ_0 * B_0 (constant)
        B_0 = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], p_start[2])
        theta_0 = self.particle.GetPitchAngle()

        sin_theta_squared = np.sin(theta_0)**2 * B_0 / B_vals
        sin_theta_squared = np.clip(sin_theta_squared, 0.0, 1.0)

        # Perpendicular velocity at each point (speed is constant for this calc)
        v_total = self.particle.GetSpeed()
        v_perp = v_total * np.sqrt(sin_theta_squared)

        # Physical constants
        q = abs(self.particle.GetCharge())
        m = self.particle.GetMass()
        gamma = self.particle.GetGamma()

        # Relativistic Larmor power at each point along trajectory
        prefactor = q**4 / (6 * np.pi * sc.epsilon_0 *
                            m**2 * sc.c**3 * gamma**2)
        P_local = prefactor * B_vals**2 * v_perp**2

        # Time-averaged power over one axial period
        # Use trapezoidal integration: <P> = (1/T) ∫ P(t) dt
        P_avg = trapezoid(P_local, t_analytic) / T_axial

        return P_avg

    def _calc_energy_vs_time(self, t: NDArray) -> NDArray:
        """
        Calculate kinetic energy as function of time including radiative losses

        To first order: E(t) = E_0 - P * t

        Parameters
        ----------
        t : NDArray
            Time array in seconds

        Returns
        -------
        NDArray
            Kinetic energy in eV at each time point
        """
        P_larmor = self._calc_larmor_power()  # Watts
        E_initial = self.particle.GetEnergy()  # eV
        P_eV_per_sec = P_larmor / sc.e

        # Linear energy loss (first-order approximation)
        E_t = E_initial - P_eV_per_sec * t

        # Ensure energy doesn't go negative
        E_t = np.maximum(E_t, 1.0)

        return E_t

    def _calc_gamma_vs_time(self, E_t: NDArray) -> NDArray:
        """
        Calculate gamma factor as function of energy

        Parameters
        ----------
        E_t : NDArray
            Kinetic energy in eV at each time point

        Returns
        -------
        NDArray
            Gamma factor at each time point
        """
        m = self.particle.GetMass()
        return 1.0 + E_t * sc.e / (m * sc.c**2)

    def _calc_speed_vs_time(self, gamma_t: NDArray) -> NDArray:
        """
        Calculate particle speed as function of gamma

        Parameters
        ----------
        gamma_t : NDArray
            Gamma factor at each time point

        Returns
        -------
        NDArray
            Speed in m/s at each time point
        """
        beta_t = np.sqrt(1 - 1 / gamma_t**2)
        return sc.c * beta_t

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

    def _calc_rho_from_z(self, z: NDArray) -> NDArray:
        """
        Calculate cylindrical radius along a field line using flux conservation

        Parameters
        ----------
        z : NDArray
            Axial positions in meters

        Returns
        -------
        NDArray
            Cylindrical radius at each z position in meters
        """
        p_start = self.particle.GetPosition()
        rho_0 = np.sqrt(p_start[0]**2 + p_start[1]**2)
        return self.field.calc_rho_along_field_line(rho_0, z)

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

        # Calculate radial position along field line
        rho = self._calc_rho_from_z(z)

        # Get azimuthal angle as function of time
        phi = self._calc_azimuthal_motion(t, z, rho)

        # Convert to Cartesian coordinates
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        # Stack into position array
        pos = np.column_stack([x, y, z])

        return t, pos

    def _calc_position_with_radiation(self, sample_rate: float, t_max: float) -> tuple[NDArray, NDArray]:
        """
        Calculate position with radiative energy loss corrections

        Applies first-order corrections to the axial motion based on
        the decreasing particle energy over time.

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
        # Get base trajectory (constant energy)
        t, z_base = self._calc_axial_motion(sample_rate, t_max)

        # Calculate energy loss
        E_t = self._calc_energy_vs_time(t)
        E_0 = self.particle.GetEnergy()

        # First-order correction: axial amplitude scales with sqrt(energy)
        energy_ratio = np.sqrt(E_t / E_0)

        z_corrected = z_base * energy_ratio

        # Calculate radial position along field line
        rho = self._calc_rho_from_z(z_corrected)

        phi = self._calc_azimuthal_motion(t, z_corrected, rho)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        # Stack into position array
        pos = np.column_stack([x, y, z_corrected])
        return t, pos

    def _calc_axial_motion(self, sample_rate: float, t_max: float) -> tuple[NDArray, NDArray]:
        """
        Calculate axial (z) position as a function of time

        Uses the field's calc_t_vs_z method to get the analytic relationship,
        then interpolates to create uniformly-sampled time points.

        Special case: If the pitch angle is very close to 90 degrees, approximate 
        as no axial motion 

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
        # Create uniformly-spaced time array
        n_points = int(np.round(t_max * sample_rate)) + 1
        t_vals = np.linspace(0, t_max, n_points)

        PERPENDICULAR_THRESHOLD = 1e-4 * np.pi / 180
        # Use approximation if we're too close to 90 degree pitch angle
        if abs(np.pi/2 - self.particle.GetPitchAngle()) < PERPENDICULAR_THRESHOLD:
            p_init = self.particle.GetPosition()
            z_pos = np.full(n_points, p_init[2])
            return t_vals, z_pos

        else:
            axial_period = 2 * np.pi / self.field.CalcOmegaAxial(self.particle)
            t_analytic, z_analytic = self.field.calc_t_vs_z(self.particle,
                                                            axial_period)
            t_to_z = interp1d(t_analytic, z_analytic, kind='cubic')
            # Interpolate, accounting for periodicity
            z_pos = t_to_z(np.mod(t_vals, axial_period))

            return t_vals, z_pos

    def _calc_azimuthal_motion(self, t: NDArray, z: NDArray,
                               rho: NDArray) -> NDArray:
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
        rho : NDArray
            Cylindrical radius at each z position in meters

        Returns
        -------
        NDArray
            Azimuthal angle in radians
        """
        p_start = self.particle.GetPosition()

        # Evaluate fields using actual rho along trajectory
        # (azimuthally symmetric, so use rho as x with y=0)
        field_mags = self.field.evaluate_field_magnitude(rho, 0.0, z)
        field_0 = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], p_start[2])

        # Get B-field z-component (dominant component)
        _, _, field_vec_z = self.field.evaluate_field(rho, 0.0, z)

        # Get radial field gradient
        field_grads = self.field.evaluate_field_gradient(rho, z)

        # Calculate magnetic moment (adiabatic invariant): μ = γmv_⊥²/(2B)
        mu_mag = (self.particle.GetGamma() * self.particle.GetMass() *
                  (np.sin(self.particle.GetPitchAngle()) * self.particle.GetSpeed())**2 /
                  (2 * field_0))

        # Calculate grad-B drift velocity: v_∇B = (μ/qB²)(B × ∇⊥B)
        # Azimuthal component of B × ∇⊥B ≈ B_z * ∂B/∂ρ
        v_grad_B = ((mu_mag / (self.particle.GetCharge() * field_mags**2)) *
                    field_grads * field_vec_z)

        # Calculate initial azimuthal angle
        phi_i = np.arctan2(p_start[1], p_start[0])

        # Integrate to get azimuthal position
        # If rho is too small everywhere, avoid division by zero
        if np.all(rho < 1e-10):
            return np.full_like(t, phi_i)

        phi = cumulative_simpson(v_grad_B / rho, x=t, initial=phi_i)

        return phi

    def _calc_velocity(self, t: NDArray, pos: NDArray, sample_rate: float,
                       E_t: NDArray = None) -> NDArray:
        """
        Calculate velocity as a function of time

        The velocity has three components:
        - Cyclotron motion at frequency omega_c in the x-y plane
        - Axial motion in the z direction
        Both depend on the time-varying pitch angle as the electron moves
        through the non-uniform field.

        Parameters
        ----------
        t : NDArray
            Time array in seconds (N,)
        pos : NDArray
            Position array (N, 3)
        sample_rate : float
            Sample rate in Hz
        E_t : NDArray, optional
            Kinetic energy in eV at each time point. If None, uses constant energy.

        Returns
        -------
        NDArray
            Velocity array (N, 3)
        """
        # Get positions along trajectory
        x_pos = pos[:, 0]
        y_pos = pos[:, 1]
        z_pos = pos[:, 2]

        p_start = self.particle.GetPosition()
        B_vals = self.field.evaluate_field_magnitude(x_pos, y_pos, z_pos)

        # Calculate time-dependent or constant gamma and speed
        if E_t is not None:
            gamma_t = self._calc_gamma_vs_time(E_t)
            speed_t = self._calc_speed_vs_time(gamma_t)
        else:
            gamma_t = self.particle.GetGamma()
            speed_t = self.particle.GetSpeed()

        # Calculate cyclotron phase by integrating omega_c
        omega_c = sc.e * B_vals / (gamma_t * self.particle.GetMass())
        psi = cumulative_simpson(omega_c, x=t, initial=0.0)

        # Calculate time-varying pitch angle from adiabatic invariant
        # sin²θ(t) * B(t) = sin²θ_0 * B_0
        field_0 = self.field.evaluate_field_magnitude(
            p_start[0], p_start[1], p_start[2])
        sin_theta_squared = np.sin(
            self.particle.GetPitchAngle())**2 * field_0 / B_vals

        # Clamp to [0, 1] to handle numerical issues
        sin_theta_squared = np.clip(sin_theta_squared, 0.0, 1.0)
        sin_theta = np.sqrt(sin_theta_squared)
        cos_theta = np.sqrt(1.0 - sin_theta_squared)

        # Determine direction of axial motion from gradient of z
        z_diff = np.gradient(z_pos, t)
        moving_positive = z_diff > 0.0

        # Velocity components (use time-dependent speed if available)
        vel_x = speed_t * np.cos(psi) * sin_theta
        vel_y = speed_t * np.sin(psi) * sin_theta
        vel_z = np.where(moving_positive, cos_theta, -cos_theta) * speed_t

        return np.column_stack([vel_x, vel_y, vel_z])

    def _calc_acceleration(self, pos: NDArray, vel: NDArray, sample_rate: float,
                           E_t: NDArray = None) -> NDArray:
        """
        Calculate acceleration using hybrid approach

        Uses Lorentz force for perpendicular (cyclotron) acceleration and
        numerical gradient for parallel (mirror force) acceleration. This
        combines physical accuracy for high-frequency cyclotron motion with
        numerical calculation for slower axial dynamics.

        Parameters
        ----------
        pos : NDArray
            Position array (N, 3)
        vel : NDArray
            Velocity array (N, 3)
        sample_rate : float
            Sample rate in Hz
        E_t : NDArray, optional
            Kinetic energy in eV at each time point. If None, uses constant energy.

        Returns
        -------
        NDArray
            Acceleration array (N, 3)
        """
        # Extract position components
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        # Evaluate magnetic field at each position
        B_x, B_y, B_z = self.field.evaluate_field(x, y, z)
        B = np.column_stack([B_x, B_y, B_z])

        # Calculate unit vector along B field
        B_mag = np.linalg.norm(B, axis=1, keepdims=True)
        B_hat = B / B_mag

        # Physical constants
        charge = self.particle.GetCharge()
        mass = self.particle.GetMass()

        # Calculate time-dependent or constant gamma
        if E_t is not None:
            gamma_t = self._calc_gamma_vs_time(E_t)
        else:
            gamma_t = self.particle.GetGamma()

        # Calculate Lorentz force: a_perp = (q/(γm)) × (v × B)
        # This gives accurate perpendicular (cyclotron) acceleration
        v_cross_B = np.cross(vel, B)
        if E_t is not None:
            # Use time-dependent gamma (need to broadcast for element-wise division)
            a_lorentz = (charge / (gamma_t[:, np.newaxis] * mass)) * v_cross_B
        else:
            a_lorentz = (charge / (gamma_t * mass)) * v_cross_B

        # Calculate parallel velocity component
        v_parallel = np.sum(vel * B_hat, axis=1, keepdims=True) * B_hat

        # Calculate parallel acceleration using numerical gradient
        # This captures the mirror force effects
        dt = 1.0 / sample_rate
        a_parallel_numerical = np.gradient(v_parallel, dt, axis=0)

        # The Lorentz force is perpendicular to B, so it has no parallel component
        # We need to add the parallel acceleration from the gradient
        # Project Lorentz acceleration onto perpendicular plane
        a_lorentz_parallel = np.sum(
            a_lorentz * B_hat, axis=1, keepdims=True) * B_hat
        a_lorentz_perp = a_lorentz - a_lorentz_parallel

        # Combine: use Lorentz for perpendicular, numerical for parallel
        acceleration = a_lorentz_perp + a_parallel_numerical

        return acceleration
