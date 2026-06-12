"""
Base field module

Provides the abstract class BaseField which allows for implementation of 'real'
fields.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional
from .Particle import Particle
from scipy.integrate import simpson, cumulative_simpson
import scipy.constants as sc
from scipy.interpolate import interp1d


class BaseField(ABC):
    """
    Base field abstract class

    Methods
    -------
    evaluate_field(x, y, z): Evaluates the field at a given position or positions
                             (abstractmethod)

    evaluate_field_magnitude(x, y, z): Evaluate field magnitude for a position or positions
    """

    @abstractmethod
    def evaluate_field(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple:
        """
        Parameters
        ----------
        x : ArrayLike
          x position(s) in metres
        y : ArrayLike
          y position(s) in metres
        z : ArrayLike
          z position(s) in metres
        """

    def evaluate_field_magnitude(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray | float:
        """
        Parameters
        ----------
        x : ArrayLike
          x position(s) in metres
        y : ArrayLike
          y position(s) in metres
        z : ArrayLike
          z position(s) in metres

        Returns
        -------
        NDArray:
            Magnetic field magnitude in Tesla
        """
        b_x, b_y, b_z = self.evaluate_field(x, y, z)
        return np.sqrt(b_x**2 + b_y**2 + b_z**2)

    def evaluate_field_gradient(self, rho: ArrayLike, z: ArrayLike, eps: float = 1e-8):
        """
        Compute field gradient with respect to radial coordinate, rho

        Parameters
        ----------
        rho, z : ArrayLike
            Positions in cylindrical coordinates to evaluate at (in metres)
        eps : float
            Step size for calculation of finite differences

        Returns
        -------
        NDArray :
            Radial gradient component in T/m
        """
        rho = np.array(rho)
        z = np.asarray(z)

        # Azimuthally symmetric so just interpret rho as x coordinate
        grad = (self.evaluate_field_magnitude(rho + eps, 0, z) -
                self.evaluate_field_magnitude(rho - eps, 0, z)) / (2 * eps)
        return grad

    def calc_rho_along_field_line(self, rho_0: float,
                                  z: ArrayLike) -> NDArray:
        """
        Calculate cylindrical radius along a field line using flux conservation

        For azimuthally symmetric fields: rho(z) = rho_0 * sqrt(B_0 / B(z))

        Parameters
        ----------
        rho_0 : float
            Cylindrical radius at z = 0 in metres
        z : ArrayLike
            Axial position(s) in metres

        Returns
        -------
        NDArray
            Cylindrical radius at each z position in metres

        Raises
        ------
        ValueError
            If rho_0 is negative
        """
        if rho_0 < 0:
            raise ValueError("Initial radius rho_0 must be non-negative")

        if rho_0 == 0.0:
            return np.zeros_like(np.asarray(z, dtype=float))

        B_0 = self.evaluate_field_magnitude(rho_0, 0.0, 0.0)
        B_z = self.evaluate_field_magnitude(rho_0, 0.0, z)
        return rho_0 * np.sqrt(B_0 / B_z)

    @abstractmethod
    def calc_z_max(self, particle: Particle) -> float:
        """
        Calculate the maximum axial position of the particle in the trap

        Parameters
        ----------
        particle : Particle
            The particle in question

        Returns
        -------
        float
            The maximum axial displacement in metres
        """

    def calc_omega_0(self, particle: Particle, n_t_points: int = 499) -> float:
        """
        Calculates the mean cyclotron frequency

        Parameters
        ----------
        particle : Particle
            The particle being trapped
        n_t_points : int
            The number of time points to use for the calculation

        Returns
        -------
        float
            The average cyclotron frequency in radians/s
        """
        if particle.get_pitch_angle() == np.pi/2:
            pos = particle.get_position()
            B0 = self.evaluate_field_magnitude(pos[0], pos[1], pos[2])
            return sc.e * B0 / (particle.get_gamma() * particle.get_mass())
        else:
            t, B = self.b_from_t(particle, n_t_points)
            phi_Ta = simpson(
                sc.e * B / (particle.get_gamma() * particle.get_mass()), t)
            return phi_Ta / t[-1]

    def b_from_t(self, particle: Particle, n_t_points: int):
        """
        Calculate the magnetic field as a function of time over 1 axial period

        Parameters
        ----------
        particle : Particle
            The particle being trapped
        n_t_points : int
            The number of time points to output B at

        Returns
        -------
        tuple[NDArray, NDArray]
            Time values in seconds, Field magnitudes in Tesla
        """
        if (n_t_points < 2):
            raise ValueError("Require at least 2 time points")

        # Initially calculate the axial period
        t, z = self.calc_t_vs_z(particle)
        t_vals = np.linspace(0.0, t[-1], n_t_points, endpoint=True)
        t_to_z = interp1d(t, z, kind='cubic')
        z = t_to_z(t_vals)
        pStart = particle.get_position()
        rho_0 = np.sqrt(pStart[0]**2 + pStart[1]**2)
        rho = self.calc_rho_along_field_line(rho_0, z)
        return t_vals, self.evaluate_field_magnitude(rho, 0.0, z)

    def cyclotron_phase_from_t(self, particle: Particle,
                               n_t_points: int = 499) -> tuple[NDArray, NDArray]:
        """
        Calculate the cumulative cyclotron phase as a function of time over one axial period

        Parameters
        ----------
        particle : Particle
            The particle being trapped
        n_t_points : int
            Number of time points (default 499)

        Returns
        -------
        tuple[NDArray, NDArray]
            Time values in seconds, cumulative phase in radians
        """
        t, B = self.b_from_t(particle, n_t_points)
        return t, cumulative_simpson(
            sc.e * B / (particle.get_gamma() * particle.get_mass()), x=t, initial=0.0)

    def calc_omega_axial(self, particle: Particle,
                         n_points: int = 50000) -> float:
        """
        Calculate the angular axial frequency of trapped particle motion

        Parameters
        ----------
        particle : Particle
            The particle being trapped
        n_points : int
            Number of integration points to use

        Returns
        -------
        float
            Axial frequency in radians/s
        """
        pa = particle.get_pitch_angle()
        gamma = particle.get_gamma()
        p0 = gamma * particle.get_mass() * particle.get_speed()
        pStart = particle.get_position()
        rho_0 = np.sqrt(pStart[0]**2 + pStart[1]**2)

        centralField = self.evaluate_field_magnitude(pStart[0], pStart[1], 0.)
        zMax = self.calc_z_max(particle)

        # Equivalent magnetic moment
        muMag = gamma * particle.get_mass() * (np.sin(pa) * particle.get_speed())**2 / \
            (2 * centralField)

        # Use the substitution z = zMax * sin^2(u) to avoid singularities
        u_points = np.linspace(0, np.pi/2, n_points, endpoint=True)
        z = zMax * np.sin(u_points)**2
        dz_du = 2 * zMax * np.sin(u_points) * np.cos(u_points)

        rho = self.calc_rho_along_field_line(rho_0, z)
        B_mag = self.evaluate_field_magnitude(rho, 0.0, z)
        _, _, B_z = self.evaluate_field(rho, 0.0, z)

        # dt/dz = γm |B| / (p_∥ |B_z|), where p_∥ = sqrt(p0² - 2γm μ B)
        p_parallel = np.sqrt(np.maximum(
            p0**2 - 2 * gamma * particle.get_mass() * muMag * B_mag, 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            integrand = gamma * particle.get_mass() * B_mag / (p_parallel * np.abs(B_z)) * dz_du
        integrand = np.where(np.isfinite(integrand), integrand, 0.0)

        integral = float(2 * simpson(integrand, u_points) / np.pi)
        return 1 / integral

    def calc_t_vs_z(self, particle: Particle,
                    axial_period: Optional[float] = None,
                    n_points: int = 10000) -> tuple[NDArray, NDArray]:
        """
        Calculate the time versus the z position of an electron in a trap

        Parameters
        ----------
        particle : Particle
            The particle being trapped
        axial_period : float
            Optional axial period argument in seconds (default = None)
        n_points : int
            Number of integration points per quarter-period

        Returns
        -------
        tuple[NDArray, NDArray]
            Time values in seconds, z values in metres
        """

        pa = particle.get_pitch_angle()
        gamma = particle.get_gamma()
        p0 = gamma * particle.get_mass() * particle.get_speed()
        pStart = particle.get_position()
        rho_0 = np.sqrt(pStart[0]**2 + pStart[1]**2)

        centralField = self.evaluate_field_magnitude(pStart[0], pStart[1], 0.)
        zMax = self.calc_z_max(particle)
        muMag = gamma * particle.get_mass() * (np.sin(pa) * particle.get_speed())**2 / \
            (2 * centralField)

        if axial_period is None:
            axial_period = 2 * np.pi / self.calc_omega_axial(particle)

        quarter_period = axial_period / 4
        half_period = axial_period / 2

        # Use the substitution z = zMax * sin(u)^2
        u_points = np.linspace(0, np.pi / 2, n_points)
        z1 = zMax * np.sin(u_points)**2
        dz_du = 2 * zMax * np.sin(u_points) * np.cos(u_points)

        rho = self.calc_rho_along_field_line(rho_0, z1)
        B_mag = self.evaluate_field_magnitude(rho, 0.0, z1)
        _, _, B_z_comp = self.evaluate_field(rho, 0.0, z1)
        p_parallel = np.sqrt(np.maximum(
            p0**2 - 2 * gamma * particle.get_mass() * muMag * B_mag, 0.0))

        with np.errstate(divide='ignore', invalid='ignore'):
            integrand_u = gamma * particle.get_mass() * B_mag / (p_parallel *
                                                                np.abs(B_z_comp)) * dz_du
        integrand_u = np.where(np.isfinite(integrand_u), integrand_u, 0.0)

        # For axially symmetric traps, we only need to integrate one quarter
        tVals1 = cumulative_simpson(integrand_u, x=u_points, initial=0.0)
        tVals1[-1] = quarter_period

        # Return trip by symmetry, using true quarter period as the midpoint
        zVals2 = np.flip(z1[:-1])
        tVals2 = 2 * quarter_period - np.flip(tVals1[:-1])

        # First half (positive z): 0 -> zMax -> 0, with turning point
        zFirstHalf = np.concatenate((z1, zVals2))
        tFirstHalf = np.concatenate((tVals1, tVals2))

        # Second half (negative z): 0 -> -zMax -> 0, offset by T/2
        zSecondHalf = -np.concatenate((z1[1:], zVals2))
        tSecondHalf = half_period + \
            np.concatenate((tVals1[1:], tVals2))

        tVals = np.concatenate((tFirstHalf, tSecondHalf))
        zVals = np.concatenate((zFirstHalf, zSecondHalf))
        return tVals, zVals
