"""
Base field module

Provides the abstract class BaseField which allows for implementation of 'real'
fields.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray
from CRESSignalStructure.Particle import Particle
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

    def evaluate_field_magnitude(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray:
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
    def CalcZMax(self, particle: Particle) -> float:
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

    def CalcOmega0(self, particle: Particle, n_t_points: int = 499) -> float:
        """
        Calculates the mean cyclotron frequency

        Parameters
        ----------
        trap : BaseField
            The trapping field
        particle : Particle
            The particle being trapped
        n_t_points : int
            The number of time points to use for the calculation

        Returns
        -------
        float
            The average cyclotron frequency in radians/s
        """
        t, B = self.B_from_t(particle, n_t_points)
        phi_Ta = simpson(
            sc.e * B / (particle.GetGamma() * particle.GetMass()), t)
        return phi_Ta / t[-1]

    def B_from_t(self, particle: Particle, n_t_points: int):
        """
        Calculate the magnetic field as a function of time over 1 axial period

        Parameters
        ----------
        trap : BaseField
            The trapping field
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
        Ta = 1 / (self.CalcOmegaAxial(particle) / (2 * np.pi))
        t, z = self.calc_t_vs_z(particle)
        t_vals = np.linspace(0.0, t[-1], n_t_points, endpoint=True)
        t_to_z = interp1d(t, z, kind='cubic')
        z = t_to_z(t_vals)
        pStart = particle.GetPosition()
        return t_vals, self.evaluate_field_magnitude(pStart[0], pStart[1], z)

    def CalcOmegaAxial(self, particle: Particle) -> float:
        """
        Calculate the angular axial frequency of trapped particle motion

        Parameters
        ----------
        trap: BaseField
            The trapping magnetic field
        particle : Particle
            The particle being trapped

        Returns
        -------
        float
            Axial frequency in radians/s
        """
        pa = particle.GetPitchAngle()
        gamma = particle.GetGamma()
        p0 = gamma * particle.GetMass() * particle.GetSpeed()
        pStart = particle.GetPosition()

        centralField = self.evaluate_field_magnitude(pStart[0], pStart[1], 0.)
        zMax = self.CalcZMax(particle)

        # Equivalent magnetic moment
        muMag = gamma * particle.GetMass() * (np.sin(pa) * particle.GetSpeed())**2 / \
            (2 * centralField)

        # Calculate the integrand at each point
        integrationPoints = np.linspace(0, zMax, 900000, endpoint=False)
        integrand = gamma * particle.GetMass() / np.sqrt(p0**2 - 2 * gamma * particle.GetMass()
                                                         * muMag * self.evaluate_field_magnitude(pStart[0], pStart[1], integrationPoints))

        integral = float(2 * simpson(integrand, integrationPoints) / np.pi)
        return 1 / integral

    def calc_t_vs_z(self, particle: Particle,
                    axial_period: float = None) -> tuple[NDArray, NDArray]:
        """
        Calculate the time versus the z position of an electron in a trap

        Parameters
        ----------
        particle : Particle
            The particle being trapped
        axial_period : float
            Optional axial period argument in seconds (default = None)

        Returns
        -------
        tuple[NDArray, NDArray]
            Time values in seconds, z values in metres
        """

        pa = particle.GetPitchAngle()
        gamma = particle.GetGamma()
        p0 = gamma * particle.GetMass() * particle.GetSpeed()
        pStart = particle.GetPosition()

        centralField = self.evaluate_field_magnitude(pStart[0], pStart[1], 0.)
        zMax = self.CalcZMax(particle)
        muMag = gamma * particle.GetMass() * (np.sin(pa) * particle.GetSpeed())**2 / \
            (2 * centralField)

        if axial_period is None:
            axial_period = 2 * np.pi / self.CalcOmegaAxial(particle)

        quarter_period = axial_period / 4
        half_period = axial_period / 2

        def t_integrand(z):
            result = gamma * particle.GetMass() / np.sqrt(p0**2 - 2 * gamma * particle.GetMass()
                                                          * muMag * self.evaluate_field_magnitude(pStart[0], pStart[1], z))
            return result

        # For axially symmetric traps, we only need to integrate one quarter
        zVals1 = np.linspace(0.0, 0.999 * zMax, 1000)
        tVals1 = cumulative_simpson(t_integrand(zVals1), x=zVals1, initial=0.0)

        # Return trip by symmetry, using true quarter period as the midpoint
        zVals2 = np.flip(zVals1[:-1])
        tVals2 = 2 * quarter_period - np.flip(tVals1[:-1])

        # First half (positive z): 0 → zMax → 0, with turning point
        zFirstHalf = np.concatenate((zVals1, [zMax], zVals2))
        tFirstHalf = np.concatenate((tVals1, [quarter_period], tVals2))

        # Second half (negative z): 0 → -zMax → 0, offset by T/2
        zSecondHalf = -np.concatenate((zVals1[1:], [zMax], zVals2))
        tSecondHalf = half_period + \
            np.concatenate((tVals1[1:], [quarter_period], tVals2))

        tVals = np.concatenate((tFirstHalf, tSecondHalf))
        zVals = np.concatenate((zFirstHalf, zSecondHalf))
        return tVals, zVals
