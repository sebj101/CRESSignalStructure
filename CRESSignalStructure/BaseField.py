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

    def calc_t_vs_z(self, particle: Particle) -> tuple[NDArray, NDArray]:
        """
        Calculate the time versus the z position of an electron in a trap

        Parameters
        ----------
        trap : BaseField
            The trapping field
        particle : Particle
            The particle being trapped

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

        def t_integrand(z):
            result = gamma * particle.GetMass() / np.sqrt(p0**2 - 2 * gamma * particle.GetMass()
                                                          * muMag * self.evaluate_field_magnitude(pStart[0], pStart[1], z))
            return result

        # For axially symmetric traps, we should only need to do one integration
        zVals1 = np.linspace(0.0, 0.999 * zMax, 1000)
        tVals1 = cumulative_simpson(t_integrand(zVals1), x=zVals1, initial=0.0)
        zVals2 = np.flip(zVals1[:-1])
        tVals2 = 2 * tVals1[-1] - np.flip(tVals1[:-1])
        tVals3 = tVals2[-1] + np.concatenate((tVals1[1:], tVals2))
        zVals3 = -np.concatenate((zVals1[1:], zVals2))
        tVals = np.concatenate((tVals1, tVals2, tVals3))
        zVals = np.concatenate((zVals1, zVals2, zVals3))
        return tVals, zVals
