"""
BaseSpectrumCalculator.y

Contains abstract base class BaseSpectrumCalculator from which other spectrum
calculators should inherit.

S. Jones
"""
from abc import ABC, abstractmethod
from numpy.typing import NDArray, ArrayLike
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from scipy.special import jvp, j1
import numpy as np
from scipy.integrate import quad
import scipy.constants as sc


class BaseSpectrumCalculator(ABC):
    """
    Base class for spectrum calculators
    """

    def __init__(self, trap, waveguide: CircularWaveguide,
                 particle: Particle):
        self.__trap = trap
        self.__waveguide = waveguide
        self.__particle = particle

    @abstractmethod
    def GetPeakAmp(self, order: ArrayLike, negativeFreqs=False) -> NDArray:
        """
        Calculate the complex amplitude of a peak, given the order

        Parameters
        ----------
        order : ArrayLike
            Order of the peak for which we are calculating the amplitude
        negativeFreqs : bool
            Boolean to return amps for negative frequencies (default false)

        Returns
        -------
        NDArray
            Array of complex peak amplitudes
        """

    @abstractmethod
    def GetPeakFrequency(self, order: ArrayLike, negativeFreqs=False) -> NDArray:
        """
        Calculate the frequencies at which the components occur

        Parameters
        ----------
        order : ArrayLike
            The order of the peak to calculate the frequency for
        negativeFreqs : bool
            Boolean to return negative frequencies for given order (default false)

        Returns
        -------
        NDArray
            The frequency at which a component is observed in Hertz
        """

    def GetPowerNorm(self) -> float:
        """
        Calculate the power normalisation for an electron in a circular
        waveguide coupling to the TE11 mode

        Returns
        -------
        float
            Power factor in Watts
        """

        def alphaIntegrand(rho, kc):
            """
            Integrand for alpha calculation
            """
            return rho * ((j1(kc * rho) / (kc * rho))**2 + jvp(1, kc * rho)**2)

        kc = 1.841 / self.__waveguide.wgR
        alpha, _ = quad(alphaIntegrand, 0, self.__waveguide.wgR, args=kc)
        r_gyro = np.sqrt(self.__particle.GetPosition()[
                         0]**2 + self.__particle.GetPosition()[1]**2)

        prefactor = self.__waveguide.CalcTE11Impedance(self.__trap.CalcOmega0(
            self.__particle.GetSpeed(), self.__particle.GetPitchAngle())) * (sc.e * self.__particle.GetSpeed())**2 / (8 * np.pi * alpha)

        if r_gyro == 0:
            # Handle the special case where r_gyro = 0
            # lim(x->0) j1(x)/x = 1/2, so (j1(x)/x)^2 = 1/4
            return prefactor * (jvp(1, kc * r_gyro)**2 + 0.25)
        else:
            return prefactor * (jvp(1, kc * r_gyro)**2 + (j1(kc * r_gyro) / (kc * r_gyro))**2)

    def GetPeakPower(self, order: ArrayLike) -> NDArray:
        """
        Calculates the power in a given order sideband

        Parameters
        ----------
        order : ArrayLike
            Order of sideband to calculate

        Returns
        -------
        NDArray
            Power in peaks of order 'order' in Watts
        """

        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")

        a = self.GetPeakAmp(order)
        return np.abs(a)**2 * self.GetPowerNorm()
