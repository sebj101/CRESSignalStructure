"""
PowerSpectrumCalculator.py

Class to calculate power spectrum for CRES signals

S. Jones 29-07-25
"""

from src.BaseTrap import BaseTrap
from src.CircularWaveguide import CircularWaveguide
from src.QTNMTraps import HarmonicTrap, BathtubTrap
from src.Particle import Particle
import numpy as np
import scipy.constants as sc
from scipy.special import jv, jvp, j1
from scipy.integrate import quad


class PowerSpectrumCalculator:
    def __init__(self, trap: BaseTrap, waveguide: CircularWaveguide,
                 particle: Particle):
        self.__trap = trap
        self.__waveguide = waveguide
        self.__particle = particle

    def GetPeakFrequency(self, order: int):
        """
        Calculate the frequencies at which the components occur

        Parameters
        ----------
        order : int
            The order of the peak to calculate the frequency for

        Returns
        -------
        tuple(float, float)
            The frequency or frequencies at which components are observed in 
            radians/s
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise TypeError("Order must be positive")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        # Treat the n = 0 case separately
        v0 = self.__particle.GetSpeed()
        pa = self.__particle.GetPitchAngle()
        return (self.__trap.CalcOmega0(v0, pa) + order * self.__trap.CalcOmegaAxial(pa, v0),
                self.__trap.CalcOmega0(v0, pa) - order * self.__trap.CalcOmegaAxial(pa, v0))

    def GetPeakAmp(self, order: int):
        """
        Calculate the peak amplitude in the sidebands for a given peak order

        Parameters
        ----------
        order : int
            Order of sideband

        Returns
        -------
        tuple(float, float)
            Peak amplitudes
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise TypeError("Order must be positive")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        kc = 1.841 / self.__waveguide.wgR  # Consider only TE11 mode
        f1, f2 = self.GetPeakFrequency(order)
        pitchAngle = self.__particle.GetPitchAngle()
        v0 = self.__particle.GetSpeed()
        zmax = self.__trap.CalcZMax(pitchAngle)
        beta1 = np.sqrt((f1 / sc.c)**2 - kc**2)
        beta2 = np.sqrt((f2 / sc.c)**2 - kc**2)

        if isinstance(self.__trap, HarmonicTrap):
            q = self.__trap.Calcq(v0, pitchAngle)
            MArr = np.arange(-6, 7, 1)
            amp1 = np.sum(jv(MArr, q) * jv(order - 2 * MArr, beta1 * zmax))
            amp2 = np.sum(jv(MArr, q) * jv(-order - 2 * MArr, beta2 * zmax))
            return (amp1, amp2)
        else:
            raise TypeError("Trap type currently not supported")

    def GetPowerNorm(self):
        """
        Calculate the power normalisation for the electron in this case

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

    def GetSpectrumPowers(self, order: int):
        """
        Calculates the power in a given order sideband

        Parameters
        ----------
        order : int
            Order of sideband

        Returns
        -------
        tuple
            Power in sidebands of order 'order' in Watts
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise TypeError("Order must be positive")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        a1, a2 = self.GetPeakAmp(order)
        return (a1**2 * self.GetPowerNorm(), a2**2 * self.GetPowerNorm())
