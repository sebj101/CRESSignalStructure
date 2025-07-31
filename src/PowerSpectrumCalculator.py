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
from scipy.special import jv


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
            amp2 = np.sum(jv(MArr, q) * jv(order - 2 * MArr, beta2 * zmax))
            return (amp1, amp2)
        else:
            raise TypeError("Trap type currently not supported")
