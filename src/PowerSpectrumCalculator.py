"""
PowerSpectrumCalculator.py

Class to calculate power spectrum for CRES signals

S. Jones 29-07-25
"""

from CircularWaveguide import CircularWaveguide
from BaseTrap import BaseTrap
from QTNMTraps import HarmonicTrap, BathtubTrap
from Particle import Particle
import numpy as np


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
            The frequency or frequencies at which components are observed
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise ValueError("Cannot have negative orders")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        # Treat the n = 0 case separately
        v0 = self.__particle.GetSpeed()
        pa = self.__particle.GetPitchAngle()
        return (self.__trap.CalcOmega0(v0, pa) + order * self.__trap.CalcOmegaAxial(pa, v0),
                self.__trap.CalcOmega0(v0, pa) - order * self.__trap.CalcOmegaAxial(pa, v0))
