"""
QTNM trap module

Module containing implementations of BaseTrap.

Implementations:
---------------

HarmonicTrap: Harmonic magnetic field providing trapping field 
BathtubTrap: Two-coil bathtub magnetic field providing trapping field
"""

import numpy as np
from src.BaseTrap import BaseTrap
import scipy.constants as sc
from numpy.typing import ArrayLike


class HarmonicTrap(BaseTrap):
    """
    Trap class representing a harmonic magnetic field
    """

    __B0 = 0.0
    __L0 = 0.0

    def __init__(self, B0: float, L0: float, gradB: float = 0.0):
        """
        Constructor for HarmonicTrap

        Parameters
        ----------
        B0 : float 
            Magnetic field strength at the trap centre in T
        L0 : float 
            Characteristic length of the trap in m
        gradB : float 
            Gradient of the magnetic field in T/m
        """
        if not isinstance(B0, (int, float)):
            raise TypeError("B0 must be a number")
        if B0 <= 0:
            raise ValueError("B0 must be positive")
        if not np.isfinite(B0):
            raise ValueError("B0 must be finite")

        if not isinstance(L0, (int, float)):
            raise TypeError("L0 must be a number")
        if L0 <= 0:
            raise ValueError("L0 must be positive")
        if not np.isfinite(L0):
            raise ValueError("L0 must be finite")

        self.__B0 = B0
        self.__L0 = L0
        self.SetGradB(gradB)

    def CalcZMax(self, pitchAngle: ArrayLike):
        """
        Calc the maximum axial position

        Parameters
        ----------
        pitchAngle : float representing the pitch angle in radians
        """
        pitchAngle = self._ValidatePitchAngle(pitchAngle)
        result = np.where(np.abs(pitchAngle) < 1e-10, np.inf,
                          self.__L0 / np.tan(pitchAngle))
        result = np.where(np.abs(pitchAngle - np.pi/2) < 1e-10, 0.0, result)
        return result

    def CalcOmegaAxial(self, pitchAngle: ArrayLike, v: ArrayLike):
        """
        Get the axial frequency of the electron's motion in radians/s

        Parameters:
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """
        v = self._ValidateVelocity(v)
        pitchAngle = self._ValidatePitchAngle(pitchAngle)
        return v * np.sin(pitchAngle) / self.__L0

    def CalcOmega0(self, v: ArrayLike, pitchAngle: ArrayLike):
        """
        Get the average cyclotron frequency in radians/s

        Parameters
        ----------
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        v = self._ValidateVelocity(v)
        pitchAngle = self._ValidatePitchAngle(pitchAngle)
        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        return sc.e * self.__B0 / (sc.m_e * gamma) * (1 + self.CalcZMax(pitchAngle)**2 / (2 * self.__L0**2))


class BathtubTrap(BaseTrap):
    """
    Trap class representing a bathtub magnetic field
    """

    __B0 = 0.0
    __L0 = 0.0
    __L1 = 0.0

    def __init__(self, B0: float, L0: float, L1: float, gradB: float = 0.0):
        """
        Constructor for BathtubTrap

        Parameters
        ----------
        B0 : float 
            Magnetic field strength at the trap centre in T
        L0 : float 
            Characteristic length of trap quadratic section in m
        L1 : float 
            Length of the flat region of the trap in m
        gradB : float 
            Radial gradient of the magnetic field in T/m
        """
        if not isinstance(B0, (int, float)):
            raise TypeError("B0 must be a number")
        if B0 <= 0:
            raise ValueError("B0 must be positive")
        if not np.isfinite(B0):
            raise ValueError("B0 must be finite")

        if not isinstance(L0, (int, float)):
            raise TypeError("L0 must be a number")
        if L0 <= 0:
            raise ValueError("L0 must be positive")
        if not np.isfinite(L0):
            raise ValueError("L0 must be finite")

        if not isinstance(L1, (int, float)):
            raise TypeError("L1 must be a number")
        if L1 <= 0:
            raise ValueError("L1 must be positive")
        if not np.isfinite(L1):
            raise ValueError("L1 must be finite")

        self.__B0 = B0
        self.__L0 = L0
        self.__L1 = L1
        self.SetGradB(gradB)

    def CalcZMax(self, pitchAngle: ArrayLike):
        """
        Calc the maximum axial position

        Parameters
        ----------
        pitchAngle: float representing the pitch angle in radians
        """
        pitchAngle = self._ValidatePitchAngle(pitchAngle)
        result = np.where(np.abs(pitchAngle) < 1e-10, np.inf,
                          self.__L0 / np.tan(pitchAngle))
        result = np.where(np.abs(pitchAngle - np.pi/2) < 1e-10, 0.0, result)
        return result

    def CalcOmegaAxial(self, pitchAngle: ArrayLike, v: ArrayLike):
        """
        Get the axial frequency of the electron's motion in radians/s

        Parameters
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """
        pitchAngle = self._ValidatePitchAngle(pitchAngle)
        v = self._ValidateVelocity(v)
        wa = v * np.sin(pitchAngle) / self.__L0

        return wa / (1 + self.__L1 * np.tan(pitchAngle) / (self.__L0 * np.pi))

    def CalcOmega0(self, v: ArrayLike, pitchAngle: ArrayLike):
        """
        Get the average cyclotron frequency in radians/s

        Parameters
        ----------
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """
        pitchAngle = self._ValidatePitchAngle(pitchAngle)
        v = self._ValidateVelocity(v)

        beta = v / sc.c
        gamma = 1 / np.sqrt(1 - beta**2)
        prefac = sc.e * self.__B0 / (sc.m_e * gamma)

        zmax = self.CalcZMax(pitchAngle)
        denominator = 1 + self.__L1 * np.tan(pitchAngle) / (self.__L0 * np.pi)
        correction = np.where(np.isinf(zmax) | (np.abs(denominator) < 1e-15),
                              1.0,
                              (1 + zmax**2 / (2 * self.__L0**2)) / denominator)
        return prefac * correction
