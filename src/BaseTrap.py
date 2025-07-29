'''
BaseTrap.py

This file contains the BaseTrap class, which is an abstract class representing a
generic electron trap.

S. Jones 29-07-25
'''
from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as sc


class BaseTrap(ABC):
    """
    Base class for electron traps

    Attributes
    ----------
    __gradB: float representing the gradient of the magnetic field
    """

    __gradB = 0.0

    def _ValidatePitchAngle(self, pitchAngle):
        """Validate pitch angle parameter"""
        if not isinstance(pitchAngle, (int, float)):
            raise TypeError("Pitch angle must be a number")
        if not np.isfinite(pitchAngle):
            raise ValueError("Pitch angle must be finite")
        if pitchAngle <= 0 or pitchAngle >= np.pi:
            raise ValueError("Pitch angle must be in range (0, π)")

    def _ValidateVelocity(self, v):
        """Validate speed parameter"""
        if not isinstance(v, (int, float)):
            raise TypeError("Speed must be a number")
        if v <= 0:
            raise ValueError("Speed must be positive")
        if v >= sc.c:
            raise ValueError(f"Speed {v} m/s exceeds speed of light")
        if not np.isfinite(v):
            raise ValueError("Speed must be finite")

    @abstractmethod
    def CalcZMax(self, pitchAngle):
        """
        Calculate the maximum axial position

        Parameters
        ----------
        pitchAngle: float representing the pitch angle in radians
        """

    @abstractmethod
    def CalcOmegaAxial(self, pitchAngle, v):
        """
        Get the axial frequency of the electron's motion

        Parameters
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """

    @abstractmethod
    def CalcOmega0(self, v, pitchAngle):
        """
        Get the average cyclotron frequency

        Parameters
        ----------
        v: float representing the speed of the electron in m/s
        pitchAngle: float representing the pitch angle in radians
        """

    def GetGradB(self):
        """
        Getter for the gradient of the magnetic field

        Returns:
        --------
            float: Gradient of the magnetic field in Tesla per metre
        """
        return self.__gradB

    def SetGradB(self, gradB):
        """
        Setter for the gradient of the magnetic field

        Parameters:
        -----------
            gradB (float): Gradient of the magnetic field in Tesla per metre
        """
        if not isinstance(gradB, (int, float)):
            raise TypeError("Gradient must be a number")
        if not np.isfinite(gradB):
            raise ValueError("Gradient must be finite")

        self.__gradB = gradB
