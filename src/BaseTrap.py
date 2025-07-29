'''
BaseTrap.py

This file contains the BaseTrap class, which is an abstract class representing a
generic electron trap.

S. Jones 29-07-25
'''
from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike


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
        pitchAngle = np.asarray(pitchAngle)

        if not np.issubdtype(pitchAngle.dtype, np.number):
            raise TypeError("Pitch angle must be numeric")

        if not np.all(np.isfinite(pitchAngle)):
            raise ValueError("Pitch angle must be finite")

        if np.any(pitchAngle <= 0) or np.any(pitchAngle >= np.pi):
            raise ValueError("Pitch angle must be in range (0, π)")

        return pitchAngle

    def _ValidateVelocity(self, v):
        """Validate speed parameter"""
        v = np.asarray(v)

        if not np.issubdtype(v.dtype, np.number):
            raise TypeError("Velocity must be numeric")

        if np.any(v <= 0):
            raise ValueError("Velocity must be positive")

        if np.any(v >= sc.c):
            raise ValueError(f"Velocity exceeds speed of light")

        if not np.all(np.isfinite(v)):
            raise ValueError("Velocity must be finite")

        return v

    @abstractmethod
    def CalcZMax(self, pitchAngle: ArrayLike) -> ArrayLike:
        """
        Calculate the maximum axial position

        Parameters
        ----------
        pitchAngle : ArrayLike 
            Pitch angle in radians
        """

    @abstractmethod
    def CalcOmegaAxial(self, pitchAngle: ArrayLike, v: ArrayLike) -> ArrayLike:
        """
        Get the axial frequency of the electron's motion

        Parameters
        ----------
        pitchAngle: float representing the pitch angle in radians
        v: float representing the speed of the electron in m/s
        """

    @abstractmethod
    def CalcOmega0(self, v: ArrayLike, pitchAngle: ArrayLike) -> ArrayLike:
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

        Returns
        -------
            float: Gradient of the magnetic field in Tesla per metre
        """
        return self.__gradB

    def SetGradB(self, gradB: float):
        """
        Setter for the gradient of the magnetic field

        Parameters
        ----------
        gradB : float 
            Gradient of the magnetic field in Tesla per metre
        """
        if not isinstance(gradB, (int, float)):
            raise TypeError("Gradient must be a number")
        if not np.isfinite(gradB):
            raise ValueError("Gradient must be finite")

        self.__gradB = gradB
