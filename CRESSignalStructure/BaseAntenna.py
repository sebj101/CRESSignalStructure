"""
BaseAntenna.py

Abstract base class for antennas detecting CRES radiation in free space.

This module provides the foundation for modeling different antenna types
that can detect cyclotron radiation from trapped electrons.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseAntenna(ABC):
    """
    Abstract base class for antennas detecting CRES radiation

    This class defines the interface that all antenna implementations must follow,
    including methods for calculating effective length, impedance, and position.
    """

    @abstractmethod
    def GetEffectiveLength(self, frequency: float, theta: float, phi: float) -> NDArray:
        """
        Get the effective length vector of the antenna

        The effective length relates the incident electric field to the
        open-circuit voltage at the antenna terminals: V_oc = E · l_eff

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        theta : float
            Polar angle in radians (angle from antenna axis/z-axis)
        phi : float
            Azimuthal angle in radians (angle in x-y plane)

        Returns
        -------
        NDArray
            3-vector representing effective length in meters [l_x, l_y, l_z]
        """
        pass

    @abstractmethod
    def GetImpedance(self, frequency: float) -> complex:
        """
        Get the antenna impedance at a given frequency

        Parameters
        ----------
        frequency : float
            Frequency in Hz

        Returns
        -------
        complex
            Antenna impedance in Ohms (resistance + j*reactance)
        """
        pass

    @abstractmethod
    def GetPosition(self) -> NDArray:
        """
        Get the antenna position in the trap coordinate system

        Returns
        -------
        NDArray
            3-vector position in meters [x, y, z]
        """
        pass

    @abstractmethod
    def GetOrientation(self) -> NDArray:
        """
        Get the antenna orientation unit vector

        For dipole antennas, this is the direction of the antenna axis.
        For loop antennas, this is the normal to the loop plane.

        Returns
        -------
        NDArray
            3-vector unit direction
        """
        pass

    def GetGain(self, theta: float, phi: float) -> float:
        """
        Get the antenna gain pattern (default implementation)

        Can be overridden by subclasses for specific gain patterns.

        Parameters
        ----------
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        float
            Dimensionless gain (linear, not dB)
        """
        # Default: omnidirectional (unity gain)
        return 1.0

    def _validate_frequency(self, frequency: float) -> float:
        """
        Validate frequency parameter

        Parameters
        ----------
        frequency : float
            Frequency in Hz

        Returns
        -------
        float
            Validated frequency

        Raises
        ------
        TypeError
            If frequency is not a number
        ValueError
            If frequency is not positive and finite
        """
        if not isinstance(frequency, (int, float)):
            raise TypeError("Frequency must be a number")
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        if not np.isfinite(frequency):
            raise ValueError("Frequency must be finite")
        return float(frequency)

    def _validate_angles(self, theta: float, phi: float) -> tuple[float, float]:
        """
        Validate angular parameters

        Parameters
        ----------
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        tuple[float, float]
            Validated (theta, phi)

        Raises
        ------
        TypeError
            If angles are not numbers
        ValueError
            If angles are not finite
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("Theta must be a number")
        if not isinstance(phi, (int, float)):
            raise TypeError("Phi must be a number")
        if not np.isfinite(theta):
            raise ValueError("Theta must be finite")
        if not np.isfinite(phi):
            raise ValueError("Phi must be finite")
        return float(theta), float(phi)

    def _validate_position(self, position: ArrayLike) -> NDArray:
        """
        Validate position vector

        Parameters
        ----------
        position : ArrayLike
            3-vector position

        Returns
        -------
        NDArray
            Validated position array

        Raises
        ------
        TypeError
            If position is not numeric
        ValueError
            If position is not a 3-vector or not finite
        """
        position = np.asarray(position)
        if not np.issubdtype(position.dtype, np.number):
            raise TypeError("Position must be numeric")
        if position.shape != (3,):
            raise ValueError("Position must be a 3-vector")
        if not np.all(np.isfinite(position)):
            raise ValueError("Position must be finite")
        return position

    def _validate_direction(self, direction: ArrayLike) -> NDArray:
        """
        Validate and normalize direction vector

        Parameters
        ----------
        direction : ArrayLike
            3-vector direction

        Returns
        -------
        NDArray
            Validated and normalized unit direction vector

        Raises
        ------
        TypeError
            If direction is not numeric
        ValueError
            If direction is not a 3-vector, not finite, or has zero length
        """
        direction = np.asarray(direction)
        if not np.issubdtype(direction.dtype, np.number):
            raise TypeError("Direction must be numeric")
        if direction.shape != (3,):
            raise ValueError("Direction must be a 3-vector")
        if not np.all(np.isfinite(direction)):
            raise ValueError("Direction must be finite")

        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Direction vector must have non-zero length")

        return direction / norm
