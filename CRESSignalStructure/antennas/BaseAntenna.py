"""
BaseAntenna.py

Abstract base class for antennas detecting CRES radiation in free space.

This module provides the foundation for modeling different antenna types
that can detect cyclotron radiation from trapped electrons.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


class BaseAntenna(ABC):
    """
    Abstract base class for antennas detecting CRES radiation

    This class defines the interface that all antenna implementations must follow,
    including methods for calculating effective length, impedance, and polarisation.
    """

    def __init__(self, pos: NDArray, z_ax: NDArray, x_ax: NDArray):
        self._pos = self._validate_position(pos)
        self._z_ax = self._validate_direction(z_ax)
        self._x_ax = self._validate_direction(x_ax)

        # Make x-axis orthogonal to z-axis (Gram–Schmidt) and normalize
        x_vec_proj = self._x_ax - np.dot(self._x_ax, self._z_ax) * self._z_ax
        x_norm = np.linalg.norm(x_vec_proj)
        if x_norm == 0.0:
            raise ValueError("x_ax must not be parallel to z_ax.")
        self._x_ax = x_vec_proj / x_norm

        self._y_ax = np.cross(self._z_ax, self._x_ax)

        logger.debug("Initialised %s at pos=%s, z_ax=%s, x_ax=%s",
                     type(self).__name__, self._pos, self._z_ax, self._x_ax)

    @abstractmethod
    def GetETheta(self, pos: NDArray) -> NDArray:
        """
        Get the component of the antenna radiation pattern in the theta direction

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape(N,3)) to calculate the field at
            in metres

        Returns
        -------
        NDArray
            An (N,3) array of electric field vectors 
        """
        pass

    @abstractmethod
    def GetEPhi(self, pos: NDArray) -> NDArray:
        """
        Get the component of the antenna radiation pattern in the phi direction

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape(N,3)) to calculate the field at
            in metres

        Returns
        -------
        NDArray
            An (N,3) array of electric field vectors 
        """
        pass

    def GetTheta(self, pos: NDArray) -> NDArray:
        """
        Calculates polar angle theta w.r.t. antenna axes

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape(N,3)) in metres

        Returns
        -------
        NDArray
            An (N,1) array of angles in radians
        """
        r = self._pos - pos
        rHat = r / np.linalg.norm(r, axis=1, keepdims=True)
        return np.acos(np.dot(rHat, self._z_ax))

    def GetPhi(self, pos: NDArray) -> NDArray:
        """
        Calculates azimuthal angle phi w.r.t. antenna axes

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape(N,3)) in metres

        Returns
        -------
        NDArray
            An (N,1) array of angles in radians
        """
        r = self._pos - pos
        rHat = r / np.linalg.norm(r, axis=1, keepdims=True)
        phi = np.atan2(np.dot(rHat, self._y_ax), np.dot(rHat, self._x_ax))
        return phi

    @abstractmethod
    def GetEffectiveLength(self, frequency: float, pos: NDArray) -> NDArray:
        """
        Get the effective length of the antenna

        The effective length relates the incident electric field to the
        open-circuit voltage at the antenna terminals: V_oc = E · l_eff.
        Subclasses with a single dominant polarisation (e.g. dipoles)
        should implement this in terms of GetETheta / GetEPhi.

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres.
            Each vector is the point at which the radiation pattern is
            evaluated — typically a synthetic point at unit distance from
            the antenna toward each source.  Only the direction from the
            antenna matters; magnitude is normalised internally by
            GetETheta / GetEPhi.

        Returns
        -------
        NDArray
            (N, 3) array of effective length vectors in meters
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

    def GetPosition(self) -> NDArray:
        """
        Get the antenna position in the trap coordinate system

        Returns
        -------
        NDArray
            3-vector position in meters [x, y, z]
        """
        return self._pos.copy()

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
        return self._z_ax.copy()

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
