"""
IsotropicAntenna.py

Implementation of an ideal isotropic antenna model for CRES signal detection.

This module provides a concrete implementation of the BaseAntenna class
for an isotropic antenna - a theoretical antenna with uniform gain in all
directions and no polarization preference.
"""

import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike, NDArray
from CRESSignalStructure.BaseAntenna import BaseAntenna


class IsotropicAntenna(BaseAntenna):
    """
    Ideal isotropic antenna model

    An isotropic antenna is a theoretical reference antenna that radiates
    uniformly in all directions with no polarization preference. It is
    commonly used as a reference for antenna gain calculations (0 dBi).

    This antenna has:
    - Unity gain (G = 1) in all directions
    - No directional pattern
    - Equal response to all polarizations
    - Effective length independent of angle

    While not physically realizable, it serves as a useful reference and
    simplification for modeling omnidirectional reception.
    """

    def __init__(self, position: ArrayLike, impedance: complex = 50.0 + 0j,
                 effective_length: float = 0.01):
        """
        Constructor for IsotropicAntenna

        Parameters
        ----------
        position : ArrayLike
            3-vector position in meters [x, y, z]
        impedance : complex, optional
            Antenna impedance in Ohms (default 50 + 0j Ω)
        effective_length : float, optional
            Effective length magnitude in meters (default 0.01 m = 1 cm)
            This sets the overall sensitivity of the antenna

        Raises
        ------
        TypeError
            If parameters have incorrect types
        ValueError
            If parameters have invalid values
        """
        # Validate position
        self._position = self._validate_position(position)

        # Validate impedance
        if not isinstance(impedance, (int, float, complex)):
            raise TypeError("Impedance must be a number")
        if not np.isfinite(impedance):
            raise ValueError("Impedance must be finite")
        if np.real(impedance) < 0:
            raise ValueError("Impedance real part (resistance) must be non-negative")
        self._impedance = complex(impedance)

        # Validate effective length
        if not isinstance(effective_length, (int, float)):
            raise TypeError("Effective length must be a number")
        if effective_length <= 0:
            raise ValueError("Effective length must be positive")
        if not np.isfinite(effective_length):
            raise ValueError("Effective length must be finite")
        self._effective_length = float(effective_length)

        # Set a default orientation (not physically meaningful for isotropic)
        self._orientation = np.array([0.0, 0.0, 1.0])

    def GetEffectiveLength(self, frequency: float, theta: float, phi: float) -> NDArray:
        """
        Get the effective length vector of the isotropic antenna

        For an isotropic antenna, the effective length is constant in magnitude
        and has no directional or polarization dependence. We return a vector
        that is perpendicular to the direction of arrival to represent that
        the antenna couples to the transverse electric field.

        Parameters
        ----------
        frequency : float
            Frequency in Hz (not used for isotropic antenna)
        theta : float
            Polar angle in radians (angle from z-axis)
        phi : float
            Azimuthal angle in radians (angle in x-y plane)

        Returns
        -------
        NDArray
            3-vector effective length in meters

        Notes
        -----
        The effective length is oriented perpendicular to the propagation
        direction and has constant magnitude. For an isotropic antenna,
        we choose an arbitrary transverse direction.
        """
        frequency = self._validate_frequency(frequency)
        theta, phi = self._validate_angles(theta, phi)

        # Direction of incoming wave (k-vector direction)
        k_hat = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Choose an arbitrary direction perpendicular to k_hat
        # Use a stable method to find perpendicular vector
        if np.abs(k_hat[2]) < 0.9:
            # k is not too close to z-axis, use z-cross-k
            perp = np.array([-k_hat[1], k_hat[0], 0.0])
        else:
            # k is close to z-axis, use x-cross-k
            perp = np.array([0.0, -k_hat[2], k_hat[1]])

        # Normalize and scale by effective length
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-10:
            l_eff = self._effective_length * perp / perp_norm
        else:
            # Degenerate case (shouldn't happen)
            l_eff = np.array([self._effective_length, 0.0, 0.0])

        return l_eff

    def GetImpedance(self, frequency: float) -> complex:
        """
        Get the antenna impedance

        For an isotropic antenna, the impedance is constant and independent
        of frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz

        Returns
        -------
        complex
            Antenna impedance in Ohms (resistance + j*reactance)
        """
        frequency = self._validate_frequency(frequency)
        return self._impedance

    def GetPosition(self) -> NDArray:
        """
        Get the antenna position

        Returns
        -------
        NDArray
            3-vector position in meters [x, y, z]
        """
        return self._position.copy()

    def GetOrientation(self) -> NDArray:
        """
        Get the antenna orientation

        For an isotropic antenna, orientation is not physically meaningful
        since the antenna responds equally in all directions. This returns
        an arbitrary unit vector (z-direction).

        Returns
        -------
        NDArray
            3-vector unit direction (arbitrary for isotropic antenna)
        """
        return self._orientation.copy()

    def GetGain(self, theta: float, phi: float) -> float:
        """
        Get the antenna gain pattern for an isotropic antenna

        An isotropic antenna has unity gain (0 dBi) in all directions.

        Parameters
        ----------
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        float
            Dimensionless gain (always 1.0 for isotropic antenna)
        """
        theta, phi = self._validate_angles(theta, phi)
        return 1.0

    def GetEffectiveLengthMagnitude(self) -> float:
        """
        Get the effective length magnitude

        Returns
        -------
        float
            Effective length in meters
        """
        return self._effective_length

    def SetEffectiveLength(self, effective_length: float) -> None:
        """
        Set the effective length magnitude

        Parameters
        ----------
        effective_length : float
            Effective length in meters (must be positive)

        Raises
        ------
        TypeError
            If effective_length is not a number
        ValueError
            If effective_length is not positive and finite
        """
        if not isinstance(effective_length, (int, float)):
            raise TypeError("Effective length must be a number")
        if effective_length <= 0:
            raise ValueError("Effective length must be positive")
        if not np.isfinite(effective_length):
            raise ValueError("Effective length must be finite")
        self._effective_length = float(effective_length)
