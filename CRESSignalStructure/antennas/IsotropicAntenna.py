"""
IsotropicAntenna.py

Implementation of an ideal isotropic antenna model for CRES signal detection.

This module provides a concrete implementation of the BaseAntenna class
for an isotropic antenna - a theoretical antenna with uniform gain in all
directions and no polarization preference.
"""

import numpy as np
from numpy.typing import NDArray
from .BaseAntenna import BaseAntenna


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

    def __init__(self, position: NDArray, impedance: complex = 50.0 + 0j,
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
        # Axes are arbitrary for an isotropic antenna
        super().__init__(position, np.array(
            [0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))

        # Validate impedance
        if not isinstance(impedance, (int, float, complex)):
            raise TypeError("Impedance must be a number")
        if not np.isfinite(impedance):
            raise ValueError("Impedance must be finite")
        if np.real(impedance) < 0:
            raise ValueError(
                "Impedance real part (resistance) must be non-negative")
        self._impedance = complex(impedance)

        # Validate effective length
        if not isinstance(effective_length, (int, float)):
            raise TypeError("Effective length must be a number")
        if effective_length <= 0:
            raise ValueError("Effective length must be positive")
        if not np.isfinite(effective_length):
            raise ValueError("Effective length must be finite")
        self._effective_length = float(effective_length)

    def GetETheta(self, pos: NDArray) -> NDArray:
        """
        Get the theta component of the isotropic radiation pattern

        Returns a unit vector in the θ̂ direction at each observation point.
        Zeroed at the poles (θ = 0, π) where θ̂ is undefined.

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of unit θ̂ vectors
        """
        pos = np.atleast_2d(pos)
        r = pos - self._pos                                          # (N, 3)
        r_hat = r / np.linalg.norm(r, axis=1, keepdims=True)        # (N, 3)
        cos_theta = np.dot(r_hat, self._z_ax)                       # (N,)

        # v = sin(θ)·θ̂
        v = cos_theta[:, np.newaxis] * r_hat - self._z_ax            # (N, 3)
        sin_theta = np.linalg.norm(v, axis=1, keepdims=True)        # (N, 1)

        safe_sin = np.where(sin_theta > 1e-10, sin_theta, 1.0)
        return np.where(sin_theta > 1e-10, v / safe_sin, 0.0)       # (N, 3)

    def GetEPhi(self, pos: NDArray) -> NDArray:
        """
        Get the phi component of the isotropic radiation pattern

        Returns a unit vector in the φ̂ direction at each observation point.
        Zeroed at the poles (θ = 0, π) where φ̂ is undefined.

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of unit φ̂ vectors
        """
        pos = np.atleast_2d(pos)
        r = pos - self._pos                                          # (N, 3)
        r_hat = r / np.linalg.norm(r, axis=1, keepdims=True)        # (N, 3)

        # cross(ẑ, r̂) = sin(θ)·φ̂
        v = np.cross(self._z_ax, r_hat)                              # (N, 3)
        sin_theta = np.linalg.norm(v, axis=1, keepdims=True)        # (N, 1)

        safe_sin = np.where(sin_theta > 1e-10, sin_theta, 1.0)
        return np.where(sin_theta > 1e-10, v / safe_sin, 0.0)       # (N, 3)

    def GetEffectiveLength(self, frequency: float, pos: NDArray) -> NDArray:
        """
        Get the effective length vector of the isotropic antenna

        An isotropic antenna has no polarisation preference, so the
        effective length direction is an arbitrary unit vector perpendicular
        to the direction of arrival.  This means it couples to only one
        polarisation component per call.

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of effective length vectors in meters
        """
        self._validate_frequency(frequency)
        pos = np.atleast_2d(pos)

        # Direction from antenna toward each position
        r = pos - self._pos
        k_hat = r / np.linalg.norm(r, axis=1, keepdims=True)  # (N, 3)

        # Choose a perpendicular direction per k_hat.
        # Not close to z-axis: cross(z, k) = [-k_y, k_x, 0]
        # Close to z-axis:     cross(x, k) = [0, -k_z, k_y]
        perp = np.where(
            (np.abs(k_hat[:, 2:3]) < 0.9),
            np.stack([-k_hat[:, 1], k_hat[:, 0],
                      np.zeros(len(k_hat))], axis=-1),
            np.stack([np.zeros(len(k_hat)), -
                      k_hat[:, 2], k_hat[:, 1]], axis=-1)
        )

        perp_norm = np.linalg.norm(perp, axis=1, keepdims=True)
        safe_norm = np.where(perp_norm > 1e-10, perp_norm, 1.0)
        l_eff = self._effective_length * perp / safe_norm

        # Fallback for degenerate cases
        degenerate = (perp_norm.ravel() < 1e-10)
        if np.any(degenerate):
            l_eff[degenerate] = np.array([self._effective_length, 0.0, 0.0])

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
        _ = self._validate_frequency(frequency)
        return self._impedance

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
        _, _ = self._validate_angles(theta, phi)
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
