"""
DipoleAntennas.py

Implementations of dipole antenna models for CRES signal detection.

This module provides concrete implementations of the BaseAntenna class
for short dipole and half-wave dipole antennas.
"""

import logging
import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike, NDArray
from .BaseAntenna import BaseAntenna

logger = logging.getLogger(__name__)


class ShortDipoleAntenna(BaseAntenna):
    """
    Short dipole antenna model (length << wavelength)

    A short dipole (also called Hertzian dipole) is an idealized antenna
    where the current distribution is approximately uniform along its length.
    This approximation is valid when l << λ (typically l < λ/10).

    The radiation pattern is approximately sin(θ) where θ is the angle
    from the antenna axis.
    """

    def __init__(self, position: NDArray, orientation: ArrayLike,
                 length: float, resistance: float = 1.0):
        """
        Constructor for ShortDipoleAntenna

        Parameters
        ----------
        position : ArrayLike
            3-vector position in meters [x, y, z]
        orientation : ArrayLike
            3-vector direction of dipole axis (will be normalized)
        length : float
            Physical length in meters (should be << wavelength)
        resistance : float
            Radiation resistance in Ohms (default 1.0 Ω)
            For a short dipole: R_rad ≈ 20π²(l/λ)² Ω

        Raises
        ------
        TypeError
            If parameters have incorrect types
        ValueError
            If parameters have invalid values
        """
        # Validate orientation first so we can derive x_ax for the base class
        orientation_hat = self._validate_direction(orientation)

        # x_ax is arbitrary for a dipole (azimuthally symmetric)
        ref = np.array([0.0, 0.0, 1.0]) if abs(
            orientation_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_ax = np.cross(orientation_hat, ref)
        x_ax /= np.linalg.norm(x_ax)

        super().__init__(position, orientation_hat, x_ax)

        if not isinstance(length, (int, float)):
            raise TypeError("Length must be a number")
        if length <= 0:
            raise ValueError("Length must be positive")
        if not np.isfinite(length):
            raise ValueError("Length must be finite")
        self._length = float(length)

        if not isinstance(resistance, (int, float)):
            raise TypeError("Resistance must be a number")
        if resistance < 0:
            raise ValueError("Resistance must be non-negative")
        if not np.isfinite(resistance):
            raise ValueError("Resistance must be finite")
        self._resistance = float(resistance)

        logger.info("Created ShortDipoleAntenna at pos=%s, orientation=%s, "
                    "length=%.4e m, resistance=%.4e Ohm",
                    self._pos, self._z_ax, self._length, self._resistance)

    def GetETheta(self, pos: NDArray) -> NDArray:
        """
        Get the theta component of the short dipole radiation pattern

        For a short dipole the far-field pattern is E_θ ∝ sin(θ).  Using the
        identity sin(θ)·θ̂ = cos(θ)·r̂ − ẑ avoids any division by sin(θ) and
        is well-defined at the poles.

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of E_theta vectors

        Raises
        ------
        ValueError
            If any position equals the antenna position
        """
        pos = np.atleast_2d(pos)
        r = pos - self._pos   # (N, 3)
        r_mag = np.linalg.norm(r, axis=1, keepdims=True)

        if np.any(r_mag == 0.0):
            raise ValueError("Position cannot equal antenna position")

        r_hat = r / r_mag
        cos_theta = np.dot(r_hat, self._z_ax)
        E_theta = cos_theta[:, np.newaxis] * r_hat - self._z_ax

        return E_theta

    def GetEPhi(self, pos: NDArray) -> NDArray:
        """
        Get the phi component of the short dipole radiation pattern

        A short dipole has no phi component (azimuthally symmetric).

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of zeros
        """
        return np.zeros((np.atleast_2d(pos).shape[0], 3))

    def GetEffectiveLength(self, frequency: float, pos: NDArray) -> NDArray:
        """
        Get the effective length vector of the short dipole

        Derived from the radiation pattern via reciprocity:
        l_eff = -length * E_theta(pos)

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
        return -self._length * self.GetETheta(pos)

    def GetImpedance(self, frequency: float) -> complex:
        """
        Get the antenna impedance for a short dipole

        For a short dipole (l << λ), the impedance consists of:
        - Radiation resistance: R_rad ≈ 20π²(l/λ)²
        - Capacitive reactance: X_c ≈ -120(ln(l/a) - 1) / (kl)

        where k = 2π/λ is the wave number and a is the wire radius
        (approximated as l/100 here).

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

        wavelength = sc.c / frequency
        k = 2 * np.pi / wavelength

        # Radiation resistance (small for short dipoles)
        R_rad = 20 * np.pi**2 * (self._length / wavelength)**2

        # Use provided resistance or calculated value
        R = max(self._resistance, R_rad)

        # Capacitive reactance (negative, dominates for short dipoles)
        # Assume wire radius a ≈ l/100
        a = self._length / 100
        X_c = -120 * (np.log(self._length / a) - 1) / (k * self._length)

        return R + 1j * X_c

    def GetGain(self, theta: float, phi: float) -> float:
        """
        Get the antenna gain pattern for a short dipole

        The short dipole has a sin²(θ_d) pattern where θ_d is the angle
        between the dipole axis and the direction to the observation point.

        Parameters
        ----------
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        float
            Dimensionless gain (maximum gain is 1.5 for short dipole)
        """
        theta, phi = self._validate_angles(theta, phi)

        # Direction to observation point
        r_hat = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Angle between dipole axis and observation direction
        cos_theta_d = np.dot(self._z_ax, r_hat)
        sin_theta_d = np.sqrt(1 - cos_theta_d**2)

        # Short dipole gain pattern: G(θ) = 1.5 * sin²(θ_d)
        gain = 1.5 * sin_theta_d**2

        return gain


class HalfWaveDipoleAntenna(BaseAntenna):
    """
    Half-wave dipole antenna model (length ≈ λ/2)

    A half-wave dipole is a resonant antenna with length approximately
    equal to half the wavelength. It has a sinusoidal current distribution
    and is one of the most common antenna types.

    The radiation pattern is given by:
    F(θ) = cos((π/2)cos(θ)) / sin(θ)

    At resonance, the input impedance is approximately 73 + j42.5 Ω.
    """

    def __init__(self, position: NDArray, orientation: ArrayLike,
                 resonant_frequency: float, wire_radius: float = 0.0):
        """
        Constructor for HalfWaveDipoleAntenna

        Parameters
        ----------
        position : ArrayLike
            3-vector position in meters [x, y, z]
        orientation : ArrayLike
            3-vector direction of dipole axis (will be normalized)
        resonant_frequency : float
            Resonant frequency in Hz (sets length to λ/2)
        wire_radius : float, optional
            Radius of the dipole wire in meters
            If None, defaults to λ/1000

        Raises
        ------
        TypeError
            If parameters have incorrect types
        ValueError
            If parameters have invalid values
        """
        # Validate orientation first so we can derive x_ax for the base class
        orientation_hat = self._validate_direction(orientation)

        # x_ax is arbitrary for a dipole (azimuthally symmetric)
        ref = np.array([0.0, 0.0, 1.0]) if abs(
            orientation_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_ax = np.cross(orientation_hat, ref)
        x_ax /= np.linalg.norm(x_ax)

        super().__init__(position, orientation_hat, x_ax)

        if not isinstance(resonant_frequency, (int, float)):
            raise TypeError("Resonant frequency must be a number")
        if resonant_frequency <= 0:
            raise ValueError("Resonant frequency must be positive")
        if not np.isfinite(resonant_frequency):
            raise ValueError("Resonant frequency must be finite")
        self._f0 = float(resonant_frequency)

        # Calculate length at resonance
        self._lambda0 = sc.c / self._f0
        self._length = self._lambda0 / 2

        # Set wire radius
        if wire_radius == 0.0:
            self._wire_radius = self._lambda0 / 1000
        else:
            if not isinstance(wire_radius, (int, float)):
                raise TypeError("Wire radius must be a number")
            if wire_radius < 0:
                raise ValueError("Wire radius must be positive")
            if not np.isfinite(wire_radius):
                raise ValueError("Wire radius must be finite")
            self._wire_radius = float(wire_radius)

        logger.info("Created HalfWaveDipoleAntenna at pos=%s, orientation=%s, "
                    "resonant_frequency=%.5e Hz, length=%.4e m",
                    self._pos, self._z_ax, self._f0, self._length)

    def GetETheta(self, pos: NDArray) -> NDArray:
        """
        Get the theta component of the half-wave dipole radiation pattern

        The half-wave dipole pattern is F(θ) = cos((π/2)cos(θ)) / sin(θ).
        Letting v = cos(θ)·r̂ − ẑ  (so that |v| = sin(θ) and v/|v| = θ̂),
        the full vector pattern is:
            F(θ)·θ̂ = cos((π/2)cos(θ)) · v / |v|²
        The result is zeroed where sin(θ) ≈ 0; the field is zero at the
        poles so this is exact.

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of E_theta vectors

        Raises
        ------
        ValueError
            If any position equals the antenna position
        """
        pos = np.atleast_2d(pos)
        r = pos - self._pos                                          # (N, 3)
        r_mag = np.linalg.norm(r, axis=1, keepdims=True)            # (N, 1)

        if np.any(r_mag == 0.0):
            raise ValueError("Position cannot equal antenna position")

        r_hat = r / r_mag                                            # (N, 3)
        cos_theta = np.dot(r_hat, self._z_ax)                       # (N,)

        v = cos_theta[:, np.newaxis] * r_hat - self._z_ax            # (N, 3)
        sin_theta_sq = np.sum(v * v, axis=1)                         # (N,)

        # cos((π/2)cos(θ)) / sin²(θ), zeroed at the poles.
        # safe_sin_sq avoids the divide-by-zero that np.where would otherwise
        # evaluate before selecting.
        safe_sin_sq = np.where(sin_theta_sq > 1e-20, sin_theta_sq, 1.0)
        pattern = np.where(sin_theta_sq > 1e-20,
                           np.cos(0.5 * np.pi * cos_theta) / safe_sin_sq,
                           0.0)                                      # (N,)

        return pattern[:, np.newaxis] * v                            # (N, 3)

    def GetEPhi(self, pos: NDArray) -> NDArray:
        """
        Get the phi component of the half-wave dipole radiation pattern

        A half-wave dipole has no phi component (azimuthally symmetric).

        Parameters
        ----------
        pos : NDArray
            Array of N position 3-vectors (shape (N,3)) in metres

        Returns
        -------
        NDArray
            (N, 3) array of zeros
        """
        return np.zeros((np.atleast_2d(pos).shape[0], 3))

    def GetEffectiveLength(self, frequency: float, pos: NDArray) -> NDArray:
        """
        Get the effective length vector of the half-wave dipole

        Derived from the radiation pattern via reciprocity:
        l_eff = -(λ/π) * E_theta(pos)

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
        wavelength = sc.c / frequency
        return -(wavelength / np.pi) * self.GetETheta(pos)

    def GetImpedance(self, frequency: float) -> complex:
        """
        Get the antenna impedance for a half-wave dipole

        Near resonance, the impedance is approximately:
        Z ≈ 73 + j42.5 Ω at f = f0
        Z ≈ 73 + j*X(f) away from resonance

        where X(f) accounts for detuning from resonance.

        Parameters
        ----------
        frequency : float
            Frequency in Hz

        Returns
        -------
        complex
            Antenna impedance in Ohms (resistance + j*reactance)

        Notes
        -----
        At resonance (f = f0): Z ≈ 73 + j42.5 Ω
        Below resonance: capacitive (negative reactance)
        Above resonance: inductive (positive reactance)
        """
        frequency = self._validate_frequency(frequency)

        # Radiation resistance (approximately constant near resonance)
        R_rad = 73.0

        # Calculate reactance based on detuning
        # For a center-fed dipole, reactance varies approximately linearly
        # with (f - f0) / f0 near resonance
        delta_f = (frequency - self._f0) / self._f0

        # At resonance, inductive reactance is +42.5 Ω
        # Reactance changes by ~100 Ω per 10% frequency change
        X = 42.5 + 1000 * delta_f

        return R_rad + 1j * X

    def GetGain(self, theta: float, phi: float) -> float:
        """
        Get the antenna gain pattern for a half-wave dipole

        The half-wave dipole has a directional pattern given by:
        G(θ) = 1.643 * [cos((π/2)cos(θ_d)) / sin(θ_d)]²

        where θ_d is the angle from the dipole axis and 1.643 is the
        maximum directivity.

        Parameters
        ----------
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        float
            Dimensionless gain (maximum gain is ~1.643)
        """
        theta, phi = self._validate_angles(theta, phi)

        # Direction to observation point
        r_hat = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Angle between dipole axis and observation direction
        cos_theta_d = np.dot(self._z_ax, r_hat)
        sin_theta_d = np.sqrt(1 - cos_theta_d**2)

        # Avoid division by zero at the poles
        if sin_theta_d < 1e-10:
            return 0.0

        # Half-wave dipole pattern
        numerator = np.cos((np.pi / 2) * cos_theta_d)
        pattern = numerator / sin_theta_d

        # Maximum directivity for half-wave dipole
        gain = 1.643 * pattern**2

        return gain

    def GetResonantFrequency(self) -> float:
        """
        Get the resonant frequency of the dipole

        Returns
        -------
        float
            Resonant frequency in Hz
        """
        return self._f0

    def GetLength(self) -> float:
        """
        Get the physical length of the dipole

        Returns
        -------
        float
            Length in meters (λ/2 at resonance)
        """
        return self._length
