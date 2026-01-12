"""
DipoleAntennas.py

Implementations of dipole antenna models for CRES signal detection.

This module provides concrete implementations of the BaseAntenna class
for short dipole and half-wave dipole antennas.
"""

import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike, NDArray
from CRESSignalStructure.BaseAntenna import BaseAntenna


class ShortDipoleAntenna(BaseAntenna):
    """
    Short dipole antenna model (length << wavelength)

    A short dipole (also called Hertzian dipole) is an idealized antenna
    where the current distribution is approximately uniform along its length.
    This approximation is valid when l << λ (typically l < λ/10).

    The radiation pattern is approximately sin(θ) where θ is the angle
    from the antenna axis.
    """

    def __init__(self, position: ArrayLike, orientation: ArrayLike,
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
        # Validate inputs
        self._position = self._validate_position(position)
        self._orientation = self._validate_direction(orientation)

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

    def GetEffectiveLength(self, frequency: float, theta: float, phi: float) -> NDArray:
        """
        Get the effective length vector of the short dipole

        For a short dipole, the effective length is approximately equal to
        the physical length in the direction of the antenna, projected
        perpendicular to the direction of propagation.

        Parameters
        ----------
        frequency : float
            Frequency in Hz (not used for short dipole approximation)
        theta : float
            Polar angle in radians (measured from propagation direction)
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        NDArray
            3-vector effective length in meters

        Notes
        -----
        The effective length is given by:
        l_eff = l * (d̂ - (d̂·k̂)k̂) * sin(θ_d)
        where d̂ is the dipole direction, k̂ is the propagation direction,
        and θ_d is the angle between them.
        """
        frequency = self._validate_frequency(frequency)
        theta, phi = self._validate_angles(theta, phi)

        # Direction of incoming wave (k-vector direction)
        k_hat = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Project dipole direction perpendicular to propagation direction
        # E_effective = E - (E·k̂)k̂  (only transverse E-field components matter)
        projection = self._orientation - np.dot(self._orientation, k_hat) * k_hat

        # Effective length is physical length times projection
        l_eff = self._length * projection

        return l_eff

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

    def GetPosition(self) -> NDArray:
        """
        Get the antenna position

        Returns
        -------
        NDArray
            3-vector position in meters
        """
        return self._position.copy()

    def GetOrientation(self) -> NDArray:
        """
        Get the antenna orientation

        Returns
        -------
        NDArray
            3-vector unit direction along dipole axis
        """
        return self._orientation.copy()

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
        cos_theta_d = np.dot(self._orientation, r_hat)
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

    def __init__(self, position: ArrayLike, orientation: ArrayLike,
                 resonant_frequency: float, wire_radius: float = None):
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
        # Validate inputs
        self._position = self._validate_position(position)
        self._orientation = self._validate_direction(orientation)

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
        if wire_radius is None:
            self._wire_radius = self._lambda0 / 1000
        else:
            if not isinstance(wire_radius, (int, float)):
                raise TypeError("Wire radius must be a number")
            if wire_radius <= 0:
                raise ValueError("Wire radius must be positive")
            if not np.isfinite(wire_radius):
                raise ValueError("Wire radius must be finite")
            self._wire_radius = float(wire_radius)

    def GetEffectiveLength(self, frequency: float, theta: float, phi: float) -> NDArray:
        """
        Get the effective length vector of the half-wave dipole

        For a half-wave dipole with sinusoidal current distribution,
        the effective length in the direction of the antenna is:
        l_eff = λ/π ≈ 0.318λ

        This is then projected perpendicular to the propagation direction.

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        NDArray
            3-vector effective length in meters
        """
        frequency = self._validate_frequency(frequency)
        theta, phi = self._validate_angles(theta, phi)

        wavelength = sc.c / frequency

        # Effective length for half-wave dipole
        l_eff_magnitude = wavelength / np.pi

        # Direction of incoming wave
        k_hat = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Project perpendicular to propagation direction
        projection = self._orientation - np.dot(self._orientation, k_hat) * k_hat
        norm = np.linalg.norm(projection)

        if norm < 1e-10:
            # Wave propagating along dipole axis - no coupling
            return np.zeros(3)

        # Normalize and scale
        l_eff = l_eff_magnitude * projection / norm * norm

        return l_eff

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

    def GetPosition(self) -> NDArray:
        """
        Get the antenna position

        Returns
        -------
        NDArray
            3-vector position in meters
        """
        return self._position.copy()

    def GetOrientation(self) -> NDArray:
        """
        Get the antenna orientation

        Returns
        -------
        NDArray
            3-vector unit direction along dipole axis
        """
        return self._orientation.copy()

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
        cos_theta_d = np.dot(self._orientation, r_hat)
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
