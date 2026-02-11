"""
Antenna Models

This subpackage provides antenna models for CRES signal detection,
including abstract base classes and concrete implementations for
various antenna types.

Available Antennas
------------------
BaseAntenna : Abstract base class for all antennas
IsotropicAntenna : Ideal omnidirectional antenna with uniform gain
ShortDipoleAntenna : Short dipole antenna (length << wavelength)
HalfWaveDipoleAntenna : Half-wave dipole antenna (length ≈ λ/2)
"""

from .BaseAntenna import BaseAntenna
from .IsotropicAntenna import IsotropicAntenna
from .DipoleAntennas import ShortDipoleAntenna, HalfWaveDipoleAntenna

__all__ = [
    'BaseAntenna',
    'IsotropicAntenna',
    'ShortDipoleAntenna',
    'HalfWaveDipoleAntenna',
]
