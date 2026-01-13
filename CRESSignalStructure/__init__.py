"""
CRESSignalStructure

A Python library for modeling CRES (Cyclotron Radiation Emission Spectroscopy)
signals in quantum mechanics experiments.

This package provides classes for modeling electron traps, waveguides, antennas,
particles, and calculating power spectra and time-domain signals.

Basic Usage
-----------
>>> import numpy as np
>>> from CRESSignalStructure import (
...     Particle, HarmonicField, IsotropicAntenna,
...     TrajectoryGenerator, ReceiverChain, AntennaSignalGenerator
... )
>>>
>>> # Setup components
>>> field = HarmonicField(R_COIL=0.03, I_COIL=400, B_BKG=1.0)
>>> particle = Particle(18.6e3, np.array([0.01, 0, 0]), 89.5*np.pi/180)
>>> antenna = IsotropicAntenna(position=np.array([0.02, 0, 0]))
>>> receiver = ReceiverChain(sample_rate=200e6, lo_frequency=26e9)
>>>
>>> # Generate trajectory
>>> traj_gen = TrajectoryGenerator(field, particle)
>>> trajectory = traj_gen.generate(sample_rate=5e9, t_max=10e-6)
>>>
>>> # Generate signal
>>> sig_gen = AntennaSignalGenerator(trajectory, antenna, receiver)
>>> time, signal = sig_gen.generate_signal()

Package Organization
--------------------
Particles:
    Particle - Particle physics and relativistic kinematics

Magnetic Fields:
    BaseField - Abstract base class for magnetic fields
    HarmonicField - Harmonic magnetic field (single coil)
    BathtubField - Bathtub magnetic field (two coils)

Traps:
    BaseTrap - Abstract base class for electron traps
    HarmonicTrap - Harmonic trap configuration
    BathtubTrap - Bathtub trap configuration

Antennas:
    BaseAntenna - Abstract base class for antennas
    IsotropicAntenna - Ideal isotropic antenna (omnidirectional)
    ShortDipoleAntenna - Short dipole antenna (length << wavelength)
    HalfWaveDipoleAntenna - Half-wave dipole antenna (length ≈ λ/2)

Trajectories:
    Trajectory - Container for trajectory data
    TrajectoryGenerator - Generate electron trajectories with grad-B drift

Waveguides:
    CircularWaveguide - TE11 mode in circular waveguide

Receivers:
    ReceiverChain - Downmixing and digitization chain

Signal Generation:
    AntennaSignalGenerator - Generate signals from trajectories using antenna models
    SignalGenerator - Generate signals from spectrum calculators (frequency domain)

Spectrum Calculators:
    BaseSpectrumCalculator - Abstract base for spectrum calculations
    PowerSpectrumCalculator - Analytical power spectrum calculator
    NumericalSpectrumCalculator - Numerical power spectrum calculator
"""

# Particle
from .Particle import Particle

# Magnetic fields
from .BaseField import BaseField
from .RealFields import HarmonicField, BathtubField

# Traps
from .BaseTrap import BaseTrap
from .QTNMTraps import HarmonicTrap, BathtubTrap

# Antennas
from .antennas import BaseAntenna, IsotropicAntenna, ShortDipoleAntenna, HalfWaveDipoleAntenna

# Trajectories
from .TrajectoryGenerator import Trajectory, TrajectoryGenerator

# Waveguides
from .CircularWaveguide import CircularWaveguide

# Receivers
from .ReceiverChain import ReceiverChain

# Signal generation
from .AntennaSignalGenerator import AntennaSignalGenerator
from .SignalGenerator import SignalGenerator

# Spectrum calculators
from .BaseSpectrumCalculator import BaseSpectrumCalculator
from .PowerSpectrumCalculator import PowerSpectrumCalculator
from .NumericalSpectrumCalculator import NumericalSpectrumCalculator

# Define public API
__all__ = [
    # Particle
    'Particle',

    # Fields
    'BaseField',
    'HarmonicField',
    'BathtubField',

    # Traps
    'BaseTrap',
    'HarmonicTrap',
    'BathtubTrap',

    # Antennas
    'BaseAntenna',
    'IsotropicAntenna',
    'ShortDipoleAntenna',
    'HalfWaveDipoleAntenna',

    # Trajectories
    'Trajectory',
    'TrajectoryGenerator',

    # Waveguides
    'CircularWaveguide',

    # Receivers
    'ReceiverChain',

    # Signal generation
    'AntennaSignalGenerator',
    'SignalGenerator',

    # Spectrum calculators
    'BaseSpectrumCalculator',
    'PowerSpectrumCalculator',
    'NumericalSpectrumCalculator',
]

# Version info
__version__ = '0.1.0'
__author__ = 'Seb Jones'
