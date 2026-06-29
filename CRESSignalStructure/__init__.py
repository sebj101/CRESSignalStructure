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
...     Electron, HarmonicField, IsotropicAntenna,
...     TrajectoryGenerator, ReceiverChain, AntennaSignalGenerator
... )
>>>
>>> # Setup components
>>> field = HarmonicField(radius=0.03, current=400, background=1.0)
>>> electron = Electron(18.6e3, np.array([1e-3, 0, 0]), 89.5*np.pi/180)
>>> antenna = IsotropicAntenna(position=np.array([0.05, 0, 0]))
>>> receiver = ReceiverChain(sample_rate=1e9, lo_frequency=26.8e9)
>>>
>>> # Generate trajectory
>>> traj_gen = TrajectoryGenerator(field, electron)
>>> trajectory = traj_gen.generate(sample_rate=10e9, t_max=10e-6)
>>>
>>> # Generate signal
>>> sig_gen = AntennaSignalGenerator(trajectory, antenna, receiver)
>>> time, signal = sig_gen.generate_signal()

Package Organization
--------------------
Particles:
    Particle - Particle physics and relativistic kinematics
    Electron - Convenience subclass of Particle with fixed electron mass/charge

Magnetic Fields:
    HarmonicField - Harmonic magnetic field (single coil)
    BathtubField - Bathtub magnetic field (two coils)

Traps:
    HarmonicTrap - Harmonic trap configuration
    BathtubTrap - Bathtub trap configuration

Antennas:
    IsotropicAntenna - Ideal isotropic antenna (omnidirectional)
    ShortDipoleAntenna - Short dipole antenna (length << wavelength)
    HalfWaveDipoleAntenna - Half-wave dipole antenna (length ≈ λ/2)
    HFSSAntenna - Antenna model driven by HFSS simulation exports
    HFSSDataParser - Parser for HFSS far-field CSV exports

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
    SpectrumCalculator - Unified spectrum calculator (analytical and numerical)

Scattering:
    InelasticCrossSection - Inelastic electron scattering cross-section
    ElasticCrossSection - Elastic electron scattering cross-section
    GasModel - Gas mixture model for scattering simulations
    ScatteringSimulator - Simulator for events with scattering
    ScatteringResult - Container for scattering simulation results
    scatter_to_pitch_angle - Utility for computing post-scatter pitch angles
"""

# Particle
from .Particle import Particle, Electron

# Magnetic fields
from .RealFields import HarmonicField, BathtubField

# Traps
from .QTNMTraps import HarmonicTrap, BathtubTrap

# Antennas
from .antennas import (
    IsotropicAntenna, ShortDipoleAntenna, 
    HalfWaveDipoleAntenna, HFSSAntenna, HFSSDataParser
)

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
from .SpectrumCalculator import SpectrumCalculator

# Datasets (sample data paths)
from .datasets import get_dipole_antenna_paths

# Scattering
from .scattering import (BaseCrossSection, InelasticCrossSection,
                         ElasticCrossSection, GasModel, ScatteringSimulator,
                         ScatteringResult, scatter_to_pitch_angle)

# Define public API
__all__ = [
    # Particle
    'Particle',
    'Electron',

    # Fields
    'HarmonicField',
    'BathtubField',

    # Traps
    'HarmonicTrap',
    'BathtubTrap',

    # Antennas
    'IsotropicAntenna',
    'ShortDipoleAntenna',
    'HalfWaveDipoleAntenna',
    'HFSSAntenna',
    'HFSSDataParser',

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
    'SpectrumCalculator',

    # Scattering
    'BaseCrossSection',
    'InelasticCrossSection',
    'ElasticCrossSection',
    'GasModel',
    'ScatteringSimulator',
    'ScatteringResult',
    'scatter_to_pitch_angle',

    # Datasets
    'get_dipole_antenna_paths',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Seb Jones'

# Library-level logging: do not add any handlers other than NullHandler.
# Applications using this library are responsible for configuring logging.
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
