# CRESSignalStructure

A Python library for modelling CRES (Cyclotron Radiation Emission Spectroscopy) signals in waveguides and from antennas. This package provides tools for simulating electron behavior in magnetic traps, calculating electromagnetic field distributions in waveguides, computing power spectra, and generating time-domain signals from antenna and waveguide systems.

## Overview

CRES is a technique used to measure the energy of electrons via the cyclotron radiation they emit when trapped in a magnetic field. This library models the physics of:

- Electron traps with various magnetic field configurations (analytic and numerical from current loops)
- Waveguide electromagnetic field distributions (TE11 mode in circular waveguides)
- Antenna radiation patterns (isotropic, short dipole, half-wave dipole)
- Electron trajectory generation with grad-B drift effects
- Relativistic particle kinematics
- Power spectrum calculations for CRES signals
- Time-domain signal generation with receiver chain modelling (downmixing, amplification, digitization)

## Features

- **Flexible Trap Models**: Support for both analytic field forms and for fields generated from arbitrary current loops
- **Electromagnetic Field Calculations**: Accurate mode field distributions in waveguides and Lienard-Wiechert fields for antennas
- **Antenna Models**: Isotropic, short dipole, and half-wave dipole antenna implementations
- **Trajectory Generation**: Full electron trajectory simulation including grad-B drift effects
- **Relativistic Particle Physics**: Full relativistic treatment of electron kinematics
- **Power Spectrum Analysis**: Calculate frequency spectra for analytical and numerical traps
- **Time-Domain Signals**: Generate realistic signals including receiver chain effects (downmixing, amplification, digitization)
- **Batch Processing**: Multi-core ensemble generation with HDF5 output
- **Vectorized Operations**: NumPy-based implementations for efficient numerical computations

## Installation

### Prerequisites

- Python >= 3.9
- Numpy
- Scipy >= 1.12
- pip

### Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/CRESSignalStructure.git
cd CRESSignalStructure
pip install -e .
```

### Install with Testing Dependencies

To run tests, install the optional test dependencies:

```bash
pip install -e ".[test]"
```

## Quick Start

### Example 1: Power Spectrum Calculation

Here's a simple example of calculating a CRES power spectrum:

```python
from CRESSignalStructure import HarmonicTrap, CircularWaveguide, Particle, PowerSpectrumCalculator

# Create a harmonic trap with magnetic field parameters
trap = HarmonicTrap(B0=1.0, L0=0.2)  # B0 in Tesla, L0 in m

# Create a circular waveguide (TE11 mode)
waveguide = CircularWaveguide(radius=0.005)  # 5mm radius

# Create an electron with specific energy and pitch angle
particle = Particle(
    mass=9.10938e-31,      # electron mass in kg
    charge=-1.602176e-19,  # electron charge in C
    kinetic_energy=18.6e3, # 18.6 keV
    pitch_angle=1.5        # radians
)

# Calculate power spectrum
calculator = PowerSpectrumCalculator(trap, waveguide, particle)

# Get peak frequencies for the fundamental (order=0)
freq = calculator.GetPeakFrequency(order=0)
print(f"Cyclotron frequency: {freq/1e9:.3f} GHz")

# Get power in the fundamental peak
power = calculator.GetPeakPower(order=0)
print(f"Power: {power*1e15:.3f} fW")
```

### Example 2: Antenna Signal Generation

Generate time-domain signals from electron trajectories:

```python
import numpy as np
from CRESSignalStructure import (
    HarmonicField, Electron, IsotropicAntenna,
    TrajectoryGenerator, ReceiverChain, AntennaSignalGenerator
)

# Create magnetic field
field = HarmonicField(radius=0.05, current=400, background=1.0)

# Create electron with 18.6 keV energy at 89.5 degree pitch angle
electron = Electron(
    ke=18.6e3,
    startPos=np.array([0.01, 0, 0]),
    pitchAngle=89.5 * np.pi / 180
)

# Create antenna
antenna = IsotropicAntenna(position=np.array([0.05, 0, 0]))

# Create receiver chain (200 MHz sampling, 26 GHz LO)
receiver = ReceiverChain(sample_rate=200e6, lo_frequency=26e9)

# Generate trajectory
traj_gen = TrajectoryGenerator(field, electron)
trajectory = traj_gen.generate(sample_rate=5e9, t_max=10e-6)

# Generate downmixed signal
sig_gen = AntennaSignalGenerator(trajectory, antenna, receiver)
time, signal = sig_gen.generate_signal()

print(f"Generated {len(signal)} samples over {time[-1]*1e6:.2f} μs")
```

## Core Components

### Traps

- **BaseTrap**: Abstract base class for electron traps
- **HarmonicTrap**: Models harmonic magnetic field configurations with quadratic spatial dependence
- **BathtubTrap**: Models bathtub-shaped magnetic field profiles with flat bottom regions

### Magnetic Fields

- **BaseField**: Abstract base class for magnetic field calculations
- **HarmonicField**: Harmonic magnetic field implementation
- **BathtubField**: Bathtub magnetic field implementation

### Particles

- **Particle**: General charged particle class with:
  - Mass, charge, kinetic energy, and pitch angle
  - Relativistic calculations (Lorentz factor, velocity, momentum)
  - Position tracking
- **Electron**: Convenience class for electrons with fixed charge and mass

### Waveguide

The **CircularWaveguide** class models TE11 mode electromagnetic fields in circular waveguides:
- Electric and magnetic field distributions
- Phase and group velocities
- Characteristic impedance
- Cutoff frequencies

### Power Spectrum Calculators

- **BaseSpectrumCalculator**: Abstract base class for spectrum calculations
- **PowerSpectrumCalculator**: Analytical calculations for harmonic and bathtub traps
- **NumericalSpectrumCalculator**: Numerical integration approach for arbitrary magnetic field configurations

### Antennas

The antenna module provides classes for modelling different antenna types:
- **BaseAntenna**: Abstract base class for antenna implementations
- **IsotropicAntenna**: Ideal omnidirectional antenna
- **ShortDipoleAntenna**: Hertzian dipole antenna (length << wavelength)
- **HalfWaveDipoleAntenna**: Half-wave dipole antenna (length ≈ λ/2)

### Trajectories

- **Trajectory**: Container class for electron trajectory data (position, velocity, time)
- **TrajectoryGenerator**: Generates electron trajectories including grad-B drift effects

### Signal Generation

- **SignalGenerator**: Generates time-domain signals from power spectrum calculators (frequency domain approach)
- **AntennaSignalGenerator**: Generates signals from electron trajectories using antenna models with Lienard-Wiechert fields
- **ReceiverChain**: Models signal processing chain (downmixing, amplification, digitization)

### Data I/O

- **CRESWriter**: HDF5 file writer for simulation data persistence
- **EnsembleGenerator**: Orchestrates batch generation of CRES simulations with multiprocessing support

## Examples

Example Jupyter notebooks are provided in the repository:

- [TestPowerSpec.ipynb](TestPowerSpec.ipynb) - Basic power spectrum calculations
- [HarmonicComparison.ipynb](HarmonicComparison.ipynb) - Comparison of harmonic trap models
- [RealisticFields.ipynb](RealisticFields.ipynb) - Working with realistic magnetic field configurations
- [SignalGenExample.ipynb](SignalGenExample.ipynb) - Generating downmixed and sampled signals

## Testing

Run the test suite using pytest:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=CRESSignalStructure
```

## Project Structure

```
CRESSignalStructure/
├── CRESSignalStructure/                # Main package directory
│   ├── __init__.py
│   ├── BaseTrap.py                     # Abstract trap base class
│   ├── QTNMTraps.py                    # Harmonic and bathtub trap implementations
│   ├── BaseField.py                    # Abstract field base class
│   ├── RealFields.py                   # Field implementations
│   ├── Particle.py                     # Particle and Electron classes
│   ├── CircularWaveguide.py            # Waveguide calculations
│   ├── BaseSpectrumCalculator.py       # Abstract spectrum calculator base class
│   ├── PowerSpectrumCalculator.py      # Analytical spectrum calculator
│   ├── NumericalSpectrumCalculator.py  # Numerical spectrum calculator
│   ├── TrajectoryGenerator.py          # Trajectory generation with grad-B drift
│   ├── ReceiverChain.py                # Signal processing chain
│   ├── SignalGenerator.py              # Frequency-domain signal generation
│   ├── AntennaSignalGenerator.py       # Antenna-based signal generation
│   ├── CRESWriter.py                   # HDF5 file I/O
│   ├── EnsembleGenerator.py            # Batch simulation orchestration
│   └── antennas/                       # Antenna models
│       ├── __init__.py
│       ├── BaseAntenna.py              # Abstract antenna base class
│       ├── IsotropicAntenna.py         # Isotropic antenna
│       └── DipoleAntennas.py           # Dipole antenna implementations
├── tests/                              # Unit tests
│   ├── unit/                           # Unit tests
│   └── integration/                    # Integration tests
├── *.ipynb                             # Example notebooks
├── pyproject.toml                      # Project configuration
└── README.md                           # This file
```

## Dependencies

- **numpy** (>=1.20): Numerical computations and array operations
- **scipy** (>=1.7): Scientific functions (constants, Bessel functions, integration)
- **h5py**: HDF5 file I/O for simulation data persistence

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or support, please open an issue on the GitHub repository or email seb.jones@ucl.ac.uk.
