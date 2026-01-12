# CRESSignalStructure

A Python library for modeling CRES (Cyclotron Radiation Emission Spectroscopy) signals, currently just in waveguides. This package provides tools for simulating electron behavior in magnetic traps, calculating electromagnetic field distributions in waveguides, and computing the resulting power spectra from combining the above.

## Overview

CRES is a technique used to measure the energy of electrons through the cyclotron radiation they emit when trapped in a magnetic field. This library models the physics of:

- Electron traps with various magnetic field configurations made up of a number of current loops
- Waveguide electromagnetic field distributions (currently just the TE11 mode of a circular waveguide but with further expansions planned)
- Relativistic particle kinematics
- Power spectrum calculations for CRES signals

## Features

- **Flexible Trap Models**: Support for both analytic field forms and for fields generated from arbitrary current loops
- **Electromagnetic Field Calculations**: Accurate mode field distributions in waveguides
- **Relativistic Particle Physics**: Full relativistic treatment of electron kinematics
- **Power Spectrum Analysis**: Calculate frequency spectra for analytical and numerical traps
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
print(f"Power: {power_plus*1e15:.3f} fW")
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

### Particle

The **Particle** class represents charged particles with:
- Mass, charge, kinetic energy, and pitch angle
- Relativistic calculations (Lorentz factor, velocity, momentum)
- Position tracking

### Waveguide

The **CircularWaveguide** class models TE11 mode electromagnetic fields in circular waveguides:
- Electric and magnetic field distributions
- Phase and group velocities
- Characteristic impedance
- Cutoff frequencies

### Power Spectrum Calculators

- **PowerSpectrumCalculator**: Analytical calculations for harmonic and bathtub traps
- **NumericalSpectrumCalculator**: Numerical integration approach for arbitrary magnetic field configurations

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
├── CRESSignalStructure/           # Main package directory
|   ├── __init__.py
│   ├── BaseTrap.py                     # Abstract trap base class
│   ├── QTNMTraps.py                    # Harmonic and bathtub trap implementations
│   ├── BaseField.py                    # Abstract field base class
│   ├── RealFields.py                   # Field implementations
│   ├── Particle.py                     # Particle physics and kinematics
│   ├── CircularWaveguide.py            # Waveguide calculations
|   ├── BaseSpectrumCalculator.py       # Abstract spectrum calculator base class
│   ├── PowerSpectrumCalculator.py      # Analytical spectrum calculator
│   └── NumericalSpectrumCalculator.py  # Numerical spectrum calculator
├── tests/                         # Unit tests
├── *.ipynb                        # Example notebooks
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Dependencies

- **numpy** (>=1.20): Numerical computations and array operations
- **scipy** (>=1.7): Scientific functions (constants, Bessel functions, integration)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or support, please open an issue on the GitHub repository or email seb.jones@ucl.ac.uk.
