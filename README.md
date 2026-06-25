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

- Python >= 3.10
- Numpy
- Scipy >= 1.12
- pip

### Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/sebj101/CRESSignalStructure.git
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
from CRESSignalStructure import (
    HarmonicTrap, 
    CircularWaveguide, 
    Electron, 
    SpectrumCalculator
)

# Create a harmonic trap with magnetic field parameters
trap = HarmonicTrap(B0=1.0, L0=0.2)  # B0 in Tesla, L0 in m

# Create a circular waveguide (TE11 mode)
waveguide = CircularWaveguide(radius=0.005)  # 5mm radius

# Create an electron with specific energy and pitch angle
particle = Electron(
    ke=18.6e3,      # eV
    startPos=np.array([1e-3, 0.0, 0.0]),  # metres
    pitchAngle=89.9*np.pi/180.0  # radians
)

# Calculate power spectrum
calculator = SpectrumCalculator(trap, waveguide, particle)

# Get peak frequencies for the fundamental (order=0)
freq = calculator.get_peak_frequency(order=0)
print(f"Cyclotron frequency: {freq/1e9:.3f} GHz")

# Get power in the fundamental peak
power = calculator.get_peak_power(order=0)
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

### Spectrum Calculator (for waveguide geometries)

- **SpectrumCalculator**: Class for calculating power spectra in waveguides

### Scattering

Models of both elastic and inelastic scattering are included (currently just for waveguide geometries) for hydrogen and helium via the following classes:
- **BaseCrossSection**: Abstract base class for cross section models
- **InelasticCrossSection**: Rudd model of impact ionisation is used here -- the class is configurable according to the species involved.
- **ElasticCrossSection**: A screened Rutherford model is used for elastic cross sections -- the class is configurable via an inputted Z and A
- **GasModel**: Allows for mixtures of gases to be modelled by inputting cross section models along with the relevant gas number densities
- **ScatteringSimulator**: Generates time series signals with scattering included by calling `SpectrumCalculator` multiple times

### Antennas

The antenna module provides classes for modelling different antenna types:
- **BaseAntenna**: Abstract base class for antenna implementations
- **IsotropicAntenna**: Ideal omnidirectional antenna
- **ShortDipoleAntenna**: Hertzian dipole antenna (length << wavelength)
- **HalfWaveDipoleAntenna**: Half-wave dipole antenna (length ≈ λ/2)
- **HFSSAntenna**: Antenna driven by HFSS simulation exports

#### HFSSAntenna

`HFSSAntenna` loads a radiation pattern and port impedance from CSV files
exported by HFSS.  It is a drop-in replacement for the analytical antenna classes
anywhere an antenna model is required.

**Required HFSS exports**

Three CSV files must be provided:

| File | HFSS report type | Required columns |
|------|------------------|------------------|
| `EFields.csv` | Far Fields – rE components (real/imag) at a single frequency | `Phi[deg]`, `Theta[deg]`, `re(rETheta)[mV]`, `im(rETheta)[mV]`, `re(rEPhi)[mV]`, `im(rEPhi)[mV]` |
| `GainTotal.csv` | Far Fields – GainTotal magnitude at the same frequency | `Phi[deg]`, `Theta[deg]`, `mag(GainTotal)` |
| `ZParameters.csv` | S/Z Parameters – frequency sweep of port impedance | `Freq [GHz]`, `re(Z(1,1)) []`, `im(Z(1,1)) []` |

The E-field and gain exports must cover the full sphere on a **complete rectangular
grid** — every (phi, theta) combination present.  A 1° step (phi: −180° to +180°,
theta: 0° to 180°) is typical.

**Coordinate system**

The HFSS coordinate frame is mapped to the lab (trap) frame via two unit vectors
passed to the constructor:

- `z_ax`: direction of the HFSS +Z axis (bore-sight) in the lab frame
- `x_ax`: direction of the HFSS +X axis (phi = 0 reference) in the lab frame

The Y axis is derived as `y_ax = z_ax × x_ax` (right-handed).

**Usage**

Sample HFSS data for a half-wave dipole antenna is included in the repository.
Use `get_dipole_antenna_paths()` to locate the files from any working directory:

```python
import numpy as np
from CRESSignalStructure import HFSSAntenna, get_dipole_antenna_paths
from CRESSignalStructure.antennas.HFSSDataParser import HFSSDataParser

DATA = get_dipole_antenna_paths()

# Find the resonant frequency from the zero-crossing of Im(Z)
parser = HFSSDataParser()
z_data = parser.parse_impedance(DATA["impedance"])
i = np.where(np.diff(np.sign(z_data.impedance.imag)))[0][0]
frac = -z_data.impedance.imag[i] / (z_data.impedance.imag[i + 1]
                                    - z_data.impedance.imag[i])
f0 = z_data.frequency[i] + frac * (z_data.frequency[i + 1] - z_data.frequency[i])

# Construct the antenna — dipole axis along y, 10 cm from the trap centre
antenna = HFSSAntenna(
    position=np.array([0.1, 0.0, 0.0]),
    z_ax=np.array([0.0, 1.0, 0.0]),   # dipole axis along y
    x_ax=np.array([1.0, 0.0, 0.0]),   # phi = 0 along x
    efield_path=DATA["efield"],
    gain_path=DATA["gain"],
    impedance_path=DATA["impedance"],
    pattern_frequency=f0,
)
```

`get_dipole_antenna_paths()` returns a `dict` with keys `"efield"`, `"gain"`,
and `"impedance"`.  It requires a git clone with `pip install -e .` (editable
install); it raises `FileNotFoundError` if the data directory cannot be found.

For a full comparison between `HFSSAntenna` and `HalfWaveDipoleAntenna`, see
`examples/HFSSvsHalfWaveDipole.py` and
`examples/AntennaSignalGenerator_HFSSvsDipole.ipynb`.

### Trajectories

- **Trajectory**: Container class for electron trajectory data (position, velocity, time)
- **TrajectoryGenerator**: Generates electron trajectories including grad-B drift effects

### Signal Generation

- **SignalGenerator**: Generates time-domain signals from power spectrum calculators (frequency domain approach applicable for waveguides)
- **AntennaSignalGenerator**: Generates signals from electron trajectories using antenna models with Lienard-Wiechert fields
- **ReceiverChain**: Models signal processing chain (downmixing, amplification, digitization)

### Data I/O

- **CRESWriter**: HDF5 file writer for simulation data persistence
- **EnsembleGenerator**: Orchestrates batch generation of CRES simulations with multiprocessing support

## Examples

Example Jupyter notebooks are provided in the repository:

- [GettingStarted.ipynb](examples/GettingStarted.ipynb) - Shows the basic mechanics of generating spectra, both with waveguide geometries and with antennas
- [HarmonicComparison.ipynb](examples/armonicComparison.ipynb) - Comparison of harmonic trap models
- [RealisticFields.ipynb](examples/RealisticFields.ipynb) - Working with realistic magnetic field configurations
- [SignalGenExample.ipynb](examples/SignalGenExample.ipynb) - Generating downmixed and sampled signals
- [DopplerEffectDemo.ipynb](examples/DopplerEffectDemo.ipynb) - Demonstration of the power of the Doppler effect in a long trap
- [AntennaSignalGenerator_HFSSvsDipole.ipynb](examples/AntennaSignalGenerator_HFSSvsDipole.ipynb) - Full signal generation comparison: `HFSSAntenna` vs `HalfWaveDipoleAntenna`
- [MultiAntennaBeamforming.ipynb](examples/MultiAntennaBeamforming.ipynb) - Multi-antenna beamforming with an array of `HalfWaveDipole` antennas

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
│   ├── AntennaSignalGenerator.py       # Antenna-based signal generation
│   ├── BaseField.py                    # Abstract field base class
│   ├── BaseTrap.py                     # Abstract trap base class
│   ├── CircularWaveguide.py            # Waveguide calculations
│   ├── CRESWriter.py                   # HDF5 file I/O
│   ├── EnsembleGenerator.py            # Batch simulation orchestration
│   ├── Particle.py                     # Particle and Electron classes
│   ├── QTNMTraps.py                    # Harmonic and bathtub trap implementations
│   ├── RealFields.py                   # Field implementations
│   ├── ReceiverChain.py                # Signal processing chain
│   ├── SignalGenerator.py              # Frequency-domain signal generation
│   ├── SpectrumCalculator.py           # Waveguide spectrum calculator class
│   ├── TrajectoryGenerator.py          # Trajectory generation with grad-B drift
│   ├── antennas/                       # Antenna models
│   │   ├── __init__.py
│   │   ├── BaseAntenna.py              # Abstract antenna base class
│   │   ├── HFSSAntenna.py              # Class for antennas simulated using HFSS
│   │   ├── HFSSDataParser.py           # Parser class for HFSS field data
│   │   ├── IsotropicAntenna.py         # Isotropic antenna
│   │   └── DipoleAntennas.py           # Dipole antenna implementations
│   └── scattering/                     # Scattering sub-folder
│       ├── __init__.py
│       ├── BaseCrossSection.py         # Abstract cross-section base class
│       ├── CrossSections.py            # Cross-section model implementations
│       ├── GasModel.py                 # Gas mixture code
│       ├── ScatteringSimulator.py      # Simulator for events with scattering
│       └── scattering_utils.py         # Utilities for use with scattering models
├── examples/                           
├── tests/                              # Tests
│   ├── unit/                           # Unit tests
│   └── integration/                    # Integration tests
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
