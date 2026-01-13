# Package Import Guide

The `CRESSignalStructure` package now provides clean, organized imports through `__init__.py`.

## Before (Old Way)

```python
# Required explicit submodule imports
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.IsotropicAntenna import IsotropicAntenna
from CRESSignalStructure.TrajectoryGenerator import TrajectoryGenerator, Trajectory
from CRESSignalStructure.ReceiverChain import ReceiverChain
from CRESSignalStructure.AntennaSignalGenerator import AntennaSignalGenerator
```

## After (New Way)

```python
# Clean imports from package level
from CRESSignalStructure import (
    Particle,
    HarmonicField,
    IsotropicAntenna,
    TrajectoryGenerator,
    Trajectory,
    ReceiverChain,
    AntennaSignalGenerator,
)
```

## Available Exports

### Core Physics
- `Particle` - Particle physics and relativistic kinematics

### Magnetic Fields
- `BaseField` - Abstract base class
- `HarmonicField` - Single coil harmonic field
- `BathtubField` - Two-coil bathtub field

### Electron Traps
- `BaseTrap` - Abstract base class
- `HarmonicTrap` - Harmonic trap configuration
- `BathtubTrap` - Bathtub trap configuration

### Antennas
- `BaseAntenna` - Abstract base class
- `IsotropicAntenna` - Omnidirectional antenna
- `ShortDipoleAntenna` - Short dipole (l << λ)
- `HalfWaveDipoleAntenna` - Half-wave dipole (l ≈ λ/2)

### Trajectories
- `Trajectory` - Trajectory data container
- `TrajectoryGenerator` - Generate electron trajectories with grad-B drift

### Waveguides
- `CircularWaveguide` - TE11 mode circular waveguide

### Receiver Chain
- `ReceiverChain` - Downmixing and digitization

### Signal Generation
- `AntennaSignalGenerator` - Generate signals from trajectories
- `SignalGenerator` - Generate signals from spectrum calculators

### Spectrum Calculators
- `BaseSpectrumCalculator` - Abstract base class
- `PowerSpectrumCalculator` - Analytical spectrum calculator
- `NumericalSpectrumCalculator` - Numerical spectrum calculator

## Complete Example

```python
import numpy as np
from CRESSignalStructure import (
    Particle,
    HarmonicField,
    IsotropicAntenna,
    TrajectoryGenerator,
    ReceiverChain,
    AntennaSignalGenerator,
)

# Setup magnetic field
field = HarmonicField(R_COIL=0.03, I_COIL=400, B_BKG=1.0)

# Setup particle
particle = Particle(
    kinetic_energy=18.6e3,  # eV
    position=np.array([0.01, 0, 0]),  # meters
    pitch_angle=89.5 * np.pi / 180  # radians
)

# Setup antenna
antenna = IsotropicAntenna(position=np.array([0.02, 0, 0]))

# Setup receiver
receiver = ReceiverChain(
    sample_rate=200e6,  # 200 MHz
    lo_frequency=26e9,  # 26 GHz
    receiver_gain=1e6   # Linear gain
)

# Generate trajectory
traj_gen = TrajectoryGenerator(field, particle)
trajectory = traj_gen.generate(sample_rate=5e9, t_max=10e-6)

# Generate signal
sig_gen = AntennaSignalGenerator(trajectory, antenna, receiver)
time, signal = sig_gen.generate_signal()

print(f"Generated {len(signal)} samples")
print(f"Duration: {time[-1]*1e6:.2f} μs")
```

## Package Metadata

```python
import CRESSignalStructure

print(CRESSignalStructure.__version__)  # '0.1.0'
print(CRESSignalStructure.__author__)   # 'Seb Jones'
print(len(CRESSignalStructure.__all__))  # 20 exported symbols
```

## Documentation

For detailed documentation on each class, use Python's built-in help:

```python
from CRESSignalStructure import TrajectoryGenerator
help(TrajectoryGenerator)
```

Or access the package-level documentation:

```python
import CRESSignalStructure
help(CRESSignalStructure)
```
