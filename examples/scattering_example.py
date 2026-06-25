"""
scattering_example.py

Simulates a CRES signal with gas scattering for an 18.6 keV electron
in a HarmonicField trap. Uses real cross-section models (BEB inelastic
and screened Rutherford elastic) for an 80/20 mixture of atomic tritium
and helium-4 at a combined number density of 1e12 cm^-3.

The BathtubField (single coil at z=0 plus uniform background) creates a trapping 
minimum between the two coils. The trap depth is ~4 mT, trapping electrons with 
pitch angles above ~86.5 deg.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
import scipy.constants as sc

from CRESSignalStructure import (
    Particle, BathtubField, CircularWaveguide,
    InelasticCrossSection, ElasticCrossSection, GasModel, ScatteringSimulator
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Physical setup ---

# Harmonic trapping field: coil (R=3cm, trap depth = 4mT) + 1T background
# On-axis field at centre ~ mu_0*I/(2R) = 4 mT above the 1T background
COIL_RADIUS = 3e-2                                  # metres
TRAP_DEPTH = 4e-3                                   # Tesla
CURRENT = 2 * COIL_RADIUS * TRAP_DEPTH / sc.mu_0    # Amperes
TRAP_LENGTH = 15e-2                                 # metres
field = BathtubField(radius=COIL_RADIUS, current=CURRENT, Z1=-TRAP_LENGTH/2,
                     Z2=TRAP_LENGTH/2, background=np.array([0.0, 0.0, 1.0]))

# Circular waveguide: R=5mm gives TE11 cutoff ~ 16 GHz,
# well below the ~27 GHz cyclotron frequency
waveguide = CircularWaveguide(radius=5e-3)

# 18.6 keV electron at 89.5 deg pitch angle with small radial offset
electron = Particle(18.6e3, np.array([0.001, 0.0, 0.0]),
                    pitchAngle=np.radians(89.5))

f0 = field.calc_omega_0(electron) / (2 * np.pi)

# Combined number density: 2e12 cm^-3 = 2e18 m^-3, split 80/20 between T and He
n_total = 1e12 * 1e6   # m^-3
n_T = 0.8 * n_total
n_He = 0.2 * n_total

# Atomic tritium (Z=1, A=3.016): inelastic uses "H" parameters (same electron structure)
T_inelastic = InelasticCrossSection("H")
T_elastic = ElasticCrossSection(Z=1, A=3.016)

# Helium-4 (Z=2, A=4.003)
He_inelastic = InelasticCrossSection("He")
He_elastic = ElasticCrossSection(Z=2, A=4.003)

gas = GasModel([
    (T_inelastic, n_T),
    (T_elastic, n_T),
    (He_inelastic, n_He),
    (He_elastic, n_He),
])

# --- Simulator ---

sample_rate = 1e9       # 500 MHz digitiser
lo_freq = 26.7e9       # LO frequency, giving IF ~ 40 MHz
max_event_time = 1e-3   # 1 ms total event duration

simulator = ScatteringSimulator(
    trap=field, waveguide=waveguide, gas_model=gas,
    sample_rate=sample_rate, lo_freq=lo_freq,
    max_event_time=max_event_time
)

# --- Run ---

logger.info("Running scattering simulation...")
rng = np.random.default_rng()
result = simulator.simulate(electron, max_order=3, rng=rng)

logger.info("Event duration: %.3f ms", result.times[-1] * 1e3)
logger.info("Scatters: %d", len(result.scatter_times))
logger.info("Escaped: %s", result.escaped)
for i, p in enumerate(result.particles):
    tag = "Initial" if i == 0 else f"After scatter {i}"
    logger.info("  %s: E = %.1f eV, pitch angle = %.2f deg",
                tag, p.get_energy(), np.degrees(p.get_pitch_angle()))

# --- Plot spectrogram ---

nperseg = 2**14
SFT = ShortTimeFFT.from_window('hann', fs=sample_rate, nperseg=nperseg,
                               noverlap=0, scale_to='magnitude',
                               fft_mode='centered')
Sz1 = SFT.stft(result.signal)
t_lo, t_hi, f_lo, f_hi = SFT.extent(len(result.signal))

f_if = f0 - lo_freq
power = np.abs(Sz1)**2

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(power, origin='lower',
               extent=(t_lo * 1e3, t_hi * 1e3, f_lo / 1e6, f_hi / 1e6),
               aspect='auto', cmap='viridis')
for ts in result.scatter_times:
    ax.axvline(ts * 1e3, color='r', linestyle='--', alpha=0.7, label='Scatter')
ax.set(
    xlabel="Time [ms]",
    ylabel="Frequency [MHz]",
    ylim=(0, 500),
)
ax.set_title("CRES Signal with Scattering (HarmonicField)")
fig.colorbar(im, label=r"Power [$V^2$]")

handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend([handles[0]], [labels[0]])

plt.tight_layout()
plt.savefig("scattering_example.png", dpi=150)
plt.show()