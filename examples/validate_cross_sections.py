"""
validate_cross_sections.py

Validation plots for InelasticCrossSection and ElasticCrossSection models.

Produces a 2x2 figure:
  1. Total inelastic cross-sections vs energy (T) for H, H2, He,
     with Shah et al. 1987 (H) and Shah et al. 1988 (He) data overlaid
  2. Total inelastic cross-sections vs (T - I) for H, H2, He
     (centred representation as in Rudd et al.)
  3. Differential inelastic cross-sections dsigma/dW at T = 18.6 keV
     for H, H2, He
  4. Total elastic cross-sections vs energy for H (Z=1, A=1) and
     He (Z=2, A=4)

References
----------
Shah et al. 1987, J. Phys. B 20, 3501 (H ionization data)
Shah et al. 1988, J. Phys. B 21, 2751 (He ionization data)

Run from the repo root:
    python examples/validate_cross_sections.py
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from CRESSignalStructure.scattering.CrossSections import (
    InelasticCrossSection, ElasticCrossSection,
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Shah et al. 1987, J. Phys. B 20, 3501 — H ionization (106 points)
# -----------------------------------------------------------------------
energies_shah_H = np.array([
    14.6,   14.8,   15.0,   15.1,   15.2,   15.4,   15.6,   15.9,   16.1,
    16.4,   16.6,   16.9,   17.1,   17.4,   17.6,   17.9,   18.1,   18.4,
    18.7,   19.0,   19.3,   19.6,   20.0,   20.4,   20.9,   21.4,   22.0,
    22.6,   23.3,   24.0,   24.8,   25.6,   26.6,   27.3,   28.3,   29.3,
    30.5,   31.6,   32.8,   34.1,   35.4,   36.7,   38.1,   39.6,   41.2,
    42.9,   44.7,   46.6,   48.6,   50.7,   52.9,   55.2,   57.6,   60.1,
    63.0,   66.0,   69.0,   72.1,   75.5,   79.5,   84.0,   89.0,   94.0,
    102.0,  103.0,  113.0,  121.0,  130.2,  138.2,  148.2,  158.2,  168.2,
    178.2,  188.2,  198.2,  213.2,  228.2,  248.2,  268.2,  288.0,  317.9,
    347.9,  387.9,  427.9,  467.9,  508.2,  548.2,  598.2,  668.2,  748.2,
    818.2,  898.2,  998.2,  1100.0, 1200.0, 1300.0, 1506.7, 1662.7, 1848.1,
    1998.1, 2198.1, 2448.1, 2698.1, 2998.1, 3298.1, 3648.1,
])
xsec_shah_H = np.array([
    0.544e-21, 0.661e-21, 0.762e-21, 0.820e-21, 0.870e-21, 0.990e-21,
    1.08e-21,  1.25e-21,  1.37e-21,  1.45e-21,  1.63e-21,  1.68e-21,
    1.73e-21,  1.96e-21,  2.07e-21,  2.15e-21,  2.22e-21,  2.35e-21,
    2.50e-21,  2.61e-21,  2.75e-21,  2.81e-21,  2.93e-21,  3.11e-21,
    3.34e-21,  3.39e-21,  3.61e-21,  3.76e-21,  4.01e-21,  4.15e-21,
    4.30e-21,  4.44e-21,  4.57e-21,  4.75e-21,  4.95e-21,  5.01e-21,
    5.10e-21,  5.27e-21,  5.39e-21,  5.53e-21,  5.59e-21,  5.74e-21,
    5.83e-21,  5.89e-21,  6.02e-21,  6.07e-21,  6.08e-21,  6.23e-21,
    6.27e-21,  6.19e-21,  6.23e-21,  6.21e-21,  6.13e-21,  6.14e-21,
    6.11e-21,  6.11e-21,  6.01e-21,  5.96e-21,  5.91e-21,  5.84e-21,
    5.78e-21,  5.59e-21,  5.40e-21,  5.42e-21,  5.23e-21,  5.07e-21,
    5.05e-21,  4.83e-21,  4.62e-21,  4.55e-21,  4.43e-21,  4.28e-21,
    4.10e-21,  3.98e-21,  3.79e-21,  3.61e-21,  3.43e-21,  3.31e-21,
    3.03e-21,  2.84e-21,  2.66e-21,  2.50e-21,  2.31e-21,  2.15e-21,
    2.00e-21,  1.86e-21,  1.77e-21,  1.59e-21,  1.47e-21,  1.38e-21,
    1.26e-21,  1.13e-21,  1.05e-21,  0.982e-21, 0.914e-21, 0.807e-21,
    0.721e-21, 0.673e-21, 0.631e-21, 0.577e-21, 0.525e-21, 0.472e-21,
    0.437e-21, 0.403e-21, 0.370e-21, 0.339e-21,
])

# -----------------------------------------------------------------------
# Shah et al. 1988, J. Phys. B 21, 2751 — He ionization (57 points)
# Raw values are in units of 10^-21 m^2
# -----------------------------------------------------------------------
energies_shah_He = np.array([
    26.6, 27.6, 28.6, 29.6, 30.6, 32.1, 33.6, 38.6, 43.6, 48.6, 53.6, 58.6,
    68.6, 78.6, 88.6, 90.2, 95.2, 100,  105,  110,  115,  120,  130,  140,
    150,  160,  170,  195,  220,  250,  280,  325,  375,  430,  500,  570,
    650,  750,  870,  1000, 1150, 1320, 1520, 1750, 2010, 2300, 2650, 3000,
    3500, 4000, 4600, 5300, 6100, 7000, 8000, 9000, 10000,
])
xsec_shah_He = np.array([
    0.242, 0.366, 0.480, 0.604, 0.715, 0.871, 1.05,  1.52,  1.90,  2.26,
    2.50,  2.73,  3.05,  3.29,  3.45,  3.53,  3.60,  3.67,  3.74,  3.70,
    3.67,  3.70,  3.69,  3.67,  3.60,  3.58,  3.55,  3.42,  3.25,  3.13,
    2.89,  2.65,  2.53,  2.32,  2.09,  1.87,  1.77,  1.61,  1.44,  1.28,
    1.19,  1.07,  0.955, 0.872, 0.796, 0.693, 0.615, 0.551, 0.520, 0.448,
    0.398, 0.337, 0.308, 0.276, 0.250, 0.224, 0.195,
]) * 1e-21  # Convert to m^2

# -----------------------------------------------------------------------
# Cross-section model objects
# -----------------------------------------------------------------------
xsec_H = InelasticCrossSection("H")
xsec_H2 = InelasticCrossSection("H2")
xsec_He = InelasticCrossSection("He")

elastic_H = ElasticCrossSection(Z=1, A=1.0)
elastic_He = ElasticCrossSection(Z=2, A=4.0)

# Binding energies (eV)
I_H = 13.6058  # Rydberg energy
I_H2 = 15.43
I_He = 24.59

# -----------------------------------------------------------------------
# Energy grids
# -----------------------------------------------------------------------
N_POINTS = 1000

# Grid in T for total cross-section plots
T_min = I_He + 1.0
T_max = 20e3
T_grid = np.geomspace(T_min, T_max, N_POINTS)

# Grid in (T - I) for centred representation
TmI_min = 0.5
TmI_max = 20e3
TmI_grid = np.geomspace(TmI_min, TmI_max, N_POINTS)

# Grid in W for differential cross-section
W_min = 1e-1
W_max = 1e2
N_W = 200
W_grid = np.geomspace(W_min, W_max, N_W)

# -----------------------------------------------------------------------
# Compute cross-sections
# -----------------------------------------------------------------------

# Total inelastic vs T
sigma_H = np.array([xsec_H.total_cross_section(T) for T in T_grid])
sigma_H2 = np.array([xsec_H2.total_cross_section(T) for T in T_grid])
sigma_He = np.array([xsec_He.total_cross_section(T) for T in T_grid])

# Total inelastic vs (T - I) (centred)
sigma_H_c = np.array([xsec_H.total_cross_section(TmI + I_H) for TmI in TmI_grid])
sigma_H2_c = np.array([xsec_H2.total_cross_section(TmI + I_H2) for TmI in TmI_grid])
sigma_He_c = np.array([xsec_He.total_cross_section(TmI + I_He) for TmI in TmI_grid])

# SDCS at T = 18.6 keV
T_sdcs = 18.6e3
sdcs_H = np.array([xsec_H.sdcs(T_sdcs, W) for W in W_grid])
sdcs_H2 = np.array([xsec_H2.sdcs(T_sdcs, W) for W in W_grid])
sdcs_He = np.array([xsec_He.sdcs(T_sdcs, W) for W in W_grid])

# Total elastic vs T
sigma_el_H = np.array([elastic_H.total_cross_section(T) for T in T_grid])
sigma_el_He = np.array([elastic_He.total_cross_section(T) for T in T_grid])

# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'

SCALE = 1e20  # Plot in units of 10^-20 m^2

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# --- Top-left: total inelastic vs T with Shah data ---
ax = axes[0, 0]
ax.loglog(T_grid, sigma_H * SCALE, 'r', label='H (BEB)')
ax.loglog(T_grid, sigma_H2 * SCALE, color='tab:blue', label='H2 (BEB)')
ax.loglog(T_grid, sigma_He * SCALE, 'g', label='He (BEB)')
ax.plot(energies_shah_H, xsec_shah_H * SCALE, 'x', color='r',
        ms=4, label='Shah 1987 (H)')
ax.plot(energies_shah_He, xsec_shah_He * SCALE, 'x', color='g',
        ms=4, label='Shah 1988 (He)')
ax.set_xlabel('T [eV]')
ax.set_ylabel('Cross-section [1e-20 m²]')
ax.set_title('Total ionisation cross-section')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Top-right: total inelastic vs (T - I) ---
ax = axes[0, 1]
ax.loglog(TmI_grid, sigma_H_c * SCALE, 'r', label='H')
ax.loglog(TmI_grid, sigma_H2_c * SCALE, color='tab:blue', label='H2')
ax.loglog(TmI_grid, sigma_He_c * SCALE, 'g', label='He')
ax.set_xlabel('T - I [eV]')
ax.set_ylabel('Cross-section [1e-20 m²]')
ax.set_title('Total ionisation (centred)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Bottom-left: SDCS at 18.6 keV ---
ax = axes[1, 0]
ax.loglog(W_grid, sdcs_H, 'r', label='H')
ax.loglog(W_grid, sdcs_H2, color='tab:blue', label='H2')
ax.loglog(W_grid, sdcs_He, 'g', label='He')
ax.set_xlabel('W [eV]')
ax.set_ylabel('d sigma / dW [m²/eV]')
ax.set_title('Differential cross-section (T = 18.6 keV)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Bottom-right: total elastic ---
ax = axes[1, 1]
ax.loglog(T_grid, sigma_el_H * SCALE, color='tab:purple', label='H (Z=1)')
ax.loglog(T_grid, sigma_el_He * SCALE, color='tab:cyan', label='He (Z=2)')
ax.set_xlabel('T [eV]')
ax.set_ylabel('Cross-section [1e-20 m²]')
ax.set_title('Total elastic cross-section')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('Cross-section model validation', fontsize=14)
plt.tight_layout()
plt.savefig("examples/validate_cross_sections.png", dpi=150, bbox_inches="tight")
logger.info("Figure saved to examples/validate_cross_sections.png")
plt.close(fig)
