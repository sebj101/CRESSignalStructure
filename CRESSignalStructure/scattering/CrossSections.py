"""
CrossSections.py

Cross-section models implemented as subclasses of BaseCrossSection.

Models:
    SimpleCrossSectionModel - toy model with constant cross-section
    InelasticCrossSection - BEB inelastic model (Kim & Rudd) with Rudd 1991
                            angular distribution
    ElasticCrossSection - screened Rutherford elastic model with Gauvin & Drouin
                          correction
"""

import dataclasses
import numpy as np
from numpy.random import Generator
import scipy.constants as sc

from .BaseCrossSection import BaseCrossSection
from .scattering_utils import scatter_to_pitch_angle


_A0 = sc.physical_constants['Bohr radius'][0]
_RYDBERG_EV = sc.physical_constants['Rydberg constant times hc in eV'][0]


class SimpleCrossSectionModel(BaseCrossSection):
    """
    A simple cross section model, with a one-sided normal distribution for
    energy loss and a normal distribution for pitch angle change. The total xsec
    is constant at all energies.

    cross_sec : float
        Total cross-section in m^2
        Standard deviation of the energy loss distribution
    mu_pitch_angle_change : float
        Mean of the pitch angle change distribution
    sigma_pitch_angle_change : float
        Standard deviation of the pitch angle change distribution
    """

    def __init__(self, cross_sec: float, sigma_energy_loss: float,
                 mu_pitch_angle_change: float, sigma_pitch_angle_change: float) -> None:
        if cross_sec < 0.0:
            raise ValueError("Cross-sections cannot be negative")
        if sigma_energy_loss < 0.0:
            raise ValueError("sigma_energy_loss cannot be negative")
        if sigma_pitch_angle_change < 0.0:
            raise ValueError("sigma_pitch_angle_change cannot be negative")

        self._xsec = cross_sec
        self._sigma_e = sigma_energy_loss
        self._mu_p = mu_pitch_angle_change
        self._sigma_p = sigma_pitch_angle_change

    def total_cross_section(self, energy: float) -> float:
        return self._xsec

    def sample_post_scatter(self, energy: float, pitch_angle: float,
                            rng: Generator) -> tuple[float, float]:
        energy_loss = abs(rng.normal(loc=0, scale=self._sigma_e))
        pitch_angle_change = rng.normal(loc=self._mu_p, scale=self._sigma_p)
        new_pitch_angle = pitch_angle + pitch_angle_change
        if new_pitch_angle > np.pi/2:
            new_pitch_angle = np.pi - new_pitch_angle

        return energy - energy_loss, new_pitch_angle


# ---------------------------------------------------------------------------
# Inelastic cross-section: BEB model (Kim & Rudd 1994) with Rudd 1991 DDCS
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _InelasticParams:
    binding_energy: float
    orbital_ke: float
    n_electrons: int
    ni: float
    osc_coeffs: tuple
    A1: float
    A2: float
    A3: float


_INELASTIC_SPECIES = {
    "H": _InelasticParams(
        binding_energy=_RYDBERG_EV,
        orbital_ke=_RYDBERG_EV,
        n_electrons=1,
        ni=0.4343,
        osc_coeffs=(-2.2473e-2, 1.1775, -4.6264e-1, 8.9064e-2, 0.0),
        A1=0.74, A2=0.87, A3=-0.60,
    ),
    "H2": _InelasticParams(
        binding_energy=15.43,
        orbital_ke=25.68,
        n_electrons=2,
        ni=1.173,
        osc_coeffs=(0.0, 1.1262, 6.3982, -7.8055, 2.144),
        A1=0.74, A2=0.87, A3=-0.60,
    ),
    "He": _InelasticParams(
        binding_energy=24.59,
        orbital_ke=39.51,
        n_electrons=2,
        ni=1.605,
        osc_coeffs=(0.0, 1.2178e1, -2.9585e1, 3.1251e1, -1.2175e1),
        A1=0.85, A2=0.36, A3=-0.10,
    ),
}


class InelasticCrossSection(BaseCrossSection):
    """
    BEB (Binary Encounter Bethe) inelastic cross-section model with Rudd 1991
    angular distribution.

    Supports species: "H" (atomic hydrogen), "H2" (molecular hydrogen),
    "He" (helium). For molecular tritium (T2), use "H2" since the electron
    structure is identical.

    References
    ----------
    Kim & Rudd, Phys. Rev. A 50, 3954 (1994) — BEB model
    Rudd, Phys. Rev. A 44, 1644 (1991) — angular distributions

    Parameters
    ----------
    species : str
        Gas species: "H", "H2", or "He"
    """

    _BETA = 0.60
    _GAMMA_RUDD = 10.0
    _G_B = 2.9
    _N_RUDD = 2.4
    _G5 = 0.33
    _N_W_BINS = 400
    _N_THETA_BINS = 500

    def __init__(self, species: str) -> None:
        if species not in _INELASTIC_SPECIES:
            raise ValueError(
                f"Unknown species '{species}'. "
                f"Supported: {list(_INELASTIC_SPECIES.keys())}")
        self.__params = _INELASTIC_SPECIES[species]

    def total_cross_section(self, energy: float) -> float:
        """
        BEB total inelastic cross-section.

        Parameters
        ----------
        energy : float
            Electron kinetic energy in eV

        Returns
        -------
        float
            Cross-section in m^2. Returns 0 below ionization threshold.
        """
        p = self.__params
        if energy <= p.binding_energy:
            return 0.0

        t = energy / p.binding_energy
        u = p.orbital_ke / p.binding_energy
        S = self._S()
        D = self._D(energy)
        return (S / (t + u + 1)
                * (D * np.log(t)
                   + (2 - p.ni / p.n_electrons)
                   * ((t - 1) / t - np.log(t) / (t + 1))))

    def sdcs(self, energy: float, W: float) -> float:
        """
        BEB singly-differential cross-section dσ/dW.

        Parameters
        ----------
        energy : float
            Incident electron kinetic energy in eV
        W : float
            Ejected electron kinetic energy in eV

        Returns
        -------
        float
            Cross-section in m^2/eV
        """
        p = self.__params
        if energy <= p.binding_energy or W < 0:
            return 0.0
        if W > energy - p.binding_energy:
            return 0.0

        B = p.binding_energy
        t = energy / B
        u = p.orbital_ke / B
        w = W / B
        b, c, d, e, f = p.osc_coeffs

        diff_osc = (b * (w + 1)**-2 + c * (w + 1)**-3 + d * (w + 1)**-4
                    + e * (w + 1)**-5 + f * (w + 1)**-6)
        prefactor = self._S() / (B * (t + u + 1))
        term1 = (p.ni / p.n_electrons - 2) * (1 / (w + 1) + 1 / (t - w)) / (t + 1)
        term2 = (2 - p.ni / p.n_electrons) * ((w + 1)**-2 + (t - w)**-2)
        term3 = np.log(t) * diff_osc / (p.n_electrons * (w + 1))
        return prefactor * (term1 + term2 + term3)

    def ddcs(self, energy: float, W: float, theta: float) -> float:
        """
        Rudd 1991 doubly-differential cross-section d²σ/dWdθ.

        Parameters
        ----------
        energy : float
            Incident electron kinetic energy in eV
        W : float
            Outgoing electron kinetic energy in eV
        theta : float
            Scattering angle in radians

        Returns
        -------
        float
            Cross-section in m^2/eV/rad
        """
        p = self.__params
        if energy <= p.binding_energy or W < 0:
            return 0.0
        if W > energy - p.binding_energy:
            return 0.0
        return self._G1(W, energy) * (self._f_BE(W, energy, theta)
                                       + self._G4fb(W, energy, theta))

    def sample_post_scatter(self, energy: float, pitch_angle: float,
                            rng: Generator) -> tuple[float, float]:
        p = self.__params
        W = self._sample_secondary_energy(energy, rng)
        new_energy = energy - W - p.binding_energy
        theta = self._sample_scattering_angle(energy, new_energy, rng)
        new_pitch = scatter_to_pitch_angle(pitch_angle, theta, rng)
        return new_energy, new_pitch

    # --- Private: BEB helper ---

    def _S(self) -> float:
        p = self.__params
        return 4 * np.pi * p.n_electrons * (_A0 * _RYDBERG_EV / p.binding_energy)**2

    def _D(self, energy: float) -> float:
        p = self.__params
        t = energy / p.binding_energy
        t_term = (t + 1) / 2
        b, c, d, e, f = p.osc_coeffs
        result = ((b / 2) * (1 - t_term**-2)
                  + (c / 3) * (1 - t_term**-3)
                  + (d / 4) * (1 - t_term**-4)
                  + (e / 5) * (1 - t_term**-5)
                  + (f / 6) * (1 - t_term**-6))
        return result / p.n_electrons

    # --- Private: Rudd 1991 angular model helpers ---
    # All take W, T in eV; normalization by I is done internally.

    def _G2(self, W: float, T: float) -> float:
        B = self.__params.binding_energy
        return np.sqrt((W / B + 1) / (T / B))

    def _G3(self, W: float, T: float) -> float:
        B = self.__params.binding_energy
        w = W / B
        g2 = self._G2(W, T)
        return self._BETA * np.sqrt(np.abs(1 - g2**2) / w) if w > 0 else 0.0

    def _G4(self, W: float, T: float) -> float:
        B = self.__params.binding_energy
        w = W / B
        t = T / B
        if t == 0:
            return 0.0
        return self._GAMMA_RUDD * (1 - w / t)**3 / (t * (w + 1))

    def _f_BE(self, W: float, T: float, theta: float) -> float:
        g2 = self._G2(W, T)
        g3 = self._G3(W, T)
        if g3 == 0:
            return 0.0
        return 1.0 / (1 + ((np.cos(theta) - g2) / g3)**2)

    def _f_b(self, theta: float) -> float:
        return 1.0 / (1 + ((np.cos(theta) + 1) / self._G5)**2)

    def _g_BE(self, W: float, T: float) -> float:
        g2 = self._G2(W, T)
        g3 = self._G3(W, T)
        if g3 == 0:
            return 0.0
        return 2 * np.pi * g3 * (np.arctan((1 - g2) / g3)
                                  + np.arctan((1 + g2) / g3))

    def _F(self, T: float) -> float:
        p = self.__params
        t = T / p.binding_energy
        if t <= 0:
            return 0.0
        return (p.A1 * np.log(t) + p.A2 + p.A3 / t) / t

    def _f_1(self, W: float, T: float) -> float:
        B = self.__params.binding_energy
        w = W / B
        t = T / B
        n = self._N_RUDD
        return ((w + 1)**(-n) + (t - w)**(-n)
                - ((w + 1) * (t - w))**(-n / 2))

    def _G1(self, W: float, T: float) -> float:
        B = self.__params.binding_energy
        g_be = self._g_BE(W, T)
        g4 = self._G4(W, T)
        denom = g_be + g4 * self._G_B
        if denom == 0:
            return 0.0
        return self._S() * self._F(T) * self._f_1(W, T) / (B * denom)

    def _G4fb(self, W: float, T: float, theta: float) -> float:
        return self._G4(W, T) * self._f_b(theta)

    # --- Private: sampling ---

    def _sample_secondary_energy(self, energy: float, rng: Generator) -> float:
        p = self.__params
        W_max = (energy - p.binding_energy) / 2
        if W_max <= 0:
            return 0.0

        W_min = 1e-5 * W_max
        log_W = np.linspace(np.log(W_min), np.log(W_max), self._N_W_BINS)
        W_grid = np.exp(log_W)
        sdcs_vals = np.array([self.sdcs(energy, W) for W in W_grid])

        cdf = np.cumsum(0.5 * (sdcs_vals[:-1] + sdcs_vals[1:])
                        * np.diff(W_grid))
        cdf = np.insert(cdf, 0, 0.0)
        if cdf[-1] <= 0:
            return 0.0
        cdf /= cdf[-1]

        u = rng.uniform()
        return float(np.interp(u, cdf, W_grid))

    def _sample_scattering_angle(self, energy: float, W: float,
                                 rng: Generator) -> float:
        theta_grid = np.linspace(1e-6, np.pi - 1e-6, self._N_THETA_BINS)
        ddcs_vals = np.array([self.ddcs(energy, W, th) for th in theta_grid])

        # Weight by sin(theta) for solid angle
        weighted = ddcs_vals * np.sin(theta_grid)
        cdf = np.cumsum(0.5 * (weighted[:-1] + weighted[1:])
                        * np.diff(theta_grid))
        cdf = np.insert(cdf, 0, 0.0)
        if cdf[-1] <= 0:
            return 0.0
        cdf /= cdf[-1]

        u = rng.uniform()
        return float(np.interp(u, cdf, theta_grid))


# ---------------------------------------------------------------------------
# Elastic cross-section: screened Rutherford with Gauvin & Drouin correction
# ---------------------------------------------------------------------------

_ELASTIC_CORRECTION = {
    1: (0.734357, 2.45719),
    2: (1.25, 12.0),
    3: (1.262598, 10.17333),
    4: (1.492947, 4.38619),
    5: (1.434886, 4.016957),
}


class ElasticCrossSection(BaseCrossSection):
    """
    Screened Rutherford elastic cross-section model with Gauvin & Drouin
    correction factor.

    The cross-section is per atom. For molecular targets (H2, T2), use
    twice the molecular number density in the GasModel.

    References
    ----------
    Gauvin & Drouin, Scanning 15, 3 (1993) — total cross-section correction

    Parameters
    ----------
    Z : int
        Atomic number of target (1-5 supported)
    A : float
        Atomic mass of target in amu
    """

    def __init__(self, Z: int, A: float) -> None:
        if Z not in _ELASTIC_CORRECTION:
            raise ValueError(
                f"Z={Z} not supported. Supported: {list(_ELASTIC_CORRECTION.keys())}")
        if A <= 0:
            raise ValueError("Atomic mass must be positive")

        self.__Z = Z
        self.__A = A
        self.__lam, self.__beta = _ELASTIC_CORRECTION[Z]

    def _screening_param(self, energy_keV: float) -> float:
        return 3.4e-3 * self.__Z**(2 / 3) / energy_keV

    def total_cross_section(self, energy: float) -> float:
        """
        Total elastic cross-section using screened Rutherford with
        Gauvin & Drouin correction factor.

        Parameters
        ----------
        energy : float
            Electron kinetic energy in eV

        Returns
        -------
        float
            Cross-section in m^2
        """
        E_keV = energy / 1e3
        alpha = self._screening_param(E_keV)
        ME_keV = sc.m_e * sc.c**2 / (sc.e * 1e3)

        # Rutherford total cross-section (Gauvin & Drouin formula)
        # 5.21e-21 cm^2 = 5.21e-25 m^2
        sigma_R = (5.21e-25
                   * (self.__Z / E_keV)**2
                   * np.pi
                   * ((E_keV + ME_keV) / (E_keV + 2 * ME_keV))**2
                   / (alpha * (1 + alpha)))

        # Correction factor
        gamma_corr = self.__lam * (1 - np.exp(-self.__beta * np.sqrt(E_keV)))
        return sigma_R * gamma_corr

    def dcs(self, energy: float, theta: float) -> float:
        """
        Screened Rutherford differential cross-section dσ/dΩ.

        Parameters
        ----------
        energy : float
            Electron kinetic energy in eV
        theta : float
            Scattering angle in radians

        Returns
        -------
        float
            Differential cross-section in m^2/sr
        """
        E_J = energy * sc.e
        a = self.__Z * sc.e**2 / (16 * np.pi * sc.epsilon_0 * E_J)
        alpha = self._screening_param(energy / 1e3)
        return a**2 / (np.sin(theta / 2)**2 + alpha)**2

    def sample_post_scatter(self, energy: float, pitch_angle: float,
                            rng: Generator) -> tuple[float, float]:
        theta = self._sample_scattering_angle(energy, rng)

        # Elastic recoil energy loss
        gamma = 1 + energy * sc.e / (sc.m_e * sc.c**2)
        beta = np.sqrt(1 - 1 / gamma**2)
        p = gamma * sc.m_e * beta * sc.c
        q = 2 * p * np.sin(theta / 2)
        M = self.__A * sc.u
        E_recoil_eV = q**2 / (2 * M * sc.e)

        new_energy = energy - E_recoil_eV
        new_pitch = scatter_to_pitch_angle(pitch_angle, theta, rng)
        return new_energy, new_pitch

    def _sample_scattering_angle(self, energy: float,
                                 rng: Generator) -> float:
        alpha = self._screening_param(energy / 1e3)
        u = rng.uniform()
        cos_theta = 1 - 2 * alpha * (1 - u) / (u + alpha)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))
