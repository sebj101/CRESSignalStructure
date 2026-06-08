"""
BaseCrossSection.py

Abstract base class for gas scattering cross-section models.
Subclasses implement the physics for specific gas species (H, T, He, H2, T2)
and scattering mechanisms (elastic, inelastic).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseCrossSection(ABC):

    @abstractmethod
    def total_cross_section(self, energy: float) -> float:
        """
        Total scattering cross-section at a given electron kinetic energy.

        Parameters
        ----------
        energy : float
            Electron kinetic energy in eV

        Returns
        -------
        float
            Total cross-section in m^2
        """

    @abstractmethod
    def sample_post_scatter(self, energy: float, pitch_angle: float,
                            rng: np.random.Generator) -> tuple[float, float]:
        """
        Sample the post-scatter electron state.

        Parameters
        ----------
        energy : float
            Pre-scatter electron kinetic energy in eV
        pitch_angle : float
            Pre-scatter pitch angle in radians
        rng : np.random.Generator
            Random number generator for sampling

        Returns
        -------
        tuple[float, float]
            (new_energy_eV, new_pitch_angle_rad)
        """
