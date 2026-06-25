"""
GasModel.py

Represents a gas mixture for electron scattering in CRES simulations.
Combines multiple cross-section models with their number densities to
compute total scattering rates and sample scatter events.
"""

import logging
import numpy as np
from .BaseCrossSection import BaseCrossSection

logger = logging.getLogger(__name__)


class GasModel:

    def __init__(self, species: list[tuple[BaseCrossSection, float]]):
        """
        Parameters
        ----------
        species : list[tuple[BaseCrossSection, float]]
            List of (cross_section_model, number_density) pairs.
            Number density is in m^-3.
        """
        if not species:
            raise ValueError("Species list must not be empty")

        for model, density in species:
            if not isinstance(model, BaseCrossSection):
                raise TypeError(
                    "Cross-section model must be a BaseCrossSection instance")
            if not isinstance(density, (int, float)):
                raise TypeError("Number density must be a number")
            if density < 0:
                raise ValueError("Number density must be non-negative")
            if not np.isfinite(density):
                raise ValueError("Number density must be finite")

        self.__species = [(model, float(density)) for model, density in species]
        for model, density in self.__species:
            logger.info(
                "GasModel species: %s, number_density=%.3e m^-3",
                type(model).__name__, density
            )

    def total_scatter_rate(self, energy: float, speed: float) -> float:
        """
        Total scattering rate for all species.

        Parameters
        ----------
        energy : float
            Electron kinetic energy in eV
        speed : float
            Electron speed in m/s

        Returns
        -------
        float
            Scattering rate in Hz (sum of n_i * sigma_i(E) * v)
        """
        rate = 0.0
        for model, density in self.__species:
            rate += density * model.total_cross_section(energy) * speed
        return rate

    def sample_time_to_scatter(self, energy: float, speed: float,
                               rng: np.random.Generator) -> float:
        """
        Sample the time until the next scatter from an exponential distribution.

        Parameters
        ----------
        energy : float
            Electron kinetic energy in eV
        speed : float
            Electron speed in m/s
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        float
            Time to next scatter in seconds
        """
        rate = self.total_scatter_rate(energy, speed)
        if rate <= 0:
            return np.inf
        return rng.exponential(1.0 / rate)

    def sample_scatter(self, energy: float, pitch_angle: float,
                       speed: float,
                       rng: np.random.Generator) -> tuple[float, float]:
        """
        Sample a scatter event: pick which species caused the scatter
        (proportional to partial rates), then sample the post-scatter state.

        Parameters
        ----------
        energy : float
            Pre-scatter electron kinetic energy in eV
        pitch_angle : float
            Pre-scatter pitch angle in radians
        speed : float
            Electron speed in m/s
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        tuple[float, float]
            (new_energy_eV, new_pitch_angle_rad)
        """
        partial_rates = np.array([
            density * model.total_cross_section(energy) * speed
            for model, density in self.__species
        ])
        total = partial_rates.sum()
        if total <= 0:
            return energy, pitch_angle

        probabilities = partial_rates / total
        chosen_idx = rng.choice(len(self.__species), p=probabilities)
        chosen_model = self.__species[chosen_idx][0]
        return chosen_model.sample_post_scatter(energy, pitch_angle, rng)
