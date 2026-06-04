"""
CrossSections.py

Cross-section models are implemented here as subclasses of BaseCrossSection
"""

from .BaseCrossSection import BaseCrossSection
from numpy.random import Generator
import numpy as np

class SimpleCrossSectionModel(BaseCrossSection):
    """
    A simple cross section model, with a one-sided normal distribution for 
    energy loss and a normal distribution for pitch angle change. The total xsec
    is constant at all energies.

    Parameters
    ----------
    cross_sec : float
        Total cross-section in m^-2
    sigma_energy_loss : float
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
        self._sigma_p = sigma_energy_loss

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
    