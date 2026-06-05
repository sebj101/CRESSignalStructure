"""
scattering_utils.py

Utility functions for scattering simulations.
"""

import numpy as np


def scatter_to_pitch_angle(pitch_angle: float, scattering_angle: float,
                           rng: np.random.Generator) -> float:
    """
    Convert a scattering angle to a new pitch angle.

    Given an electron with pitch angle alpha (relative to B-field) that
    undergoes a scatter by angle theta (relative to its velocity direction),
    compute the new pitch angle after the scatter. The azimuthal angle of
    the scatter is drawn uniformly from [0, 2*pi).

    Parameters
    ----------
    pitch_angle : float
        Current pitch angle in radians, in (0, pi)
    scattering_angle : float
        Scattering angle in radians, in [0, pi]
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    float
        New pitch angle in radians
    """
    phi = rng.uniform(0, 2 * np.pi)
    cos_new = (np.cos(pitch_angle) * np.cos(scattering_angle)
               + np.sin(pitch_angle) * np.sin(scattering_angle) * np.cos(phi))
    return np.arccos(np.clip(cos_new, -1.0, 1.0))
