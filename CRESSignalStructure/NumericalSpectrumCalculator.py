"""
NumericalSpectrumCalculator.py

Contains a class for numerical calculations of power spectra from arbitrary 
field maps

S. Jones 17-10-25
"""

from CRESSignalStructure.RealFields import BaseField, HarmonicField, BathtubField
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.BaseSpectrumCalculator import BaseSpectrumCalculator
import numpy as np
import scipy.constants as sc
from scipy.optimize import brentq
from scipy.integrate import simpson, cumulative_simpson
from scipy.interpolate import interp1d
from numpy.typing import ArrayLike, NDArray


def calc_zmax(trap: BaseField, particle: Particle) -> float:
    """
    Calculate the maximum axial displacement of the particle in the trap

    Parameters
    ----------
    trap : BaseField
        The trapping magnetic field
    particle : Particle
        The particle being trapped

    Returns
    -------
    float
        The maximum axial displacement in metres
    """
    pa = particle.GetPitchAngle()
    pStart = particle.GetPosition()

    centralField = trap.evaluate_field_magnitude(pStart[0], pStart[1], 0.)
    muCentre = centralField / (np.sin(pa)**2)

    def zMaxEqn(z):
        result = 1.0 - muCentre / trap.evaluate_field_magnitude(pStart[0],
                                                                pStart[1], z)
        return result

    # Determine where the bounds of the equation solver should be
    upperZBound = 1.0
    if isinstance(trap, HarmonicField):
        upperZBound = trap.coil.radius * 80.0
    elif isinstance(trap, BathtubField):
        upperZBound = np.max([trap.coil1.z, trap.coil2.z])
    else:
        upperZBound = 2.0

    zmax, _ = brentq(zMaxEqn, 0.0, upperZBound, full_output=True)
    return zmax


def calc_omega_axial(trap: BaseField, particle: Particle) -> float:
    """
    Calculate the angular axial frequency of trapped particle motion

    Parameters
    ----------
    trap: BaseField
        The trapping magnetic field
    particle : Particle
        The particle being trapped

    Returns
    -------
    float
        Axial frequency in radians/s
    """
    pa = particle.GetPitchAngle()
    ke = particle.GetEnergy() * sc.e
    pStart = particle.GetPosition()

    centralField = trap.evaluate_field_magnitude(pStart[0], pStart[1], 0.)

    zMax = calc_zmax(trap, particle)

    # Equivalent magnetic moment
    muMag = ke * np.sin(pa)**2 / centralField

    # Calculate the integrand at each point
    integrationPoints = np.linspace(0, zMax, 2000, endpoint=False)
    integrand = 1.0 / np.sqrt((2 / particle.GetMass()) * (ke - muMag *
                              trap.evaluate_field_magnitude(pStart[0], pStart[1], integrationPoints)))

    integral = 2 * np.trapezoid(integrand, integrationPoints) / np.pi
    return 1 / integral


def calc_t_vs_z(trap: BaseField, particle: Particle):
    """
    Calculate the time versus the z position of an electron in a trap

    Parameters
    ----------
    trap : BaseField
        The trapping field
    particle : Particle
        The particle being trapped

    Returns
    -------
    tuple[NDArray, NDArray]
        Time values in seconds, z values in metres
    """

    pa = particle.GetPitchAngle()
    ke = particle.GetEnergy() * sc.e
    pStart = particle.GetPosition()

    centralField = trap.evaluate_field_magnitude(pStart[0], pStart[1], 0.)
    zMax = calc_zmax(trap, particle)
    muMag = ke * np.sin(pa)**2 / centralField

    def t_integrand(z):
        result = np.sqrt(particle.GetMass() / 2) / np.sqrt(ke - muMag *
                                                           trap.evaluate_field_magnitude(pStart[0], pStart[1], z))
        return result

    # For axially symmetric traps, we should only need to do one integration
    zVals1 = np.linspace(0.0, 0.999 * zMax, 100)
    tVals1 = cumulative_simpson(t_integrand(zVals1), x=zVals1, initial=0.0)
    zVals2 = np.flip(zVals1[:-1])
    tVals2 = 2 * tVals1[-1] - np.flip(tVals1[:-1])
    tVals3 = tVals2[-1] + np.concatenate((tVals1[1:], tVals2))
    zVals3 = -np.concatenate((zVals1[1:], zVals2))
    tVals = np.concatenate((tVals1, tVals2, tVals3))
    zVals = np.concatenate((zVals1, zVals2, zVals3))
    return tVals, zVals


def B_from_t(trap: BaseField, part: Particle, n_t_points: int):
    """
    Calculate the magnetic field as a function of time over 1 axial period

    Parameters
    ----------
    trap : BaseField
        The trapping field
    particle : Particle
        The particle being trapped
    n_t_points : int
        The number of time points to output B at

    Returns
    -------
    tuple[NDArray, NDArray]
        Time values in seconds, Field magnitudes in Tesla
    """
    if (n_t_points < 2):
        raise ValueError("Require at least 2 time points")

    # Initially calculate the axial period
    Ta = 1 / (calc_omega_axial(trap, part) / (2 * np.pi))
    t_vals = np.linspace(0.0, Ta, n_t_points)

    t, z = calc_t_vs_z(trap, part)
    t_to_z = interp1d(t, z, kind='cubic')
    z = t_to_z(t_vals)
    pStart = part.GetPosition()
    return t_vals, trap.evaluate_field_magnitude(pStart[0], pStart[1], z)


def cyclotron_phase_from_t(trap: BaseField, particle: Particle,
                           n_t_points: int = 499) -> tuple[NDArray, NDArray]:
    """
    Calculates the cyclotron phase as a function of time.

    Parameters
    ----------
    trap : BaseField
        The trapping field
    particle : Particle
        The particle being trapped
    n_t_points : int
        The number of time points to output B at

    Returns
    -------
    tuple[NDArray, NDArray]
        Time values in seconds, Phase in radians
    """
    t, B = B_from_t(trap, particle, n_t_points)
    return t, cumulative_simpson(sc.e * B / (particle.GetGamma() * particle.GetMass()),
                                 x=t, initial=0.0)


def calc_omega_0(trap: BaseField, particle: Particle,
                 n_t_points: int = 499) -> float:
    """
    Calculates the mean cyclotron frequency

    Parameters
    ----------
    trap : BaseField
        The trapping field
    particle : Particle
        The particle being trapped
    n_t_points : int
        The number of time points to use for the calculation

    Returns
    -------
    float
        The average cyclotron frequency in radians/s
    """
    t, B = B_from_t(trap, particle, n_t_points)
    phi_Ta = simpson(sc.e * B / (particle.GetGamma() * particle.GetMass()), t)
    return phi_Ta / t[-1]


class NumericalSpectrumCalculator(BaseSpectrumCalculator):
    def __init__(self, field: BaseField, waveguide: CircularWaveguide,
                 particle: Particle):
        super().__init__(field, waveguide, particle)
        self.__trap = field
        self.__waveguide = waveguide
        self.__particle = particle

    def GetPeakFrequency(self, order: ArrayLike) -> NDArray:
        """
        Calculate the frequencies at which the components occur

        Parameters
        ----------
        order : int
            The order of the peak to calculate the frequency for

        Returns
        -------
        ArrayLike
            The frequencies at which a component is observed in Hertz
        """
        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")

        f0 = calc_omega_0(self.__trap, self.__particle) / (2 * np.pi)
        fa = calc_omega_axial(self.__trap, self.__particle) / (2 * np.pi)
        return f0 + order * fa

    def GetPeakAmp(self, order: ArrayLike) -> NDArray:
        """
        Calculate the complex amplitude of a peak, given the order

        Parameters
        ----------
        order : ArrayLike
            Order of the peak for which we are calculating the amplitude 
        """
        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")

        # Get the z position array
        t1, z = calc_t_vs_z(self.__trap, self.__particle)
        # NB these t values are not uniformly distributed
        interp_t1_z = interp1d(t1, z, kind='cubic')

        N_T_POINTS = 499
        omega_a = calc_omega_axial(self.__trap, self.__particle)
        omega_0 = calc_omega_0(self.__trap, self.__particle, N_T_POINTS)
        t, phi_t = cyclotron_phase_from_t(
            self.__trap, self.__particle, N_T_POINTS)
        Ta = t[-1]
        z_t = interp_t1_z(t)  # These are uniform in t

        # Get propagation constant
        omega_c = 1.841 * sc.c / self.__waveguide.wgR
        v_phase = sc.c / np.sqrt(1 - (omega_c / omega_0)**2)
        kLambda = omega_0 / v_phase

        # Reshape order for broadcasting: (len(order), 1) to broadcast with (N_T_POINTS,)
        order_reshaped = order.reshape(-1, 1)

        # Common phase term that doesn't depend on order
        common_phase = np.exp(1j * (phi_t + kLambda * z_t - omega_0 * t))

        # Order-dependent phase term, broadcasted over all time points
        order_phase = np.exp(-1j * order_reshaped * omega_a * t)

        # Full integrand: shape (len(order), N_T_POINTS)
        integrand = common_phase * order_phase

        # Integrate along time axis (axis=1) for each order
        return simpson(integrand, t, axis=1) / Ta
