"""
SpectrumCalculator.py

Unified spectrum calculator for CRES signals.

Accepts either a BaseTrap (analytical) or BaseField (numerical) and dispatches
accordingly. The analytical path uses closed-form Bessel-function expressions;
the numerical path integrates the cyclotron phase over the axial period.
"""

from typing import overload

from .BaseTrap import BaseTrap
from .BaseField import BaseField
from .CircularWaveguide import CircularWaveguide
from .QTNMTraps import HarmonicTrap, BathtubTrap
from .Particle import Particle
import numpy as np
import scipy.constants as sc
from scipy.special import jv, jvp, j1
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from numpy.typing import ArrayLike, NDArray


class SpectrumCalculator:
    def __init__(self, trap: BaseTrap | BaseField,
                 waveguide: CircularWaveguide, particle: Particle):
        """
        Parameters
        ----------
        trap : BaseTrap | BaseField
            Analytical trap (BaseTrap subclass) or numerical field (BaseField subclass)
        waveguide : CircularWaveguide
        particle : Particle
        """
        if not isinstance(trap, (BaseTrap, BaseField)):
            raise TypeError("trap must be an instance of BaseTrap or BaseField")
        self._trap = trap
        self._waveguide = waveguide
        self._particle = particle

    def get_particle(self) -> Particle:
        return self._particle

    def get_power_norm(self) -> float:
        """
        Calculate the power normalisation for an electron in a circular
        waveguide coupling to the TE11 mode

        Returns
        -------
        float
            Power factor in Watts
        """
        def alphaIntegrand(rho, kc):
            return rho * ((j1(kc * rho) / (kc * rho))**2 + jvp(1, kc * rho)**2)

        kc = 1.841 / self._waveguide.wgR
        alpha, _ = quad(alphaIntegrand, 0, self._waveguide.wgR, args=kc)
        r_gyro = np.sqrt(self._particle.get_position()[0]**2 +
                         self._particle.get_position()[1]**2)

        if isinstance(self._trap, BaseTrap):
            prefactor = self._waveguide.calc_te11_impedance(
                self._trap.calc_omega_0(self._particle.get_speed(),
                                      self._particle.get_pitch_angle())
            ) * (sc.e * self._particle.get_speed())**2 / (8 * np.pi * alpha)
        elif isinstance(self._trap, BaseField):
            prefactor = self._waveguide.calc_te11_impedance(
                self._trap.calc_omega_0(self._particle)
            ) * (sc.e * self._particle.get_speed())**2 / (8 * np.pi * alpha)
        else:
            prefactor = 0.0

        if r_gyro == 0:
            return prefactor * (jvp(1, kc * r_gyro)**2 + 0.25)
        else:
            return prefactor * (jvp(1, kc * r_gyro)**2 +
                                (j1(kc * r_gyro) / (kc * r_gyro))**2)

    def get_peak_power(self, order: ArrayLike) -> NDArray:
        """
        Calculate the power in a given sideband order

        Parameters
        ----------
        order : ArrayLike
            Sideband order(s)

        Returns
        -------
        NDArray
            Power in Watts
        """
        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")
        return np.abs(self.get_peak_amp(order))**2 * self.get_power_norm()

    def apply_phase_shifts(self, amps: NDArray, orders: ArrayLike,
                           phi_c: float, phi_a: float,
                           negativeFreqs: bool = False) -> NDArray:
        """
        Apply initial cyclotron and axial phase shifts to complex amplitudes

        Parameters
        ----------
        amps : NDArray
            Complex peak amplitudes
        orders : ArrayLike
            Sideband orders corresponding to each amplitude
        phi_c : float
            Initial cyclotron phase in radians
        phi_a : float
            Initial axial phase in radians, applied as n*phi_a for order n
        negativeFreqs : bool
            If True, applies conjugate phase

        Returns
        -------
        NDArray
            Phase-shifted complex amplitudes
        """
        sign = -1 if negativeFreqs else 1
        return amps * np.exp(sign * 1j * (phi_c + np.asarray(orders) * phi_a))

    @overload
    def get_peak_frequency(self, order: int, negativeFreqs: bool = False) -> float: ...
    @overload
    def get_peak_frequency(self, order: NDArray, negativeFreqs: bool = False) -> NDArray: ...

    def get_peak_frequency(self, order, negativeFreqs=False):
        """
        Calculate the frequencies at which spectral components occur

        Parameters
        ----------
        order : int or ArrayLike
            Sideband order(s)
        negativeFreqs : bool
            If True, return negative-frequency components

        Returns
        -------
        float or NDArray
            Frequencies in Hertz
        """
        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")

        if isinstance(self._trap, BaseTrap):
            v0 = self._particle.get_speed()
            pa = self._particle.get_pitch_angle()
            f0 = self._trap.calc_omega_0(v0, pa) / (2 * np.pi)
            fa = self._trap.calc_omega_axial(v0, pa) / (2 * np.pi)
        else:
            f0 = self._trap.calc_omega_0(self._particle) / (2 * np.pi)
            fa = self._trap.calc_omega_axial(self._particle) / (2 * np.pi)

        if negativeFreqs:
            return -f0 - order * fa
        else:
            return f0 + order * fa

    def get_peak_amp(self, order: ArrayLike, negativeFreqs=False) -> NDArray:
        """
        Calculate the complex amplitude of spectral peaks

        Parameters
        ----------
        order : ArrayLike
            Sideband order(s)
        negativeFreqs : bool
            If True, return amplitudes for negative-frequency components

        Returns
        -------
        NDArray
            Complex peak amplitudes
        """
        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")

        if isinstance(self._trap, BaseTrap):
            return self._get_peak_amp_analytical(order, negativeFreqs)
        else:
            return self._get_peak_amp_numerical(order, negativeFreqs)

    def _get_peak_amp_analytical(self, order: NDArray, negativeFreqs: bool) -> NDArray:
        kc = 1.841 / self._waveguide.wgR
        f1 = self.get_peak_frequency(order)
        pitchAngle = self._particle.get_pitch_angle()
        v0 = self._particle.get_speed()
        zmax = self._trap.calc_z_max(pitchAngle)
        beta = np.sqrt((f1 * 2 * np.pi / sc.c)**2 - kc**2)

        if isinstance(self._trap, HarmonicTrap):
            q = self._trap.calc_q(v0, pitchAngle)
            MArr = np.arange(-6, 7, 1)
            order_2d = np.atleast_1d(order)[:, np.newaxis]
            beta_2d = np.atleast_1d(beta)[:, np.newaxis]
            amp = np.sum(jv(MArr, q) * jv(order_2d - 2 * MArr, beta_2d * zmax), axis=1)
            if np.ndim(order) == 0:
                return amp[0]
            return amp

        elif isinstance(self._trap, BathtubTrap):
            omega_a = v0 * np.sin(pitchAngle) / self._trap.get_l0()
            t1 = self._trap.calc_t1(v0, pitchAngle)
            t2 = t1 + np.pi / omega_a
            T = 2 * t2
            omegaAx = self._trap.calc_omega_axial(v0, pitchAngle)
            deltaOmega = self._trap.calc_omega_0(
                v0, pitchAngle) - sc.e * self._trap.get_b0() / (self._particle.get_gamma() * sc.m_e)
            L1 = self._trap.get_l1()

            M_SUM_RANGE = 40
            MArr = np.arange(-M_SUM_RANGE, 21, 1)

            def CalcAlpha_n(n):
                n = np.asarray(n)
                n_broadcast = n[:, np.newaxis]
                omega_term = deltaOmega + n * omegaAx
                A = np.exp(-1j * omega_term * t1 / 2)
                A *= t1 * np.sinc(omega_term * t1 / (2 * np.pi))
                jv_term = jv(MArr, deltaOmega / (2 * omegaAx))
                exp_n_term = np.exp(-1j * n * np.pi / 2)
                sinc_arg = (deltaOmega * t1 / 2 - n_broadcast * np.pi *
                            omegaAx / (2 * omega_a) + MArr * np.pi) / np.pi
                sinc_term = np.sinc(sinc_arg)
                B_sum = np.sum(jv_term * sinc_term, axis=1)
                B = B_sum * exp_n_term * \
                    np.exp(-1j * omega_term * t1 / 2) * np.pi / omega_a
                sign_term = (-1.0)**n
                C = sign_term * A
                D = sign_term * B
                return (A + B + C + D) / T

            def CalcBeta_n(n, klambda):
                n = np.asarray(n)
                vz0 = v0 * np.cos(pitchAngle)
                omegaTerm = n * omegaAx
                E = t1 * np.exp(-1j * omegaTerm * t1 / 2)
                E *= np.sinc((klambda * vz0 - omegaTerm) * t1 / (2 * np.pi))
                G = ((-1.0)**n) * t1 * np.exp(-1j * omegaTerm * t1 / 2)
                G *= np.sinc((klambda * vz0 + omegaTerm) * t1 / (2 * np.pi))
                n_broadcast = n[:, np.newaxis]
                F = np.exp(1j * klambda * L1 / 2 - 1j * omegaTerm * t1 / 2) * np.pi / omega_a
                jv_term = jv(MArr, klambda * zmax)
                sinc_arg = MArr * np.pi / 2 - n_broadcast * \
                    np.pi * omegaAx / (2 * omega_a)
                i_m_n = (1j)**(MArr - n_broadcast)
                sinc_term = np.sinc(sinc_arg / np.pi)
                F_sum = np.sum(jv_term * i_m_n * sinc_term, axis=1)
                F *= F_sum
                H = ((-1.0)**n) * np.exp(-1j * klambda * L1 / 2 -
                                         1j * omegaTerm * t1 / 2) * np.pi / omega_a
                i_minus_m_n = (1j)**(-MArr - n_broadcast)
                H_sum = np.sum(jv_term * i_minus_m_n * sinc_term, axis=1)
                H *= H_sum
                return (E + F + G + H) / T

            order_1d = np.atleast_1d(order)
            beta_1d = np.atleast_1d(beta)
            secondSum = np.arange(-20, 21, 1)
            alpha = CalcAlpha_n(secondSum)
            amps = np.array([
                np.sum(alpha * CalcBeta_n(n_out - secondSum, b))
                for n_out, b in zip(order_1d, beta_1d)
            ])
            if negativeFreqs:
                amps = np.conjugate(amps)
            if np.ndim(order) == 0:
                return amps[0]
            return amps

        else:
            raise TypeError("Trap type currently not supported")

    def _get_peak_amp_numerical(self, order: NDArray, negativeFreqs: bool) -> NDArray:
        t1, z = self._trap.calc_t_vs_z(self._particle)
        interp_t1_z = interp1d(t1, z, kind='cubic')

        N_T_POINTS = 499
        omega_a = self._trap.calc_omega_axial(self._particle)
        omega_0 = self._trap.calc_omega_0(self._particle, N_T_POINTS)
        t, phi_t = self._trap.cyclotron_phase_from_t(self._particle, N_T_POINTS)
        Ta = t[-1]
        z_t = interp_t1_z(t)

        omega_c = 1.841 * sc.c / self._waveguide.wgR
        v_phase = sc.c / np.sqrt(1 - (omega_c / omega_0)**2)
        kLambda = omega_0 / v_phase

        order_1d = np.atleast_1d(order)
        order_reshaped = order_1d.reshape(-1, 1)
        common_phase = np.exp(1j * (phi_t + kLambda * z_t - omega_0 * t))
        order_phase = np.exp(-1j * order_reshaped * omega_a * t)
        integrand = common_phase * order_phase

        amp = simpson(integrand, t, axis=1) / Ta
        if negativeFreqs:
            amp = np.conjugate(amp)
        if np.ndim(order) == 0:
            return amp[0]
        return amp
