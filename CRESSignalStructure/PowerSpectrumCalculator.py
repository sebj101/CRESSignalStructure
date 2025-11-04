"""
PowerSpectrumCalculator.py

Class to calculate power spectrum for CRES signals

S. Jones 29-07-25
"""

from CRESSignalStructure.BaseTrap import BaseTrap
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.QTNMTraps import HarmonicTrap, BathtubTrap
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.BaseSpectrumCalculator import BaseSpectrumCalculator
import numpy as np
import scipy.constants as sc
from scipy.special import jv, jvp, j1
from scipy.integrate import quad
from numpy.typing import ArrayLike, NDArray


class PowerSpectrumCalculator(BaseSpectrumCalculator):
    def __init__(self, trap: BaseTrap, waveguide: CircularWaveguide,
                 particle: Particle):
        super().__init__(trap, waveguide, particle)
        self.__trap = trap
        self.__waveguide = waveguide
        self.__particle = particle

    def GetPeakFrequency(self, order: ArrayLike, negativeFreqs=False) -> NDArray:
        """
        Calculate the frequencies at which the components occur

        Parameters
        ----------
        order : ArrayLike
            The order of the peak to calculate the frequency for
        negativeFreqs : bool
            Boolean to return negative frequencies (default false)

        Returns
        -------
        NDArray
            The frequency/frequencies at which components are observed in Hertz
        """
        order = np.asarray(order)
        if not np.issubdtype(order.dtype, np.integer):
            raise TypeError("Order must be an integer")
        if not np.any(np.isfinite(order)):
            raise ValueError("Order must be finite")

        v0 = self.__particle.GetSpeed()
        pa = self.__particle.GetPitchAngle()
        f0 = self.__trap.CalcOmega0(v0, pa) / (2 * np.pi)
        fa = self.__trap.CalcOmegaAxial(v0, pa) / (2*np.pi)
        if negativeFreqs == True:
            return -f0 - order * fa
        else:
            return f0 + order * fa

    def GetPeakAmp(self, order: ArrayLike, negativeFreqs=False) -> NDArray:
        """
        Calculate the peak amplitude in the sidebands for a given peak order

        Parameters
        ----------
        order : ArrayLike
            Order of peak to calculate for
        negativeFreqs : bool
            Boolean to return amps for negative frequencies (default false)

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

        kc = 1.841 / self.__waveguide.wgR  # Consider only TE11 mode
        f1 = self.GetPeakFrequency(order)
        pitchAngle = self.__particle.GetPitchAngle()
        v0 = self.__particle.GetSpeed()
        zmax = self.__trap.CalcZMax(pitchAngle)
        beta = np.sqrt((f1 * 2 * np.pi / sc.c)**2 - kc**2)

        if isinstance(self.__trap, HarmonicTrap):
            q = self.__trap.Calcq(v0, pitchAngle)
            MArr = np.arange(-6, 7, 1)
            amp = np.sum(jv(MArr, q) * jv(order - 2 * MArr, beta * zmax))
            return amp
        elif isinstance(self.__trap, BathtubTrap):
            omega_a = v0 * np.sin(pitchAngle) / self.__trap.GetL0()
            t1 = self.__trap.CalcT1(v0, pitchAngle)
            t2 = t1 + np.pi / omega_a
            T = 2 * t2
            omegaAx = self.__trap.CalcOmegaAxial(v0, pitchAngle)
            deltaOmega = self.__trap.CalcOmega0(
                v0, pitchAngle) - sc.e * self.__trap.GetB0() / (self.__particle.GetGamma() * sc.m_e)
            L1 = self.__trap.GetL1()

            M_SUM_RANGE = 40
            MArr = np.arange(-M_SUM_RANGE, 21, 1)

            def CalcAlpha_n(n):
                """Vectorized version that handles array of n values"""
                n = np.asarray(n)

                # Broadcast n_array for vectorized computation
                # n_array shape: (N,), we want final result shape: (N,)
                # Shape: (N, 1) for broadcasting with MArr
                n_broadcast = n[:, np.newaxis]

                # Vectorized A computation
                omega_term = deltaOmega + n * omegaAx  # Shape: (N,)
                A = np.exp(-1j * omega_term * t1 / 2)
                A *= t1 * np.sinc(omega_term * t1 / (2 * np.pi))

                # Vectorized B computation - this is the tricky part due to the sum over MArr
                # Shape: (len(MArr),) - independent of n
                jv_term = jv(MArr, deltaOmega/(2 * omegaAx))
                exp_n_term = np.exp(-1j * n * np.pi/2)  # Shape: (N,)

                # The sinc term involves both n and MArr, so we need broadcasting
                sinc_arg = (deltaOmega * t1 / 2 - n_broadcast * np.pi *
                            omegaAx / (2 * omega_a) + MArr * np.pi) / np.pi
                # sinc_arg shape: (N, len(MArr))
                sinc_term = np.sinc(sinc_arg)

                # Sum over MArr dimension (axis=1), result shape: (N,)
                # Broadcasting: jv_term is (M,), sinc_term is (N,M)
                B_sum = np.sum(jv_term * sinc_term, axis=1)
                B = B_sum * exp_n_term * \
                    np.exp(-1j * omega_term * t1 / 2) * np.pi / omega_a

                # Vectorized C and D
                sign_term = (-1.0)**n  # Shape: (N,)
                C = sign_term * A
                D = sign_term * B

                return (A + B + C + D) / T

            def CalcBeta_n(n, klambda):
                """Vectorized version of the above handling an array like n"""
                n = np.asarray(n)

                vz0 = v0 * np.cos(pitchAngle)

                # E term is independent of M so can be easily vectorized
                omegaTerm = n * omegaAx          # Shape (N, )
                E = t1 * np.exp(-1j * omegaTerm * t1 / 2)
                E *= np.sinc((klambda * vz0 - omegaTerm) * t1 / (2 * np.pi))

                # Similar situation with G term
                G = ((-1.0)**n) * t1 * np.exp(-1j * omegaTerm * t1 / 2)
                G *= np.sinc((klambda * vz0 + omegaTerm) * t1 / (2 * np.pi))

                # Broadcast n_array for vectorized computation
                # n_array shape: (N,), we want final result shape: (N,)
                # Shape: (N, 1) for broadcasting with MArr
                n_broadcast = n[:, np.newaxis]

                F = np.exp(1j * klambda * L1 / 2 - 1j *
                           omegaTerm * t1 / 2) * np.pi / omega_a

                # Vectorized B computation
                # Shape: (M,) - independent of n
                jv_term = jv(MArr, klambda * zmax)

                # The sinc term involves both n and MArr, so we need broadcasting
                sinc_arg = MArr * np.pi / 2 - n_broadcast * \
                    np.pi * omegaAx / (2 * omega_a)
                i_m_n = (1j)**(MArr - n_broadcast)
                # sinc_arg shape: (N, M)
                sinc_term = np.sinc(sinc_arg / np.pi)

                # Sum over MArr dimension (axis=1), result shape: (N,)
                F_sum = np.sum(jv_term * i_m_n * sinc_term, axis=1)
                F *= F_sum

                # Now do H term
                H = ((-1.0)**n) * np.exp(-1j * klambda * L1 / 2 -
                                         1j * omegaTerm * t1 / 2) * np.pi / omega_a
                i_minus_m_n = (1j)**(-MArr - n_broadcast)
                H_sum = np.sum(jv_term * i_minus_m_n * sinc_term, axis=1)
                H *= H_sum

                return (E + F + G + H) / T

            secondSum = np.arange(-20, 21, 1)
            amp = np.sum(CalcAlpha_n(secondSum) *
                         CalcBeta_n(order - secondSum, beta))

            if negativeFreqs == True:
                return np.conjugate(amp)
            else:
                return amp

        else:
            raise TypeError("Trap type currently not supported")
