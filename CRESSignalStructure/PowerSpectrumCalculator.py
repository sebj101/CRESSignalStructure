"""
PowerSpectrumCalculator.py

Class to calculate power spectrum for CRES signals

S. Jones 29-07-25
"""

from CRESSignalStructure.BaseTrap import BaseTrap
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.QTNMTraps import HarmonicTrap, BathtubTrap
from CRESSignalStructure.Particle import Particle
import numpy as np
import scipy.constants as sc
from scipy.special import jv, jvp, j1
from scipy.integrate import quad


class PowerSpectrumCalculator:
    def __init__(self, trap: BaseTrap, waveguide: CircularWaveguide,
                 particle: Particle):
        self.__trap = trap
        self.__waveguide = waveguide
        self.__particle = particle

    def GetPeakFrequency(self, order: int):
        """
        Calculate the frequencies at which the components occur

        Parameters
        ----------
        order : int
            The order of the peak to calculate the frequency for

        Returns
        -------
        tuple(float, float)
            The frequency or frequencies at which components are observed in
            radians/s
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise TypeError("Order must be positive")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        # Treat the n = 0 case separately
        v0 = self.__particle.GetSpeed()
        pa = self.__particle.GetPitchAngle()
        return ((self.__trap.CalcOmega0(v0, pa) + order * self.__trap.CalcOmegaAxial(v0, pa)) / (2*np.pi),
                (self.__trap.CalcOmega0(v0, pa) - order * self.__trap.CalcOmegaAxial(v0, pa)) / (2*np.pi))

    def GetPeakAmp(self, order: int):
        """
        Calculate the peak amplitude in the sidebands for a given peak order

        Parameters
        ----------
        order : int
            Order of sideband

        Returns
        -------
        tuple(float, float)
            Peak amplitudes
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise TypeError("Order must be positive")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        kc = 1.841 / self.__waveguide.wgR  # Consider only TE11 mode
        f1, f2 = self.GetPeakFrequency(order)
        pitchAngle = self.__particle.GetPitchAngle()
        v0 = self.__particle.GetSpeed()
        zmax = self.__trap.CalcZMax(pitchAngle)
        beta1 = np.sqrt((f1 * 2 * np.pi / sc.c)**2 - kc**2)
        beta2 = np.sqrt((f2 * 2 * np.pi / sc.c)**2 - kc**2)

        if isinstance(self.__trap, HarmonicTrap):
            q = self.__trap.Calcq(v0, pitchAngle)
            MArr = np.arange(-6, 7, 1)
            amp1 = np.sum(jv(MArr, q) * jv(order - 2 * MArr, beta1 * zmax))
            amp2 = np.sum(jv(MArr, q) * jv(-order - 2 * MArr, beta2 * zmax))
            return (amp1, amp2)
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
            amp1 = np.sum(CalcAlpha_n(secondSum) *
                          CalcBeta_n(order - secondSum, beta1))
            amp2 = np.sum(CalcAlpha_n(secondSum) *
                          CalcBeta_n(-order - secondSum, beta2))
            return (amp1, amp2)

        else:
            raise TypeError("Trap type currently not supported")

    def GetPowerNorm(self):
        """
        Calculate the power normalisation for the electron in this case

        Returns
        -------
        float
            Power factor in Watts
        """

        def alphaIntegrand(rho, kc):
            """
            Integrand for alpha calculation
            """
            return rho * ((j1(kc * rho) / (kc * rho))**2 + jvp(1, kc * rho)**2)

        kc = 1.841 / self.__waveguide.wgR
        alpha, _ = quad(alphaIntegrand, 0, self.__waveguide.wgR, args=kc)
        r_gyro = np.sqrt(self.__particle.GetPosition()[
                         0]**2 + self.__particle.GetPosition()[1]**2)

        prefactor = self.__waveguide.CalcTE11Impedance(self.__trap.CalcOmega0(
            self.__particle.GetSpeed(), self.__particle.GetPitchAngle())) * (sc.e * self.__particle.GetSpeed())**2 / (8 * np.pi * alpha)

        if r_gyro == 0:
            # Handle the special case where r_gyro = 0
            # lim(x->0) j1(x)/x = 1/2, so (j1(x)/x)^2 = 1/4
            return prefactor * (jvp(1, kc * r_gyro)**2 + 0.25)
        else:
            return prefactor * (jvp(1, kc * r_gyro)**2 + (j1(kc * r_gyro) / (kc * r_gyro))**2)

    def GetSpectrumPowers(self, order: int):
        """
        Calculates the power in a given order sideband

        Parameters
        ----------
        order : int
            Order of sideband

        Returns
        -------
        tuple
            Power in sidebands of order 'order' in Watts
        """
        if not isinstance(order, int):
            raise TypeError("Order must be an integer")
        if order < 0:
            raise TypeError("Order must be positive")
        if not np.isfinite(order):
            raise ValueError("Order must be finite")

        a1, a2 = self.GetPeakAmp(order)
        return (np.abs(a1)**2 * self.GetPowerNorm(), np.abs(a2)**2 * self.GetPowerNorm())
