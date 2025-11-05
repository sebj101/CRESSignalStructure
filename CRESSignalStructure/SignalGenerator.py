"""
SignalGenerator.py

Contains implementation of SignalGenerator class which allows for time series 
signal generation of CRES signals. This includes simulation of:
- Downmixing
- Filtering
- Sampling
- Signal chirp
"""

from CRESSignalStructure.BaseSpectrumCalculator import BaseSpectrumCalculator
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt
import scipy.constants as sc


class SignalGenerator:
    def __init__(self, spectrum_calc: BaseSpectrumCalculator,
                 sample_rate: float, lo_freq: float, acq_time: float):
        """
        Constructor for SignalGenerator class

        Parameters
        ----------
        spectrum_calc : BaseSpectrumCalculator
            Instance of PowerSpectrumCalculator or NumericalSpectrumCalculator
        sample_rate : float
            Digitizer sample rate in Hertz
        lo_freq : float
            Local oscillator frequency in Hertz
        acq_time : float
            Total acquisition time in seconds
        """
        self.__sample_rate = sample_rate
        self.__lo_freq = lo_freq
        self.__acq_time = acq_time
        self.__spec_calc = spectrum_calc

    def GenerateSignal(self, max_order: int) -> NDArray:
        """
        Main method orchestrating signal generation

        Parameters
        ----------
        max_order : int
            Maximum order of sideband to calculate

        Returns
        -------
        NDArray 
            A 1D array of complex numbers representing the time series signal
        """
        FAST_SAMPLE_FACTOR = 5
        FAST_SAMPLE_FREQ = FAST_SAMPLE_FACTOR * self.__sample_rate
        N_SAMPLES = int(self.__acq_time * FAST_SAMPLE_FREQ)
        times_fast_sample = np.linspace(
            0, N_SAMPLES / FAST_SAMPLE_FREQ, N_SAMPLES)

        # Get the fourier amplitudes and frequencies
        orders = np.arange(-max_order, max_order+1, 1)
        fourier_freqs = self.__spec_calc.GetPeakFrequency(orders, False)
        fourier_freqs = np.append(
            fourier_freqs, self.__spec_calc.GetPeakFrequency(orders, True))
        fourier_amps = self.__spec_calc.GetPeakAmp(orders, False)
        fourier_amps = np.append(
            fourier_amps, self.__spec_calc.GetPeakAmp(orders, True))

        # Calculate the chirp rate
        beta = self.__spec_calc.__particle.GetBeta()
        gamma = self.__spec_calc.__particle.GetGamma()
        central_freq = self.__spec_calc.GetPeakFrequency(0)
        chirp_rate_ang = sc.e**2 * (2 * np.pi * central_freq)**3 * gamma * \
            beta**2 / (6 * np.pi * sc.epsilon_0 * sc.c) / \
            (sc.m_e * sc.c**2)  # radians per second squared

        # Generate the LO signal for I/Q demodulation and add in the chirp
        lo_signal = np.exp(-2j * np.pi * self.__lo_freq * times_fast_sample)
        rf_signal_dm = np.sum(fourier_amps[:, np.newaxis] * np.exp(2j * np.pi * fourier_freqs[:, np.newaxis]
                              * times_fast_sample), axis=0) * lo_signal * np.exp(1j * chirp_rate_ang * times_fast_sample**2)

        # Generate a Butterworth filter and filter the signal
        sos = butter(N=8, Wn=self.__sample_rate / 2, btype='low', output='sos',
                     fs=self.__sample_rate * FAST_SAMPLE_FACTOR)
        rf_signal_filtered = sosfilt(sos, rf_signal_dm, zi=None)

        # Return reduced signal
        return rf_signal_filtered[::FAST_SAMPLE_FACTOR]
