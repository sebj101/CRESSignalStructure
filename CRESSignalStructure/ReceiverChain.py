"""
ReceiverChain.py

Classes for modeling downmixing and digitization of CRES signals.

This module provides tools for simulating the receiver chain, including
local oscillator mixing and analog-to-digital conversion.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


class ReceiverChain:
    """
    Model for downmixing and digitization of RF signals

    This class simulates the receiver chain for CRES signal detection:
    1. Mixing with a local oscillator (LO) to produce IF signal
    2. Low-pass filtering to remove upper sideband
    3. Optional amplification with receiver chain gain
    4. Analog-to-digital conversion at specified sample rate

    The downmixing process converts a high-frequency RF signal at frequency
    f_RF to an intermediate frequency (IF) signal at f_IF = |f_RF - f_LO|.

    Attributes
    ----------
    sample_rate : float
        ADC sample rate in Hz
    lo_frequency : float
        Local oscillator frequency in Hz
    receiver_gain : float
        Linear voltage gain of receiver chain (dimensionless)
    """

    def __init__(self, sample_rate: float, lo_frequency: float,
                 receiver_gain: float = 1.0):
        """
        Initialize ReceiverChain

        Parameters
        ----------
        sample_rate : float
            ADC sample rate in Hz
        lo_frequency : float
            Local oscillator frequency in Hz
        receiver_gain : float, optional
            Linear voltage gain of the receiver chain (default 1.0)
            For gain in dB: receiver_gain = 10^(gain_dB/20)

        Raises
        ------
        TypeError
            If parameters are not numeric
        ValueError
            If parameters are not positive and finite

        Examples
        --------
        >>> # Create digitizer with 200 MHz sample rate, 26 GHz LO, 60 dB gain
        >>> digitizer = ReceiverChain(
        ...     sample_rate=200e6,
        ...     lo_frequency=26e9,
        ...     receiver_gain=10**(60/20)
        ... )
        """
        self._sample_rate = self._validate_positive_finite(
            sample_rate, "sample_rate")
        self._lo_frequency = self._validate_positive_finite(
            lo_frequency, "lo_frequency")
        self._receiver_gain = self._validate_positive_finite(
            receiver_gain, "receiver_gain")

    def _validate_positive_finite(self, value, name: str) -> float:
        """
        Validate that a parameter is a positive finite number

        Parameters
        ----------
        value : numeric
            Value to validate
        name : str
            Parameter name for error messages

        Returns
        -------
        float
            Validated value as float

        Raises
        ------
        TypeError
            If value is not numeric
        ValueError
            If value is not positive and finite
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return float(value)

    def _downmix(self, time: NDArray, signal: NDArray) -> NDArray:
        """
        Downmix an RF signal to IF using the local oscillator

        This performs IQ (in-phase/quadrature) downconversion:
        I(t) = signal(t) * cos(2π f_LO t)
        Q(t) = signal(t) * sin(2π f_LO t)

        The complex IF signal is: IF(t) = I(t) + j*Q(t)

        Parameters
        ----------
        time : NDArray
            Time array in seconds, shape (N,)
        signal : NDArray
            Real-valued RF signal, shape (N,)

        Returns
        -------
        NDArray
            Complex IF signal after downmixing, shape (N,)
            Real part is I (in-phase), imaginary part is Q (quadrature)

        Raises
        ------
        ValueError
            If time and signal have different lengths or if signal frequency
            would cause aliasing
        """
        # Validate inputs
        time = np.asarray(time)
        signal = np.asarray(signal)

        if time.ndim != 1:
            raise ValueError("Time must be a 1D array")
        if signal.ndim != 1:
            raise ValueError("Signal must be a 1D array")
        if len(time) != len(signal):
            raise ValueError(
                f"Time and signal must have same length "
                f"(got {len(time)} and {len(signal)})"
            )

        # Generate LO signals for IQ mixing
        lo_signal = np.exp(-2j * np.pi * self._lo_frequency * time)

        # Combine into complex IF signal
        if_signal = signal * lo_signal
        return if_signal

    def _lowpass_filter(self, if_signal: NDArray) -> NDArray:
        """
        Apply low-pass filter to remove upper sideband after mixing

        Uses a Butterworth filter for the low-pass filter with the cutoff 
        frequency set at the Nyquist frequency of the digitizer 

        Parameters
        ----------
        if_signal : NDArray
            Complex IF signal to filter

        Returns
        -------
        NDArray
            Filtered complex IF signal
        """

        cutoff = self._sample_rate / 2.
        # We originally generate the signal at 5 times the sample rate so use
        # half that number for the Nyquist frequency
        nyquist_freq = cutoff * 5.
        normalised_cutoff = cutoff / nyquist_freq
        filter_coeffs = butter(6, normalised_cutoff, btype='low', output='sos')
        return sosfiltfilt(filter_coeffs, if_signal)

    def _apply_gain(self, signal: NDArray) -> NDArray:
        """
        Apply receiver chain gain to signal

        Parameters
        ----------
        signal : NDArray
            Input signal (real or complex)

        Returns
        -------
        NDArray
            Amplified signal
        """
        return signal * self._receiver_gain

    def digitize(self, time: NDArray, signal: NDArray,
                 oversample_factor: int) -> tuple[NDArray, NDArray]:
        """
        Complete digitization pipeline: downmix, filter, gain, and resample

        This method performs the full receiver chain processing:
        1. Downmix RF signal to IF using local oscillator
        2. Apply receiver gain
        3. Resample to ADC sample rate

        Parameters
        ----------
        time : NDArray
            Time array in seconds at the signal's native sample rate
        signal : NDArray
            Real-valued RF signal
        oversample_factor : int
            Oversampling factor versus

        Returns
        -------
        tuple[NDArray, NDArray]
            time_digitized : Time array at ADC sample rate
            signal_digitized : Complex digitized IF signal
        """
        # Validate that input sample rate matches or exceeds ADC sample rate
        if len(time) < 2:
            raise ValueError("Need at least 2 time points")

        # Downmix to IF
        if_signal = self._downmix(time, signal)
        # Apply low-pass filter to remove upper sideband
        if_signal = self._lowpass_filter(if_signal)
        # Apply receiver gain
        if_signal = self._apply_gain(if_signal)

        # Resample to ADC sample rate with integer decimation
        time_resampled = time[::oversample_factor]
        if_signal_resampled = if_signal[::oversample_factor]
        return time_resampled, if_signal_resampled

    def get_sample_rate(self) -> float:
        """
        Get the ADC sample rate

        Returns
        -------
        float
            Sample rate in Hz
        """
        return self._sample_rate

    def get_lo_frequency(self) -> float:
        """
        Get the local oscillator frequency

        Returns
        -------
        float
            LO frequency in Hz
        """
        return self._lo_frequency

    def get_receiver_gain(self) -> float:
        """
        Get the receiver chain gain (linear)

        Returns
        -------
        float
            Linear voltage gain
        """
        return self._receiver_gain

    def get_receiver_gain_db(self) -> float:
        """
        Get the receiver chain gain in dB

        Returns
        -------
        float
            Gain in dB (20*log10(linear_gain))
        """
        return 20 * np.log10(self._receiver_gain)

    def set_receiver_gain(self, gain: float) -> None:
        """
        Set the receiver chain gain (linear)

        Parameters
        ----------
        gain : float
            Linear voltage gain (must be positive)

        Raises
        ------
        TypeError
            If gain is not numeric
        ValueError
            If gain is not positive and finite
        """
        self._receiver_gain = self._validate_positive_finite(
            gain, "receiver_gain")
