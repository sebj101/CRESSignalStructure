"""
Integration tests for SignalGenerator class.

These tests generate full time-domain signals and verify peak positions via FFT.
They cover both analytical (BaseTrap) and numerical (BaseField) spectrum calculator
paths via the unified SpectrumCalculator class.

Note: tests using BaseField are slower due to the numerical integration involved
in computing the axial frequency and amplitudes.
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure import (
    SignalGenerator,
    SpectrumCalculator,
    HarmonicTrap,
    BathtubTrap,
    HarmonicField,
    CircularWaveguide,
    Electron,
)

KE = 18600.0        # eV
WG_RADIUS = 5e-3    # m
B0 = 1.0            # T
L0 = 0.5            # m
PITCH_87 = np.deg2rad(87.0)
SAMPLE_RATE = 1e9   # Hz (1 GHz ADC)
ACQ_TIME = 5e-6     # s  (200 kHz frequency resolution)
FREQ_BIN = 1 / ACQ_TIME  # Hz

# HarmonicField parameters matching setup from test_trap_configs.py
R_COIL = 3e-2               # m
TRAP_DEPTH = 4e-3           # T
I_COIL = 2 * TRAP_DEPTH * R_COIL / sc.mu_0
BKG_FIELD = 1.0             # T


def _fft_peak_near(signal, sample_rate, f_centre, bandwidth):
    """Return the FFT bin frequency closest to f_centre within +/-bandwidth."""
    freqs = np.fft.fftfreq(len(signal), d=1 / sample_rate)
    spectrum = np.abs(np.fft.fft(signal))
    mask = np.abs(freqs - f_centre) < bandwidth
    if not np.any(mask):
        raise ValueError(f"No FFT bins found within {bandwidth} Hz of {f_centre} Hz")
    return freqs[mask][np.argmax(spectrum[mask])]


class TestSignalPeakFrequenciesAnalytical:
    """Full-pipeline signal tests using SpectrumCalculator (analytical traps)"""

    def test_ninety_degree_mainband_at_correct_if_frequency(self):
        """
        For a 90° electron the mainband should be at the relativistic cyclotron
        frequency for the field at the start position. After downmixing, the
        IF peak should be at f_0 - lo_freq.
        """
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, np.array([0.0, 0.0, 0.0]), np.pi / 2)
        spec = SpectrumCalculator(trap, wg, particle)

        f_0 = spec.GetPeakFrequency(0)
        lo_freq = f_0 - 200e6
        f_if_expected = 200e6

        sg = SignalGenerator(spec, SAMPLE_RATE, lo_freq, ACQ_TIME)
        _, signal = sg.GenerateSignal(0)

        f_if_measured = _fft_peak_near(signal, SAMPLE_RATE, f_if_expected, 50e6)
        assert abs(f_if_measured - f_if_expected) < 2 * FREQ_BIN

    def test_sideband_separation_in_generated_signal_harmonic_trap(self):
        """
        For a sub-90° electron with HarmonicTrap, the sideband peaks in the
        generated signal should be separated by f_axial.
        """
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, np.array([1e-4, 0.0, 0.0]), PITCH_87)
        spec = SpectrumCalculator(trap, wg, particle)

        f_0 = spec.GetPeakFrequency(0)
        f_axial = trap.CalcOmegaAxial(particle.GetSpeed(), PITCH_87) / (2 * np.pi)
        lo_freq = f_0 - 200e6

        sg = SignalGenerator(spec, SAMPLE_RATE, lo_freq, ACQ_TIME)
        _, signal = sg.GenerateSignal(9)

        f_n0 = _fft_peak_near(signal, SAMPLE_RATE, 200e6, 10e6)
        f_n1 = _fft_peak_near(signal, SAMPLE_RATE, 200e6 + f_axial, 10e6)
        assert abs((f_n1 - f_n0) - f_axial) < 2 * FREQ_BIN

    def test_sideband_separation_in_generated_signal_bathtub_trap(self):
        """
        For a sub-90° electron with BathtubTrap, the sideband peaks in the
        generated signal should also be separated by f_axial.
        """
        trap = BathtubTrap(B0=1.0, L0=0.5, L1=0.05)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, np.array([1e-4, 0.0, 0.0]), PITCH_87)
        spec = SpectrumCalculator(trap, wg, particle)

        f_0 = spec.GetPeakFrequency(0)
        f_axial = trap.CalcOmegaAxial(particle.GetSpeed(), PITCH_87) / (2 * np.pi)
        lo_freq = f_0 - 200e6

        sg = SignalGenerator(spec, SAMPLE_RATE, lo_freq, ACQ_TIME)
        _, signal = sg.GenerateSignal(1)

        f_n0 = _fft_peak_near(signal, SAMPLE_RATE, 200e6, 5e6)
        f_n1 = _fft_peak_near(signal, SAMPLE_RATE, 200e6 + f_axial, 5e6)
        assert abs((f_n1 - f_n0) - f_axial) < 2 * FREQ_BIN


class TestSignalPeakFrequenciesNumerical:
    """
    Signal tests using NumericalSpectrumCalculator with a HarmonicField.

    Note: NumericalSpectrumCalculator cannot handle pitch angle = 90° because
    the numerical root-finding for z_max requires a sign change, which does not
    occur when z_max = 0 (i.e. a 90° electron).
    """

    def test_sideband_separation_in_generated_signal_numerical(self):
        """
        For a sub-90° electron with HarmonicField (NumericalSpectrumCalculator),
        the sideband peaks in the generated signal should be separated by f_axial.
        """
        trap = HarmonicField(R_COIL, I_COIL, BKG_FIELD)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, np.array([0.0, 0.0, 0.0]), PITCH_87)
        spec = SpectrumCalculator(trap, wg, particle)

        f_axial = trap.CalcOmegaAxial(particle) / (2 * np.pi)
        f_0 = spec.GetPeakFrequency(0)
        lo_freq = f_0 - 200e6

        sg = SignalGenerator(spec, SAMPLE_RATE, lo_freq, ACQ_TIME)
        _, signal = sg.GenerateSignal(1)

        f_n0 = _fft_peak_near(signal, SAMPLE_RATE, 200e6, 5e6)
        f_n1 = _fft_peak_near(signal, SAMPLE_RATE, 200e6 + f_axial, 5e6)
        assert abs((f_n1 - f_n0) - f_axial) < 2 * FREQ_BIN
