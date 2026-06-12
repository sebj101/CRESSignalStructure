"""
Unit tests for SignalGenerator class
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure import (
    SignalGenerator,
    SpectrumCalculator,
    HarmonicTrap,
    BathtubTrap,
    CircularWaveguide,
    Electron,
)

KE = 18600.0            # eV
WG_RADIUS = 5e-3        # m
B0 = 1.0                # T
L0 = 0.5                # m — gives f_axial ~25 MHz at 87°, within 1 GHz Nyquist
PITCH = np.deg2rad(87.0)
POS = np.array([1e-4, 0.0, 0.0])


def _make_harmonic_spec(pitch=PITCH, pos=POS):
    trap = HarmonicTrap(B0, L0)
    wg = CircularWaveguide(WG_RADIUS)
    particle = Electron(KE, pos, pitch)
    return SpectrumCalculator(trap, wg, particle), trap, particle


class TestSignalGeneratorConstruction:
    """Tests for SignalGenerator constructor"""

    def test_valid_signal_generator_creation(self):
        """Test creating a valid SignalGenerator"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, 1e9, lo, 1e-6)
        assert sg is not None

    def test_non_float_sample_rate_raises_type_error(self):
        """Test that a non-float sample rate raises TypeError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        with pytest.raises(TypeError, match="Sample rate must be a float"):
            SignalGenerator(spec, 1, lo, 1e-6)

    def test_non_positive_sample_rate_raises_value_error(self):
        """Test that a negative sample rate raises ValueError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        with pytest.raises(ValueError, match="Sample rate must be positive and finite"):
            SignalGenerator(spec, -1e9, lo, 1e-6)

    def test_zero_sample_rate_raises_value_error(self):
        """Test that zero sample rate raises ValueError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        with pytest.raises(ValueError, match="Sample rate must be positive and finite"):
            SignalGenerator(spec, 0.0, lo, 1e-6)

    def test_infinite_sample_rate_raises_value_error(self):
        """Test that infinite sample rate raises ValueError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        with pytest.raises(ValueError, match="Sample rate must be positive and finite"):
            SignalGenerator(spec, np.inf, lo, 1e-6)

    def test_non_float_lo_freq_raises_type_error(self):
        """Test that a non-float LO frequency raises TypeError"""
        spec, _, _ = _make_harmonic_spec()
        with pytest.raises(TypeError, match="LO frequency must be a float"):
            SignalGenerator(spec, 1e9, 27000000000, 1e-6)

    def test_non_positive_lo_freq_raises_value_error(self):
        """Test that a non-positive LO frequency raises ValueError"""
        spec, _, _ = _make_harmonic_spec()
        with pytest.raises(ValueError, match="LO frequency must be positive and finite"):
            SignalGenerator(spec, 1e9, -1e9, 1e-6)

    def test_non_float_acq_time_raises_type_error(self):
        """Test that a non-float acquisition time raises TypeError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        with pytest.raises(TypeError, match="Acquisition time must be a float"):
            SignalGenerator(spec, 1e9, lo, 1)

    def test_non_positive_acq_time_raises_value_error(self):
        """Test that a non-positive acquisition time raises ValueError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        with pytest.raises(ValueError, match="Acquisition time must be positive and finite"):
            SignalGenerator(spec, 1e9, lo, -1e-6)


class TestSignalGeneratorOutputFormat:
    """Tests for the format and shape of GenerateSignal output"""

    def test_generate_signal_returns_two_arrays(self):
        """Test that GenerateSignal returns a (times, signal) tuple of arrays"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, 1e9, lo, 1e-6)
        times, signal = sg.generate_signal(0)
        assert isinstance(times, np.ndarray)
        assert isinstance(signal, np.ndarray)

    def test_times_has_correct_length(self):
        """Test that the times array length matches sample_rate * acq_time"""
        sample_rate = 1e9
        acq_time = 1e-6
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, sample_rate, lo, acq_time)
        times, _ = sg.generate_signal(0)
        assert len(times) == int(acq_time * sample_rate)

    def test_signal_has_correct_length(self):
        """Test that signal array length matches times array length"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, 1e9, lo, 1e-6)
        times, signal = sg.generate_signal(0)
        assert len(signal) == len(times)

    def test_signal_is_complex_valued(self):
        """Test that the returned signal is complex"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, 1e9, lo, 1e-6)
        _, signal = sg.generate_signal(0)
        assert np.issubdtype(signal.dtype, np.complexfloating)

    def test_non_integer_max_order_raises_type_error(self):
        """Test that a float max_order raises TypeError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, 1e9, lo, 1e-6)
        with pytest.raises(TypeError, match="max_order must be an integer"):
            sg.generate_signal(1.0)

    def test_negative_max_order_raises_value_error(self):
        """Test that a negative max_order raises ValueError"""
        spec, _, _ = _make_harmonic_spec()
        lo = spec.get_peak_frequency(0) - 200e6
        sg = SignalGenerator(spec, 1e9, lo, 1e-6)
        with pytest.raises(ValueError, match="max_order must be finite and >= 0"):
            sg.generate_signal(-1)


class TestSidebandSpacingAnalytical:
    """Tests that sideband separations equal the axial frequency (analytical traps)"""

    def test_n1_sideband_minus_mainband_equals_axial_frequency_harmonic_trap(self):
        """n=1 sideband minus mainband should equal the axial frequency"""
        spec, trap, particle = _make_harmonic_spec()
        f_axial = trap.calc_omega_axial(particle.get_speed(), PITCH) / (2 * np.pi)
        assert np.isclose(spec.get_peak_frequency(1) - spec.get_peak_frequency(0), f_axial)

    def test_negative_n1_sideband_spacing_harmonic_trap(self):
        """Mainband minus n=-1 sideband should also equal the axial frequency"""
        spec, trap, particle = _make_harmonic_spec()
        f_axial = trap.calc_omega_axial(particle.get_speed(), PITCH) / (2 * np.pi)
        assert np.isclose(spec.get_peak_frequency(0) - spec.get_peak_frequency(-1), f_axial)

    def test_higher_order_sidebands_evenly_spaced_harmonic_trap(self):
        """Successive sidebands should be uniformly separated by f_axial"""
        spec, trap, particle = _make_harmonic_spec()
        f_axial = trap.calc_omega_axial(particle.get_speed(), PITCH) / (2 * np.pi)
        orders = np.array([0, 1, 2, 3])
        freqs = np.array([spec.get_peak_frequency(n) for n in orders])
        assert np.allclose(np.diff(freqs), f_axial)

    def test_sideband_spacing_bathtub_trap(self):
        """n=1 sideband minus mainband equals axial frequency for BathtubTrap"""
        trap = BathtubTrap(B0=1.0, L0=0.5, L1=0.05)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, POS, PITCH)
        spec = SpectrumCalculator(trap, wg, particle)

        f_axial = trap.calc_omega_axial(particle.get_speed(), PITCH) / (2 * np.pi)
        assert np.isclose(spec.get_peak_frequency(1) - spec.get_peak_frequency(0), f_axial)


class TestNinetyDegreeMainbandAnalytical:
    """Tests that the 90° mainband equals the local cyclotron frequency"""

    def test_ninety_degree_mainband_equals_cyclotron_frequency_at_start_position(self):
        """
        For a 90° electron at the trap centre (z=0), the mainband should equal the
        relativistic cyclotron frequency at B0. The axial correction in CalcOmega0
        vanishes because z_max=0 at 90°.
        """
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        particle_90 = Electron(KE, np.array([0.0, 0.0, 0.0]), np.pi / 2)
        spec_90 = SpectrumCalculator(trap, wg, particle_90)

        f_expected = sc.e * B0 / (2 * np.pi * sc.m_e * particle_90.get_gamma())
        assert np.isclose(spec_90.get_peak_frequency(0), f_expected)

    def test_ninety_degree_electron_has_zero_sideband_amplitudes(self):
        """For a 90° electron z_max=0, so all sideband amplitudes (n≠0) vanish"""
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        particle_90 = Electron(KE, np.array([0.0, 0.0, 0.0]), np.pi / 2)
        spec_90 = SpectrumCalculator(trap, wg, particle_90)
        for n in (-2, -1, 1, 2):
            assert np.isclose(np.abs(spec_90.get_peak_amp(n)), 0.0, atol=1e-12)
