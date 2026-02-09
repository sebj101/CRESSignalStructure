"""
Unit tests for ReceiverChain class
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure.ReceiverChain import ReceiverChain


class TestReceiverChainConstruction:
    """Tests for ReceiverChain constructor"""

    def test_valid_receiver_chain_creation(self):
        """Test creating a valid receiver chain"""
        sample_rate = 1e9  # 1 GHz
        lo_frequency = 26e9  # 26 GHz
        receiver_gain = 1000.0  # 60 dB

        rc = ReceiverChain(sample_rate, lo_frequency, receiver_gain)

        assert rc.get_sample_rate() == sample_rate
        assert rc.get_lo_frequency() == lo_frequency
        assert rc.get_receiver_gain() == receiver_gain

    def test_receiver_chain_with_default_gain(self):
        """Test creating receiver chain with default gain (1.0)"""
        rc = ReceiverChain(1e9, 26e9)
        assert rc.get_receiver_gain() == 1.0

    def test_negative_sample_rate_raises_error(self):
        """Test that negative sample rate raises ValueError"""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            ReceiverChain(-1e9, 26e9)

    def test_zero_sample_rate_raises_error(self):
        """Test that zero sample rate raises ValueError"""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            ReceiverChain(0.0, 26e9)

    def test_infinite_sample_rate_raises_error(self):
        """Test that infinite sample rate raises ValueError"""
        with pytest.raises(ValueError, match="sample_rate must be finite"):
            ReceiverChain(np.inf, 26e9)

    def test_nan_sample_rate_raises_error(self):
        """Test that NaN sample rate raises ValueError"""
        with pytest.raises(ValueError, match="sample_rate must be finite"):
            ReceiverChain(np.nan, 26e9)

    def test_negative_lo_frequency_raises_error(self):
        """Test that negative LO frequency raises ValueError"""
        with pytest.raises(ValueError, match="lo_frequency must be positive"):
            ReceiverChain(1e9, -26e9)

    def test_zero_lo_frequency_raises_error(self):
        """Test that zero LO frequency raises ValueError"""
        with pytest.raises(ValueError, match="lo_frequency must be positive"):
            ReceiverChain(1e9, 0.0)

    def test_infinite_lo_frequency_raises_error(self):
        """Test that infinite LO frequency raises ValueError"""
        with pytest.raises(ValueError, match="lo_frequency must be finite"):
            ReceiverChain(1e9, np.inf)

    def test_negative_receiver_gain_raises_error(self):
        """Test that negative receiver gain raises ValueError"""
        with pytest.raises(ValueError, match="receiver_gain must be positive"):
            ReceiverChain(1e9, 26e9, -10.0)

    def test_zero_receiver_gain_raises_error(self):
        """Test that zero receiver gain raises ValueError"""
        with pytest.raises(ValueError, match="receiver_gain must be positive"):
            ReceiverChain(1e9, 26e9, 0.0)

    def test_infinite_receiver_gain_raises_error(self):
        """Test that infinite receiver gain raises ValueError"""
        with pytest.raises(ValueError, match="receiver_gain must be finite"):
            ReceiverChain(1e9, 26e9, np.inf)


class TestDownmixing:
    """Tests for downmixing functionality"""

    def test_downmix_single_tone(self):
        """Test downmixing a single-frequency signal"""
        sample_rate = 1e9
        lo_frequency = 26e9
        signal_frequency = 26.1e9  # 100 MHz above LO

        rc = ReceiverChain(sample_rate, lo_frequency)

        # Create test signal
        duration = 1e-6  # 1 microsecond
        time = np.linspace(0, duration, int(5 * sample_rate * duration))
        signal = np.cos(2 * np.pi * signal_frequency * time)

        # Downmix
        if_signal = rc._downmix(time, signal)

        # Check output is complex
        assert np.iscomplexobj(if_signal)
        assert len(if_signal) == len(time)

    def test_downmix_frequency_translation(self):
        """Test that downmixing correctly translates frequency"""
        sample_rate = 1e9
        lo_frequency = 26e9
        signal_frequency = 26.1e9  # 100 MHz above LO
        expected_if_frequency = signal_frequency - lo_frequency  # 100 MHz

        rc = ReceiverChain(sample_rate, lo_frequency)

        # Create test signal with sufficient duration and sampling
        duration = 10e-6  # 10 microseconds for better frequency resolution
        time = np.linspace(0, duration, int(5 * sample_rate * duration))
        signal = np.cos(2 * np.pi * signal_frequency * time)

        # Downmix
        if_signal = rc._downmix(time, signal)

        # Compute FFT to verify frequency
        fft = np.fft.fft(if_signal)
        freqs = np.fft.fftfreq(len(time), time[1] - time[0])

        # Find peak frequency (positive frequencies only)
        positive_freqs = freqs > 0
        peak_idx = np.argmax(np.abs(fft[positive_freqs]))
        peak_freq = freqs[positive_freqs][peak_idx]

        # Check that peak is near expected IF frequency
        assert np.isclose(peak_freq, expected_if_frequency, rtol=0.001)

    def test_downmix_array_length_mismatch_raises_error(self):
        """Test that mismatched time and signal lengths raise error"""
        rc = ReceiverChain(1e9, 26e9)
        time = np.linspace(0, 1e-6, 1000)
        signal = np.ones(500)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            rc._downmix(time, signal)

    def test_downmix_multidimensional_arrays_raise_error(self):
        """Test that multidimensional arrays raise error"""
        rc = ReceiverChain(1e9, 26e9)
        time = np.linspace(0, 1e-6, 1000).reshape(10, 100)
        signal = np.ones((10, 100))

        with pytest.raises(ValueError, match="1D array"):
            rc._downmix(time, signal)

    def test_downmix_preserves_amplitude(self):
        """Test that downmixing preserves signal amplitude"""
        rc = ReceiverChain(1e9, 26e9)

        # Create test signal with known amplitude
        duration = 1e-6
        time = np.linspace(0, duration, int(5e9 * duration))
        amplitude = 5.0
        signal = amplitude * np.cos(2 * np.pi * 26.1e9 * time)

        # Downmix
        if_signal = rc._downmix(time, signal)

        # Check that peak amplitude is preserved (approximately)
        # Note: amplitude of complex exponential mixing is half the original
        assert np.max(np.abs(if_signal)) > amplitude * 0.4


class TestGainApplication:
    """Tests for gain application"""

    def test_apply_gain_scales_signal(self):
        """Test that gain correctly scales the signal"""
        gain = 100.0
        rc = ReceiverChain(1e9, 26e9, gain)

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        amplified = rc._apply_gain(signal)

        expected = signal * gain
        assert np.allclose(amplified, expected)

    def test_apply_gain_works_with_complex_signals(self):
        """Test that gain works with complex signals"""
        gain = 10.0
        rc = ReceiverChain(1e9, 26e9, gain)

        signal = np.array([1+1j, 2+2j, 3+3j])
        amplified = rc._apply_gain(signal)

        expected = signal * gain
        assert np.allclose(amplified, expected)

    def test_unity_gain_leaves_signal_unchanged(self):
        """Test that unity gain doesn't change signal"""
        rc = ReceiverChain(1e9, 26e9, 1.0)

        signal = np.random.randn(100)
        amplified = rc._apply_gain(signal)

        assert np.allclose(amplified, signal)


class TestLowpassFilter:
    """Tests for low-pass filtering"""

    def test_lowpass_filter_output_length(self):
        """Test that filter preserves signal length"""
        rc = ReceiverChain(1e9, 26e9)

        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        filtered = rc._lowpass_filter(signal)

        assert len(filtered) == len(signal)

    def test_lowpass_filter_returns_complex(self):
        """Test that filter returns complex signal"""
        rc = ReceiverChain(1e9, 26e9)

        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        filtered = rc._lowpass_filter(signal)

        assert np.iscomplexobj(filtered)

    def test_lowpass_filter_attenuates_high_frequencies(self):
        """Test that low-pass filter attenuates frequencies above cutoff"""
        sample_rate = 1e9
        rc = ReceiverChain(sample_rate, 26e9)

        # Create signal with high and low frequency components
        duration = 10e-6
        # Signal is at 5x the sample rate
        time = np.linspace(0, duration, int(5 * sample_rate * duration))

        # Low frequency component (well below Nyquist)
        low_freq = 100e6  # 100 MHz
        # High frequency component (above Nyquist)
        high_freq = 600e6  # 600 MHz (above 500 MHz Nyquist for 1 GHz ADC)

        signal = (np.exp(2j * np.pi * low_freq * time) +
                  0.5 * np.exp(2j * np.pi * high_freq * time))

        filtered = rc._lowpass_filter(signal)

        # FFT to check frequency content
        fft_before = np.fft.fft(signal)
        fft_after = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(time), time[1] - time[0])

        # Find power at high frequency
        high_freq_idx = np.argmin(np.abs(freqs - high_freq))

        # High frequency should be attenuated
        assert np.abs(fft_after[high_freq_idx]) < np.abs(fft_before[high_freq_idx])


class TestDigitizationPipeline:
    """Tests for complete digitization pipeline"""

    def test_digitize_returns_correct_types(self):
        """Test that digitize returns time and signal arrays"""
        rc = ReceiverChain(1e9, 26e9)

        duration = 1e-6
        oversample_factor = 5
        time = np.linspace(0, duration, int(oversample_factor * 1e9 * duration))
        signal = np.cos(2 * np.pi * 26.1e9 * time)

        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)

        assert isinstance(time_dig, np.ndarray)
        assert isinstance(signal_dig, np.ndarray)
        assert np.iscomplexobj(signal_dig)

    def test_digitize_output_length(self):
        """Test that digitization produces correct output length"""
        sample_rate = 1e9
        rc = ReceiverChain(sample_rate, 26e9)

        duration = 1e-6
        oversample_factor = 5
        input_length = int(oversample_factor * sample_rate * duration)
        time = np.linspace(0, duration, input_length)
        signal = np.cos(2 * np.pi * 26.1e9 * time)

        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)

        expected_length = input_length // oversample_factor
        assert len(time_dig) == expected_length
        assert len(signal_dig) == expected_length

    def test_digitize_applies_gain(self):
        """Test that digitization applies receiver gain"""
        gain = 100.0
        rc = ReceiverChain(1e9, 26e9, gain)

        duration = 1e-6
        oversample_factor = 5
        time = np.linspace(0, duration, int(oversample_factor * 1e9 * duration))
        # Use a test signal with known frequency content
        signal = np.cos(2 * np.pi * 26.1e9 * time)

        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)

        # Signal should be scaled by gain (check that RMS is significantly amplified)
        # Original signal has RMS ≈ 1/sqrt(2), after gain should be ~70
        rms = np.sqrt(np.mean(np.abs(signal_dig)**2))
        assert rms > 10.0  # Should be significantly amplified

    def test_digitize_too_few_time_points_raises_error(self):
        """Test that fewer than 2 time points raises error"""
        rc = ReceiverChain(1e9, 26e9)

        time = np.array([0.0])
        signal = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2 time points"):
            rc.digitize(time, signal, 1)

    def test_digitize_time_signal_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise error in digitize"""
        rc = ReceiverChain(1e9, 26e9)

        time = np.linspace(0, 1e-6, 1000)
        signal = np.ones(500)

        with pytest.raises(ValueError, match="same length"):
            rc.digitize(time, signal, 5)

    def test_digitize_produces_expected_if_frequency(self):
        """Test that digitization produces expected IF frequency"""
        sample_rate = 1e9
        lo_frequency = 26e9
        signal_frequency = 26.1e9
        expected_if = signal_frequency - lo_frequency  # 100 MHz

        rc = ReceiverChain(sample_rate, lo_frequency)

        duration = 10e-6
        oversample_factor = 5
        time = np.linspace(0, duration, int(oversample_factor * sample_rate * duration))
        signal = np.cos(2 * np.pi * signal_frequency * time)

        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)

        # Compute FFT
        fft = np.fft.fft(signal_dig)
        freqs = np.fft.fftfreq(len(time_dig), time_dig[1] - time_dig[0])

        # Find peak frequency
        positive_freqs = freqs > 0
        peak_idx = np.argmax(np.abs(fft[positive_freqs]))
        peak_freq = freqs[positive_freqs][peak_idx]

        # Should be close to expected IF frequency
        assert np.isclose(peak_freq, expected_if, rtol=0.05)


class TestGetters:
    """Tests for getter methods"""

    def test_get_sample_rate(self):
        """Test getting sample rate"""
        sample_rate = 1e9
        rc = ReceiverChain(sample_rate, 26e9)
        assert rc.get_sample_rate() == sample_rate

    def test_get_lo_frequency(self):
        """Test getting LO frequency"""
        lo_frequency = 26e9
        rc = ReceiverChain(1e9, lo_frequency)
        assert rc.get_lo_frequency() == lo_frequency

    def test_get_receiver_gain(self):
        """Test getting receiver gain (linear)"""
        gain = 1000.0
        rc = ReceiverChain(1e9, 26e9, gain)
        assert rc.get_receiver_gain() == gain

    def test_get_receiver_gain_db(self):
        """Test getting receiver gain in dB"""
        linear_gain = 1000.0
        expected_db = 20 * np.log10(linear_gain)  # 60 dB
        rc = ReceiverChain(1e9, 26e9, linear_gain)
        assert np.isclose(rc.get_receiver_gain_db(), expected_db)

    def test_get_receiver_gain_db_for_unity_gain(self):
        """Test that unity gain gives 0 dB"""
        rc = ReceiverChain(1e9, 26e9, 1.0)
        assert np.isclose(rc.get_receiver_gain_db(), 0.0)

    def test_get_receiver_gain_db_for_various_gains(self):
        """Test dB conversion for various gain values"""
        test_gains = [1.0, 10.0, 100.0, 1000.0]
        expected_dbs = [0.0, 20.0, 40.0, 60.0]

        for gain, expected_db in zip(test_gains, expected_dbs):
            rc = ReceiverChain(1e9, 26e9, gain)
            assert np.isclose(rc.get_receiver_gain_db(), expected_db)


class TestSetters:
    """Tests for setter methods"""

    def test_set_receiver_gain(self):
        """Test setting receiver gain"""
        rc = ReceiverChain(1e9, 26e9, 10.0)
        new_gain = 100.0
        rc.set_receiver_gain(new_gain)
        assert rc.get_receiver_gain() == new_gain

    def test_set_receiver_gain_updates_db_value(self):
        """Test that setting gain updates dB value"""
        rc = ReceiverChain(1e9, 26e9, 10.0)
        new_gain = 1000.0
        rc.set_receiver_gain(new_gain)
        expected_db = 20 * np.log10(new_gain)
        assert np.isclose(rc.get_receiver_gain_db(), expected_db)

    def test_set_negative_gain_raises_error(self):
        """Test that setting negative gain raises error"""
        rc = ReceiverChain(1e9, 26e9, 10.0)
        with pytest.raises(ValueError, match="receiver_gain must be positive"):
            rc.set_receiver_gain(-5.0)

    def test_set_zero_gain_raises_error(self):
        """Test that setting zero gain raises error"""
        rc = ReceiverChain(1e9, 26e9, 10.0)
        with pytest.raises(ValueError, match="receiver_gain must be positive"):
            rc.set_receiver_gain(0.0)

    def test_set_infinite_gain_raises_error(self):
        """Test that setting infinite gain raises error"""
        rc = ReceiverChain(1e9, 26e9, 10.0)
        with pytest.raises(ValueError, match="receiver_gain must be finite"):
            rc.set_receiver_gain(np.inf)

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_short_signal(self):
        """Test with short but valid signal length for filtering"""
        rc = ReceiverChain(1e9, 26e9)

        # Need at least ~50 points for the filter (padlen=21 requires more)
        duration = 100e-9  # 100 ns
        oversample_factor = 5
        time = np.linspace(0, duration, int(oversample_factor * 1e9 * duration))
        signal = np.cos(2 * np.pi * 26.1e9 * time)

        # Should not raise error
        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)
        assert len(time_dig) >= 1

    def test_lo_frequency_matches_signal_frequency(self):
        """Test when LO frequency equals signal frequency (zero IF)"""
        sample_rate = 1e9
        lo_frequency = 26e9
        rc = ReceiverChain(sample_rate, lo_frequency)

        duration = 1e-6
        oversample_factor = 5
        time = np.linspace(0, duration, int(oversample_factor * sample_rate * duration))
        # Signal at exactly LO frequency -> should produce DC
        signal = np.cos(2 * np.pi * lo_frequency * time)

        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)

        # After filtering and downsampling, should have low frequency content
        assert np.all(np.isfinite(signal_dig))

    def test_signal_below_lo_frequency(self):
        """Test with signal frequency below LO (negative IF)"""
        sample_rate = 1e9
        lo_frequency = 26e9
        signal_frequency = 25.9e9  # 100 MHz below LO

        rc = ReceiverChain(sample_rate, lo_frequency)

        duration = 10e-6
        oversample_factor = 5
        time = np.linspace(0, duration, int(oversample_factor * sample_rate * duration))
        signal = np.cos(2 * np.pi * signal_frequency * time)

        time_dig, signal_dig = rc.digitize(time, signal, oversample_factor)

        # Should handle negative IF frequency
        assert np.all(np.isfinite(signal_dig))
        assert np.max(np.abs(signal_dig)) > 0

    def test_large_gain_values(self):
        """Test with very large gain values (80 dB)"""
        gain_db = 80
        gain_linear = 10**(gain_db/20)  # 10,000
        rc = ReceiverChain(1e9, 26e9, gain_linear)

        assert np.isclose(rc.get_receiver_gain_db(), gain_db)

        # Test that large gain doesn't cause issues
        signal = np.array([0.001, 0.002, 0.003])
        amplified = rc._apply_gain(signal)
        assert np.allclose(amplified, signal * gain_linear)

    def test_very_small_gain(self):
        """Test with very small gain value (attenuation)"""
        gain = 0.01  # -40 dB
        rc = ReceiverChain(1e9, 26e9, gain)

        expected_db = 20 * np.log10(gain)
        assert np.isclose(rc.get_receiver_gain_db(), expected_db)


class TestSignalProcessingAccuracy:
    """Tests for signal processing accuracy and consistency"""

    def test_downmix_preserves_phase_relationships(self):
        """Test that downmixing preserves relative phase"""
        rc = ReceiverChain(1e9, 26e9)

        duration = 10e-6  # Longer duration for better phase estimation
        time = np.linspace(0, duration, int(5e9 * duration))

        # Create two signals with known phase relationship
        freq = 26.1e9
        signal1 = np.cos(2 * np.pi * freq * time)
        signal2 = np.cos(2 * np.pi * freq * time + np.pi/4)  # 45° phase shift

        if1 = rc._downmix(time, signal1)
        if2 = rc._downmix(time, signal2)

        # Use mean angle over middle section for better estimate
        mid_start = len(if1) // 3
        mid_end = 2 * len(if1) // 3
        phase1 = np.angle(np.mean(if1[mid_start:mid_end]))
        phase2 = np.angle(np.mean(if2[mid_start:mid_end]))
        phase_diff = np.angle(np.exp(1j * (phase2 - phase1)))  # Wrap to [-π, π]

        # Should be close to π/4
        assert np.isclose(np.abs(phase_diff), np.pi/4, atol=0.15)

    def test_gain_linearity(self):
        """Test that gain application is linear"""
        rc = ReceiverChain(1e9, 26e9, 10.0)

        signal1 = np.array([1.0, 2.0, 3.0])
        signal2 = np.array([2.0, 4.0, 6.0])  # Double of signal1

        amp1 = rc._apply_gain(signal1)
        amp2 = rc._apply_gain(signal2)

        # amp2 should be double of amp1
        assert np.allclose(amp2, 2 * amp1)

    def test_filter_does_not_introduce_dc_offset(self):
        """Test that filtering doesn't add DC offset to zero-mean signal"""
        rc = ReceiverChain(1e9, 26e9)

        # Create zero-mean signal
        signal = np.random.randn(10000) + 1j * np.random.randn(10000)
        signal = signal - np.mean(signal)  # Ensure zero mean

        filtered = rc._lowpass_filter(signal)

        # Mean should remain close to zero
        assert np.abs(np.mean(filtered)) < 0.1