"""
test_antenna_signal_generator.py

Integration tests for the AntennaSignalGenerator signal generation pipeline.
Covers output structure and CRES physics (mainband and sideband structure).
"""

import numpy as np
import scipy.constants as sc
from scipy.signal import periodogram
import pytest

from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.TrajectoryGenerator import TrajectoryGenerator
from CRESSignalStructure.AntennaSignalGenerator import AntennaSignalGenerator
from CRESSignalStructure.antennas import IsotropicAntenna
from CRESSignalStructure.ReceiverChain import ReceiverChain


# ---------------------------------------------------------------------------
# Shared physical parameters
# ---------------------------------------------------------------------------

R_COIL = 3e-2                               # m
I_COIL = 2 * 4e-3 * R_COIL / sc.mu_0       # A

KE           = 18.6e3    # eV
ADC_RATE     = 1e9     # Hz
LO_OFFSET    = 50e6       # Hz below f_c → IF frequency
OVERSAMPLING = 5         # signal rate = ADC_RATE * OVERSAMPLING = 1 GHz
TRAJ_RATE    = 7e9       # Hz  (> ADC_RATE × OVERSAMPLING = 1 GHz)
ANTENNA_POS  = np.array([0.1, 0.0, 0.0])


def _make_field():
    return HarmonicField(R_COIL, I_COIL, 1.0)


def _make_particle(pitch_angle):
    return Particle(ke=KE,
                    startPos=np.array([0.001, 0.0, 0.0]),
                    pitchAngle=pitch_angle)


# ---------------------------------------------------------------------------
# Module-scope fixtures — expensive: computed once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def perpendicular_signal():
    """
    Signal pipeline for a 90-degree (perpendicular) pitch angle electron.
    """
    field    = _make_field()
    particle = _make_particle(np.pi / 2)
    f_c      = field.calc_omega_0(particle) / (2 * np.pi)

    traj     = TrajectoryGenerator(field, particle).generate(
                   sample_rate=TRAJ_RATE, t_max=10e-6)
    receiver = ReceiverChain(sample_rate=ADC_RATE,
                             lo_frequency=f_c - LO_OFFSET)
    gen      = AntennaSignalGenerator(
                   traj, IsotropicAntenna(ANTENNA_POS), receiver,
                   oversampling_factor=OVERSAMPLING)

    time, signal = gen.generate_signal(return_time=True)
    return time, signal, LO_OFFSET, receiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spectrum(signal, sample_rate):
    """Return (freqs, spectrum) for a complex signal."""
    freqs, Pxx   = periodogram(signal, sample_rate, window='hann', 
                               scaling='spectrum')
    return freqs, Pxx


def _power_in_window(freqs, psd, f_lo, f_hi):
    """Sum spectrum in the frequency window [f_lo, f_hi]."""
    return np.sum(psd[(freqs >= f_lo) & (freqs <= f_hi)])


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestAntennaSignalGeneratorConstruction:

    def setup_method(self):
        field         = _make_field()
        particle      = _make_particle(np.pi / 2)
        f_c           = field.calc_omega_0(particle) / (2 * np.pi)
        self.traj     = TrajectoryGenerator(field, particle).generate(
                            sample_rate=TRAJ_RATE, t_max=0.1e-6)
        self.antenna  = IsotropicAntenna(ANTENNA_POS)
        self.receiver = ReceiverChain(sample_rate=ADC_RATE,
                                      lo_frequency=f_c - LO_OFFSET)

    def test_valid_construction(self):
        """Constructs without error given valid inputs."""
        gen = AntennaSignalGenerator(self.traj, self.antenna, self.receiver,
                                     oversampling_factor=OVERSAMPLING)
        assert gen is not None

    def test_wrong_trajectory_type(self):
        """Raises TypeError if trajectory is not a Trajectory."""
        with pytest.raises(TypeError):
            AntennaSignalGenerator("not_a_trajectory", self.antenna,
                                   self.receiver,
                                   oversampling_factor=OVERSAMPLING)

    def test_wrong_antenna_type(self):
        """Raises TypeError if antenna is not a BaseAntenna."""
        with pytest.raises(TypeError):
            AntennaSignalGenerator(self.traj, "not_an_antenna",
                                   self.receiver,
                                   oversampling_factor=OVERSAMPLING)

    def test_wrong_receiver_chain_type(self):
        """Raises TypeError if receiver chain has wrong type."""
        with pytest.raises(TypeError):
            AntennaSignalGenerator(self.traj, self.antenna,
                                   "not_a_receiver",
                                   oversampling_factor=OVERSAMPLING)

    def test_non_integer_oversampling(self):
        """Raises TypeError if oversampling factor is not an int."""
        with pytest.raises(TypeError):
            AntennaSignalGenerator(self.traj, self.antenna, self.receiver,
                                   oversampling_factor=2.5)

    def test_negative_oversampling(self):
        """Raises ValueError if oversampling factor is less than 1."""
        with pytest.raises(ValueError):
            AntennaSignalGenerator(self.traj, self.antenna, self.receiver,
                                   oversampling_factor=0)

    def test_trajectory_sample_rate_too_low(self):
        """Raises ValueError when trajectory sample rate ≤ ADC rate × oversampling."""
        field     = _make_field()
        particle  = _make_particle(np.pi / 2)
        # 500 MHz < ADC_RATE (200 MHz) × OVERSAMPLING (5) = 1 GHz
        slow_traj = TrajectoryGenerator(field, particle).generate(
                        sample_rate=500e6, t_max=0.1e-6)
        with pytest.raises(ValueError):
            AntennaSignalGenerator(slow_traj, self.antenna, self.receiver,
                                   oversampling_factor=OVERSAMPLING)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

class TestSignalOutputStructure:

    def test_returns_tuple_of_two_arrays(self, perpendicular_signal):
        """generate_signal(return_time=True) returns a (time, signal) tuple."""
        time, signal, *_ = perpendicular_signal
        assert isinstance(time, np.ndarray)
        assert isinstance(signal, np.ndarray)

    def test_signal_only_mode(self):
        """generate_signal(return_time=False) returns a single 1D array."""
        field    = _make_field()
        particle = _make_particle(np.pi / 2)
        f_c      = field.calc_omega_0(particle) / (2 * np.pi)
        traj     = TrajectoryGenerator(field, particle).generate(
                       sample_rate=TRAJ_RATE, t_max=1e-6)
        gen      = AntennaSignalGenerator(
                       traj, IsotropicAntenna(ANTENNA_POS),
                       ReceiverChain(sample_rate=ADC_RATE,
                                     lo_frequency=f_c - LO_OFFSET),
                       oversampling_factor=OVERSAMPLING)
        result   = gen.generate_signal(return_time=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_signal_is_complex(self, perpendicular_signal):
        """Digitized signal array is complex-valued."""
        _, signal, *_ = perpendicular_signal
        assert np.iscomplexobj(signal)

    def test_time_and_signal_same_length(self, perpendicular_signal):
        """Time and signal arrays have the same length."""
        time, signal, *_ = perpendicular_signal
        assert len(time) == len(signal)

    def test_time_spacing_matches_adc_rate(self, perpendicular_signal):
        """Mean time step equals 1 / ADC sample rate."""
        time, _, _, receiver = perpendicular_signal
        np.testing.assert_allclose(
            np.mean(np.diff(time)), 1.0 / receiver.get_sample_rate(), rtol=1e-3)

    def test_signal_has_nonzero_power(self, perpendicular_signal):
        """Signal carries non-zero power."""
        _, signal, *_ = perpendicular_signal
        assert np.mean(np.abs(signal) ** 2) > 0.0


# ---------------------------------------------------------------------------
# Mainband physics (90-degree pitch angle)
# ---------------------------------------------------------------------------

class TestMainbandPhysics:

    def test_mainband_frequency_matches_expected(self, perpendicular_signal):
        """
        Peak IF frequency matches f_c - f_LO within 0.1%.

        A perpendicularly-orbiting electron radiates at the cyclotron frequency;
        after downmixing by the LO this must appear at exactly f_c - f_LO.
        """
        _, signal, f_IF, receiver = perpendicular_signal
        freqs, psd = _spectrum(signal, receiver.get_sample_rate())
        peak_f     = abs(freqs[np.argmax(psd)])
        np.testing.assert_allclose(peak_f, f_IF, rtol=0.001)

    def test_all_power_in_mainband_for_perpendicular_electron(
            self, perpendicular_signal):
        """
        For a 90-degree pitch angle electron, >90% of power is in the mainband.

        A perpendicular electron has no axial motion and therefore no sideband
        modulation. All radiated power must be concentrated at the cyclotron
        frequency, with no sidebands in the IF signal.
        """
        time, signal, f_IF, receiver = perpendicular_signal
        adc_rate   = receiver.get_sample_rate()
        freqs, psd = _spectrum(signal, adc_rate)
        freq_res   = 1.0 / time[-1]
        peak_f     = freqs[np.argmax(psd)]

        mainband_power = _power_in_window(
            freqs, psd, peak_f - 3 * freq_res, peak_f + 3 * freq_res)

        assert mainband_power / np.sum(psd) > 0.95