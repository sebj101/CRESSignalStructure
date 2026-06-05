"""
Unit tests for ScatteringSimulator
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure import (
    HarmonicField,
    CircularWaveguide,
    Electron,
    SignalGenerator,
    SpectrumCalculator,
    ScatteringSimulator,
)
from CRESSignalStructure.scattering import BaseCrossSection, GasModel

KE = 18600.0
WG_RADIUS = 5e-3
PITCH = np.deg2rad(89.0)
POS = np.array([1e-4, 0.0, 0.0])
SAMPLE_RATE = 1e9
MAX_ORDER = 3
R_COIL = 3e-2
TRAP_DEPTH = 4e-3
I_COIL = 2 * R_COIL * TRAP_DEPTH / sc.mu_0
B0 = 1.0 


class ConstantCrossSection(BaseCrossSection):
    """Test stub: fixed cross section, no change to particle."""

    def __init__(self, sigma=1e-20):
        self.__sigma = sigma

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy, pitch_angle


class EnergyLossCrossSection(BaseCrossSection):
    """Test stub: fixed cross section, removes fixed energy per scatter."""

    def __init__(self, sigma=1e-20, energy_loss=10.0):
        self.__sigma = sigma
        self.__energy_loss = energy_loss

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy - self.__energy_loss, pitch_angle


class EscapingCrossSection(BaseCrossSection):
    """Test stub: always sets pitch angle to near zero (untrapped)."""

    def __init__(self, sigma=1e-20):
        self.__sigma = sigma

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy, 0.01  # Very shallow angle


def _make_components():
    trap = HarmonicField(R_COIL, I_COIL, B0)
    wg = CircularWaveguide(WG_RADIUS)
    particle = Electron(KE, POS, PITCH)
    return trap, wg, particle


def _make_lo_freq(trap, wg, particle):
    spec = SpectrumCalculator(trap, wg, particle)
    return spec.get_peak_frequency(0) - 200e6


class TestScatteringSimulatorConstruction:

    def test_valid_construction(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 1e16)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        assert sim is not None

    def test_invalid_trap_raises(self):
        gas = GasModel([(ConstantCrossSection(), 1e16)])
        with pytest.raises(TypeError, match="BaseTrap or BaseField"):
            ScatteringSimulator("not_a_trap", CircularWaveguide(WG_RADIUS),
                                gas, SAMPLE_RATE, 26e9, 1e-6)

    def test_invalid_gas_model_raises(self):
        trap, wg, _ = _make_components()
        with pytest.raises(TypeError, match="GasModel"):
            ScatteringSimulator(trap, wg, "not_a_gas", SAMPLE_RATE,
                                26e9, 1e-6)

    def test_negative_sample_rate_raises(self):
        trap, wg, _ = _make_components()
        gas = GasModel([(ConstantCrossSection(), 1e16)])
        with pytest.raises(ValueError, match="positive and finite"):
            ScatteringSimulator(trap, wg, gas, -1e9, 26e9, 1e-6)


class TestTrapping:

    def test_high_pitch_angle_is_trapped(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 1e16)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        assert sim.is_trapped(np.deg2rad(89.0), particle) is True

    def test_zero_pitch_not_trapped(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 1e16)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        assert sim.is_trapped(0.0, particle) is False

    def test_pi_pitch_not_trapped(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 1e16)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        assert sim.is_trapped(np.pi, particle) is False


class TestZeroScatterCase:
    """With zero gas density, no scatters occur."""

    def test_no_scatters_occur(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 0.0)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert len(result.scatter_times) == 0
        assert len(result.particles) == 1
        assert result.escaped is False

    def test_zero_density_matches_signal_generator(self):
        """Output should match SignalGenerator when no scattering occurs."""
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 0.0)])
        lo = _make_lo_freq(trap, wg, particle)
        acq_time = 1e-6

        # ScatteringSimulator
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, acq_time)
        scat_result = sim.simulate(particle, MAX_ORDER,
                                   np.random.default_rng(42))

        # SignalGenerator
        spec = SpectrumCalculator(trap, wg, particle)
        sig_gen = SignalGenerator(spec, SAMPLE_RATE, lo, acq_time)
        _, sig_ref = sig_gen.generate_signal(MAX_ORDER)

        assert len(scat_result.signal) == len(sig_ref)
        np.testing.assert_allclose(scat_result.signal, sig_ref, rtol=1e-10)


class TestScatteringBehavior:

    def test_scatters_recorded(self):
        """With high density, scatters should occur."""
        trap, wg, particle = _make_components()
        # Very high density to guarantee scatters within 1 microsecond
        gas = GasModel([(ConstantCrossSection(1e-18), 1e17)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert len(result.scatter_times) > 0

    def test_scatter_times_are_non_decreasing(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(1e-18), 1e17)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        if len(result.scatter_times) > 1:
            diffs = np.diff(result.scatter_times)
            assert np.all(diffs >= 0)

    def test_scatter_times_within_event_duration(self):
        trap, wg, particle = _make_components()
        max_time = 1e-6
        gas = GasModel([(ConstantCrossSection(1e-18), 1e17)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, max_time)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        for t in result.scatter_times:
            assert t <= max_time

    def test_energy_loss_creates_multiple_particles(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(EnergyLossCrossSection(1e-18, 10.0), 1e17)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert len(result.particles) > 1
        # Each successive particle should have lower energy
        for i in range(1, len(result.particles)):
            assert result.particles[i].get_energy() < \
                result.particles[i - 1].get_energy()

    def test_particle_escapes_on_bad_pitch_base_field(self):
        """Escape test using BaseField which has a physical trapping limit."""
        field = HarmonicField(radius=0.03, current=400, background=1.0)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, POS, PITCH)
        gas = GasModel([(EscapingCrossSection(1e-18), 1e17)])
        spec = SpectrumCalculator(field, wg, particle)
        lo = spec.get_peak_frequency(0) - 200e6
        sim = ScatteringSimulator(field, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert result.escaped is True
        assert len(result.scatter_times) >= 1


class TestSignalOutput:

    def test_signal_is_complex(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 0.0)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert np.iscomplexobj(result.signal)

    def test_times_match_sample_rate(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 0.0)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        if len(result.times) > 1:
            dt = result.times[1] - result.times[0]
            assert dt == pytest.approx(1.0 / SAMPLE_RATE, rel=1e-6)

    def test_times_and_signal_same_length(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(1e-18), 1e17)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert len(result.times) == len(result.signal)

    def test_signal_is_not_all_zeros(self):
        trap, wg, particle = _make_components()
        gas = GasModel([(ConstantCrossSection(), 0.0)])
        lo = _make_lo_freq(trap, wg, particle)
        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 1e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))
        assert np.any(result.signal != 0)
