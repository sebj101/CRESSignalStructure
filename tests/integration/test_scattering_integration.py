"""
Integration tests for scattering simulation pipeline.
Tests the full chain: GasModel -> ScatteringSimulator -> EnsembleGenerator -> CRESWriter.
"""

import pytest
import numpy as np
import h5py
from CRESSignalStructure import (
    HarmonicTrap,
    CircularWaveguide,
    Electron,
    SpectrumCalculator,
    ScatteringSimulator,
)
from CRESSignalStructure.scattering import BaseCrossSection, GasModel
from CRESSignalStructure.EnsembleGenerator import generate_scattering_ensemble

KE = 18600.0
WG_RADIUS = 5e-3
B0 = 1.0
L0 = 0.5
PITCH = np.deg2rad(89.0)
POS = np.array([1e-4, 0.0, 0.0])
SAMPLE_RATE = 1e9
MAX_ORDER = 3


class ConstantCrossSection(BaseCrossSection):

    def __init__(self, sigma=1e-20):
        self.__sigma = sigma

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy, pitch_angle


class EnergyLossCrossSection(BaseCrossSection):

    def __init__(self, sigma=1e-20, energy_loss=10.0):
        self.__sigma = sigma
        self.__energy_loss = energy_loss

    def total_cross_section(self, energy):
        return self.__sigma

    def sample_post_scatter(self, energy, pitch_angle, rng):
        return energy - self.__energy_loss, pitch_angle


class TestScatteringPipeline:

    def test_full_simulation_produces_valid_signal(self):
        """End-to-end: create all components, run simulation, check output."""
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, POS, PITCH)
        gas = GasModel([(EnergyLossCrossSection(1e-19, 5.0), 1e18)])
        spec = SpectrumCalculator(trap, wg, particle)
        lo = spec.get_peak_frequency(0) - 200e6

        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 5e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(123))

        assert len(result.times) > 0
        assert len(result.signal) == len(result.times)
        assert np.iscomplexobj(result.signal)
        assert not result.escaped

    def test_frequency_shifts_after_scatter(self):
        """After energy loss, the cyclotron frequency should shift."""
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        particle = Electron(KE, POS, PITCH)
        # Large energy loss per scatter to make the frequency shift visible
        gas = GasModel([(EnergyLossCrossSection(1e-19, 100.0), 1e18)])
        spec = SpectrumCalculator(trap, wg, particle)
        lo = spec.get_peak_frequency(0) - 200e6

        sim = ScatteringSimulator(trap, wg, gas, SAMPLE_RATE, lo, 10e-6)
        result = sim.simulate(particle, MAX_ORDER, np.random.default_rng(42))

        if len(result.particles) > 1:
            # Cyclotron frequency depends on gamma, which depends on energy.
            # Lower energy -> lower gamma -> higher cyclotron freq.
            gamma_0 = result.particles[0].get_gamma()
            gamma_1 = result.particles[1].get_gamma()
            assert gamma_1 < gamma_0  # Lost energy -> lower gamma


class TestScatteringEnsemble:

    def test_ensemble_writes_hdf5(self, tmp_path):
        output_file = tmp_path / "scattering_ensemble.h5"
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        gas = GasModel([(ConstantCrossSection(1e-20), 1e16)])

        sim_config = {
            'sample_rate': SAMPLE_RATE,
            'lo_freq': 26.5e9,
            'max_event_time': 1e-6,
            'max_order': MAX_ORDER,
        }

        def particle_gen(i):
            return Electron(KE, POS, PITCH)

        generate_scattering_ensemble(
            output_file=str(output_file),
            n_events=3,
            particle_generator=particle_gen,
            trap=trap,
            waveguide=wg,
            gas_model=gas,
            sim_config=sim_config,
            use_multiprocessing=False,
            verbose=False,
        )

        with h5py.File(output_file, 'r') as f:
            assert 'Data' in f
            for i in range(1, 4):
                sig_name = f'signal{i}'
                assert sig_name in f['Data']
                attrs = f['Data'][sig_name].attrs
                assert 'Energy [eV]' in attrs
                assert 'n_scatters' in attrs
                assert 'escaped' in attrs

    def test_ensemble_with_scatters_records_metadata(self, tmp_path):
        output_file = tmp_path / "scattering_meta.h5"
        trap = HarmonicTrap(B0, L0)
        wg = CircularWaveguide(WG_RADIUS)
        # High density to guarantee scatters
        gas = GasModel([(EnergyLossCrossSection(1e-18, 5.0), 1e19)])

        sim_config = {
            'sample_rate': SAMPLE_RATE,
            'lo_freq': 26.5e9,
            'max_event_time': 5e-6,
            'max_order': MAX_ORDER,
        }

        def particle_gen(i):
            return Electron(KE, POS, PITCH)

        generate_scattering_ensemble(
            output_file=str(output_file),
            n_events=2,
            particle_generator=particle_gen,
            trap=trap,
            waveguide=wg,
            gas_model=gas,
            sim_config=sim_config,
            use_multiprocessing=False,
            verbose=False,
        )

        with h5py.File(output_file, 'r') as f:
            attrs = f['Data']['signal1'].attrs
            n_scatters = attrs['n_scatters']
            assert n_scatters > 0
            assert 'scatter_times [seconds]' in attrs
            assert 'scatter_energies [eV]' in attrs
            assert 'scatter_pitch_angles [degrees]' in attrs
            assert 'scatter_cyclotron_frequencies [Hertz]' in attrs
            assert 'scatter_downmixed_cyclotron_frequencies [Hertz]' in attrs
            assert len(attrs['scatter_times [seconds]']) == n_scatters
