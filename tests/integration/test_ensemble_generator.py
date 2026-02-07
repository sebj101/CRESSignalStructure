"""
Integration tests for EnsembleGenerator.py
Verifies that parameter sampling and file generation work as expected.
"""
import pytest
import numpy as np
import h5py
from CRESSignalStructure.EnsembleGenerator import generate_uniform_ensemble
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.Particle import Particle


class TestEnsembleGenerator:

    def test_uniform_sampling_ranges(self, tmp_path):
        """
        Critical Test: Verifies that generated particles strictly adhere 
        to the user-defined min/max ranges.
        """
        output_file = tmp_path / "test_sampling.h5"
        
        ranges = {
            'energy': (18500.0, 18600.0),
            'pitch': (np.radians(88.0), np.radians(89.0)),
            'r': (0.0, 0.005),
            'z': (-0.001, 0.001),
            'theta': (0.0, 2*np.pi)
        }
        
        sim_config = {
            'sample_rate': 1e9,
            'lo_freq': 25e9,
            'acq_time': 1e-6, 
            'max_order': 1
        }
        
        trap = HarmonicField(radius=0.0125, current=50.0, background=1.0)
        wg = CircularWaveguide(radius=0.01)

        # 2. Run Generation
        n_events = 5
        generate_uniform_ensemble(
            output_file=str(output_file),
            n_events=n_events,
            trap=trap,
            waveguide=wg,
            sim_config=sim_config,
            ranges=ranges,
            verbose=False
        )

        # 3. Verification
        with h5py.File(output_file, 'r') as f:
            assert 'Data' in f, "File missing /Data group"
            
            for i in range(1, n_events + 1):
                sig_name = f'signal{i}'
                assert sig_name in f['Data'], f"Missing {sig_name}"
                
                attrs = f['Data'][sig_name].attrs
                
                # Check Energy
                e = attrs['Energy [eV]']
                assert ranges['energy'][0] <= e <= ranges['energy'][1]

                # Check Pitch
                p_deg = attrs['Pitch angle [degrees]']
                p_rad = np.radians(p_deg)
                # Allow tiny float tolerance
                assert ranges['pitch'][0] - 1e-5 <= p_rad <= ranges['pitch'][1] + 1e-5
                
                # Check Radius
                pos = attrs['Starting position [metres]']
                r = np.sqrt(pos[0]**2 + pos[1]**2)
                assert ranges['r'][0] <= r <= ranges['r'][1]

    def test_config_propagation(self, tmp_path):
        """
        Verifies that simulation configuration is correctly saved.
        """
        output_file = tmp_path / "test_config.h5"
        
        test_fs = 123.45e6
        test_lo = 24.99e9
        
        sim_config = {
            'sample_rate': test_fs,
            'lo_freq': test_lo,
            'acq_time': 1e-6,
            'max_order': 1
        }
        
        ranges = {
            'energy': (18600, 18600),
            'pitch': (np.radians(89.0), np.radians(89.999))
        }
        
        trap = HarmonicField(radius=0.0125, current=50.0, background=1.0)
        wg = CircularWaveguide(radius=0.01)

        generate_uniform_ensemble(
            output_file=str(output_file),
            n_events=1,
            trap=trap,
            waveguide=wg,
            sim_config=sim_config,
            ranges=ranges,
            verbose=False
        )

        with h5py.File(output_file, 'r') as f:
            attrs = f['Data']['signal1'].attrs
            
            # Check calculated frequency attributes
            assert abs(attrs['LO frequency [Hertz]'] - test_lo) < 1.0
            
            # Time step is 1/fs
            dt_expected = 1.0 / test_fs
            assert abs(attrs['Time step [seconds]'] - dt_expected) < 1e-12
