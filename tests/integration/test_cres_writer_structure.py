"""
Integration tests for CRESWriter class
"""

import numpy as np
import pytest
import h5py
import os
from CRESSignalStructure.CRESWriter import CRESWriter
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.QTNMTraps import HarmonicTrap

class TestCRESWriterStructure:
    """Tests for HDF5 file structure and legacy compliance"""

    def test_file_creation_and_hierarchy(self, tmp_path):
        """Test that the file is created with the correct /Data/signalX hierarchy"""
        
        d = tmp_path / "subdir"
        d.mkdir()
        filename = d / "test_legacy.h5"
        
        # Setup dummy physics objects
        trap = HarmonicTrap(B0=1.0, L0=0.05)
        wg = CircularWaveguide(radius=0.01) # NEW
        config = {'sample_rate': 100e6, 'lo_freq': 25e9}
        
        # Run Writer
        with CRESWriter(str(filename), mode='w') as writer:
            # UPDATED: Now passes 'wg'
            writer.set_global_config(trap, wg, config)
            
            # Write 2 events
            for i in range(2):
                p = Particle(ke=18600.0, startPos=np.array([0.001, 0., 0.]))
                t = np.linspace(0, 1e-5, 100)
                sig = np.zeros(100, dtype='complex64')
                writer.write_event(p, t, sig)

        # Verification
        assert os.path.exists(filename)
        
        with h5py.File(filename, 'r') as f:
            assert 'Data' in f, "Root group 'Data' is missing"
            assert 'signal1' in f['Data']
            assert 'signal2' in f['Data']
            dset = f['Data']['signal1']
            assert dset.shape == (100,)
            assert dset.dtype == 'complex64'

    def test_metadata_attributes(self, tmp_path):
        """Test that physics parameters are correctly attached as attributes"""
        
        filename = tmp_path / "test_metadata.h5"
        
        # Specific values to verify later
        test_energy = 18600.0
        test_pitch = np.pi / 2
        test_pos = np.array([0.005, 0.0, 0.0])
        test_B0 = 0.95
        test_lo = 25.5e9
        
        trap = HarmonicTrap(B0=test_B0, L0=0.05)
        wg = CircularWaveguide(radius=0.01) 
        config = {'sample_rate': 100e6, 'lo_freq': test_lo}
        
        with CRESWriter(str(filename), mode='w') as writer:
            writer.set_global_config(trap, wg, config)
            p = Particle(ke=test_energy, startPos=test_pos, pitchAngle=test_pitch)
            writer.write_event(p, np.zeros(10), np.zeros(10))

        with h5py.File(filename, 'r') as f:
            attrs = f['Data']['signal1'].attrs
            
            # Check Keys match legacy expectation
            assert 'Energy [eV]' in attrs
            assert attrs['Energy [eV]'] == test_energy
            assert attrs['Pitch angle [degrees]'] == 90.0
            
            # Check Array
            np.testing.assert_array_equal(attrs['Starting position [metres]'], test_pos)
            
            # Check Derived
            assert attrs['B_bkg [Tesla]'] == test_B0
            assert attrs['LO frequency [Hertz]'] == test_lo
            assert 'r_wg [metres]' in attrs

    def test_context_manager_safety(self, tmp_path):
        """Test that the file is closed properly even if an error occurs"""
        filename = tmp_path / "test_safety.h5"
        writer = None
        
        with pytest.raises(ValueError):
            with CRESWriter(str(filename)) as w:
                writer = w
                raise ValueError("Intentional Crash")
        
        assert writer is not None
        assert writer.file.id.valid == 0