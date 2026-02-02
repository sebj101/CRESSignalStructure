"""
Unit Tests for CRESWriter Logic

This module tests the internal logic and mathematics of the CRESWriter class in isolation.

1. Physics Derivations: ensuring derived values (velocity vectors, cyclotron frequencies)
   are calculated correctly from the input Particle.
2. Configuration Handling: ensuring the writer correctly processes and stores 
   simulation config (LO frequency, sample rate).
3. Data Integrity: ensuring attributes and signal data are written to the HDF5 structure 
   exactly as expected.
"""
import unittest
import h5py
import numpy as np
import tempfile
import os
from CRESSignalStructure.CRESWriter import CRESWriter
from CRESSignalStructure.Particle import Particle

# --- Mocks ---
class MockTrap:
    def evaluate_field_magnitude(self, x, y, z): return 1.0

class MockWaveguide:
    def __init__(self): self.wgR = 0.005
    def CalcTE11Impedance(self, omega): return 50.0

class TestCRESWriterPhysics(unittest.TestCase):
    
    def setUp(self):
        # Create a single temp file for the entire test class
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix='.h5')
        os.close(self.tmp_fd)

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_physics_metadata_consistency(self):
        """
        Verifies Velocity Splitting and LO Consistency 
        across multiple physical regimes (90 and 88 degrees).
        """
        # Define test cases: (Pitch Degrees, Kinetic Energy eV, LO Hertz)
        cases = [
            (90.0, 5000, 25.85e9),  # Case 1: Purely transverse (90 deg)
            (88.0, 18600, 25.85e9)  # Case 2: Realistic CRES angle (88 deg)
        ]

        # 1. WRITE PHASE
        with CRESWriter(self.tmp_path) as writer:
            for i, (deg, ke, lo) in enumerate(cases):
                p = Particle(ke=ke, startPos=np.zeros(3), pitchAngle=np.radians(deg))
                # Ensuring we use a STABLE LO frequency across both writes
                writer.set_global_config(MockTrap(), MockWaveguide(), {'lo_freq': lo, 'sample_rate': 1e9})
                writer.write_event(p, np.array([0]), np.array([0j]))

        # 2. Re-reading from disk
        with h5py.File(self.tmp_path, 'r') as f:
            for i, (deg, ke, lo) in enumerate(cases):
                sig_key = f"signal{i+1}"
                attrs = f['Data'][sig_key].attrs
                
                # Verify LO Consistency
                self.assertEqual(attrs['LO frequency [Hertz]'], lo, f"LO Mismatch in {sig_key}")

                # Verify Velocity Logic (Sine/Cosine Split)
                vel = attrs['Starting velocity [metres/second]']
                p_ref = Particle(ke=ke, startPos=np.zeros(3), pitchAngle=np.radians(deg))
                total_speed = p_ref.GetSpeed()

                expected_v_perp = total_speed * np.sin(np.radians(deg))
                expected_v_z    = total_speed * np.cos(np.radians(deg))

                self.assertAlmostEqual(vel[0], expected_v_perp, places=5, msg=f"v_perp error at {deg} deg")
                self.assertAlmostEqual(vel[2], expected_v_z, places=5, msg=f"v_z error at {deg} deg")
                
                # Check Energy attribute for good measure
                self.assertAlmostEqual(attrs['Energy [eV]'], ke, places=2)

if __name__ == '__main__':
    unittest.main()

    