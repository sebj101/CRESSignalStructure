"""
Unit tests for scattering_utils.scatter_to_pitch_angle
"""

import numpy as np
from CRESSignalStructure.scattering import scatter_to_pitch_angle


class TestScatterToPitchAngle:

    def test_zero_scattering_preserves_pitch(self):
        rng = np.random.default_rng(42)
        pitch = 1.2
        for _ in range(100):
            result = scatter_to_pitch_angle(pitch, 0.0, rng)
            assert np.isclose(result, pitch, atol=1e-12)

    def test_pi_scattering_gives_supplement(self):
        rng = np.random.default_rng(42)
        pitch = 1.2
        for _ in range(100):
            result = scatter_to_pitch_angle(pitch, np.pi, rng)
            assert np.isclose(result, np.pi - pitch, atol=1e-12)

    def test_result_always_in_valid_range(self):
        rng = np.random.default_rng(42)
        for pitch in [0.01, 0.5, np.pi / 2, 2.5, np.pi - 0.01]:
            for theta in [0.0, 0.1, 0.5, np.pi / 2, np.pi]:
                for _ in range(50):
                    result = scatter_to_pitch_angle(pitch, theta, rng)
                    assert 0 <= result <= np.pi

    def test_small_scatter_mean_preserves_pitch(self):
        rng = np.random.default_rng(42)
        pitch = np.pi / 3
        theta = 0.01
        results = [scatter_to_pitch_angle(pitch, theta, rng) for _ in range(5000)]
        mean_result = np.mean(results)
        # For small theta, the mean pitch should be approximately preserved
        # (azimuthal symmetry means the average deflection is zero)
        assert np.isclose(mean_result, pitch, atol=0.02)

    def test_known_geometry_phi_zero(self):
        # When phi = 0: cos(a') = cos(a)*cos(t) + sin(a)*sin(t) = cos(a - t)
        # We can't control phi directly, but we can verify the formula
        # by checking that the result is between |a - t| and min(a + t, 2*pi - a - t)
        rng = np.random.default_rng(42)
        pitch = np.pi / 4
        theta = np.pi / 6
        results = [scatter_to_pitch_angle(pitch, theta, rng) for _ in range(1000)]
        # All results should be between pitch - theta and pitch + theta
        assert all(r >= pitch - theta - 1e-10 for r in results)
        assert all(r <= pitch + theta + 1e-10 for r in results)

    def test_perpendicular_pitch_symmetry(self):
        # At pitch = pi/2, the distribution of new pitch angles should be
        # symmetric about pi/2 for any scattering angle
        rng = np.random.default_rng(42)
        theta = 0.3
        results = [scatter_to_pitch_angle(np.pi / 2, theta, rng)
                   for _ in range(5000)]
        mean_result = np.mean(results)
        assert np.isclose(mean_result, np.pi / 2, atol=0.02)
