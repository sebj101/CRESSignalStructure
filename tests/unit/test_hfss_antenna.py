"""
Unit tests for HFSSAntenna.

A minimal synthetic dataset is written to temporary CSV files for each test:
  - E-field pattern: rETheta = sin(theta) mV, rEPhi = 0  (theta-only polarisation)
  - Gain pattern:    G = 1.5 * sin^2(theta)               (dipole-like)
  - Impedance:       four frequency points around 26 GHz

The antenna is placed at the origin with z_ax = [0,0,1] and x_ax = [1,0,0],
so the standard spherical-coordinate geometry applies:
  - source at [1,0,0] → theta = pi/2, phi = 0   (equator, main beam)
  - source at [0,0,1] → theta = 0               (null along bore-sight)
"""

import numpy as np
import pytest
import scipy.constants as sc
from pathlib import Path

from CRESSignalStructure.antennas.HFSSAntenna import HFSSAntenna

_ETA0 = sc.mu_0 * sc.c   # ~376.73 Ω

# ---------------------------------------------------------------------------
# Synthetic dataset constants
# ---------------------------------------------------------------------------

_THETA_DEG = [0, 90, 180]
_PHI_DEG   = [-180, 0, 180]

# Four impedance points: cubic interp1d requires at least 4
_FREQ_GHZ  = [24.0, 25.0, 26.0, 27.0]
_RE_Z      = [65.0, 70.0, 73.0, 75.0]
_IM_Z      = [20.0, 30.0, 42.5, 55.0]

_PATTERN_FREQ = 26e9   # Hz


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def _efield_csv(tmp_path):
    lines = [
        "Phi[deg],Theta[deg],re(rETheta)[mV],im(rETheta)[mV],"
        "re(rEPhi)[mV],im(rEPhi)[mV]"
    ]
    for phi in _PHI_DEG:
        for theta in _THETA_DEG:
            re_t = np.sin(np.deg2rad(theta))   # sin(theta) in mV
            lines.append(f"{phi},{theta},{re_t},0.0,0.0,0.0")
    return _write(tmp_path, "efield.csv", "\n".join(lines))


def _gain_csv(tmp_path):
    lines = ["Phi[deg],Theta[deg],mag(GainTotal)"]
    for phi in _PHI_DEG:
        for theta in _THETA_DEG:
            g = 1.5 * np.sin(np.deg2rad(theta)) ** 2
            lines.append(f"{phi},{theta},{g}")
    return _write(tmp_path, "gain.csv", "\n".join(lines))


def _impedance_csv(tmp_path):
    lines = ['Freq [GHz],"re(Z(1,1)) []","im(Z(1,1)) []"']
    for f, r, i in zip(_FREQ_GHZ, _RE_Z, _IM_Z):
        lines.append(f"{f},{r},{i}")
    return _write(tmp_path, "impedance.csv", "\n".join(lines))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_paths(tmp_path):
    return _efield_csv(tmp_path), _gain_csv(tmp_path), _impedance_csv(tmp_path)


@pytest.fixture
def antenna(csv_paths):
    efield_path, gain_path, impedance_path = csv_paths
    return HFSSAntenna(
        position=np.array([0.0, 0.0, 0.0]),
        z_ax=np.array([0.0, 0.0, 1.0]),
        x_ax=np.array([1.0, 0.0, 0.0]),
        efield_path=efield_path,
        gain_path=gain_path,
        impedance_path=impedance_path,
        pattern_frequency=_PATTERN_FREQ,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_construction_succeeds(self, antenna):
        assert antenna is not None

    def test_position_stored(self, antenna):
        np.testing.assert_array_equal(antenna.get_position(), [0.0, 0.0, 0.0])

    def test_orientation_is_z_ax(self, antenna):
        np.testing.assert_array_almost_equal(antenna.get_orientation(), [0.0, 0.0, 1.0])

    def test_accepts_path_objects(self, csv_paths):
        efield_path, gain_path, impedance_path = csv_paths
        ant = HFSSAntenna(
            np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]),
            Path(efield_path), Path(gain_path), Path(impedance_path), _PATTERN_FREQ,
        )
        assert ant is not None

    def test_gram_schmidt_applied_to_x_ax(self, csv_paths):
        """A slightly non-orthogonal x_ax is silently corrected."""
        efield_path, gain_path, impedance_path = csv_paths
        ant = HFSSAntenna(
            np.array([0., 0., 0.]), np.array([0., 0., 1.]),
            np.array([1., 0., 0.2]),   # not exactly orthogonal to z_ax
            efield_path, gain_path, impedance_path, _PATTERN_FREQ,
        )
        assert abs(np.dot(ant._x_ax, ant._z_ax)) < 1e-12

    def test_x_ax_parallel_to_z_ax_raises(self, csv_paths):
        efield_path, gain_path, impedance_path = csv_paths
        with pytest.raises(ValueError):
            HFSSAntenna(
                np.array([0., 0., 0.]), np.array([0., 0., 1.]),
                np.array([0., 0., 1.]),   # parallel → singular Gram-Schmidt
                efield_path, gain_path, impedance_path, _PATTERN_FREQ,
            )

    def test_invalid_position_raises(self, csv_paths):
        efield_path, gain_path, impedance_path = csv_paths
        with pytest.raises(ValueError, match="3-vector"):
            HFSSAntenna(
                np.array([0., 0.]),   # 2-vector
                np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                efield_path, gain_path, impedance_path, _PATTERN_FREQ,
            )

    def test_negative_pattern_frequency_raises(self, csv_paths):
        efield_path, gain_path, impedance_path = csv_paths
        with pytest.raises(ValueError):
            HFSSAntenna(
                np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                efield_path, gain_path, impedance_path, -26e9,
            )

    def test_missing_efield_file_raises(self, csv_paths, tmp_path):
        _, gain_path, impedance_path = csv_paths
        with pytest.raises(FileNotFoundError):
            HFSSAntenna(
                np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                tmp_path / "missing.csv", gain_path, impedance_path, _PATTERN_FREQ,
            )

    def test_missing_gain_file_raises(self, csv_paths, tmp_path):
        efield_path, _, impedance_path = csv_paths
        with pytest.raises(FileNotFoundError):
            HFSSAntenna(
                np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                efield_path, tmp_path / "missing.csv", impedance_path, _PATTERN_FREQ,
            )

    def test_missing_impedance_file_raises(self, csv_paths, tmp_path):
        efield_path, gain_path, _ = csv_paths
        with pytest.raises(FileNotFoundError):
            HFSSAntenna(
                np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                efield_path, gain_path, tmp_path / "missing.csv", _PATTERN_FREQ,
            )


# ---------------------------------------------------------------------------
# get_impedance
# ---------------------------------------------------------------------------

class TestGetImpedance:

    def test_returns_complex(self, antenna):
        assert isinstance(antenna.get_impedance(26e9), complex)

    def test_at_data_point(self, antenna):
        Z = antenna.get_impedance(26e9)
        assert Z.real == pytest.approx(73.0, rel=1e-5)
        assert Z.imag == pytest.approx(42.5, rel=1e-5)

    def test_invalid_frequency_raises(self, antenna):
        with pytest.raises(ValueError):
            antenna.get_impedance(-1e9)


# ---------------------------------------------------------------------------
# get_gain
# ---------------------------------------------------------------------------

class TestGetGain:

    def test_returns_float(self, antenna):
        assert isinstance(antenna.get_gain(np.pi / 2, 0.0), float)

    def test_at_equator(self, antenna):
        assert antenna.get_gain(np.pi / 2, 0.0) == pytest.approx(1.5, rel=1e-5)

    def test_at_pole_is_zero(self, antenna):
        assert antenna.get_gain(0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self, antenna):
        for theta in np.linspace(0.0, np.pi, 10):
            assert antenna.get_gain(float(theta), 0.0) >= 0.0

    def test_invalid_angle_type_raises(self, antenna):
        with pytest.raises(TypeError):
            antenna.get_gain("bad", 0.0)


# ---------------------------------------------------------------------------
# get_e_theta
# ---------------------------------------------------------------------------

class TestGetETheta:

    def test_shape(self, antenna):
        E = antenna.get_e_theta(np.array([[1., 0., 0.], [0., 1., 0.]]))
        assert E.shape == (2, 3)

    def test_is_complex(self, antenna):
        E = antenna.get_e_theta(np.array([[1., 0., 0.]]))
        assert np.iscomplexobj(E)

    def test_zero_along_boresight(self, antenna):
        """Along z_ax (theta=0) the rETheta pattern is zero."""
        E = antenna.get_e_theta(np.array([[0., 0., 1.]]))
        np.testing.assert_array_almost_equal(np.abs(E), np.zeros((1, 3)))

    def test_nonzero_at_equator(self, antenna):
        """rETheta = sin(90°) = 1 mV at the equator."""
        E = antenna.get_e_theta(np.array([[1., 0., 0.]]))
        assert np.linalg.norm(E) > 0

    def test_accepts_1d_single_point(self, antenna):
        E = antenna.get_e_theta(np.array([1., 0., 0.]))
        assert E.shape == (1, 3)

    def test_transverse_to_propagation_direction(self, antenna):
        """E_theta must be perpendicular to the direction from antenna to source."""
        pos = np.array([[1., 0., 0.], [0., 1., 0.]])
        E = antenna.get_e_theta(pos)
        r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)
        dots = np.sum(np.real(E) * r_hat, axis=1)
        np.testing.assert_array_almost_equal(dots, np.zeros(2))


# ---------------------------------------------------------------------------
# get_e_phi
# ---------------------------------------------------------------------------

class TestGetEPhi:

    def test_shape(self, antenna):
        E = antenna.get_e_phi(np.array([[1., 0., 0.], [0., 1., 0.]]))
        assert E.shape == (2, 3)

    def test_is_complex(self, antenna):
        E = antenna.get_e_phi(np.array([[1., 0., 0.]]))
        assert np.iscomplexobj(E)

    def test_zero_everywhere_for_zero_phi_pattern(self, antenna):
        """Our test data has rEPhi = 0 everywhere."""
        pos = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        E = antenna.get_e_phi(pos)
        np.testing.assert_array_almost_equal(np.abs(E), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# get_effective_length
# ---------------------------------------------------------------------------

class TestGetEffectiveLength:

    def test_shape(self, antenna):
        l = antenna.get_effective_length(26e9, np.array([[1., 0., 0.], [0., 1., 0.]]))
        assert l.shape == (2, 3)

    def test_is_complex(self, antenna):
        l = antenna.get_effective_length(26e9, np.array([[1., 0., 0.]]))
        assert np.iscomplexobj(l)

    def test_zero_at_null(self, antenna):
        """At theta=0 the pattern is zero so l_eff must vanish."""
        l = antenna.get_effective_length(26e9, np.array([[0., 0., 1.]]))
        np.testing.assert_array_almost_equal(np.abs(l), np.zeros((1, 3)), decimal=10)

    def test_nonzero_at_equator(self, antenna):
        l = antenna.get_effective_length(26e9, np.array([[1., 0., 0.]]))
        assert np.linalg.norm(l) > 0

    def test_magnitude_matches_formula_at_equator(self, antenna):
        """
        |l_eff| = sqrt(G * lambda^2 * Re(Z) / (pi * eta))

        At equator (theta=pi/2, phi=0): G=1.5, Re(Z)=73 at 26 GHz.
        """
        l = antenna.get_effective_length(26e9, np.array([[1., 0., 0.]]))

        wavelength = sc.c / 26e9
        expected = np.sqrt(1.5 * wavelength**2 * 73.0 / (np.pi * _ETA0))

        assert np.linalg.norm(l) == pytest.approx(expected, rel=1e-4)

    def test_accepts_1d_single_point(self, antenna):
        l = antenna.get_effective_length(26e9, np.array([1., 0., 0.]))
        assert l.shape == (1, 3)

    def test_source_at_antenna_position_raises(self, antenna):
        with pytest.raises(ValueError):
            antenna.get_effective_length(26e9, np.array([[0., 0., 0.]]))

    def test_invalid_frequency_raises(self, antenna):
        with pytest.raises(ValueError):
            antenna.get_effective_length(-1.0, np.array([[1., 0., 0.]]))
