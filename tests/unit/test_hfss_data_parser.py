"""
Unit tests for HFSSDataParser.

Tests cover all three parse methods (parse_efield, parse_gain, parse_impedance),
including unit conversions, grid reshaping, error handling, and edge cases.
"""

import numpy as np
import pytest

from CRESSignalStructure.antennas.HFSSDataParser import (
    EFieldData, GainData, HFSSDataParser, ImpedanceData,
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, filename, content):
    p = tmp_path / filename
    p.write_text(content)
    return p


def _efield_csv(tmp_path, phi_vals, theta_vals,
                re_theta, im_theta, re_phi, im_phi, filename="efield.csv"):
    lines = [
        "Phi[deg],Theta[deg],re(rETheta)[mV],im(rETheta)[mV],"
        "re(rEPhi)[mV],im(rEPhi)[mV]"
    ]
    idx = 0
    for phi in phi_vals:
        for theta in theta_vals:
            lines.append(
                f"{phi},{theta},{re_theta[idx]},{im_theta[idx]},"
                f"{re_phi[idx]},{im_phi[idx]}"
            )
            idx += 1
    return _write_csv(tmp_path, filename, "\n".join(lines))


def _gain_csv(tmp_path, phi_vals, theta_vals, gains, filename="gain.csv"):
    lines = ["Phi[deg],Theta[deg],mag(GainTotal)"]
    idx = 0
    for phi in phi_vals:
        for theta in theta_vals:
            lines.append(f"{phi},{theta},{gains[idx]}")
            idx += 1
    return _write_csv(tmp_path, filename, "\n".join(lines))


def _impedance_csv(tmp_path, freqs_ghz, re_z, im_z, filename="impedance.csv"):
    # Column names containing commas must be quoted for valid CSV.
    lines = ['Freq [GHz],"re(Z(1,1)) []","im(Z(1,1)) []"']
    for f, r, i in zip(freqs_ghz, re_z, im_z):
        lines.append(f"{f},{r},{i}")
    return _write_csv(tmp_path, filename, "\n".join(lines))


# ---------------------------------------------------------------------------
# parse_efield
# ---------------------------------------------------------------------------

class TestParseEField:

    def test_returns_efield_data_instance(self, tmp_path):
        path = _efield_csv(tmp_path, [0, 90], [0, 90],
                           [1, 2, 3, 4], [0]*4, [0]*4, [0]*4)
        assert isinstance(HFSSDataParser().parse_efield(path), EFieldData)

    def test_phi_and_theta_converted_to_radians(self, tmp_path):
        path = _efield_csv(tmp_path, [0, 90], [0, 90],
                           [1, 2, 3, 4], [0]*4, [0]*4, [0]*4)
        result = HFSSDataParser().parse_efield(path)
        np.testing.assert_array_almost_equal(result.phi, np.deg2rad([0, 90]))
        np.testing.assert_array_almost_equal(result.theta, np.deg2rad([0, 90]))

    def test_grid_shape_is_n_theta_by_n_phi(self, tmp_path):
        phis = [0, 45, 90]
        thetas = [0, 30, 60, 90]
        n = len(phis) * len(thetas)
        path = _efield_csv(tmp_path, phis, thetas,
                           list(range(n)), [0]*n, [0]*n, [0]*n)
        result = HFSSDataParser().parse_efield(path)
        assert result.E_theta.shape == (len(thetas), len(phis))
        assert result.E_phi.shape == (len(thetas), len(phis))

    def test_mv_to_v_conversion(self, tmp_path):
        path = _efield_csv(tmp_path, [0], [0],
                           [1000.0], [0.0], [500.0], [0.0])
        result = HFSSDataParser().parse_efield(path)
        assert result.E_theta[0, 0].real == pytest.approx(1.0)
        assert result.E_phi[0, 0].real == pytest.approx(0.5)

    def test_complex_field_assembled_correctly(self, tmp_path):
        path = _efield_csv(tmp_path, [0], [0],
                           [3.0], [4.0], [1.0], [2.0])
        result = HFSSDataParser().parse_efield(path)
        # 3 + 4j mV → 0.003 + 0.004j V
        assert result.E_theta[0, 0].real == pytest.approx(3e-3)
        assert result.E_theta[0, 0].imag == pytest.approx(4e-3)
        assert result.E_phi[0, 0].real == pytest.approx(1e-3)
        assert result.E_phi[0, 0].imag == pytest.approx(2e-3)

    def test_grid_values_placed_at_correct_indices(self, tmp_path):
        # phi=[0,90] (outer loop), theta=[0,90] (inner loop)
        # row order: (phi=0,theta=0)→10, (phi=0,theta=90)→20,
        #            (phi=90,theta=0)→30, (phi=90,theta=90)→40
        path = _efield_csv(tmp_path, [0, 90], [0, 90],
                           [10, 20, 30, 40], [0]*4, [0]*4, [0]*4)
        result = HFSSDataParser().parse_efield(path)
        assert result.E_theta[0, 0].real == pytest.approx(10e-3)  # theta=0, phi=0
        assert result.E_theta[1, 0].real == pytest.approx(20e-3)  # theta=90, phi=0
        assert result.E_theta[0, 1].real == pytest.approx(30e-3)  # theta=0, phi=90
        assert result.E_theta[1, 1].real == pytest.approx(40e-3)  # theta=90, phi=90

    def test_accepts_string_path(self, tmp_path):
        path = _efield_csv(tmp_path, [0], [0], [1.0], [0.0], [0.0], [0.0])
        result = HFSSDataParser().parse_efield(str(path))
        assert isinstance(result, EFieldData)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HFSSDataParser().parse_efield(tmp_path / "missing.csv")

    def test_missing_columns_raises_value_error(self, tmp_path):
        content = "Phi[deg],Theta[deg],re(rETheta)[mV]\n0,0,1.0"
        path = _write_csv(tmp_path, "bad.csv", content)
        with pytest.raises(ValueError, match="missing required columns"):
            HFSSDataParser().parse_efield(path)

    def test_irregular_grid_raises_value_error(self, tmp_path):
        # phi_vals=[0,90], theta_vals=[0,90] → expects 4 rows, only 3 given
        content = (
            "Phi[deg],Theta[deg],re(rETheta)[mV],im(rETheta)[mV],"
            "re(rEPhi)[mV],im(rEPhi)[mV]\n"
            "0,0,1,0,0,0\n"
            "0,90,2,0,0,0\n"
            "90,0,3,0,0,0\n"
        )
        path = _write_csv(tmp_path, "irr.csv", content)
        with pytest.raises(ValueError):
            HFSSDataParser().parse_efield(path)


# ---------------------------------------------------------------------------
# parse_gain
# ---------------------------------------------------------------------------

class TestParseGain:

    def test_returns_gain_data_instance(self, tmp_path):
        path = _gain_csv(tmp_path, [0, 90], [0, 90], [1.0, 2.0, 3.0, 4.0])
        assert isinstance(HFSSDataParser().parse_gain(path), GainData)

    def test_phi_and_theta_converted_to_radians(self, tmp_path):
        path = _gain_csv(tmp_path, [0, 90], [0, 90], [1.0, 2.0, 3.0, 4.0])
        result = HFSSDataParser().parse_gain(path)
        np.testing.assert_array_almost_equal(result.phi, np.deg2rad([0, 90]))
        np.testing.assert_array_almost_equal(result.theta, np.deg2rad([0, 90]))

    def test_grid_shape_is_n_theta_by_n_phi(self, tmp_path):
        phis = [0, 45, 90]
        thetas = [0, 30, 60, 90]
        n = len(phis) * len(thetas)
        path = _gain_csv(tmp_path, phis, thetas, list(range(n)))
        result = HFSSDataParser().parse_gain(path)
        assert result.gain.shape == (len(thetas), len(phis))

    def test_gain_values_placed_at_correct_indices(self, tmp_path):
        path = _gain_csv(tmp_path, [0, 90], [0, 90], [1.0, 2.0, 3.0, 4.0])
        result = HFSSDataParser().parse_gain(path)
        assert result.gain[0, 0] == pytest.approx(1.0)  # theta=0, phi=0
        assert result.gain[1, 0] == pytest.approx(2.0)  # theta=90, phi=0
        assert result.gain[0, 1] == pytest.approx(3.0)  # theta=0, phi=90
        assert result.gain[1, 1] == pytest.approx(4.0)  # theta=90, phi=90

    def test_gain_is_real_valued(self, tmp_path):
        path = _gain_csv(tmp_path, [0], [0], [1.5])
        result = HFSSDataParser().parse_gain(path)
        assert np.isrealobj(result.gain)

    def test_accepts_string_path(self, tmp_path):
        path = _gain_csv(tmp_path, [0], [0], [1.0])
        result = HFSSDataParser().parse_gain(str(path))
        assert isinstance(result, GainData)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HFSSDataParser().parse_gain(tmp_path / "missing.csv")

    def test_missing_columns_raises_value_error(self, tmp_path):
        content = "Phi[deg],Theta[deg]\n0,0"
        path = _write_csv(tmp_path, "bad.csv", content)
        with pytest.raises(ValueError, match="missing required columns"):
            HFSSDataParser().parse_gain(path)

    def test_irregular_grid_raises_value_error(self, tmp_path):
        # phi_vals=[0,90], theta_vals=[0,90] → expects 4, only 3 given
        content = "Phi[deg],Theta[deg],mag(GainTotal)\n0,0,1.0\n0,90,2.0\n90,0,3.0"
        path = _write_csv(tmp_path, "irr.csv", content)
        with pytest.raises(ValueError):
            HFSSDataParser().parse_gain(path)


# ---------------------------------------------------------------------------
# parse_impedance
# ---------------------------------------------------------------------------

class TestParseImpedance:

    def test_returns_impedance_data_instance(self, tmp_path):
        path = _impedance_csv(tmp_path, [1.0, 2.0], [50, 60], [10, 20])
        assert isinstance(HFSSDataParser().parse_impedance(path), ImpedanceData)

    def test_frequency_converted_from_ghz_to_hz(self, tmp_path):
        path = _impedance_csv(tmp_path, [1.0, 2.0], [50, 60], [10, 20])
        result = HFSSDataParser().parse_impedance(path)
        np.testing.assert_array_almost_equal(result.frequency, [1e9, 2e9])

    def test_impedance_is_complex(self, tmp_path):
        path = _impedance_csv(tmp_path, [1.0], [50.0], [25.0])
        result = HFSSDataParser().parse_impedance(path)
        assert result.impedance[0].real == pytest.approx(50.0)
        assert result.impedance[0].imag == pytest.approx(25.0)

    def test_negative_reactance_preserved(self, tmp_path):
        path = _impedance_csv(tmp_path, [1.0], [73.0], [-42.5])
        result = HFSSDataParser().parse_impedance(path)
        assert result.impedance[0].imag == pytest.approx(-42.5)

    def test_rows_sorted_by_ascending_frequency(self, tmp_path):
        path = _impedance_csv(tmp_path, [3.0, 1.0, 2.0],
                              [30, 10, 20], [3, 1, 2])
        result = HFSSDataParser().parse_impedance(path)
        np.testing.assert_array_almost_equal(result.frequency, [1e9, 2e9, 3e9])
        np.testing.assert_array_almost_equal(result.impedance.real, [10, 20, 30])
        np.testing.assert_array_almost_equal(result.impedance.imag, [1, 2, 3])

    def test_extra_geometry_columns_ignored(self, tmp_path):
        content = (
            'dL [mm],R_wire [mm],Freq [GHz],"re(Z(1,1)) []","im(Z(1,1)) []"\n'
            "10.0,0.5,1.0,50.0,10.0\n"
            "10.0,0.5,2.0,60.0,20.0\n"
        )
        path = _write_csv(tmp_path, "impedance_extra.csv", content)
        result = HFSSDataParser().parse_impedance(path)
        np.testing.assert_array_almost_equal(result.frequency, [1e9, 2e9])
        np.testing.assert_array_almost_equal(result.impedance.real, [50.0, 60.0])

    def test_duplicate_frequency_raises_value_error(self, tmp_path):
        path = _impedance_csv(tmp_path, [1.0, 1.0], [50, 50], [10, 10])
        with pytest.raises(ValueError, match="strictly increasing"):
            HFSSDataParser().parse_impedance(path)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HFSSDataParser().parse_impedance(tmp_path / "missing.csv")

    def test_missing_columns_raises_value_error(self, tmp_path):
        content = 'Freq [GHz],"re(Z(1,1)) []"\n1.0,50.0'
        path = _write_csv(tmp_path, "bad.csv", content)
        with pytest.raises(ValueError, match="missing required columns"):
            HFSSDataParser().parse_impedance(path)

    def test_accepts_string_path(self, tmp_path):
        path = _impedance_csv(tmp_path, [1.0], [50.0], [0.0])
        result = HFSSDataParser().parse_impedance(str(path))
        assert isinstance(result, ImpedanceData)
