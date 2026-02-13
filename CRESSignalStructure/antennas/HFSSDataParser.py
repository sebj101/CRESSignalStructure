"""
HFSSDataParser.py

Parser for HFSS-exported antenna simulation data files.

Handles three CSV export formats produced by HFSS:
- Far-field E-field components (rETheta, rEPhi) on a (phi, theta) grid
- Total gain pattern on a (phi, theta) grid
- Port impedance Z(1,1) as a function of frequency
"""

import numpy as np
from numpy.typing import NDArray
import csv
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EFieldData:
    """
    Container for parsed far-field E-field data.

    The field values are rE components (distance-weighted, in V), as exported
    by HFSS. Phi and theta are in radians; arrays are shaped (n_theta, n_phi).
    """
    phi: NDArray       # (n_phi,) in radians, ascending
    theta: NDArray     # (n_theta,) in radians, ascending
    E_theta: NDArray   # (n_theta, n_phi) complex, in V (converted from mV)
    E_phi: NDArray     # (n_theta, n_phi) complex, in V (converted from mV)


@dataclass
class GainData:
    """Container for parsed total gain pattern data."""
    phi: NDArray       # (n_phi,) in radians, ascending
    theta: NDArray     # (n_theta,) in radians, ascending
    gain: NDArray      # (n_theta, n_phi) real, dimensionless linear


@dataclass
class ImpedanceData:
    """Container for parsed port impedance data."""
    frequency: NDArray   # (n_freq,) in Hz
    impedance: NDArray   # (n_freq,) complex, in Ohms


class HFSSDataParser:
    """
    Parser for HFSS antenna simulation CSV exports.

    Supports the three file types produced by a standard HFSS far-field and
    port-parameter export:

    1. E-field file: columns Phi[deg], Theta[deg], im(rEPhi)[mV], re(rEPhi)[mV],
       re(rETheta)[mV], im(rETheta)[mV]
    2. Gain file:    columns Phi[deg], Theta[deg], mag(GainTotal)
    3. Z-parameter file: columns dL [mm], R_wire [mm], Freq [GHz],
       re(Z(1,1)) [], im(Z(1,1)) []

    Usage
    -----
    parser = HFSSDataParser()
    e_data = parser.parse_efield("EFields.csv")
    gain_data = parser.parse_gain("GainTotal.csv")
    z_data = parser.parse_impedance("ZParameters.csv")
    """

    # ------------------------------------------------------------------ #
    # Public parsing methods                                               #
    # ------------------------------------------------------------------ #

    def parse_efield(self, filepath: str | Path) -> EFieldData:
        """
        Parse an HFSS far-field E-field CSV export.

        Expected columns (in any order):
            Phi[deg], Theta[deg],
            re(rETheta)[mV], im(rETheta)[mV],
            re(rEPhi)[mV],   im(rEPhi)[mV]

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.

        Returns
        -------
        EFieldData
            Parsed data with phi/theta grids in radians and complex E-field
            arrays in Volts, shaped (n_theta, n_phi).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required columns are missing or the data grid is irregular.
        """
        filepath = self._check_file(filepath)
        raw = self._read_csv(filepath)

        required = {"Phi[deg]", "Theta[deg]",
                    "re(rETheta)[mV]", "im(rETheta)[mV]",
                    "re(rEPhi)[mV]",   "im(rEPhi)[mV]"}
        self._check_columns(raw.keys(), required, filepath)

        phi_deg   = np.asarray(raw["Phi[deg]"],       dtype=float)
        theta_deg = np.asarray(raw["Theta[deg]"],     dtype=float)
        re_Etheta = np.asarray(raw["re(rETheta)[mV]"], dtype=float)
        im_Etheta = np.asarray(raw["im(rETheta)[mV]"], dtype=float)
        re_Ephi   = np.asarray(raw["re(rEPhi)[mV]"],  dtype=float)
        im_Ephi   = np.asarray(raw["im(rEPhi)[mV]"],  dtype=float)

        phi_vals, theta_vals = self._extract_grid_axes(phi_deg, theta_deg,
                                                       filepath)

        n_phi   = len(phi_vals)
        n_theta = len(theta_vals)

        E_theta_flat = (re_Etheta + 1j * im_Etheta) * 1e-3  # mV -> V
        E_phi_flat   = (re_Ephi   + 1j * im_Ephi)   * 1e-3

        E_theta_grid, E_phi_grid = self._reshape_to_grid(
            E_theta_flat, E_phi_flat, phi_deg, theta_deg,
            phi_vals, theta_vals, filepath)

        return EFieldData(
            phi=np.deg2rad(phi_vals),
            theta=np.deg2rad(theta_vals),
            E_theta=E_theta_grid,
            E_phi=E_phi_grid,
        )

    def parse_gain(self, filepath: str | Path) -> GainData:
        """
        Parse an HFSS total gain CSV export.

        Expected columns (in any order):
            Phi[deg], Theta[deg], mag(GainTotal)

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.

        Returns
        -------
        GainData
            Parsed data with phi/theta grids in radians and gain array
            shaped (n_theta, n_phi).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required columns are missing or the data grid is irregular.
        """
        filepath = self._check_file(filepath)
        raw = self._read_csv(filepath)

        required = {"Phi[deg]", "Theta[deg]", "mag(GainTotal)"}
        self._check_columns(raw.keys(), required, filepath)

        phi_deg   = np.asarray(raw["Phi[deg]"],     dtype=float)
        theta_deg = np.asarray(raw["Theta[deg]"],   dtype=float)
        gain_flat = np.asarray(raw["mag(GainTotal)"], dtype=float)

        phi_vals, theta_vals = self._extract_grid_axes(phi_deg, theta_deg,
                                                        filepath)

        gain_grid, _ = self._reshape_to_grid(
            gain_flat, gain_flat, phi_deg, theta_deg,
            phi_vals, theta_vals, filepath)

        return GainData(
            phi=np.deg2rad(phi_vals),
            theta=np.deg2rad(theta_vals),
            gain=gain_grid,
        )

    def parse_impedance(self, filepath: str | Path) -> ImpedanceData:
        """
        Parse an HFSS Z-parameter CSV export.

        Expected columns (in any order):
            Freq [GHz], re(Z(1,1)) [], im(Z(1,1)) []

        Additional geometry columns (dL [mm], R_wire [mm]) are accepted but
        ignored; it is assumed that the file contains data for a single antenna
        geometry so the frequency axis is unique.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.

        Returns
        -------
        ImpedanceData
            Parsed data with frequency array in Hz and complex impedance in
            Ohms, both sorted by ascending frequency.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required columns are missing or the frequency axis is not
            monotonically increasing after sorting.
        """
        filepath = self._check_file(filepath)
        raw = self._read_csv(filepath)

        required = {"Freq [GHz]", "re(Z(1,1)) []", "im(Z(1,1)) []"}
        self._check_columns(raw.keys(), required, filepath)

        freq_ghz = np.asarray(raw["Freq [GHz]"],    dtype=float)
        re_z     = np.asarray(raw["re(Z(1,1)) []"], dtype=float)
        im_z     = np.asarray(raw["im(Z(1,1)) []"], dtype=float)

        sort_idx = np.argsort(freq_ghz)
        freq_hz  = freq_ghz[sort_idx] * 1e9   # GHz -> Hz
        impedance = (re_z + 1j * im_z)[sort_idx]

        if not np.all(np.diff(freq_hz) > 0):
            raise ValueError(
                f"{filepath.name}: frequency axis is not strictly increasing "
                "after sorting. Check for duplicate frequency entries.")

        return ImpedanceData(frequency=freq_hz, impedance=impedance)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_file(filepath: str | Path) -> Path:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"HFSS data file not found: {path}")
        return path

    @staticmethod
    def _read_csv(filepath: Path) -> dict[str, list]:
        """Read CSV into a dict of column-name -> list of float strings."""
        with open(filepath, newline="") as fh:
            reader = csv.DictReader(fh)
            columns: dict[str, list] = {
                field.strip(): [] for field in reader.fieldnames
            }
            for row in reader:
                for key in columns:
                    columns[key].append(row[key].strip())
        return columns

    @staticmethod
    def _check_columns(present, required: set, filepath: Path) -> None:
        missing = required - set(present)
        if missing:
            raise ValueError(
                f"{filepath.name}: missing required columns: {missing}")

    @staticmethod
    def _extract_grid_axes(phi_deg: NDArray,
                           theta_deg: NDArray,
                           filepath: Path
                           ) -> tuple[NDArray, NDArray]:
        """
        Extract unique, sorted phi and theta grid axes and validate regularity.

        Raises ValueError if the (phi, theta) pairs do not form a complete
        regular rectangular grid.
        """
        phi_vals   = np.unique(phi_deg)
        theta_vals = np.unique(theta_deg)
        expected   = len(phi_vals) * len(theta_vals)
        if len(phi_deg) != expected:
            raise ValueError(
                f"{filepath.name}: data has {len(phi_deg)} rows but a regular "
                f"grid of {len(phi_vals)} phi × {len(theta_vals)} theta values "
                f"requires {expected} rows.")
        return phi_vals, theta_vals

    @staticmethod
    def _reshape_to_grid(data1: NDArray,
                         data2: NDArray,
                         phi_deg: NDArray,
                         theta_deg: NDArray,
                         phi_vals: NDArray,
                         theta_vals: NDArray,
                         filepath: Path
                         ) -> tuple[NDArray, NDArray]:
        """
        Reorder flat data arrays into (n_theta, n_phi) grids.

        HFSS exports vary phi fastest (outer loop theta, inner loop phi), but
        this is verified rather than assumed.
        """
        n_phi   = len(phi_vals)
        n_theta = len(theta_vals)

        phi_idx   = np.searchsorted(phi_vals,   phi_deg)
        theta_idx = np.searchsorted(theta_vals, theta_deg)

        grid1 = np.empty((n_theta, n_phi), dtype=data1.dtype)
        grid2 = np.empty((n_theta, n_phi), dtype=data2.dtype)

        try:
            grid1[theta_idx, phi_idx] = data1
            grid2[theta_idx, phi_idx] = data2
        except IndexError as exc:
            raise ValueError(
                f"{filepath.name}: index out of range while mapping flat data "
                f"to grid. {exc}") from exc

        return grid1, grid2