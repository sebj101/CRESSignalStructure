"""
HFSSAntenna.py

Antenna model built from HFSS simulation exports.

The far-field pattern (E-field components and gain) is loaded from CSV files
exported by HFSS and used to evaluate effective length, gain, and impedance.
"""

import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator, interp1d
import scipy.constants as sc

from .BaseAntenna import BaseAntenna

logger = logging.getLogger(__name__)
from .HFSSDataParser import HFSSDataParser, EFieldData, GainData, ImpedanceData


# Free-space wave impedance eta = mu_0 * c
_ETA0 = sc.mu_0 * sc.c  # ~376.73 Ohm


class HFSSAntenna(BaseAntenna):
    """
    Antenna model driven by HFSS simulation data.

    The radiation pattern (rETheta, rEPhi) and total gain are taken from an
    HFSS far-field export at a single simulation frequency.  The impedance
    Z(1,1) is loaded from a separate frequency-swept export and used to
    interpolate the input impedance at any requested frequency.

    Effective length is derived from the gain and impedance data using:

        |l_eff(theta, phi)| = sqrt( G(theta, phi) * lambda^2 * Re(Z(f)) / (pi * eta) )

    The direction and phase of l_eff follow the normalised complex rE
    pattern, so both polarisation and phase information from the HFSS
    simulation are preserved.

    Parameters
    ----------
    position : ArrayLike
        3-vector antenna phase-centre position in the trap frame, metres.
    z_ax : ArrayLike
        Unit vector along the HFSS +Z axis (bore-sight) expressed in the
        trap frame.
    x_ax : ArrayLike
        Unit vector along the HFSS +X axis (phi = 0 reference) expressed in
        the trap frame.  Need not be exactly orthogonal to z_ax; a
        Gram-Schmidt correction is applied in the base class.
    efield_path : str
        Path to the HFSS far-field E-field CSV (EFields.csv).
    gain_path : str
        Path to the HFSS total gain CSV (GainTotal.csv).
    impedance_path : str
        Path to the HFSS Z-parameter CSV (ZParameters.csv).
    pattern_frequency : float
        Frequency in Hz at which the E-field and gain data were simulated.
        Used to set the wavelength when computing the normalised effective
        length pattern.
    """

    def __init__(self, position: ArrayLike, z_ax: ArrayLike, x_ax: ArrayLike,
                 efield_path: str, gain_path: str, impedance_path: str,
                 pattern_frequency: float):

        super().__init__(position, z_ax, x_ax)

        pattern_frequency = self._validate_frequency(pattern_frequency)
        self._pattern_frequency = pattern_frequency

        parser = HFSSDataParser()
        e_data: EFieldData = parser.parse_efield(efield_path)
        g_data: GainData = parser.parse_gain(gain_path)
        z_data: ImpedanceData = parser.parse_impedance(impedance_path)

        self._pattern_theta = e_data.theta   # (n_theta,) radians, 0..pi
        self._pattern_phi = e_data.phi       # (n_phi,)   radians, -pi..pi

        self._E_theta_re_interp = self._make_2d_interp(e_data.E_theta.real)
        self._E_theta_im_interp = self._make_2d_interp(e_data.E_theta.imag)
        self._E_phi_re_interp   = self._make_2d_interp(e_data.E_phi.real)
        self._E_phi_im_interp   = self._make_2d_interp(e_data.E_phi.imag)
        self._gain_interp       = self._make_2d_interp(g_data.gain)

        # Precompute the normalised pattern magnitude for use in l_eff
        # |rE|^2 = |rE_theta|^2 + |rE_phi|^2
        rE_sq = np.abs(e_data.E_theta)**2 + np.abs(e_data.E_phi)**2
        self._rE_sq_interp = self._make_2d_interp(rE_sq)

        # Impedance: interpolate real and imaginary parts separately so we can
        # safely extrapolate. Warn if the requested frequency falls outside the
        # simulation range.
        self._z_freq = z_data.frequency
        self._z_re_interp = interp1d(z_data.frequency, z_data.impedance.real,
                                     kind='cubic', fill_value='extrapolate')
        self._z_im_interp = interp1d(z_data.frequency, z_data.impedance.imag,
                                     kind='cubic', fill_value='extrapolate')

        logger.info("Created HFSSAntenna at pos=%s, pattern_frequency=%.5e Hz, "
                    "efield='%s', gain='%s', impedance='%s'",
                    self._pos, self._pattern_frequency,
                    efield_path, gain_path, impedance_path)

    # ------------------------------------------------------------------ #
    # BaseAntenna abstract methods                                         #
    # ------------------------------------------------------------------ #

    def GetETheta(self, pos: NDArray) -> NDArray:
        """
        Theta component of the far-field pattern as a Cartesian vector.

        Returns the complex rETheta value interpolated from the HFSS pattern,
        multiplied by the theta-hat unit vector at each source position.
        The result is in the lab coordinates.

        Parameters
        ----------
        pos : NDArray
            (N, 3) source positions in metres.

        Returns
        -------
        NDArray
            (N, 3) complex array, units V (rE component, i.e. r * E_theta).
        """
        pos = np.atleast_2d(pos)
        theta, phi = self._pos_to_antenna_angles(pos)
        rE_theta = self._interp_complex(self._E_theta_re_interp,
                                        self._E_theta_im_interp, theta, phi)
        theta_hat = self._theta_hat_lab(theta, phi)     # (N, 3) real
        return rE_theta[:, np.newaxis] * theta_hat

    def GetEPhi(self, pos: NDArray) -> NDArray:
        """
        Phi component of the far-field pattern as a Cartesian vector in lab 
        coordinates.

        Parameters
        ----------
        pos : NDArray
            (N, 3) source positions in metres.

        Returns
        -------
        NDArray
            (N, 3) complex array, units V (rE component, i.e. r * E_phi).
        """
        pos = np.atleast_2d(pos)
        theta, phi = self._pos_to_antenna_angles(pos)
        rE_phi = self._interp_complex(self._E_phi_re_interp,
                                      self._E_phi_im_interp, theta, phi)
        phi_hat = self._phi_hat_lab(phi)                 # (N, 3) real
        return rE_phi[:, np.newaxis] * phi_hat

    def GetEffectiveLength(self, frequency: float, pos: NDArray) -> NDArray:
        """
        Effective length vector derived from the HFSS pattern.

        The magnitude is obtained from the gain and input resistance via:

            |l_eff| = sqrt( G * lambda^2 * Re(Z(f)) / (pi * eta) )

        The direction and relative phase follow the normalised complex rE
        pattern, preserving both polarisation and inter-angle phase from HFSS.

        Parameters
        ----------
        frequency : float
            Frequency in Hz at which to evaluate Z and lambda.
        pos : NDArray
            (N, 3) positions in metres.  Only the direction from the antenna
            matters; magnitude is normalised internally.

        Returns
        -------
        NDArray
            (N, 3) complex effective length vectors in metres.
        """
        frequency = self._validate_frequency(frequency)
        pos = np.atleast_2d(pos)

        theta, phi = self._pos_to_antenna_angles(pos)

        # Scalar pattern components
        rE_theta = self._interp_complex(self._E_theta_re_interp,
                                        self._E_theta_im_interp, theta, phi)
        rE_phi   = self._interp_complex(self._E_phi_re_interp,
                                        self._E_phi_im_interp, theta, phi)

        # Gain and |rE|^2 at each direction
        gain  = self._gain_interp((theta, phi))          # (N,)
        rE_sq = self._rE_sq_interp((theta, phi))         # (N,)

        # Effective length magnitude from gain + impedance
        wavelength = sc.c / frequency
        R_in = np.real(self.GetImpedance(frequency))
        l_mag = np.sqrt(np.maximum(gain, 0.0) * wavelength**2 * R_in
                        / (np.pi * _ETA0))               # (N,)

        # Scale the rE pattern to unit magnitude (avoid /0 for silent directions)
        rE_mag = np.sqrt(np.maximum(rE_sq, 0.0))        # (N,)
        safe_rE = np.where(rE_mag > 1e-30, rE_mag, 1.0)

        # Complex scaling: l_eff = l_mag * (rE / |rE|)
        scale_theta = l_mag * rE_theta / safe_rE         # (N,) complex
        scale_phi   = l_mag * rE_phi   / safe_rE         # (N,) complex

        # Zero out directions where |rE| is negligibly small
        zero = rE_mag < 1e-30
        scale_theta[zero] = 0.0
        scale_phi[zero]   = 0.0

        theta_hat = self._theta_hat_lab(theta, phi)      # (N, 3) real
        phi_hat   = self._phi_hat_lab(phi)               # (N, 3) real

        return (scale_theta[:, np.newaxis] * theta_hat
                + scale_phi[:, np.newaxis] * phi_hat)

    def GetImpedance(self, frequency: float) -> complex:
        """
        Input impedance interpolated from HFSS Z-parameter data.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        complex
            Z(1,1) in Ohms.  Issues a warning if frequency lies outside the
            simulated range (extrapolation is used in that case).
        """
        frequency = self._validate_frequency(frequency)

        if not (self._z_freq[0] <= frequency <= self._z_freq[-1]):
            logger.warning(
                "Requested frequency %.4f GHz is outside the simulated range "
                "[%.4f, %.4f] GHz. Extrapolating.",
                frequency / 1e9, self._z_freq[0] / 1e9, self._z_freq[-1] / 1e9)

        re = float(self._z_re_interp(frequency))
        im = float(self._z_im_interp(frequency))
        return complex(re, im)

    def GetGain(self, theta: float, phi: float) -> float:
        """
        Total gain interpolated from HFSS pattern data.

        Parameters
        ----------
        theta : float
            Polar angle in radians, measured from the antenna z-axis.
        phi : float
            Azimuthal angle in radians, measured from the antenna x-axis.

        Returns
        -------
        float
            Dimensionless gain (linear, not dB).
        """
        theta, phi = self._validate_angles(theta, phi)
        phi = self._wrap_phi(np.array([phi]))[0]
        theta = float(np.clip(theta, self._pattern_theta[0],
                              self._pattern_theta[-1]))
        return float(self._gain_interp((theta, phi)))

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                   #
    # ------------------------------------------------------------------ #

    def _pos_to_antenna_angles(self, pos: NDArray) -> tuple[NDArray, NDArray]:
        """
        Convert lab-frame source positions to antenna-frame (theta, phi).

        Parameters
        ----------
        pos : NDArray
            (N, 3) positions in metres.

        Returns
        -------
        theta : NDArray
            (N,) polar angles in radians, clipped to [0, pi].
        phi : NDArray
            (N,) azimuthal angles in radians, clipped to [-pi, pi].
        """
        r = pos - self._pos                              # (N, 3)
        r_mag = np.linalg.norm(r, axis=1, keepdims=True)

        if np.any(r_mag == 0.0):
            raise ValueError("Source position cannot equal antenna position.")

        r_hat = r / r_mag                               # (N, 3)

        cos_theta = np.clip(np.dot(r_hat, self._z_ax), -1.0, 1.0)
        theta = np.arccos(cos_theta)                    # (N,)

        x_comp = np.dot(r_hat, self._x_ax)             # (N,)
        y_comp = np.dot(r_hat, self._y_ax)             # (N,)
        phi = np.arctan2(y_comp, x_comp)               # (N,) in (-pi, pi]

        # Clip theta to the grid range (HFSS data is 0..pi)
        theta = np.clip(theta, self._pattern_theta[0], self._pattern_theta[-1])
        phi   = self._wrap_phi(phi)

        return theta, phi

    def _wrap_phi(self, phi: NDArray) -> NDArray:
        """Clamp phi to the range covered by the HFSS data."""
        phi_min = self._pattern_phi[0]
        phi_max = self._pattern_phi[-1]
        return np.clip(phi, phi_min, phi_max)

    def _theta_hat_lab(self, theta: NDArray, phi: NDArray) -> NDArray:
        """
        Compute theta-hat unit vectors in the lab frame.

        theta_hat (antenna frame) = cos(theta)*cos(phi) x_hat
                                   + cos(theta)*sin(phi) y_hat
                                   - sin(theta) z_hat
        Then rotated to lab frame via the antenna basis vectors.

        Parameters
        ----------
        theta, phi : NDArray
            (N,) angles in radians.

        Returns
        -------
        NDArray
            (N, 3) unit vectors in the lab frame.
        """
        ct = np.cos(theta)   # (N,)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)

        # theta_hat in antenna frame expressed in lab frame
        return (ct * cp)[:, np.newaxis] * self._x_ax \
             + (ct * sp)[:, np.newaxis] * self._y_ax \
             - st[:, np.newaxis]        * self._z_ax

    def _phi_hat_lab(self, phi: NDArray) -> NDArray:
        """
        Compute phi-hat unit vectors in the lab frame.

        phi_hat (antenna frame) = -sin(phi) x_hat + cos(phi) y_hat

        Parameters
        ----------
        phi : NDArray
            (N,) azimuthal angles in radians.

        Returns
        -------
        NDArray
            (N, 3) unit vectors in the lab frame.
        """
        return (-np.sin(phi))[:, np.newaxis] * self._x_ax \
             + ( np.cos(phi))[:, np.newaxis] * self._y_ax

    # ------------------------------------------------------------------ #
    # Interpolation helpers                                                #
    # ------------------------------------------------------------------ #

    def _make_2d_interp(self, data: NDArray) -> RegularGridInterpolator:
        """
        Build a RegularGridInterpolator over the (theta, phi) grid.

        Parameters
        ----------
        data : NDArray
            (n_theta, n_phi) real array.

        Returns
        -------
        RegularGridInterpolator
        """
        return RegularGridInterpolator(
            (self._pattern_theta, self._pattern_phi),
            data,
            method='linear',
            bounds_error=False,
            fill_value=None,   # extrapolate at boundaries
        )

    @staticmethod
    def _interp_complex(re_interp: RegularGridInterpolator,
                        im_interp: RegularGridInterpolator,
                        theta: NDArray,
                        phi: NDArray) -> NDArray:
        """
        Evaluate a complex quantity by interpolating real and imaginary parts.

        Parameters
        ----------
        re_interp, im_interp : RegularGridInterpolator
            Interpolators for the real and imaginary components.
        theta, phi : NDArray
            (N,) query angles in radians.

        Returns
        -------
        NDArray
            (N,) complex values.
        """
        pts = np.column_stack((theta, phi))
        return re_interp(pts) + 1j * im_interp(pts)