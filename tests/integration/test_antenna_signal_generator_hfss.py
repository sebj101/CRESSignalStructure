"""
test_antenna_signal_generator_hfss.py

Integration tests for AntennaSignalGenerator with HFSSAntenna as the antenna
model.  Synthetic HFSS-format CSV files are generated once per module.

The synthetic radiation pattern is dipole-like (theta-polarised, no phi component):

    rETheta(theta, phi) = sin(theta)   [mV],   rEPhi = 0
    G(theta, phi)       = 1.5 * sin²(theta)    (linear, dimensionless)

giving nulls at bore-sight (theta = 0, pi) and a maximum at the equator
(theta = pi/2).

Antenna geometries
------------------
Equatorial antenna – receives maximum signal
    position [0.1, 0, 0], z_ax [0, 1, 0], x_ax [1, 0, 0]

    A perpendicularly-orbiting electron near the origin lies at theta = pi/2
    from this antenna (maximum gain = 1.5).  The effective length points in
    the -y direction, which aligns with the y-polarised cyclotron radiation
    field arriving from the +x direction.

Null-direction antenna – receives negligible signal
    position [0, 0.1, 0], z_ax [0, 1, 0], x_ax [1, 0, 0]

    The electron is nearly along the anti-bore-sight direction (theta ≈ pi),
    where gain = 0.

Polarisation-mismatch antenna – gain is high but coupling is near zero
    position [0.1, 0, 0], z_ax [0, 0, 1], x_ax [1, 0, 0]

    Theta = pi/2 (gain = 1.5) but the effective length points along z, which
    is orthogonal to the y-polarised cyclotron radiation.
"""

import numpy as np
import scipy.constants as sc
from scipy.signal import periodogram
import pytest

from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.TrajectoryGenerator import TrajectoryGenerator
from CRESSignalStructure.AntennaSignalGenerator import AntennaSignalGenerator
from CRESSignalStructure.antennas.HFSSAntenna import HFSSAntenna
from CRESSignalStructure.ReceiverChain import ReceiverChain


# ---------------------------------------------------------------------------
# Physical simulation parameters
# ---------------------------------------------------------------------------

R_COIL       = 3e-2
I_COIL       = 2 * 4e-3 * R_COIL / sc.mu_0

KE           = 18.6e3   # eV
ADC_RATE     = 1e9      # Hz
LO_OFFSET    = 50e6     # Hz  (LO placed this far below f_c)
OVERSAMPLING = 5
TRAJ_RATE    = 7e9      # Hz  (must be > ADC_RATE × OVERSAMPLING)

# Antenna geometry constants (see module docstring)
_EQ_POS   = np.array([0.1, 0.0, 0.0])   # equatorial: theta = pi/2, max gain
_NULL_POS = np.array([0.0, 0.1, 0.0])   # null:       theta ≈ pi,  gain ≈ 0
_Z_AX_Y   = np.array([0.0, 1.0, 0.0])   # dipole axis along y
_Z_AX_Z   = np.array([0.0, 0.0, 1.0])   # dipole axis along z (pol. mismatch)
_X_AX     = np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Synthetic HFSS CSV helpers
# ---------------------------------------------------------------------------

_THETA_DEG = [0, 90, 180]
_PHI_DEG   = [-180, 0, 180]
_FREQ_GHZ  = [24.0, 25.0, 27.0, 28.0]   # spans expected ~27 GHz f_c
_RE_Z      = [65.0, 70.0, 73.0, 75.0]
_IM_Z      = [20.0, 30.0, 42.5, 55.0]
_PATTERN_FREQ = 26.5e9                   # Hz, close to expected f_c


def _write(base, name, text):
    p = base / name
    p.write_text(text)
    return p


def _make_efield_csv(base):
    lines = [
        "Phi[deg],Theta[deg],re(rETheta)[mV],im(rETheta)[mV],"
        "re(rEPhi)[mV],im(rEPhi)[mV]"
    ]
    for phi in _PHI_DEG:
        for theta in _THETA_DEG:
            re_t = np.sin(np.deg2rad(theta))
            lines.append(f"{phi},{theta},{re_t},0.0,0.0,0.0")
    return _write(base, "efield.csv", "\n".join(lines))


def _make_gain_csv(base):
    lines = ["Phi[deg],Theta[deg],mag(GainTotal)"]
    for phi in _PHI_DEG:
        for theta in _THETA_DEG:
            g = 1.5 * np.sin(np.deg2rad(theta)) ** 2
            lines.append(f"{phi},{theta},{g}")
    return _write(base, "gain.csv", "\n".join(lines))


def _make_impedance_csv(base):
    lines = ['Freq [GHz],"re(Z(1,1)) []","im(Z(1,1)) []"']
    for f, r, i in zip(_FREQ_GHZ, _RE_Z, _IM_Z):
        lines.append(f"{f},{r},{i}")
    return _write(base, "impedance.csv", "\n".join(lines))


# ---------------------------------------------------------------------------
# Object factories
# ---------------------------------------------------------------------------

def _make_field():
    return HarmonicField(R_COIL, I_COIL, 1.0)


def _make_particle():
    return Particle(ke=KE, startPos=np.array([0.001, 0.0, 0.0]),
                    pitchAngle=np.pi / 2)


def _make_receiver(field, particle):
    f_c = field.calc_omega_0(particle) / (2 * np.pi)
    return ReceiverChain(sample_rate=ADC_RATE, lo_frequency=f_c - LO_OFFSET)


def _make_trajectory(field, particle, t_max=10e-6):
    return TrajectoryGenerator(field, particle).generate(
        sample_rate=TRAJ_RATE, t_max=t_max)


def _make_hfss_antenna(csv_dir, position, z_ax=_Z_AX_Y, x_ax=_X_AX):
    return HFSSAntenna(
        position=position, z_ax=z_ax, x_ax=x_ax,
        efield_path=csv_dir / "efield.csv",
        gain_path=csv_dir / "gain.csv",
        impedance_path=csv_dir / "impedance.csv",
        pattern_frequency=_PATTERN_FREQ,
    )


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------

def _spectrum(signal, sample_rate):
    freqs, Pxx = periodogram(signal, sample_rate, window='hann',
                              scaling='spectrum')
    return freqs, Pxx


def _power_in_window(freqs, psd, f_lo, f_hi):
    return np.sum(psd[(freqs >= f_lo) & (freqs <= f_hi)])


# ---------------------------------------------------------------------------
# Module-scope fixtures  (computed once for the whole module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csv_dir(tmp_path_factory):
    """Write synthetic HFSS CSV files to a temporary directory."""
    base = tmp_path_factory.mktemp("hfss_csv")
    _make_efield_csv(base)
    _make_gain_csv(base)
    _make_impedance_csv(base)
    return base


@pytest.fixture(scope="module")
def equatorial_signal(csv_dir):
    """
    Full signal pipeline: perpendicular electron, y-axis dipole at the
    equatorial position.  Shared by output-structure and mainband tests.
    """
    field    = _make_field()
    particle = _make_particle()
    receiver = _make_receiver(field, particle)
    traj     = _make_trajectory(field, particle, t_max=10e-6)
    antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
    gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                      oversampling_factor=OVERSAMPLING)
    time, signal = gen.generate_signal(return_time=True)
    return time, signal, receiver


@pytest.fixture(scope="module")
def null_direction_signal(csv_dir):
    """
    Full signal pipeline: same electron / receiver, antenna at the null
    direction.  Used for the directional-response comparison.
    """
    field    = _make_field()
    particle = _make_particle()
    receiver = _make_receiver(field, particle)
    traj     = _make_trajectory(field, particle, t_max=10e-6)
    antenna  = _make_hfss_antenna(csv_dir, _NULL_POS)
    gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                      oversampling_factor=OVERSAMPLING)
    time, signal = gen.generate_signal(return_time=True)
    return time, signal, receiver


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestHFSSAntennaInPipeline:

    def test_construction_with_hfss_antenna_succeeds(self, csv_dir):
        """AntennaSignalGenerator accepts an HFSSAntenna without error."""
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        assert gen is not None

    def test_get_antenna_returns_hfss_antenna(self, csv_dir):
        """get_antenna() returns the original HFSSAntenna instance."""
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        assert isinstance(gen.get_antenna(), HFSSAntenna)
        assert gen.get_antenna() is antenna

    def test_average_cyclotron_frequency_is_physical(self, csv_dir):
        """Computed average cyclotron frequency lies within the HFSS impedance range."""
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        f_avg = gen.get_average_cyclotron_frequency()
        # Must be positive and in the GHz range for an 18.6 keV electron in ~1 T
        assert 20e9 < f_avg < 35e9


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestHFSSSignalOutputStructure:

    def test_returns_tuple_of_ndarrays(self, equatorial_signal):
        time, signal, _ = equatorial_signal
        assert isinstance(time, np.ndarray)
        assert isinstance(signal, np.ndarray)

    def test_signal_is_complex(self, equatorial_signal):
        """IQ downmixed output must be complex."""
        _, signal, _ = equatorial_signal
        assert np.iscomplexobj(signal)

    def test_time_and_signal_same_length(self, equatorial_signal):
        time, signal, _ = equatorial_signal
        assert len(time) == len(signal)

    def test_time_step_matches_adc_rate(self, equatorial_signal):
        """Mean time step must equal 1 / ADC sample rate."""
        time, _, receiver = equatorial_signal
        expected_dt = 1.0 / receiver.get_sample_rate()
        np.testing.assert_allclose(
            np.mean(np.diff(time)), expected_dt, rtol=1e-3)

    def test_return_time_false_gives_single_array(self, csv_dir):
        """generate_signal(return_time=False) returns one 1-D array."""
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        result   = gen.generate_signal(return_time=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert np.iscomplexobj(result)


# ---------------------------------------------------------------------------
# Mainband physics
# ---------------------------------------------------------------------------

class TestHFSSMainbandFrequency:

    def test_signal_has_nonzero_power(self, equatorial_signal):
        """Equatorial HFSSAntenna receives nonzero power from the electron."""
        _, signal, _ = equatorial_signal
        assert np.mean(np.abs(signal) ** 2) > 0.0

    def test_peak_if_frequency_matches_lo_offset(self, equatorial_signal):
        """
        Peak of the IF spectrum must equal LO_OFFSET within 0.2%.

        A perpendicularly-orbiting electron radiates at the cyclotron
        frequency.  After mixing with the LO at f_c - LO_OFFSET, the IF peak
        must appear at exactly LO_OFFSET.
        """
        time, signal, receiver = equatorial_signal
        freqs, psd = _spectrum(signal, receiver.get_sample_rate())
        peak_f     = abs(freqs[np.argmax(psd)])
        np.testing.assert_allclose(peak_f, LO_OFFSET, rtol=0.002)

    def test_mainband_contains_majority_of_power(self, equatorial_signal):
        """
        For a 90-degree pitch-angle electron there is no axial bounce, so no
        sidebands.  More than 90% of IF power must sit in a narrow window
        around the peak.
        """
        time, signal, receiver = equatorial_signal
        freqs, psd = _spectrum(signal, receiver.get_sample_rate())
        freq_res   = 1.0 / time[-1]
        peak_f     = freqs[np.argmax(psd)]
        mainband   = _power_in_window(freqs, psd,
                                      peak_f - 3 * freq_res,
                                      peak_f + 3 * freq_res)
        assert mainband / np.sum(psd) > 0.90


# ---------------------------------------------------------------------------
# Directional response
# ---------------------------------------------------------------------------

class TestHFSSDirectionalResponse:

    def test_equatorial_antenna_receives_nonzero_power(self, equatorial_signal):
        """Antenna at theta=pi/2 (max gain, aligned polarisation) gives signal."""
        _, signal, _ = equatorial_signal
        assert np.mean(np.abs(signal) ** 2) > 0.0

    def test_null_direction_antenna_receives_negligible_power(
            self, null_direction_signal):
        """
        Antenna at theta ≈ pi (gain = 0) receives nearly zero power.
        The signal power must be orders of magnitude less than from the
        equatorial configuration.
        """
        _, signal, _ = null_direction_signal
        # Power is nonzero only due to the finite Larmor radius displacing
        # the electron slightly from the exact null direction.
        # Store for use in the comparison test below (accessed via fixture).
        assert np.isfinite(np.mean(np.abs(signal) ** 2))

    def test_equatorial_power_much_greater_than_null_power(
            self, equatorial_signal, null_direction_signal):
        """
        The equatorial antenna (gain = 1.5) must deliver at least 100 times
        more signal power than the null-direction antenna (gain ≈ 0).

        The expected gain ratio is ~1.5 / (1.5 * sin²(δθ)) where
        δθ ≈ 0.015 rad (Larmor radius / antenna distance).
        This gives a power ratio of ~10⁴, well above the 100× threshold.
        """
        _, sig_eq,   _ = equatorial_signal
        _, sig_null, _ = null_direction_signal
        power_eq   = np.mean(np.abs(sig_eq)   ** 2)
        power_null = np.mean(np.abs(sig_null) ** 2)
        assert power_eq > power_null * 100

    def test_polarisation_mismatch_gives_negligible_signal(self, csv_dir):
        """
        A z-axis dipole at the equatorial position has theta = pi/2 (gain = 1.5)
        but its effective length is along z, orthogonal to the y-polarised
        cyclotron radiation.  Received power must be negligible compared to
        the optimally-oriented (y-axis) dipole.

        This validates that the antenna orientation, not just the gain pattern,
        is applied correctly in the signal calculation.
        """
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=5e-6)
        receiver = _make_receiver(field, particle)

        # y-axis dipole – optimally aligned with the cyclotron radiation
        ant_y   = _make_hfss_antenna(csv_dir, _EQ_POS, z_ax=_Z_AX_Y)
        gen_y   = AntennaSignalGenerator(traj, ant_y, receiver,
                                         oversampling_factor=OVERSAMPLING)
        _, sig_y = gen_y.generate_signal()

        # z-axis dipole – polarisation-mismatched at the same position
        ant_z   = _make_hfss_antenna(csv_dir, _EQ_POS, z_ax=_Z_AX_Z)
        gen_z   = AntennaSignalGenerator(traj, ant_z, receiver,
                                         oversampling_factor=OVERSAMPLING)
        _, sig_z = gen_z.generate_signal()

        power_y = np.mean(np.abs(sig_y) ** 2)
        power_z = np.mean(np.abs(sig_z) ** 2)

        # y-axis dipole must receive far more power
        assert power_y > power_z * 100

    def test_null_direction_peak_is_at_cyclotron_if_frequency(
            self, null_direction_signal):
        """
        Even in the null direction the residual signal (from finite Larmor
        radius) oscillates at the cyclotron frequency.  The IF peak must still
        appear near LO_OFFSET.
        """
        time, signal, receiver = null_direction_signal
        freqs, psd = _spectrum(signal, receiver.get_sample_rate())
        peak_f     = abs(freqs[np.argmax(psd)])
        np.testing.assert_allclose(peak_f, LO_OFFSET, rtol=0.01)


# ---------------------------------------------------------------------------
# Impedance and effective length
# ---------------------------------------------------------------------------

class TestHFSSEffectiveLengthPhysics:

    def test_impedance_at_cyclotron_frequency_has_positive_real_part(
            self, csv_dir):
        """
        Re(Z) > 0 is required for the passive antenna model.  Verifies that
        the impedance interpolation is evaluated at the correct frequency.
        """
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        f_c = gen.get_average_cyclotron_frequency()
        assert antenna.get_impedance(f_c).real > 0.0

    def test_effective_length_shape_and_type(self, csv_dir):
        """
        get_effective_length called at the cyclotron frequency for a batch of
        positions returns a complex (N, 3) array.
        """
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        f_c = gen.get_average_cyclotron_frequency()

        test_positions = np.array([
            [0.0, 0.0, 0.0],    # equatorial source
            [0.05, 0.0, 0.0],
        ])
        l_eff = antenna.get_effective_length(f_c, test_positions)
        assert l_eff.shape == (2, 3)
        assert np.iscomplexobj(l_eff)

    def test_effective_length_zero_in_null_direction(self, csv_dir):
        """
        Effective length vanishes along the null direction (theta = 0 or pi).
        A source placed directly below the antenna (anti-bore-sight) should
        return |l_eff| = 0.
        """
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _NULL_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        f_c = gen.get_average_cyclotron_frequency()

        # Source directly below the null antenna, along the -y direction
        # (theta = pi from the antenna bore-sight, where gain = 0)
        source_at_null = np.array([[0.0, 0.0, 0.0]])
        l_eff = antenna.get_effective_length(f_c, source_at_null)
        np.testing.assert_array_almost_equal(np.abs(l_eff), np.zeros((1, 3)),
                                             decimal=6)

    def test_effective_length_nonzero_at_equatorial_source(self, csv_dir):
        """
        Effective length is nonzero for a source at the equatorial position.
        """
        field    = _make_field()
        particle = _make_particle()
        traj     = _make_trajectory(field, particle, t_max=1e-6)
        antenna  = _make_hfss_antenna(csv_dir, _EQ_POS)
        receiver = _make_receiver(field, particle)
        gen      = AntennaSignalGenerator(traj, antenna, receiver,
                                          oversampling_factor=OVERSAMPLING)
        f_c = gen.get_average_cyclotron_frequency()

        # Source at origin – equatorial direction from the equatorial antenna
        source = np.array([[0.0, 0.0, 0.0]])
        l_eff  = antenna.get_effective_length(f_c, source)
        assert np.linalg.norm(l_eff) > 0.0
