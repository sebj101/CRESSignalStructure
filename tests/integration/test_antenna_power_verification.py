"""
test_antenna_power_verification.py

Physics consistency tests for the AntennaSignalGenerator power calculations
with a HalfWaveDipoleAntenna.

Three complementary checks:
1. Antenna identity: |l_eff|^2 = G R_rad λ^2 / (π η_0)
2. Poynting + effective area matches voltage power
3. Larmor cross-check: L-W radiation field matches predicted Poynting flux
"""

import numpy as np
import scipy.constants as sc
import pytest

from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.TrajectoryGenerator import TrajectoryGenerator
from CRESSignalStructure.AntennaSignalGenerator import AntennaSignalGenerator
from CRESSignalStructure.antennas.DipoleAntennas import HalfWaveDipoleAntenna
from CRESSignalStructure.ReceiverChain import ReceiverChain


# ---------------------------------------------------------------------------
# Shared physical parameters
# ---------------------------------------------------------------------------

R_COIL = 3e-2
I_COIL = 2 * 4e-3 * R_COIL / sc.mu_0
KE = 18.6e3           # eV
ADC_RATE = 1e9        # Hz
OVERSAMPLING = 5
TRAJ_RATE = 100e9     # Hz (high rate for accurate retarded-time interpolation)
ANTENNA_POS = np.array([0.1, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def signal_pipeline():
    """Full pipeline for a perpendicular electron with HalfWaveDipoleAntenna."""
    field = HarmonicField(R_COIL, I_COIL, 1.0)
    particle = Particle(ke=KE, startPos=np.array([0.0, 0.0, 0.0]),
                        pitchAngle=np.pi / 2)
    f_c = field.calc_omega_0(particle) / (2 * np.pi)

    traj = TrajectoryGenerator(field, particle).generate(
        sample_rate=TRAJ_RATE, t_max=2e-6)

    antenna = HalfWaveDipoleAntenna(
        position=ANTENNA_POS,
        orientation=np.array([0.0, 1.0, 0.0]),
        resonant_frequency=f_c
    )

    receiver = ReceiverChain(sample_rate=ADC_RATE, lo_frequency=f_c - 50e6)
    gen = AntennaSignalGenerator(traj, antenna, receiver,
                                 oversampling_factor=OVERSAMPLING)
    return gen, particle


@pytest.fixture(scope="module")
def retarded_data(signal_pipeline):
    """Pre-computed E-field and voltage for Sections 2 and 3."""
    gen, _ = signal_pipeline

    t_adv, spline = gen._calculate_advanced_time()

    signal_sample_rate = ADC_RATE * OVERSAMPLING
    dt = 1.0 / signal_sample_rate
    n_points = int(np.floor((t_adv[-1] - t_adv[0]) * signal_sample_rate))
    t_obs = t_adv[0] + np.arange(n_points) * dt

    ret_quantities = gen._calculate_retarded_quantities(t_obs, spline)
    E_field = gen._calculate_E_field(ret_quantities)
    voltage = gen._calculate_antenna_voltage(E_field, ret_quantities)
    valid = ret_quantities['t_ret'] >= 0

    return {
        'ret_quantities': ret_quantities,
        'E_field': E_field,
        'voltage': voltage,
        'valid': valid,
    }


# ---------------------------------------------------------------------------
# Section 1: Antenna identity check
# |l_eff(θ)|^2 = G(θ) R_rad λ^2 / (π η_0)
# ---------------------------------------------------------------------------

class TestAntennaIdentity:
    """
    Verify the antenna theorem for HalfWaveDipoleAntenna at all polar angles.

    get_effective_length and get_gain use different parametrisations of the
    radiation pattern; this test confirms they are mutually consistent.
    """

    def test_identity_holds_across_all_angles(self):
        """
        |l_eff|^2 / (G R_rad λ^2 / π η_0) = 1 within 0.1% for all non-null angles.
        """
        f = 27e9
        antenna = HalfWaveDipoleAntenna(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 1.0]),
            resonant_frequency=f
        )
        wavelength = sc.c / f
        eta_0 = sc.mu_0 * sc.c
        R_rad = antenna.get_impedance(f).real
        rhs_scale = R_rad * wavelength**2 / (np.pi * eta_0)

        thetas = np.linspace(0.05, np.pi - 0.05, 200)
        ratios = []
        for theta in thetas:
            pos = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
            l_eff = antenna.get_effective_length(f, pos)
            G = antenna.get_gain(theta, 0.0)
            if G > 1e-6:
                ratios.append(np.sum(l_eff**2) / (G * rhs_scale))

        np.testing.assert_allclose(ratios, 1.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# Section 2: Poynting flux + effective area vs voltage power
# ---------------------------------------------------------------------------

class TestPoyntingVsVoltagePower:
    """
    Verify that two independent power estimates from the signal chain agree.

    P_voltage  = <V^2> / (4 R_rad)                     — uses l_eff
    P_poynting = <V^2 / |l_eff|^2> G λ^2 / (4 π η_0)  — uses G

    The ratio equals the antenna identity ratio from Section 1. Disagreement
    catches bugs in either the l_eff or G code paths.
    """

    def test_voltage_power_matches_poynting_power(self, signal_pipeline,
                                                   retarded_data):
        """Voltage-method and Poynting-method power estimates agree within 0.1%."""
        gen, _ = signal_pipeline
        antenna = gen.get_antenna()
        f_c_avg = gen.get_average_cyclotron_frequency()
        eta_0 = sc.mu_0 * sc.c
        wavelength = sc.c / f_c_avg

        valid = retarded_data['valid']
        V = retarded_data['voltage'][valid]
        E = retarded_data['E_field'][valid]
        n_hat = retarded_data['ret_quantities']['n_hat_ret'][valid]

        pos_synth = antenna.get_position() - n_hat
        l_eff = antenna.get_effective_length(f_c_avg, pos_synth)
        l_eff_sq = np.sum(l_eff**2, axis=1)

        R_rad = antenna.get_impedance(f_c_avg).real
        P_voltage = np.mean(V**2) / (4 * R_rad)

        source_pos = np.array([[0.0, 0.0, 0.0]])
        theta_ant = float(antenna.get_theta(source_pos)[0])
        phi_ant = float(antenna.get_phi(source_pos)[0])
        G = antenna.get_gain(theta_ant, phi_ant)
        P_poynting = (np.mean(V**2 / l_eff_sq) / eta_0
                      * G * wavelength**2 / (4 * np.pi))

        np.testing.assert_allclose(P_voltage, P_poynting, rtol=1e-3)


# ---------------------------------------------------------------------------
# Section 3: Larmor cross-check
# ---------------------------------------------------------------------------

def _compute_dP_dOmega_avg(n_hat, beta_perp, omega_c, n_steps=2000):
    """
    Time-averaged dP/dΩ for a relativistic charge in circular orbit.

    Numerically integrates the exact relativistic formula (Jackson 14.38,
    observer-time version) over one full orbit.
    """
    prefactor = sc.e**2 / (16 * np.pi**2 * sc.epsilon_0 * sc.c)
    phases = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    beta = beta_perp * np.column_stack(
        [-np.sin(phases), np.cos(phases), np.zeros(n_steps)])
    beta_dot = (beta_perp * omega_c
                * np.column_stack([-np.cos(phases), -np.sin(phases),
                                   np.zeros(n_steps)]))
    cross1 = np.cross(n_hat - beta, beta_dot)
    cross2 = np.cross(n_hat, cross1)
    kappa = 1.0 - beta @ n_hat
    return prefactor * np.mean(np.sum(cross2**2, axis=1) / kappa**6)


class TestLarmorCrossCheck:
    """
    Verify that the L-W radiation field Poynting flux matches an independent
    prediction based only on particle kinematics and the relativistic
    dP/dΩ formula.
    """

    def test_LW_radiation_flux_matches_larmor_prediction(self, signal_pipeline,
                                                          retarded_data):
        """
        Time-averaged Poynting flux from the L-W radiation field agrees with the
        Larmor + angular-pattern prediction within 0.2%.
        """
        gen, particle = signal_pipeline
        eta_0 = sc.mu_0 * sc.c
        f_c_avg = gen.get_average_cyclotron_frequency()
        omega_c = 2 * np.pi * f_c_avg
        beta_perp = particle.get_beta()

        # Prediction from particle parameters alone
        R_ant = np.linalg.norm(ANTENNA_POS)
        n_hat_ant = ANTENNA_POS / R_ant
        S_predicted = _compute_dP_dOmega_avg(n_hat_ant, beta_perp, omega_c) / R_ant**2

        # Radiation-only L-W field from the signal pipeline
        valid = retarded_data['valid']
        n_hat = retarded_data['ret_quantities']['n_hat_ret'][valid]
        R_ret = retarded_data['ret_quantities']['R_ret'][valid]
        r_ret = retarded_data['ret_quantities']['r_ret'][valid]
        psi_ret = retarded_data['ret_quantities']['psi_ret'][valid]
        t_ret = retarded_data['ret_quantities']['t_ret'][valid]

        trajectory = gen.get_trajectory()
        vel, acc = trajectory.reconstruct_kinematics(r_ret, psi_ret, t_ret)
        beta = vel / sc.c
        beta_dot = acc / sc.c
        n_dot_beta = np.sum(n_hat * beta, axis=1)

        q = particle.get_charge()
        prefactor = (q / (4 * np.pi * sc.epsilon_0)
                     / (R_ret**2 * (1 - n_dot_beta)**3))

        n_dot_beta_dot = np.sum(n_hat * beta_dot, axis=1, keepdims=True)
        n_dot_n_minus_beta = np.sum(n_hat * (n_hat - beta), axis=1, keepdims=True)
        E_acc = prefactor[:, np.newaxis] * (
            R_ret[:, np.newaxis] / sc.c * (
                n_dot_beta_dot * (n_hat - beta)
                - n_dot_n_minus_beta * beta_dot
            )
        )

        S_LW_rad = np.mean(np.sum(E_acc**2, axis=1)) / eta_0

        np.testing.assert_allclose(S_LW_rad, S_predicted, rtol=2e-3)
