"""
AntennaSignalGenerator.py

Classes for generating CRES signals from electron trajectories using antenna models.

This module provides tools for calculating the electromagnetic radiation
from an electron trajectory as detected by an antenna, including proper
treatment of retarded time effects and Liénard-Wiechert fields.
"""

import numpy as np
from numpy.typing import NDArray
import scipy.constants as sc
from scipy.interpolate import CubicSpline

from CRESSignalStructure.TrajectoryGenerator import Trajectory
from CRESSignalStructure.antennas import BaseAntenna
from CRESSignalStructure.ReceiverChain import ReceiverChain


class AntennaSignalGenerator:
    """
    Generate antenna signals from electron trajectories

    This class computes the voltage signal detected by an antenna from
    a radiating electron, accounting for:
    - Retarded time effects
    - Liénard-Wiechert electromagnetic fields
    - Antenna effective length and directional response
    - Receiver chain downmixing and digitization

    The calculation uses the full Liénard-Wiechert potentials to compute
    the electric field at the antenna location, properly accounting for
    both velocity fields and acceleration (radiation) fields.
    """

    def __init__(self, trajectory: Trajectory, antenna: BaseAntenna,
                 receiver_chain: ReceiverChain, oversampling_factor: int = 5):
        """
        Initialize AntennaSignalGenerator

        Parameters
        ----------
        trajectory : Trajectory
            Electron trajectory containing position, velocity, and acceleration
        antenna : BaseAntenna
            Antenna for detecting the radiation
        receiver_chain : ReceiverChain
            Receiver chain for signal processing
        oversampling_factor : int
            Factor by which we oversample the digitizer sample rate

        Raises
        ------
        TypeError
            If inputs have incorrect types
        ValueError
            If oversampling factor is not positive
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory object")
        if not isinstance(antenna, BaseAntenna):
            raise TypeError("antenna must be a BaseAntenna object")
        if not isinstance(receiver_chain, ReceiverChain):
            raise TypeError("receiver_chain must be a ReceiverChain object")

        # Check what the trajectory sampling frequency is vs the digitizer frequency
        traj_sample_freq = trajectory.get_sample_rate()
        dig_sample_freq = receiver_chain.get_sample_rate()
        if traj_sample_freq <= dig_sample_freq:
            raise ValueError(f'Trajectory sample rate {traj_sample_freq:.2e} Hz'
                             'must not be less than the digitizer sample rate '
                             f'of {dig_sample_freq:.2e} Hz')
        if oversampling_factor < 1:
            raise ValueError('Oversample factor must be positive')

        self.__trajectory = trajectory
        self.__antenna = antenna
        self.__receiver_chain = receiver_chain
        self.__oversampling_factor = oversampling_factor

        # Calculate average cyclotron frequency for antenna calculations
        self.__avg_cyclotron_frequency = self._calculate_average_cyclotron_frequency()

    def _calculate_average_cyclotron_frequency(self) -> float:
        """
        Calculate average cyclotron frequency from trajectory

        Returns
        -------
        float
            Average cyclotron frequency in Hz
        """
        # Get positions along trajectory
        pos = self.__trajectory.position

        # Sample field magnitudes along the trajectory
        # Use a subset of points to avoid excessive computation
        n_samples = min(1000, len(pos))
        indices = np.linspace(0, len(pos) - 1, n_samples, dtype=int)

        B_samples = np.array([
            self.__trajectory.field.evaluate_field_magnitude(
                pos[i, 0], pos[i, 1], pos[i, 2]
            ) for i in indices
        ])

        # Average cyclotron frequency
        gamma = self.__trajectory.particle.GetGamma()
        mass = self.__trajectory.particle.GetMass()
        avg_B = np.mean(B_samples, dtype=float)
        omega_c_avg = sc.e * avg_B / (gamma * mass)
        f_c_avg = omega_c_avg / (2 * np.pi)

        return f_c_avg

    def _calculate_advanced_time(self) -> tuple[NDArray, CubicSpline]:
        """
        Calculate advanced time when signal reaches antenna

        For each point in the trajectory at time t, calculates the time
        t_adv = t + R(t)/c when the signal reaches the antenna, where
        R(t) is the distance from electron to antenna.

        Returns
        -------
        tuple[NDArray, CubicSpline]
            t_advanced : Advanced time array
            spline : Cubic spline for interpolating trajectory time from 
            advanced time
        """
        # Get electron positions
        r_electron = self.__trajectory.position

        # Get antenna position
        r_antenna = self.__antenna.GetPosition()

        # Calculate distance from electron to antenna at each time
        r_vec = r_antenna - r_electron  # Vector from electron to antenna
        R = np.linalg.norm(r_vec, axis=1)  # Distance

        # Calculate advanced time: t_adv = t + R/c
        t = self.__trajectory.time
        t_advanced = t + R / sc.c

        # Create spline to interpolate trajectory time from advanced time
        # This will allow us to find retarded times later
        spline = CubicSpline(t_advanced, t)

        return t_advanced, spline

    def _calculate_retarded_quantities(self, t_obs: NDArray,
                                       t_spline: CubicSpline) -> dict:
        """
        Calculate electron position, velocity, and acceleration at retarded times

        For each observation time t_obs, finds the retarded time t_ret such that
        the signal emitted at t_ret reaches the antenna at t_obs.

        Parameters
        ----------
        t_obs : NDArray
            Observation times at antenna
        t_spline : CubicSpline
            Spline for interpolating retarded time from observation time

        Returns
        -------
        dict
            Dictionary containing:
            - 'r_ret': Position at retarded time, shape (N, 3)
            - 'v_ret': Velocity at retarded time, shape (N, 3)
            - 'a_ret': Acceleration at retarded time, shape (N, 3)
            - 't_ret': Retarded times, shape (N,)
            - 'R_ret': Distance from electron to antenna, shape (N,)
            - 'n_hat_ret': Unit vector from electron to antenna, shape (N, 3)
        """
        # Find retarded times by inverting the advanced time relationship
        # t_obs corresponds to t_advanced, so we use the spline to get t_ret
        t_ret = t_spline(t_obs)

        # Create splines for trajectory quantities
        pos_spline = CubicSpline(
            self.__trajectory.time, self.__trajectory.position)
        vel_spline = CubicSpline(
            self.__trajectory.time, self.__trajectory.velocity)
        acc_spline = CubicSpline(
            self.__trajectory.time, self.__trajectory.acceleration)

        # Interpolate trajectory quantities at retarded times
        r_ret = pos_spline(t_ret)
        v_ret = vel_spline(t_ret)
        a_ret = acc_spline(t_ret)

        # Calculate distance and direction from electron to antenna
        r_antenna = self.__antenna.GetPosition()
        r_vec = r_antenna - r_ret  # Vector from electron to antenna
        R_ret = np.linalg.norm(r_vec, axis=1)  # Distance
        n_hat = r_vec / R_ret[:, np.newaxis]  # Unit vector toward antenna

        return {
            'v_ret': v_ret,
            'a_ret': a_ret,
            't_ret': t_ret,
            'R_ret': R_ret,
            'n_hat_ret': n_hat
        }

    def _calculate_E_field(self, ret_quantities: dict) -> NDArray:
        """
        Calculate electric field at antenna using Liénard-Wiechert formula.

        Parameters
        ----------
        ret_quantities : dict
            Dictionary with retarded time quantities from _calculate_retarded_quantities

        Returns
        -------
        NDArray
            Electric field vector at antenna position, shape (N, 3)
        """
        t_ret = ret_quantities['t_ret']
        n_hat = ret_quantities['n_hat_ret']
        v = ret_quantities['v_ret']
        a = ret_quantities['a_ret']
        R = ret_quantities['R_ret']

        beta = v / sc.c
        beta_dot = a / sc.c
        n_dot_beta = np.sum(n_hat * beta, axis=1)  # (N,)

        # Prefactor
        q = self.__trajectory.particle.GetCharge()
        prefactor = q / (4 * np.pi * sc.epsilon_0) / \
            (R**2 * (1 - n_dot_beta)**3)

        # Velocity field term: (n - beta)(1 - beta^2)
        beta_sq = np.sum(beta * beta, axis=1, keepdims=True)  # (N, 1)
        v_term = (n_hat - beta) * (1 - beta_sq)

        # Acceleration field term: R/c * [n·beta_dot (n - beta) - n·(n - beta) beta_dot]
        n_dot_beta_dot = np.sum(n_hat * beta_dot, axis=1,
                                keepdims=True)       # (N, 1)
        n_dot_n_minus_beta = np.sum(
            n_hat * (n_hat - beta), axis=1, keepdims=True)  # (N, 1)
        a_term = R[:, np.newaxis] / sc.c * (
            n_dot_beta_dot * (n_hat - beta) - n_dot_n_minus_beta * beta_dot)

        # Total electric field
        E_field = prefactor[:, np.newaxis] * (v_term + a_term)
        # Zero field before signal has propagated to antenna
        E_field[t_ret < 0] = 0.0
        return E_field

    def _calculate_antenna_voltage(self, E_field: NDArray, ret_quantities: dict) -> NDArray:
        """
        Calculate voltage at antenna terminals from electric field

        Uses V = E · l_eff where l_eff is the antenna effective length.

        Parameters
        ----------
        E_field : NDArray
            Electric field at antenna, shape (N, 3)
        ret_quantities : dict
            Dictionary with retarded time quantities

        Returns
        -------
        NDArray
            Real-valued voltage signal, shape (N,)
        """
        # Synthetic positions at unit distance from antenna toward each source.
        # n_hat_ret points source->antenna, so -n_hat_ret is antenna->source.
        # GetETheta normalises internally, so only direction matters.
        pos = self.__antenna.GetPosition() - ret_quantities['n_hat_ret']

        l_eff = self.__antenna.GetEffectiveLength(
            self.__avg_cyclotron_frequency, pos)

        return np.sum(E_field * l_eff, axis=1)

    def generate_signal(self, return_time: bool = True) -> tuple[NDArray, NDArray] | NDArray:
        """
        Generate complete signal including E-field calculation and digitization

        This method performs the complete signal generation pipeline:
        1. Calculate advanced times and create interpolation spline
        2. Generate observation time grid at trajectory sampling
        3. Calculate retarded time quantities via interpolation
        4. Calculate Liénard-Wiechert electric field at antenna
        5. Calculate antenna voltage from E-field
        6. Downmix and digitize signal using receiver chain

        Parameters
        ----------
        return_time : bool, optional
            If True, return (time, signal) tuple (default True)
            If False, return only signal array

        Returns
        -------
        tuple[NDArray, NDArray] or NDArray
            If return_time is True:
                time : Time array at ADC sample rate, shape (M,)
                signal : Complex digitized IF signal, shape (M,)
            If return_time is False:
                signal : Complex digitized IF signal, shape (M,)

        Notes
        -----
        The signal is generated at the ADC sample rate times the oversampling
        factor, then filtered and decimated by the receiver chain.
        """
        # Step 1: Calculate advanced time and create spline
        t_advanced, spline = self._calculate_advanced_time()

        # Step 2: Create observation time grid at oversampled rate
        # Use the range of advanced times to define observation window
        t_obs_start = t_advanced[0]
        t_obs_end = t_advanced[-1]

        # Sample rate for E-field calculation
        adc_rate = self.__receiver_chain.get_sample_rate()
        signal_sample_rate = adc_rate * self.__oversampling_factor

        # Number of points at oversampled rate
        duration = t_obs_end - t_obs_start
        n_points = int(np.ceil(duration * signal_sample_rate))

        # Create time array
        t_obs = np.linspace(t_obs_start, t_obs_end, n_points)

        ret_quantities = self._calculate_retarded_quantities(t_obs, spline)
        E_field = self._calculate_E_field(ret_quantities)
        voltage = self._calculate_antenna_voltage(E_field, ret_quantities)

        # Downmix and digitize
        t_digitized, signal_digitized = self.__receiver_chain.digitize(
            t_obs, voltage, self.__oversampling_factor)

        if return_time:
            return t_digitized, signal_digitized
        else:
            return signal_digitized

    def get_trajectory(self) -> Trajectory:
        """
        Get the trajectory object

        Returns
        -------
        Trajectory
            Trajectory object
        """
        return self.__trajectory

    def get_antenna(self) -> BaseAntenna:
        """
        Get the antenna object

        Returns
        -------
        BaseAntenna
            Antenna object
        """
        return self.__antenna

    def get_receiver_chain(self) -> ReceiverChain:
        """
        Get the receiver chain object

        Returns
        -------
        ReceiverChain
            Receiver chain object
        """
        return self.__receiver_chain

    def get_average_cyclotron_frequency(self) -> float:
        """
        Get the calculated average cyclotron frequency

        Returns
        -------
        float
            Average cyclotron frequency in Hz
        """
        return self.__avg_cyclotron_frequency
