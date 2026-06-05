"""
ScatteringSimulator.py

Simulates CRES signals with gas scattering events. Signal segments are
generated at an oversampled rate with continuous phase across scatter
boundaries, then the composite signal is filtered and decimated once.
"""

import dataclasses
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
import scipy.constants as sc

from ..Particle import Particle
from ..SpectrumCalculator import SpectrumCalculator
from ..BaseTrap import BaseTrap
from ..BaseField import BaseField
from ..CircularWaveguide import CircularWaveguide
from .GasModel import GasModel


@dataclasses.dataclass
class ScatteringResult:
    """
    Container for the output of a scattering simulation.

    Attributes
    ----------
    times : NDArray
        1D array of time values in seconds at the final sample rate
    signal : NDArray
        1D complex array of the downmixed, filtered, decimated signal (volts)
    scatter_times : list[float]
        Absolute times (seconds from event start) at which scatters occurred
    particles : list[Particle]
        Particle state for each signal segment. Length is n_scatters + 1
        (or n_scatters if the particle escaped on the last scatter).
    escaped : bool
        True if the particle left the trap after the final scatter
    """
    times: NDArray
    signal: NDArray
    scatter_times: list[float]
    particles: list[Particle]
    escaped: bool


class ScatteringSimulator:

    FAST_SAMPLE_FACTOR = 5
    IMPEDANCE = 50.0  # Ohms

    def __init__(self, trap: BaseTrap | BaseField,
                 waveguide: CircularWaveguide,
                 gas_model: GasModel,
                 sample_rate: float,
                 lo_freq: float,
                 max_event_time: float):
        """
        Parameters
        ----------
        trap : BaseTrap | BaseField
            The magnetic trap
        waveguide : CircularWaveguide
            The waveguide geometry
        gas_model : GasModel
            Gas composition and cross-section models
        sample_rate : float
            Digitizer sample rate in Hz
        lo_freq : float
            Local oscillator frequency in Hz
        max_event_time : float
            Maximum total event duration in seconds
        """
        if not isinstance(trap, (BaseTrap, BaseField)):
            raise TypeError("trap must be a BaseTrap or BaseField instance")
        if not isinstance(gas_model, GasModel):
            raise TypeError("gas_model must be a GasModel instance")

        for name, val in [("sample_rate", sample_rate),
                          ("lo_freq", lo_freq),
                          ("max_event_time", max_event_time)]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number")
            if val <= 0 or not np.isfinite(val):
                raise ValueError(f"{name} must be positive and finite")

        self.__trap = trap
        self.__waveguide = waveguide
        self.__gas_model = gas_model
        self.__sample_rate = float(sample_rate)
        self.__lo_freq = float(lo_freq)
        self.__max_event_time = float(max_event_time)

    def is_trapped(self, pitch_angle: float, particle: Particle) -> bool:
        """
        Check whether a particle with the given pitch angle is trapped.

        For BaseTrap, checks that calc_z_max returns a finite value.
        For BaseField, checks that calc_z_max succeeds (no ValueError from
        the root finder).

        Parameters
        ----------
        pitch_angle : float
            Pitch angle in radians
        particle : Particle
            Used for position information in the BaseField case

        Returns
        -------
        bool
        """
        if pitch_angle <= 0 or pitch_angle >= np.pi:
            return False

        if isinstance(self.__trap, BaseTrap):
            return bool(np.isfinite(self.__trap.calc_z_max(pitch_angle)))

        test_particle = Particle(
            particle.get_energy(), particle.get_position(), pitch_angle,
            particle.get_charge(), particle.get_mass())
        try:
            self.__trap.calc_z_max(test_particle)
            return True
        except ValueError:
            return False

    def simulate(self, particle: Particle, max_order: int,
                 rng: np.random.Generator = None,
                 phi_c: float = 0.0,
                 phi_a: float = 0.0) -> ScatteringResult:
        """
        Run a full scattering simulation.

        Parameters
        ----------
        particle : Particle
            Initial particle state
        max_order : int
            Maximum sideband order for spectrum calculation
        rng : np.random.Generator, optional
            Random number generator (default: unseeded default_rng)
        phi_c : float
            Initial cyclotron phase in radians
        phi_a : float
            Initial axial phase in radians

        Returns
        -------
        ScatteringResult
        """
        if rng is None:
            rng = np.random.default_rng()

        fast_rate = self.FAST_SAMPLE_FACTOR * self.__sample_rate
        orders = np.arange(-max_order, max_order + 1, 1)

        segments = []
        scatter_times = []
        particles = [particle]
        elapsed = 0.0
        current_phi_c = phi_c
        current_phi_a = phi_a
        phi_chirp = 0.0
        escaped = False

        while elapsed < self.__max_event_time:
            current_particle = particles[-1]
            spec_calc = SpectrumCalculator(
                self.__trap, self.__waveguide, current_particle)

            dt_scatter = self.__gas_model.sample_time_to_scatter(
                current_particle.get_energy(), current_particle.get_speed(),
                rng)
            segment_duration = min(dt_scatter,
                                   self.__max_event_time - elapsed)

            n_samples = int(segment_duration * fast_rate)
            if n_samples < 1:
                if dt_scatter < self.__max_event_time - elapsed:
                    new_e, new_pa = self.__gas_model.sample_scatter(
                        current_particle.get_energy(),
                        current_particle.get_pitch_angle(),
                        current_particle.get_speed(), rng)
                    if new_pa > np.pi / 2:
                        new_pa = np.pi - new_pa
                    scatter_times.append(elapsed)
                    if new_e <= 0 or not self.is_trapped(new_pa,
                                                        current_particle):
                        escaped = True
                        break
                    particles.append(Particle(
                        new_e, current_particle.get_position(), new_pa,
                        current_particle.get_charge(),
                        current_particle.get_mass()))
                    continue
                break

            segment = self._generate_segment(
                spec_calc, orders, n_samples, fast_rate, elapsed,
                current_phi_c, current_phi_a, phi_chirp)
            segments.append(segment)

            # Compute phase advances for continuity
            actual_dt = n_samples / fast_rate
            f0, fa = self._get_frequencies(current_particle)

            beta = current_particle.get_beta()
            gamma = current_particle.get_gamma()
            chirp_rate = (sc.e**2 * (2 * np.pi * f0)**3 * gamma * beta**2
                          / (6 * np.pi * sc.epsilon_0 * sc.c)
                          / (sc.m_e * sc.c**2))

            current_phi_c += 2 * np.pi * f0 * actual_dt
            current_phi_a += 2 * np.pi * fa * actual_dt
            phi_chirp += chirp_rate * actual_dt**2
            elapsed += actual_dt

            if dt_scatter >= self.__max_event_time - (elapsed - actual_dt):
                break

            scatter_times.append(elapsed)
            new_e, new_pa = self.__gas_model.sample_scatter(
                current_particle.get_energy(),
                current_particle.get_pitch_angle(),
                current_particle.get_speed(), rng)
            if new_pa > np.pi / 2:
                new_pa = np.pi - new_pa

            if new_e <= 0 or not self.is_trapped(new_pa, current_particle):
                escaped = True
                break

            particles.append(Particle(
                new_e, current_particle.get_position(), new_pa,
                current_particle.get_charge(),
                current_particle.get_mass()))

        # Concatenate, filter, and decimate
        if not segments:
            n_out = max(1, int(elapsed * self.__sample_rate))
            return ScatteringResult(
                times=np.arange(n_out) / self.__sample_rate,
                signal=np.zeros(n_out, dtype=complex),
                scatter_times=scatter_times,
                particles=particles,
                escaped=escaped)

        full_signal = np.concatenate(segments)

        # Truncate to a multiple of FAST_SAMPLE_FACTOR for clean decimation
        n_truncated = (len(full_signal) // self.FAST_SAMPLE_FACTOR) \
            * self.FAST_SAMPLE_FACTOR
        full_signal = full_signal[:n_truncated]

        if n_truncated == 0:
            return ScatteringResult(
                times=np.array([0.0]),
                signal=np.array([0.0 + 0.0j]),
                scatter_times=scatter_times,
                particles=particles,
                escaped=escaped)

        FILTER_ORDER = 8
        MIN_FILTER_LEN = 3 * (FILTER_ORDER + 1)

        if len(full_signal) > MIN_FILTER_LEN:
            sos = butter(N=FILTER_ORDER, Wn=self.__sample_rate / 2,
                         btype='low', output='sos', fs=fast_rate)
            filtered = sosfiltfilt(sos, full_signal) * np.sqrt(self.IMPEDANCE)
        else:
            filtered = full_signal * np.sqrt(self.IMPEDANCE)

        decimated = filtered[::self.FAST_SAMPLE_FACTOR]
        times_out = np.arange(len(decimated)) / self.__sample_rate

        return ScatteringResult(
            times=times_out,
            signal=decimated,
            scatter_times=scatter_times,
            particles=particles,
            escaped=escaped)

    def _get_frequencies(self, particle: Particle) -> tuple[float, float]:
        """Get cyclotron and axial frequencies in Hz for a particle."""
        if isinstance(self.__trap, BaseTrap):
            v = particle.get_speed()
            pa = particle.get_pitch_angle()
            f0 = self.__trap.calc_omega_0(v, pa) / (2 * np.pi)
            fa = self.__trap.calc_omega_axial(v, pa) / (2 * np.pi)
        else:
            f0 = self.__trap.calc_omega_0(particle) / (2 * np.pi)
            fa = self.__trap.calc_omega_axial(particle) / (2 * np.pi)
        return f0, fa

    def _generate_segment(self, spec_calc: SpectrumCalculator,
                          orders: NDArray, n_samples: int,
                          fast_rate: float, elapsed: float,
                          phi_c: float, phi_a: float,
                          phi_chirp: float) -> NDArray:
        """
        Generate an oversampled RF signal segment for a single particle state.
        """
        t_local = np.arange(n_samples) / fast_rate
        t_abs = elapsed + t_local

        # Fourier components (positive and negative frequencies)
        freqs_pos = spec_calc.get_peak_frequency(orders, False)
        freqs_neg = spec_calc.get_peak_frequency(orders, True)
        fourier_freqs = np.append(freqs_pos, freqs_neg)

        amps_pos = spec_calc.apply_phase_shifts(
            spec_calc.get_peak_amp(orders, False),
            orders, phi_c, phi_a, False)
        amps_neg = spec_calc.apply_phase_shifts(
            spec_calc.get_peak_amp(orders, True),
            orders, phi_c, phi_a, True)
        fourier_amps = np.append(amps_pos, amps_neg)

        # Chirp rate for this particle
        particle = spec_calc.get_particle()
        beta = particle.get_beta()
        gamma = particle.get_gamma()
        central_freq = spec_calc.get_peak_frequency(0)
        chirp_rate = (sc.e**2 * (2 * np.pi * central_freq)**3 * gamma
                      * beta**2 / (6 * np.pi * sc.epsilon_0 * sc.c)
                      / (sc.m_e * sc.c**2))

        # Build RF signal with phase continuity
        rf = np.sum(
            fourier_amps[:, np.newaxis]
            * np.exp(2j * np.pi * fourier_freqs[:, np.newaxis] * t_local),
            axis=0)

        # LO uses absolute time for continuity across segments
        lo = np.exp(-2j * np.pi * self.__lo_freq * t_abs)

        # Chirp: accumulated phase + local quadratic term
        chirp = np.exp(1j * (phi_chirp + chirp_rate * t_local**2))

        # Scale by sqrt(power_norm) per segment (IMPEDANCE applied after filter)
        power_norm = spec_calc.get_power_norm()

        return rf * lo * chirp * np.sqrt(power_norm)
