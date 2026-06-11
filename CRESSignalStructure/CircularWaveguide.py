from scipy.special import j1, jvp
import numpy as np
import scipy.constants as sc
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import dblquad


class CircularWaveguide:
    _TE11_KC_COEFF = 1.841  # First zero of J1'

    def __init__(self, radius):
        if not isinstance(radius, (int, float)):
            raise TypeError("Radius must be a number")
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if not np.isfinite(radius):
            raise ValueError("Radius must be finite")

        self.wgR = radius
        self._kc = self._TE11_KC_COEFF / self.wgR

    def __str__(self):
        return f"Waveguide with radius {self.wgR} metres"

    def _validate_frequency(self, omega: float) -> None:
        """
        Validate inputted frequency parameter
        """
        if not isinstance(omega, (int, float)):
            raise TypeError("Frequency must be a number")
        if omega <= 0:
            raise ValueError("Frequency must be positive")
        if not np.isfinite(omega):
            raise ValueError("Frequency must be finite")

        # Check that the frequency is above the cutoff frequency
        k = omega / sc.c
        if k <= self._kc:
            raise ValueError(
                f"Frequency {omega} rad/s is below cutoff frequency {self._kc * sc.c} rad/s")

    def _validate_position(self, rho, phi) -> tuple[NDArray, NDArray]:
        """
        Validate position parameters
        """
        rho = np.asarray(rho)
        phi = np.asarray(phi)

        if not np.issubdtype(rho.dtype, np.number):
            raise TypeError("Radial position must be numeric")
        if not np.issubdtype(phi.dtype, np.number):
            raise TypeError("Azimuthal angle must be numeric")

        if np.any(rho < 0):
            raise ValueError("Radial position must be non-negative")

        if not np.all(np.isfinite(rho)):
            raise ValueError("Radial position must be finite")
        if not np.all(np.isfinite(phi)):
            raise ValueError("Azimuthal angle must be finite")

        return rho, phi

    def _validate_amplitude(self, A) -> NDArray:
        """
        Validate amplitude parameter
        """
        A = np.asarray(A)
        if not np.issubdtype(A.dtype, np.number):
            raise TypeError("Amplitude must be numeric")
        if not np.all(np.isfinite(A)):
            raise ValueError("Amplitude must be finite")
        return A

    def _safe_j1_over_rho(self, kcRho) -> float:
        """
        Safely compute j1(x)/x to avoid division by zero
        """
        if np.abs(kcRho) < 1e-10:
            return 0.5
        else:
            return j1(kcRho) / kcRho

    def e_field_te11_rho_1(self, rho: ArrayLike, phi: ArrayLike, A: ArrayLike) -> NDArray:
        """
        Calculate the radial electric field for the TE11 mode

        Parameters
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        """
        rho, phi = self._validate_position(rho, phi)
        A = self._validate_amplitude(A)

        conditions = [rho > self.wgR, rho == 0.0, rho <= self.wgR]
        choices = [0.0, A * np.cos(phi) / self._kc,
                   A * self._safe_j1_over_rho(self._kc * rho) * np.cos(phi)]
        return np.select(conditions, choices)

    
    def e_field_te11_phi_1(self, rho: ArrayLike, phi: ArrayLike, A: ArrayLike) -> NDArray:
        """Calculate the radial electric field for the TE11 mode

        Parameters
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        """
        rho, phi = self._validate_position(rho, phi)
        A = self._validate_amplitude(A)

        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, -A * jvp(1, self._kc * rho, 1) * np.sin(phi)]
        return np.select(conditions, choices)

    def e_field_te11_z(self, rho, phi, A) -> NDArray:
        """
        Calculate the axial electric field for the TE11 mode

        Parameters
        ----------
        rho: float representing the radial position
        phi: float representing the azimuthal position
        A: float representing the amplitude of the mode
        """
        return np.zeros_like(rho)

    def e_field_te11_1(self, rho, phi, A) -> NDArray:
        """
        Calculate the electric field vector for mode 1 in Cartesian coordinates

        Parameters
        ----------
        rho: float
            Radial position in metres
        phi: float
            Azimuthal position in radians
        A: float
            Amplitude of the mode
        """

        return np.array([self.e_field_te11_rho_1(rho, phi, A) * np.cos(phi) - self.e_field_te11_phi_1(rho, phi, A) * np.sin(phi),
                         self.e_field_te11_rho_1(
                             rho, phi, A) * np.sin(phi) + self.e_field_te11_phi_1(rho, phi, A) * np.cos(phi),
                         self.e_field_te11_z(rho, phi, A)])

    def e_field_te11_pos_1(self, pos, A) -> NDArray:
        """
        Calculate the electric field vector for mode 1 in Cartesian coordinates

        Parameters
        ----------
        pos: np.ndarray
            Position three vector in metres
        A: float
            Amplitude of the mode
        """

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.e_field_te11_1(rho, phi, A)

    def e_field_te11_rho_2(self, rho: ArrayLike, phi: ArrayLike, A: ArrayLike) -> NDArray:
        """
        Calculate the radial electric field for the TE11 mode

        Parameters
        ----------
        rho: float representing the radial position in the waveguide
        phi: float representing the azimuthal angle in the waveguide
        A: float representing the amplitude of the mode
        """
        rho, phi = self._validate_position(rho, phi)
        A = self._validate_amplitude(A)

        conditions = [rho > self.wgR, rho == 0.0, rho <= self.wgR]
        choices = [
            0.0,
            -A * np.sin(phi) / self._kc,
            -A * self._safe_j1_over_rho(self._kc * rho) * np.sin(phi)
        ]
        return np.select(conditions, choices)

    def e_field_te11_phi_2(self, rho: ArrayLike, phi: ArrayLike, A: ArrayLike) -> NDArray:
        """
        Calculate the radial electric field for the TE11 mode

        Parameters
        ----------
        rho : float
            Radial position in the waveguide in metres
        phi : float
            Azimuthal angle in the waveguide in radians
        A : float
            Amplitude of the mode
        """
        rho, phi = self._validate_position(rho, phi)
        A = self._validate_amplitude(A)

        conditions = [rho > self.wgR, rho <= self.wgR]
        choices = [0.0, -A * jvp(1, self._kc * rho, 1) * np.cos(phi)]
        return np.select(conditions, choices)

    def e_field_te11_2(self, rho, phi, A) -> NDArray:
        """Calculate the electric field vector for mode 2 in Cartesian coordinates

        rho: float representing the radial position
        phi: float representing the azimuthal position
        """

        return np.array([self.e_field_te11_rho_2(rho, phi, A) * np.cos(phi) - self.e_field_te11_phi_2(rho, phi, A) * np.sin(phi),
                        self.e_field_te11_rho_2(
                            rho, phi, A) * np.sin(phi) + self.e_field_te11_phi_2(rho, phi, A) * np.cos(phi),
                        self.e_field_te11_z(rho, phi, A)])

    def e_field_te11_pos_2(self, pos, A) -> NDArray:
        """Calculate the electric field vector for mode 2 in Cartesian coordinates

        pos: numpy array representing the position
        A: float representing normlisation constant
        """

        rho = np.sqrt(pos[0]**2 + pos[1]**2)
        phi = np.arctan2(pos[1], pos[0])
        return self.e_field_te11_2(rho, phi, A)

    def calc_te11_impedance(self, omega) -> float:
        """
        Calculate the impedance of the TE11 mode

        Parameters
        ----------
        omega : float
            Angular frequency of the mode in rad/s

        Returns
        -------
        float
            Impedance of the mode in Ohms
        """
        self._validate_frequency(omega)

        k = omega / sc.c
        betaMode = np.sqrt(k**2 - self._kc**2)
        return k * np.sqrt(sc.mu_0 / sc.epsilon_0) / betaMode

    def calc_normalisation_factor(self) -> float:
        """
        Calculate the required normalisation factor for the waveguide.

        Returns
        -------
        float
            Required normalisation factor
        """

        def integrand_rho_phi(phi, rho):
            E_rho = self.e_field_te11_rho_1(rho, phi, 1.0)
            E_phi = self.e_field_te11_phi_1(rho, phi, 1.0)
            return (E_rho**2 + E_phi**2) * rho

        result = dblquad(
            integrand_rho_phi,
            0.0,              # rho lower limit
            self.wgR,         # rho upper limit
            0.0,              # phi lower limit
            2 * np.pi,        # phi upper limit
            epsabs=1e-10,
            epsrel=1e-8
        )

        return 1 / np.sqrt(result[0])

    def get_phase_velocity(self, omega: float) -> float:
        """
        Get the phase velocity for an EM wave in the waveguide.

        Parameters
        ----------
        omega : float
            Angular frequency of the wave in rad/s

        Returns
        -------
        float
            Phase velocity in m/s
        """
        self._validate_frequency(omega)
        omega_c = self._kc * sc.c
        return sc.c / np.sqrt(1 - (omega_c / omega)**2)

    def get_group_velocity(self, omega: float) -> float:
        """
        Get the group velocity for an EM wave in the waveguide.

        Parameters
        ----------
        omega : float
            Angular frequency of the wave in rad/s

        Returns
        -------
        float
            Group velocity in m/s
        """
        self._validate_frequency(omega)
        omega_c = self._kc * sc.c
        return sc.c * np.sqrt(1 - (omega_c / omega)**2)
