"""
Particle.py

File containing particle class for easy calculation of particle kinematics 
"""

import numpy as np
import scipy.constants as sc


class Particle:
    """
    Class representing a particle with mass, charge, initial kinetic energy and
    pitch angle
    """

    def __init__(self, ke: float, startPos: np.ndarray, pitchAngle: float = np.pi/2,
                 q: float = -sc.e, mass: float = sc.m_e) -> None:
        """
        Constructor for Particle class

        Parameters
        ----------
        ke : float 
            Initial kinetic energy in eV
        startPos : np.ndarray 
            3-vector representing the initial position of the particle
        pitchAngle : float 
            Pitch angle of the particle in radians
        q : float 
            Charge of the particle in C
        mass : float 
            Mass of the particle in kg
        """
        if not isinstance(ke, (int, float)):
            raise TypeError("Kinetic energy must be a number")
        if ke <= 0:
            raise ValueError("Kinetic energy must be positive")
        if not np.isfinite(ke):
            raise ValueError("Kinetic energy must be finite")

        if not np.issubdtype(startPos.dtype, np.number):
            raise TypeError("Position must be numeric")
        if not np.all(np.isfinite(startPos)):
            raise ValueError("Position must be finite")

        if not isinstance(q, (int, float)):
            raise TypeError("Charge must be a number")
        if not np.isfinite(q):
            raise ValueError("Charge must be finite")

        if not isinstance(mass, (int, float)):
            raise TypeError("Mass must be a number")
        if mass <= 0:
            raise ValueError("Mass must be positive")
        if not np.isfinite(mass):
            raise ValueError("Mass must be finite")

        if not isinstance(pitchAngle, (int, float)):
            raise TypeError("Pitch angle must be a number")
        if pitchAngle <= 0 or pitchAngle >= np.pi:
            raise ValueError("Pitch angle must be in range (0, pi)")
        if not np.isfinite(pitchAngle):
            raise ValueError("Pitch angle must be finite")

        self.__ke = ke
        self.__pos = startPos
        self.__q = q
        self.__mass = mass
        self.__pitchAngle = pitchAngle

    def GetGamma(self):
        """
        Get the Lorentz factor of the particle
        """
        return 1.0 + self.__ke * sc.e / (self.__mass * sc.c**2)

    def GetBeta(self):
        """
        Get the beta factor of the particle
        """
        return np.sqrt(1 - 1 / self.GetGamma()**2)

    def GetSpeed(self):
        """
        Get the speed of the particle
        """
        return sc.c * self.GetBeta()

    def GetMomentum(self):
        """
        Get the momentum of the particle
        """
        return self.GetGamma() * self.__mass * self.GetSpeed()

    def GetPitchAngle(self):
        """
        Get the pitch angle of the particle in radians
        """
        return self.__pitchAngle

    def GetPosition(self):
        """
        Get the position of the particle
        """
        return self.__pos

    def GetMass(self):
        """
        Get the mass of the particle in kg
        """
        return self.__mass

    def GetEnergy(self):
        """
        Get the energy of the particle in eV
        """
        return self.__ke
