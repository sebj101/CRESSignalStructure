from CRESSignalStructure.QTNMTraps import HarmonicTrap
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.Particle import Particle
import scipy.constants as sc
import numpy as np


def test_harmonic_trap_produces_correct_motion():
    """
    Test HarmonicTrap produces physically correct motion
    """
    B0 = 1.0
    L0 = 0.2
    trap = HarmonicTrap(B0, L0)

    KE = 18.6e3  # eV
    particle = Particle(KE, mass=sc.m_e, startPos=np.zeros(
        3), pitchAngle=89.5 * np.pi / 180, q=-sc.e)

    z_max = trap.CalcZMax(particle.GetPitchAngle())
    f_axial = trap.CalcOmegaAxial(
        particle.GetSpeed(), particle.GetPitchAngle()) / (2 * np.pi)
    f_cyc = trap.CalcOmega0(particle.GetSpeed(),
                            particle.GetPitchAngle()) / (2 * np.pi)

    assert z_max > 0, "Maximum axial displacement should be positive"
    assert f_axial > 0, "Axial frequency should be positive"
    assert f_cyc > 0, "Cyclotron frequency should positive"


def test_harmonic_field_produces_correct_motion():
    """
    Test HarmonicField produces correct motion
    """
    R_COIL = 3e-2  # metres
    TRAP_DEPTH = 4e-3  # Tesla
    I_COIL = 2 * TRAP_DEPTH * R_COIL / sc.mu_0  # Amps
    BKG_FIELD = 1.0  # Tesla

    trap = HarmonicField(R_COIL, I_COIL, BKG_FIELD)

    KE = 18.6e3  # eV
    particle = Particle(KE, mass=sc.m_e, startPos=np.zeros(
        3), pitchAngle=89.5 * np.pi / 180, q=-sc.e)

    z_max = trap.CalcZMax(particle)
    f_axial = trap.CalcOmegaAxial(particle) / (2*np.pi)
    f_cyc = trap.CalcOmega0(particle) / (2*np.pi)

    assert z_max > 0, "Maximum axial displacement should be positive"
    assert f_axial > 0, "Axial frequency should be positive"
    assert f_cyc > 0, "Cyclotron frequency should positive"
