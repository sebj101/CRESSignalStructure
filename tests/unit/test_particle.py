"""
Unit tests for Particle class
"""

import numpy as np
import pytest
import scipy.constants as sc
from CRESSignalStructure.Particle import Particle


class TestParticleConstruction:
    """Tests for Particle constructor and validation"""

    def test_valid_particle_creation(self):
        """Test creating a valid particle with default electron parameters"""
        ke = 18600.0  # eV
        pos = np.zeros(3)
        particle = Particle(ke=ke, startPos=pos)

        assert particle.GetEnergy() == ke
        assert np.array_equal(particle.GetPosition(), pos)
        assert particle.GetMass() == sc.m_e
        assert particle.GetPitchAngle() == np.pi / 2

    def test_custom_particle_parameters(self):
        """Test creating a particle with custom mass, charge, and pitch angle"""
        ke = 10000.0
        pos = np.array([1.0, 2.0, 3.0])
        pitch = np.pi / 4
        mass = 2 * sc.m_e
        charge = -2 * sc.e

        particle = Particle(ke=ke, startPos=pos, pitchAngle=pitch, q=charge, mass=mass)

        assert particle.GetEnergy() == ke
        assert np.array_equal(particle.GetPosition(), pos)
        assert particle.GetMass() == mass
        assert particle.GetPitchAngle() == pitch

    # Kinetic energy validation tests
    def test_negative_kinetic_energy_raises_error(self):
        """Test that negative kinetic energy raises ValueError"""
        with pytest.raises(ValueError, match="Kinetic energy must be positive"):
            Particle(ke=-100.0, startPos=np.array([0.0, 0.0, 0.0]))

    def test_zero_kinetic_energy_raises_error(self):
        """Test that zero kinetic energy raises ValueError"""
        with pytest.raises(ValueError, match="Kinetic energy must be positive"):
            Particle(ke=0.0, startPos=np.array([0.0, 0.0, 0.0]))

    def test_non_numeric_kinetic_energy_raises_error(self):
        """Test that non-numeric kinetic energy raises TypeError"""
        with pytest.raises(TypeError, match="Kinetic energy must be a number"):
            Particle(ke="not_a_number", startPos=np.array([0.0, 0.0, 0.0]))

    def test_infinite_kinetic_energy_raises_error(self):
        """Test that infinite kinetic energy raises ValueError"""
        with pytest.raises(ValueError, match="Kinetic energy must be finite"):
            Particle(ke=np.inf, startPos=np.array([0.0, 0.0, 0.0]))

    def test_nan_kinetic_energy_raises_error(self):
        """Test that NaN kinetic energy raises ValueError"""
        with pytest.raises(ValueError, match="Kinetic energy must be finite"):
            Particle(ke=np.nan, startPos=np.array([0.0, 0.0, 0.0]))

    # Position validation tests
    def test_non_numeric_position_raises_error(self):
        """Test that non-numeric position raises TypeError"""
        with pytest.raises(TypeError, match="Position must be numeric"):
            Particle(ke=1000.0, startPos=np.array(["a", "b", "c"]))

    def test_infinite_position_raises_error(self):
        """Test that infinite position values raise ValueError"""
        with pytest.raises(ValueError, match="Position must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, np.inf, 0.0]))

    def test_nan_position_raises_error(self):
        """Test that NaN position values raise ValueError"""
        with pytest.raises(ValueError, match="Position must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, np.nan, 0.0]))

    # Charge validation tests
    def test_non_numeric_charge_raises_error(self):
        """Test that non-numeric charge raises TypeError"""
        with pytest.raises(TypeError, match="Charge must be a number"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), q="invalid")

    def test_infinite_charge_raises_error(self):
        """Test that infinite charge raises ValueError"""
        with pytest.raises(ValueError, match="Charge must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), q=np.inf)

    def test_nan_charge_raises_error(self):
        """Test that NaN charge raises ValueError"""
        with pytest.raises(ValueError, match="Charge must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), q=np.nan)

    # Mass validation tests
    def test_negative_mass_raises_error(self):
        """Test that negative mass raises ValueError"""
        with pytest.raises(ValueError, match="Mass must be positive"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), mass=-1e-30)

    def test_zero_mass_raises_error(self):
        """Test that zero mass raises ValueError"""
        with pytest.raises(ValueError, match="Mass must be positive"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), mass=0.0)

    def test_non_numeric_mass_raises_error(self):
        """Test that non-numeric mass raises TypeError"""
        with pytest.raises(TypeError, match="Mass must be a number"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), mass="invalid")

    def test_infinite_mass_raises_error(self):
        """Test that infinite mass raises ValueError"""
        with pytest.raises(ValueError, match="Mass must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), mass=np.inf)

    def test_nan_mass_raises_error(self):
        """Test that NaN mass raises ValueError"""
        with pytest.raises(ValueError, match="Mass must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), mass=np.nan)

    # Pitch angle validation tests
    def test_negative_pitch_angle_raises_error(self):
        """Test that negative pitch angle raises ValueError"""
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=-0.1)

    def test_zero_pitch_angle_raises_error(self):
        """Test that zero pitch angle raises ValueError"""
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=0.0)

    def test_pi_pitch_angle_raises_error(self):
        """Test that pi pitch angle raises ValueError"""
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=np.pi)

    def test_pitch_angle_greater_than_pi_raises_error(self):
        """Test that pitch angle > pi raises ValueError"""
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=np.pi + 0.1)

    def test_non_numeric_pitch_angle_raises_error(self):
        """Test that non-numeric pitch angle raises TypeError"""
        with pytest.raises(TypeError, match="Pitch angle must be a number"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle="invalid")

    def test_infinite_pitch_angle_raises_error(self):
        """Test that infinite pitch angle raises ValueError (caught by range check)"""
        with pytest.raises(ValueError, match="Pitch angle must be in range"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=np.inf)

    def test_nan_pitch_angle_raises_error(self):
        """Test that NaN pitch angle raises ValueError"""
        with pytest.raises(ValueError, match="Pitch angle must be finite"):
            Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=np.nan)


class TestParticleRelativisticCalculations:
    """Tests for relativistic physics calculations"""

    def test_nonrelativistic_limit(self):
        """Test that at low energies, gamma approaches 1"""
        ke = 100.0  # 100 eV, very non-relativistic for electrons
        particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))

        gamma = particle.GetGamma()
        # For 100 eV electron, gamma should be very close to 1
        # ke = (gamma - 1) * mc^2, so gamma = 1 + ke/mc^2
        expected_gamma = 1 + (ke * sc.e) / (sc.m_e * sc.c**2)

        assert abs(gamma - 1.0) < 0.001  # Very close to 1
        assert abs(gamma - expected_gamma) < 1e-10  # Matches expected calculation

    def test_relativistic_regime(self):
        """Test particle in relativistic regime"""
        ke = 511_000.0  # 511 keV = rest mass energy of electron
        particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))

        gamma = particle.GetGamma()
        # At ke = mc^2, gamma should be 2
        expected_gamma = 1 + (ke * sc.e) / (sc.m_e * sc.c**2)

        assert abs(gamma - 2.0) < 0.01  # Should be approximately 2
        assert abs(gamma - expected_gamma) < 1e-10

    def test_gamma_always_greater_than_one(self):
        """Test that gamma is always >= 1 for any positive energy"""
        energies = [1.0, 100.0, 1000.0, 10000.0, 100000.0]

        for ke in energies:
            particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))
            assert particle.GetGamma() >= 1.0

    def test_beta_in_valid_range(self):
        """Test that beta is always in range [0, 1)"""
        energies = [1.0, 100.0, 1000.0, 18600.0, 100000.0, 511000.0]

        for ke in energies:
            particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))
            beta = particle.GetBeta()
            assert 0.0 <= beta < 1.0

    def test_beta_gamma_relation(self):
        """Test the relation: gamma = 1 / sqrt(1 - beta^2)"""
        ke = 18600.0  # Typical CRES electron energy
        particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))

        gamma = particle.GetGamma()
        beta = particle.GetBeta()

        # Check gamma = 1 / sqrt(1 - beta^2)
        expected_gamma = 1.0 / np.sqrt(1.0 - beta**2)
        assert abs(gamma - expected_gamma) < 1e-10

    def test_speed_less_than_c(self):
        """Test that particle speed is always less than speed of light"""
        energies = [100.0, 1000.0, 18600.0, 100000.0, 1e9]

        for ke in energies:
            particle = Particle(ke=ke, startPos=np.zeros(3))
            speed = particle.GetSpeed()
            assert speed < sc.c

    def test_speed_calculation(self):
        """Test that speed = beta * c"""
        ke = 18600.0
        particle = Particle(ke=ke, startPos=np.zeros(3))

        speed = particle.GetSpeed()
        beta = particle.GetBeta()
        expected_speed = beta * sc.c

        assert abs(speed - expected_speed) < 1e-10

    def test_momentum_calculation(self):
        """Test relativistic momentum: p = gamma * m * v"""
        ke = 18600.0
        particle = Particle(ke=ke, startPos=np.zeros(3))

        momentum = particle.GetMomentum()
        gamma = particle.GetGamma()
        mass = particle.GetMass()
        speed = particle.GetSpeed()

        expected_momentum = gamma * mass * speed
        assert abs(momentum - expected_momentum) < 1e-20

    def test_momentum_positive(self):
        """Test that momentum is always positive for positive energy"""
        energies = [100.0, 1000.0, 18600.0, 100000.0]

        for ke in energies:
            particle = Particle(ke=ke, startPos=np.zeros(3))
            assert particle.GetMomentum() > 0

    def test_high_energy_limit(self):
        """Test ultra-relativistic limit: beta approaches 1, speed approaches c"""
        ke = 1e9  # 1 GeV, highly relativistic for electrons
        particle = Particle(ke=ke, startPos=np.zeros(3))

        beta = particle.GetBeta()
        speed = particle.GetSpeed()

        # In ultra-relativistic limit, beta should be very close to 1
        assert beta > 0.999
        assert abs(speed - sc.c) / sc.c < 0.001  # Within 0.1% of c


class TestParticleGetters:
    """Tests for getter methods"""

    def test_get_energy(self):
        """Test GetEnergy returns the kinetic energy"""
        ke = 18600.0
        particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))
        assert particle.GetEnergy() == ke

    def test_get_position(self):
        """Test GetPosition returns the position vector"""
        pos = np.array([1.5, -2.3, 4.7])
        particle = Particle(ke=1000.0, startPos=pos)
        assert np.array_equal(particle.GetPosition(), pos)

    def test_get_mass(self):
        """Test GetMass returns the mass"""
        mass = 2 * sc.m_e
        particle = Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), mass=mass)
        assert particle.GetMass() == mass

    def test_get_pitch_angle(self):
        """Test GetPitchAngle returns the pitch angle"""
        pitch = np.pi / 3
        particle = Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]), pitchAngle=pitch)
        assert particle.GetPitchAngle() == pitch

    def test_default_electron_parameters(self):
        """Test that default parameters match electron constants"""
        particle = Particle(ke=1000.0, startPos=np.array([0.0, 0.0, 0.0]))
        assert particle.GetMass() == sc.m_e
        assert particle.GetPitchAngle() == np.pi / 2


class TestParticlePhysicsConsistency:
    """Tests for overall physics consistency"""

    def test_energy_momentum_relation(self):
        """Test relativistic energy-momentum relation: E^2 = (pc)^2 + (mc^2)^2"""
        ke = 18600.0
        particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))

        # Total energy
        E_total = (particle.GetGamma() * particle.GetMass() * sc.c**2) / sc.e  # in eV
        momentum = particle.GetMomentum()
        mass = particle.GetMass()

        # E^2 = (pc)^2 + (mc^2)^2
        E_squared = E_total**2 * sc.e**2  # Convert to Joules
        pc_squared = (momentum * sc.c)**2
        mc2_squared = (mass * sc.c**2)**2

        assert abs(E_squared - (pc_squared + mc2_squared)) / E_squared < 1e-10

    def test_consistency_across_energy_range(self):
        """Test that all calculations remain consistent across wide energy range"""
        energies = [10.0, 100.0, 1000.0, 18600.0, 100000.0, 511000.0]

        for ke in energies:
            particle = Particle(ke=ke, startPos=np.array([0.0, 0.0, 0.0]))

            # All calculations should complete without error
            gamma = particle.GetGamma()
            beta = particle.GetBeta()
            speed = particle.GetSpeed()
            momentum = particle.GetMomentum()

            # Basic sanity checks
            assert gamma >= 1.0
            assert 0.0 <= beta < 1.0
            assert 0.0 < speed < sc.c
            assert momentum > 0.0

            # Check internal consistency
            assert abs(speed - beta * sc.c) < 1e-10
            assert abs(gamma - 1.0 / np.sqrt(1.0 - beta**2)) < 1e-10