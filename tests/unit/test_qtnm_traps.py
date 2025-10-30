"""
Unit tests for QTNM analytical traps
"""

import pytest
from CRESSignalStructure.QTNMTraps import HarmonicTrap, BathtubTrap


class TestHarmonicTrap:
    """Tests for HarmonicTrap"""

    def test_valid_trap_creation(self):
        """Test creating a valid harmonic trap"""
        B0 = 1.0
        L0 = 0.2
        gradB = 4e-3
        trap = HarmonicTrap(B0=B0, L0=L0, gradB=gradB)

        assert trap.GetB0() == B0
        assert trap.GetL0() == L0
        assert trap.GetGradB() == gradB

    def test_negative_B0_raises_error(self):
        """Tests that negative values of B0 raise a ValueError"""
        with pytest.raises(ValueError, match="B0 must be positive"):
            HarmonicTrap(B0=-1.0, L0=0.2)

    def test_negative_L0_raises_error(self):
        """Tests that negative values of L0 raise a ValueError"""
        with pytest.raises(ValueError, match="L0 must be positive"):
            HarmonicTrap(B0=1.0, L0=-0.2)
