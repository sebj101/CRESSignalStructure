"""
CRESWriter.py

Handles HDF5 persistence for CRES simulations.
Matches legacy format: Group 'Data' containing individual datasets 'signalX'
with metadata stored as attributes with specific unit-labeled keys.
"""

import h5py
import numpy as np
import scipy.constants as sc
from typing import Dict, Any, Union

# Internal Imports
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.BaseTrap import BaseTrap
from CRESSignalStructure.BaseField import BaseField
from CRESSignalStructure.RealFields import HarmonicField, BathtubField, CoilField

class CRESWriter:
    def __init__(self, filename: str, mode: str = 'w'):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.current_index = 0
        
        # Global objects to query for metadata
        self._trap = None
        self._waveguide = None
        self._config = None

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode)
        if "Data" not in self.file:
            self.file.create_group("Data")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def set_global_config(self, trap: Union[BaseTrap, BaseField], 
                          waveguide: CircularWaveguide, 
                          sim_config: Dict[str, Any]):
        """
        Stores physics objects to calculate attributes for every event.
        """
        self._trap = trap
        self._waveguide = waveguide
        self._config = sim_config

    def _get_trap_params(self):
        """Helper to extract parameters trap parameters."""
        params = {'i_coil [Amps]': np.nan, 'r_coil [metres]': np.nan}
        
        if isinstance(self._trap, HarmonicField):
            params['i_coil [Amps]'] = self._trap.coil.current
            params['r_coil [metres]'] = self._trap.coil.radius

            
        return params

    def write_event(self, particle: Particle, 
                    time_array: np.ndarray, 
                    signal_array: np.ndarray):
        """
        Writes a single event with the exact legacy attribute structure.
        """
        if self.file is None:
            raise RuntimeError("CRESWriter must be used within a 'with' statement.")

        # --- SAFETY CHECK ---
        if len(time_array) != len(signal_array):
            raise ValueError(
                f"Shape Mismatch: Time array ({len(time_array)}) "
                f"does not match Signal array ({len(signal_array)})"
            )
        # ----------------------------------------------

        sig_name = f"signal{self.current_index + 1}"
        grp = self.file['Data']
        
        # 1. Write Data (Complex64)
        if np.iscomplexobj(signal_array):
            data_to_write = signal_array.astype('complex64')
        else:
            # Fallback for FFT power (real float)
            data_to_write = signal_array.astype('float32')
            
        dset = grp.create_dataset(sig_name, data=data_to_write)

        # 2. Calculate Derived Physics Values
        
        # --- Kinematics ---
        gamma = particle.GetGamma()
        mass = particle.GetMass()
        speed = particle.GetSpeed()
        pos = particle.GetPosition()
        pitch = particle.GetPitchAngle()
        
        v_z = speed * np.cos(pitch)
        v_perp = speed * np.sin(pitch)
        velocity_vector = np.array([v_perp, 0.0, v_z])

        # --- Fields & Frequencies ---
        # B_local at the starting position
        if hasattr(self._trap, 'evaluate_field_magnitude'):
            B_local = self._trap.evaluate_field_magnitude(pos[0], pos[1], pos[2])
        elif hasattr(self._trap, 'GetB0'):
            B_local = self._trap.GetB0() 
        else:
            B_local = 1.0 # Default fallback

        f_cyc = (sc.e * B_local) / (2 * np.pi * gamma * mass)
        
        f_lo = self._config.get('lo_freq', 0.0)
        sample_rate = self._config.get('sample_rate', 1e9)
        
        # --- Waveguide ---
        omega_cyc = 2 * np.pi * f_cyc
        try:
            impedance = self._waveguide.CalcTE11Impedance(omega_cyc)
        except (ValueError, AttributeError):
            impedance = np.nan

        # 3. Set Attributes (Exact Legacy Keys)
        attrs = dset.attrs
        
        # Trap / Background
        if hasattr(self._trap, 'background') and isinstance(self._trap.background, np.ndarray):
            attrs['B_bkg [Tesla]'] = abs(self._trap.background[2])
        elif hasattr(self._trap, 'GetB0'):
            attrs['B_bkg [Tesla]'] = self._trap.GetB0()
        else:
            attrs['B_bkg [Tesla]'] = B_local

        attrs['Cyclotron frequency [Hertz]'] = f_cyc
        attrs['Downmixed cyclotron frequency [Hertz]'] = abs(f_cyc - f_lo)
        attrs['Energy [eV]'] = particle.GetEnergy()
        attrs['LO frequency [Hertz]'] = f_lo
        attrs['Pitch angle [degrees]'] = np.degrees(pitch)
        attrs['Starting position [metres]'] = pos
        attrs['Starting velocity [metres/second]'] = velocity_vector
        attrs['Time step [seconds]'] = 1.0 / sample_rate
        attrs['Waveguide impedance [Ohms]'] = impedance
        
        # Coil Params
        trap_params = self._get_trap_params()
        attrs['i_coil [Amps]'] = trap_params['i_coil [Amps]']
        attrs['r_coil [metres]'] = trap_params['r_coil [metres]']
        
        # WG Params
        # Handle cases where Waveguide might be a mock or different object
        if hasattr(self._waveguide, 'wgR'):
            attrs['r_wg [metres]'] = self._waveguide.wgR
        else:
            attrs['r_wg [metres]'] = 0.005 # Default 5mm

        self.current_index += 1