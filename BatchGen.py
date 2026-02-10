"""
BatchGen.py

A command-line tool to generate large CRES datasets split across multiple files.
Usage example:
    python BatchGen.py --files 1 --events 5 --energy-min 18580 --energy-max 18600 --mp --no-fft --pitch-min 89.5 --pitch-max 90.0
"""
import os
import argparse
import time
import numpy as np
import scipy.constants as sc

# Imports
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.NumericalSpectrumCalculator import NumericalSpectrumCalculator
from CRESSignalStructure.EnsembleGenerator import generate_uniform_ensemble

def main():
    # 1. Setup Arguments
    parser = argparse.ArgumentParser(description="Generate CRES Batches")
    
    # Batch Control
    parser.add_argument("--files", type=int, default=1, help="Number of files to generate")
    parser.add_argument("--events", type=int, default=100, help="Number of events per file")
    parser.add_argument("--outdir", type=str, default="data/batch_run", help="Output directory")
    parser.add_argument("--prefix", type=str, default="run", help="Filename prefix")
    
    # Physics Ranges
    parser.add_argument("--energy-min", type=float, default=18600.0, help="Min Electron Energy (eV)")
    parser.add_argument("--energy-max", type=float, default=18600.0, help="Max Electron Energy (eV)")
    parser.add_argument("--pitch-min", type=float, default=89.5, help="Min Pitch (deg)")
    parser.add_argument("--pitch-max", type=float, default=90.0, help="Max Pitch (deg)")
    
    # Performance
    parser.add_argument("--mp", action="store_true", help="Enable Multiprocessing")
    parser.add_argument("--no-fft", action="store_true", help="Disable FFT generation (Signal only)")

    args = parser.parse_args()

    # 2. Setup Directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created output directory: {args.outdir}")

    # 3. Define Physics
    print("Initialising Physics Model...")
    
    TRAP_DEPTH =  0.032 # 32mT  #4e-3  # 4 mT
    B0 = 1.0           # 1 T
    R_COIL = 1.25e-2   # 1.25 cm
    I_COIL = 2 * TRAP_DEPTH * R_COIL / sc.mu_0
    
    trap = HarmonicField(radius=R_COIL, current=I_COIL, background=B0)
    wg = CircularWaveguide(radius=5e-3) # 5mm radius

    # 4. Calculate LO Frequency
    F_DIGITIZER = 1e9 
    
    mean_pitch_rad = np.radians((args.pitch_min + args.pitch_max) / 2)
    mean_energy = (args.energy_min + args.energy_max) / 2 
    
    ref_p = Particle(ke=mean_energy, startPos=np.zeros(3), pitchAngle=mean_pitch_rad)
    calc = NumericalSpectrumCalculator(trap, wg, ref_p)
    f_carrier = calc.GetPeakFrequency(0)

    LO_FREQ = f_carrier - (F_DIGITIZER / 4) # Locked in value
    
    print(f"  - Trap Current: {I_COIL:.4f} A")
    print(f"  - Central Freq: {f_carrier/1e9:.4f} GHz")
    print(f"  - LO Frequency: {LO_FREQ/1e9:.4f} GHz")
    print("-" * 40)

    # 5. Configuration Dictionary
    sim_config = {
        'sample_rate': F_DIGITIZER,
        'lo_freq': LO_FREQ,
        'acq_time': 40e-6,  # 40 microseconds
        'max_order': 8      
    }

    # 6. Parameter Ranges
    param_ranges = {
        'energy': (args.energy_min, args.energy_max), 
        'pitch': (np.radians(args.pitch_min), np.radians(args.pitch_max)),
        'r': (0.0, 5e-3),   # Up to 5 mm radial offset
        'z': (0.0, 0.0),    # Center of trap
        'theta': (0.0, 2*np.pi)
    }

    # 7. Execution Loop
    total_start = time.time()
    
    for i in range(args.files):
        print(f"\n[Batch {i+1}/{args.files}] Generating {args.events} events...")
        
        # Define Filenames
        file_base = f"{args.prefix}_{i:03d}"
        sig_path = os.path.join(args.outdir, f"{file_base}_signal.h5")
        fft_path = None
        if not args.no_fft:
            fft_path = os.path.join(args.outdir, f"{file_base}_fft.h5")

        # Run Generator
        generate_uniform_ensemble(
            output_file=sig_path,
            n_events=args.events,
            trap=trap,
            waveguide=wg,
            sim_config=sim_config,
            ranges=param_ranges,
            fft_output_file=fft_path,
            use_multiprocessing=args.mp,
            verbose=True
        )

    total_time = time.time() - total_start
    total_events = args.files * args.events
    print("=" * 40)
    print(f"BATCH COMPLETE.")
    print(f"Total Events: {total_events}")
    print(f"Total Time:   {total_time:.1f}s")
    print(f"Average Rate: {total_events/total_time:.1f} events/s")
    print(f"Output Dir:   {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()