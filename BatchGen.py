"""
BatchGen.py

A command-line tool to generate large CRES datasets split across multiple files.
Usage example:
    python BatchGen.py --files 1 --events 5 --energy-min 18580 --energy-max 18600 --mp --no-fft --pitch-min 89.5 --pitch-max 90.0
"""
import os
import json
import argparse
import time
import csv
import numpy as np
import scipy.constants as sc

# Imports
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.NumericalSpectrumCalculator import NumericalSpectrumCalculator
from CRESSignalStructure.EnsembleGenerator import generate_uniform_ensemble, generate_ensemble

CONFIG_KEYS = [
    'energy_min', 'energy_max', 'pitch_min', 'pitch_max',
    'bank_grid', 'bank_r2_steps', 'bank_costheta_steps', 'bank_energy_steps', 'bank_seed',
    'files', 'events', 'prefix', 'no_fft',
    'phase_seed', 'particle_seed',
]

def _save_run_config(args, outdir):
    config = {k: getattr(args, k) for k in CONFIG_KEYS}
    with open(os.path.join(outdir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

def _check_resume(args, outdir):
    """Returns True if resuming (config matches), False if fresh start, exits on mismatch."""
    config_path = os.path.join(outdir, 'run_config.json')
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        saved = json.load(f)
    current = {k: getattr(args, k) for k in CONFIG_KEYS}
    mismatches = {k: (saved[k], current[k]) for k in CONFIG_KEYS if saved.get(k) != current[k]}
    if mismatches:
        print("ERROR: Run config mismatch — cannot resume. Differing parameters:")
        for k, (old, new) in mismatches.items():
            print(f"  {k}: saved={old!r}  current={new!r}")
        print("Use a different --outdir or delete run_config.json to start fresh.")
        raise SystemExit(1)
    return True


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
    parser.add_argument("--max-workers", type=int, default=None, help="Cap number of CPU cores used (default: all available minus one)")
    parser.add_argument("--no-fft", action="store_true", help="Disable FFT generation (Signal only)")

    # Template Bank (Grid) Options
    parser.add_argument("--bank-grid", action="store_true", help="Generate a grid (template bank) in r^2 and cos(theta)")
    parser.add_argument("--bank-r2-steps", type=int, default=200, help="Number of steps in r^2 for grid bank")
    parser.add_argument("--bank-costheta-steps", type=int, default=100, help="Number of steps in cos(theta) for grid bank (uniform in cos θ ≈ uniform in θ near 90°)")
    parser.add_argument("--bank-energy-steps", type=int, default=1, help="Number of steps in energy for 3D grid bank (default=1 for 2D)")
    parser.add_argument("--bank-seed", type=int, default=None, help="RNG seed for bank generation")

    # Reproducibility seeds
    parser.add_argument("--phase-seed", type=int, default=42,
                        help="RNG seed for per-event initial phase draws (phi_c, phi_a). "
                             "Default 42; reproducible per (phase_seed, event_index).")
    parser.add_argument("--particle-seed", type=int, default=None,
                        help="RNG seed for per-event particle parameter draws (energy, pitch, "
                             "position). Default None falls back to numpy's global RNG "
                             "(non-reproducible). Recommend setting an explicit value per run.")

    args = parser.parse_args()

    # 2. Setup Directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created output directory: {args.outdir}")

    resuming = _check_resume(args, args.outdir)
    if resuming:
        print("Resuming previous run (config matches).")
    else:
        _save_run_config(args, args.outdir)

    # 3. Define Physics
    print("Initialising Physics Model...")
    
    TRAP_DEPTH =  4e-3 # 32mT  #4e-3  # 4 mT
    B0 = 1.0           # 1 T
    R_COIL = 1.25e-2   # 1.25 cm
    I_COIL = 2 * TRAP_DEPTH * R_COIL / sc.mu_0
    
    trap = HarmonicField(radius=R_COIL, current=I_COIL, background=B0)
    radius = 5e-3 # 5mm radius
    wg = CircularWaveguide(radius=radius) # 5mm radius

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
    print(f"Batch Generation Parameters:")
    print(f"  - Energy Range: {args.energy_min} eV to {args.energy_max} eV")
    print(f"  - Pitch Range: {args.pitch_min} deg to {args.pitch_max} deg")
    print(f"  - Radius: {radius*1e3:.2f} mm")
    if args.bank_grid:
        print(f"  - Events per File: (bank grid; split across files)")
    else:
        print(f"  - Events per File: {args.events}")
    print(f"  - Total Files: {args.files}")
    print(f"  - Multiprocessing: {'Enabled' if args.mp else 'Disabled'}")
    print(f"  - FFT Generation: {'Disabled' if args.no_fft else 'Enabled'}")
    print(f"  - Phase seed:    {args.phase_seed}")
    print(f"  - Particle seed: {args.particle_seed if args.particle_seed is not None else 'None (non-reproducible)'}")
    print("=" * 40)

    # 5. Configuration Dictionary
    sim_config = {
        'sample_rate': F_DIGITIZER,
        'lo_freq': LO_FREQ,
        'acq_time': 40e-6,  # 40 microseconds
        'max_order': 8,
        'phase_seed': args.phase_seed,
    }

    # 6. Parameter Ranges
    param_ranges = {
        'energy': (args.energy_min, args.energy_max), 
        'pitch': (np.radians(args.pitch_min), np.radians(args.pitch_max)),
        'r': (0.0, radius),   # Up to waveguide radius
        'z': (0.0, 0.0),    # Center of trap
        'theta': (0.0, 2*np.pi)
    }

    # 6a. Build Template Bank (Grid) if requested
    bank_particles = None
    bank_indices_split = None
    if args.bank_grid:
        rng = np.random.default_rng(args.bank_seed)

        r_min, r_max = param_ranges['r']
        pitch_min, pitch_max = param_ranges['pitch']
        e_min, e_max = param_ranges['energy']
        
        # cos is decreasing on [0, π/2], so cos(pitch_max) < cos(pitch_min)
        # Near 90°: cos(θ) ≈ -(θ - π/2), so uniform spacing in cos θ ≈ uniform spacing in θ
        cos_pitch_min = np.cos(pitch_max)  # smaller value (pitch_max closer to 90°)
        cos_pitch_max = np.cos(pitch_min)  # larger value  (pitch_min further from 90°)

        # Create 1D arrays for each dimension
        r2_vals = np.linspace(r_min**2, r_max**2, args.bank_r2_steps)
        cos_pitch_vals = np.linspace(cos_pitch_min, cos_pitch_max, args.bank_costheta_steps)
        energy_vals = np.linspace(e_min, e_max, args.bank_energy_steps)

        # Create 3D meshgrid
        r2_grid, cos_pitch_grid, energy_grid = np.meshgrid(r2_vals, cos_pitch_vals, energy_vals, indexing='xy')
        r_flat = np.sqrt(r2_grid.ravel())
        pitch_flat = np.arccos(cos_pitch_grid.ravel())
        energy_flat = energy_grid.ravel()

        # Azimuthal angle for position (random to fill disk)
        azimuth_flat = rng.uniform(0.0, 2*np.pi, size=r_flat.shape)

        # z position fixed at trap center
        z_flat = np.zeros_like(r_flat)

        bank_particles = []
        for r, phi, z, pitch, ke in zip(r_flat, azimuth_flat, z_flat, pitch_flat, energy_flat):
            pos = np.array([r * np.cos(phi), r * np.sin(phi), z])
            bank_particles.append(Particle(ke=ke, startPos=pos, pitchAngle=pitch))

        total_bank_events = len(bank_particles)
        bank_indices_split = np.array_split(np.arange(total_bank_events), args.files)

        print("Template Bank Configuration:")
        print(f"  - Grid r^2 steps: {args.bank_r2_steps}")
        print(f"  - Grid cos(theta) steps: {args.bank_costheta_steps}")
        print(f"  - Grid energy steps: {args.bank_energy_steps}")
        print(f"  - Total Bank Events: {total_bank_events}")
        print("  - Azimuth: random uniform [0, 2π)")
        print("-" * 40)

        # Save bank parameters to CSV for reproducibility
        csv_path = os.path.join(args.outdir, "bank_parameters.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Value', 'Units'])
            writer.writerow(['bank_r2_steps', args.bank_r2_steps, ''])
            writer.writerow(['bank_costheta_steps', args.bank_costheta_steps, ''])
            writer.writerow(['bank_energy_steps', args.bank_energy_steps, ''])
            writer.writerow(['bank_seed', args.bank_seed if args.bank_seed is not None else 'None', ''])
            writer.writerow(['total_bank_events', total_bank_events, ''])
            writer.writerow(['num_files', args.files, ''])
            writer.writerow(['energy_min', args.energy_min, 'eV'])
            writer.writerow(['energy_max', args.energy_max, 'eV'])
            writer.writerow(['pitch_min', args.pitch_min, 'degrees'])
            writer.writerow(['pitch_max', args.pitch_max, 'degrees'])
            writer.writerow(['r_min', param_ranges['r'][0], 'm'])
            writer.writerow(['r_max', param_ranges['r'][1], 'm'])
            writer.writerow(['z_position', param_ranges['z'][0], 'm'])
            writer.writerow(['trap_coil_radius', R_COIL, 'm'])
            writer.writerow(['trap_coil_current', I_COIL, 'A'])
            writer.writerow(['trap_B0', B0, 'T'])
            writer.writerow(['trap_depth', TRAP_DEPTH, 'T'])
            writer.writerow(['waveguide_radius', radius, 'm'])
            writer.writerow(['sample_rate', F_DIGITIZER, 'Hz'])
            writer.writerow(['lo_frequency', LO_FREQ, 'Hz'])
            writer.writerow(['acquisition_time', sim_config['acq_time'], 's'])
            writer.writerow(['max_order', sim_config['max_order'], ''])
            writer.writerow(['central_frequency', f_carrier, 'Hz'])
        print(f"Saved bank parameters to: {csv_path}")

    # 7. Execution Loop
    total_start = time.time()
    
    for i in range(args.files):
        if args.bank_grid and bank_indices_split is not None:
            batch_count = len(bank_indices_split[i])
        else:
            batch_count = args.events

        # Define Filenames
        file_base = f"{args.prefix}_{i:03d}"
        sig_path = os.path.join(args.outdir, f"{file_base}_signal.h5")
        fft_path = None
        if not args.no_fft:
            fft_path = os.path.join(args.outdir, f"{file_base}_fft.h5")

        # Resume: skip already-completed files
        sig_done = os.path.exists(sig_path)
        fft_done = fft_path is None or os.path.exists(fft_path)
        if sig_done and fft_done:
            print(f"\n[Batch {i+1}/{args.files}] Skipping — already complete.")
            continue

        print(f"\n[Batch {i+1}/{args.files}] Generating {batch_count} events...")

        # Per-file sim_config: inject batch_idx so the worker RNG produces a
        # disjoint event sequence per file (same phase_seed across the run,
        # different batch_idx per file -> independent streams).
        file_sim_config = {**sim_config, 'batch_idx': i}

        # Run Generator
        if args.bank_grid:
            indices = bank_indices_split[i]
            if len(indices) == 0:
                print("  - Skipping empty batch (no indices assigned)")
                continue

            def grid_particle_generator(j):
                return bank_particles[indices[j]]

            generate_ensemble(
                output_file=sig_path,
                n_events=len(indices),
                particle_generator=grid_particle_generator,
                trap=trap,
                waveguide=wg,
                sim_config=file_sim_config,
                fft_output_file=fft_path,
                use_multiprocessing=args.mp,
                max_workers=args.max_workers,
                verbose=True
            )
        else:
            generate_uniform_ensemble(
                output_file=sig_path,
                n_events=args.events,
                trap=trap,
                waveguide=wg,
                sim_config=file_sim_config,
                ranges=param_ranges,
                fft_output_file=fft_path,
                use_multiprocessing=args.mp,
                max_workers=args.max_workers,
                verbose=True,
                particle_seed=args.particle_seed,
                batch_idx=i,
            )

    total_time = time.time() - total_start
    if args.bank_grid and bank_particles is not None:
        total_events = len(bank_particles)
    else:
        total_events = args.files * args.events
    print("=" * 40)
    print(f"BATCH COMPLETE.")
    print(f"Total Events: {total_events}")
    print(f"Total Time:   {total_time:.1f}s")
    print(f"Average Rate: {total_events/total_time:.1f} events/s")
    print(f"Output Dir:   {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()