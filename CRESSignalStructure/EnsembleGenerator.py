"""
EnsembleGenerator.py

Orchestrates the batch generation of CRES simulation data.
Supports Multiprocessing and simultaneous Time-Series / FFT output.
"""
import numpy as np
import multiprocessing as mp
import scipy.fft
from contextlib import ExitStack
import time

# Internal Imports
from .Particle import Particle
from .SignalGenerator import SignalGenerator
from .SpectrumCalculator import SpectrumCalculator
from .CRESWriter import CRESWriter
from .scattering import ScatteringSimulator

def _worker_generate_event(args):
    """
    Worker function. Generates a single event and computes FFT.
    """
    index, particle, trap, waveguide, config = args

    try:
        # 1. Setup Generators
        spec_calc = SpectrumCalculator(trap, waveguide, particle)
        
        sig_gen = SignalGenerator(
            spectrum_calc=spec_calc, 
            sample_rate=config['sample_rate'],
            lo_freq=config['lo_freq'],
            acq_time=config['acq_time']
        )

        # 2. Generate Pure Signal (Time Domain), returns (times, signal)
        t, sig_pure = sig_gen.generate_signal(max_order=config['max_order'])

        F_DIGITIZER = config.get('sample_rate', 1e9)
        
        # FFT Calculation: fft(signal, norm='forward')
        fft_complex = scipy.fft.fft(sig_pure, norm='forward')
        
        # Frequency Axis
        dt = 1.0 / F_DIGITIZER
        N = len(sig_pure)
        freqs = scipy.fft.fftfreq(N, dt)
        
        # Power Calculation: |fft|^2
        fft_power = np.abs(fft_complex)**2
        # ---------------------------------------------

        return (index, particle, t, sig_pure, freqs, fft_power)

    except Exception as e:
        return (index, e)

def generate_ensemble(output_file, 
                      n_events, 
                      particle_generator, 
                      trap, 
                      waveguide, 
                      sim_config,
                      fft_output_file=None, 
                      use_multiprocessing=True,
                      verbose=True):
    """
    Main driver for ensemble generation.
    """
    
    # 1. Pre-calculate arguments for workers
    worker_args = []
    for i in range(n_events):
        p = particle_generator(i)
        worker_args.append((i, p, trap, waveguide, sim_config))

    # 2. Open Files
    with ExitStack() as stack:
        writer = stack.enter_context(CRESWriter(output_file))
        writer.set_global_config(trap, waveguide, sim_config)
        
        fft_writer = None
        if fft_output_file:
            fft_writer = stack.enter_context(CRESWriter(fft_output_file))
            fft_writer.set_global_config(trap, waveguide, sim_config)

        start_time = time.time()
        
        # --- Multiprocessing (Fast) ---
        if use_multiprocessing:
            num_cores = mp.cpu_count()
            if verbose: print(f"Starting generation of {n_events} events on {num_cores} cores...")
            
            # Create pool
            with mp.Pool(processes=num_cores) as pool:
                # chunksize improves efficiency for large batches (3000 events)
                results_iterator = pool.imap_unordered(_worker_generate_event, worker_args, chunksize=10)
                
                _process_results(results_iterator, n_events, writer, fft_writer, start_time, verbose)

        # --- Sequential (Debugging) ---
        else:
            if verbose: print(f"Starting generation of {n_events} events in SINGLE process mode...")
            results_iterator = map(_worker_generate_event, worker_args)
            _process_results(results_iterator, n_events, writer, fft_writer, start_time, verbose)

        print(f"Done! Saved to {output_file}")


def _process_results(results_iterator, n_events, writer, fft_writer, start_time, verbose):
    """Helper to write results to disk (runs in main thread)."""
    for i, result in enumerate(results_iterator):
        
        # Error Handling
        if len(result) == 2 and isinstance(result[1], Exception):
            print(f"WARNING: Event {result[0]} failed: {result[1]}")
            continue

        # Unpack
        idx, p, t, sig, freqs, fft_power = result

        # Write Time Series
        writer.write_event(p, t, sig)
        
        # Write FFT (if enabled)
        if fft_writer:
            fft_writer.write_event(p, freqs, fft_power)

        # Progress Bar
        if verbose and (i+1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed
            print(f"  Processed {i+1}/{n_events} events ({rate:.1f} ev/s)...")


# --- Helper Wrapper for Uniform Sampling ---
def generate_uniform_ensemble(output_file, 
                              n_events, 
                              trap, 
                              waveguide, 
                              sim_config, 
                              ranges,
                              fft_output_file=None, 
                              use_multiprocessing=True, 
                              verbose=True):
    
    def uniform_particle_generator(i):
        e_min, e_max = ranges.get('energy', (18500.0, 18600.0))
        p_min, p_max = ranges.get('pitch', (np.radians(88), np.radians(89.99)))
        z_min, z_max = ranges.get('z', (0.0, 0.0))

        r_min, r_max = ranges.get('r', (0.0, 0.001))
        theta_min, theta_max = ranges.get('theta', (0.0, 2*np.pi))

        ke = np.random.uniform(e_min, e_max)
        pitch = np.random.uniform(p_min, p_max)
        z = np.random.uniform(z_min, z_max)

        # Sample radius with uniform area density in the annulus [r_min, r_max]
        u_rho = np.random.uniform(0, 1)
        r = np.sqrt((r_max**2 - r_min**2) * u_rho + r_min**2)

        theta = np.random.uniform(theta_min, theta_max)

        pos = np.array([r * np.cos(theta), r * np.sin(theta), z])
        
        return Particle(ke=ke, startPos=pos, pitchAngle=pitch)

    generate_ensemble(
        output_file=output_file,
        n_events=n_events,
        particle_generator=uniform_particle_generator,
        trap=trap,
        waveguide=waveguide,
        sim_config=sim_config,
        fft_output_file=fft_output_file,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose
    )


def _worker_generate_scattering_event(args):
    """
    Worker function for scattering events. Generates a single event with
    gas scattering and computes FFT.
    """
    index, particle, trap, waveguide, gas_model, config, seed = args

    try:
        rng = np.random.default_rng(seed)
        sim = ScatteringSimulator(
            trap=trap,
            waveguide=waveguide,
            gas_model=gas_model,
            sample_rate=config['sample_rate'],
            lo_freq=config['lo_freq'],
            max_event_time=config['max_event_time']
        )
        result = sim.simulate(
            particle=particle,
            max_order=config['max_order'],
            rng=rng
        )

        fft_complex = scipy.fft.fft(result.signal, norm='forward')
        dt = 1.0 / config['sample_rate']
        freqs = scipy.fft.fftfreq(len(result.signal), dt)
        fft_power = np.abs(fft_complex)**2

        return (index, result, freqs, fft_power)

    except Exception as e:
        return (index, e)


def _process_scattering_results(results_iterator, n_events, writer,
                                fft_writer, start_time, verbose):
    """Helper to write scattering results to disk (runs in main thread)."""
    for i, result in enumerate(results_iterator):

        if len(result) == 2 and isinstance(result[1], Exception):
            print(f"WARNING: Event {result[0]} failed: {result[1]}")
            continue

        idx, scat_result, freqs, fft_power = result

        writer.write_scattering_event(scat_result)

        if fft_writer:
            fft_writer.write_event(
                scat_result.particles[0], freqs, fft_power)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i+1}/{n_events} events ({rate:.1f} ev/s)...")


def generate_scattering_ensemble(output_file,
                                 n_events,
                                 particle_generator,
                                 trap,
                                 waveguide,
                                 gas_model,
                                 sim_config,
                                 fft_output_file=None,
                                 use_multiprocessing=True,
                                 verbose=True):
    """
    Generate an ensemble of scattering events.

    Parameters
    ----------
    output_file : str
        Path to HDF5 output file for time-domain signals
    n_events : int
        Number of events to generate
    particle_generator : callable
        Function(index) -> Particle for initial particle state
    trap : BaseTrap | BaseField
        The magnetic trap
    waveguide : CircularWaveguide
        The waveguide
    gas_model : GasModel
        Gas model for scattering
    sim_config : dict
        Must contain: 'sample_rate', 'lo_freq', 'max_event_time', 'max_order'
    fft_output_file : str, optional
        Path to HDF5 output for FFT data
    use_multiprocessing : bool
        Whether to use multiprocessing (default True)
    verbose : bool
        Whether to print progress (default True)
    """
    ss = np.random.SeedSequence()
    child_seeds = ss.spawn(n_events)

    worker_args = []
    for i in range(n_events):
        p = particle_generator(i)
        worker_args.append(
            (i, p, trap, waveguide, gas_model, sim_config, child_seeds[i]))

    with ExitStack() as stack:
        writer = stack.enter_context(CRESWriter(output_file))
        writer.set_global_config(trap, waveguide, sim_config)

        fft_writer = None
        if fft_output_file:
            fft_writer = stack.enter_context(CRESWriter(fft_output_file))
            fft_writer.set_global_config(trap, waveguide, sim_config)

        start_time = time.time()

        if use_multiprocessing:
            num_cores = mp.cpu_count()
            if verbose:
                print(f"Starting generation of {n_events} scattering events "
                      f"on {num_cores} cores...")

            with mp.Pool(processes=num_cores) as pool:
                results_iterator = pool.imap_unordered(
                    _worker_generate_scattering_event,
                    worker_args, chunksize=10)
                _process_scattering_results(
                    results_iterator, n_events, writer, fft_writer,
                    start_time, verbose)
        else:
            if verbose:
                print(f"Starting generation of {n_events} scattering events "
                      f"in SINGLE process mode...")
            results_iterator = map(
                _worker_generate_scattering_event, worker_args)
            _process_scattering_results(
                results_iterator, n_events, writer, fft_writer,
                start_time, verbose)

        print(f"Done! Saved to {output_file}")