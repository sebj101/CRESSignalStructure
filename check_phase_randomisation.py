"""
check_phase_randomisation.py

Sanity check for an HDF5 batch produced by BatchGen.py. Audits the truth attrs
and produces plots that verify:
  - phi_c and phi_a are uniform on [0, 2*pi)
  - phi_c and phi_a are independent (2D scatter)
  - Per-event spectra show sideband structure at the expected positions
    f_dm_mean + n * f_axial for n in [-max_order, max_order], where f_dm_mean
    is the orbit-averaged downmixed cyclotron frequency (the actual visible
    carrier; f_c(z=0)-LO is biased low for low-pitch events).
  - Other sampling attrs (pitch, energy, radius, f_axial, f_cyc) look sensible

Anchoring on Mean downmixed cyclotron frequency requires regenerating with the
CRESWriter that emits Mean (downmixed) cyclotron frequency attrs. Falls back
to f_c(z=0)-LO with a warning if those attrs are missing.

Usage (run from the repo root):
    python check_phase_randomisation.py \
        --input /path/to/run_000_signal.h5 \
        --output-dir plots/phase_randomisation \
        --max-order 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.stats import kstest


def load_all_attrs(h5_path: Path) -> dict:
    """Load all per-event truth attrs into a dict of numpy arrays."""
    attrs: dict = {}
    with h5py.File(h5_path, "r") as f:
        keys = sorted(
            f["Data"].keys(),
            key=lambda s: int(s.replace("signal", "")),
        )
        for name in f["Data"][keys[0]].attrs.keys():
            attrs[name] = []
        for k in keys:
            for name, val in f["Data"][k].attrs.items():
                attrs[name].append(val)
    for name in list(attrs.keys()):
        attrs[name] = np.array(attrs[name])
    attrs["_event_keys"] = keys
    return attrs


def load_signal(h5_path: Path, event_key: str):
    """Return the complex IQ time series for one event and the sample rate."""
    with h5py.File(h5_path, "r") as f:
        sig = f["Data"][event_key][:]
        dt = float(f["Data"][event_key].attrs["Time step [seconds]"])
    return sig, 1.0 / dt


def starting_radius(positions: np.ndarray) -> np.ndarray:
    """positions: shape (N, 3). Returns sqrt(x^2 + y^2) per event."""
    return np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)


def print_summary(attrs: dict) -> None:
    n_events = len(attrs["_event_keys"])
    print(f"\n=== Sanity check summary ({n_events} events) ===\n")

    phi_c = attrs["Cyclotron phase [rad]"]
    phi_a = attrs["Axial phase [rad]"]
    f_ax = attrs["Axial frequency [Hertz]"]
    f_cyc = attrs["Cyclotron frequency [Hertz]"]
    pitch = attrs["Pitch angle [degrees]"]
    energy = attrs["Energy [eV]"]
    r_start = starting_radius(attrs["Starting position [metres]"])

    def summarise(name, arr, expect=None):
        line = (
            f"  {name:25s}  mean={arr.mean():.4g}  std={arr.std():.4g}  "
            f"min={arr.min():.4g}  max={arr.max():.4g}"
        )
        if expect is not None:
            line += f"   ({expect})"
        print(line)

    summarise("phi_c [rad]", phi_c, "expect uniform on [0, 2pi)")
    summarise("phi_a [rad]", phi_a, "expect uniform on [0, 2pi)")
    summarise("pitch [deg]", pitch)
    summarise("energy [eV]", energy)
    summarise("r_start [m]", r_start)
    summarise("f_axial [Hz]", f_ax)
    summarise("f_cyc [Hz]", f_cyc)
    if "Mean cyclotron frequency [Hertz]" in attrs:
        f_cyc_mean = attrs["Mean cyclotron frequency [Hertz]"]
        summarise("f_cyc_mean [Hz]", f_cyc_mean,
                  "orbit-averaged <f_c>, used as the truth carrier")
    else:
        print("  NOTE: 'Mean cyclotron frequency [Hertz]' not in attrs - "
              "regenerate with the updated CRESWriter to enable.")

    two_pi = 2 * np.pi
    print()
    print(f"  phi_c any nan?         {bool(np.isnan(phi_c).any())}")
    print(f"  phi_c in [0, 2pi)?     {bool((phi_c >= 0).all() and (phi_c < two_pi).all())}")
    print(f"  phi_a any nan?         {bool(np.isnan(phi_a).any())}")
    print(f"  phi_a in [0, 2pi)?     {bool((phi_a >= 0).all() and (phi_a < two_pi).all())}")
    print(f"  f_axial any nan?       {bool(np.isnan(f_ax).any())}")
    print(f"  f_axial any zero?      {bool((f_ax == 0).any())}")


def ks_test_phases(attrs: dict) -> None:
    phi_c = attrs["Cyclotron phase [rad]"]
    phi_a = attrs["Axial phase [rad]"]
    # KS test against Uniform(0, 1) after normalising to [0, 1]
    ks_c = kstest(phi_c / (2 * np.pi), "uniform")
    ks_a = kstest(phi_a / (2 * np.pi), "uniform")
    print("\n=== KS tests (vs uniform on [0, 2pi)) ===")
    print(
        f"  phi_c:  D={ks_c.statistic:.4f}  p={ks_c.pvalue:.4g}  "
        f"{'PASS' if ks_c.pvalue > 0.01 else 'FAIL'}"
    )
    print(
        f"  phi_a:  D={ks_a.statistic:.4f}  p={ks_a.pvalue:.4g}  "
        f"{'PASS' if ks_a.pvalue > 0.01 else 'FAIL'}"
    )


def plot_phase_histograms(attrs: dict, out_dir: Path) -> None:
    phi_c = attrs["Cyclotron phase [rad]"]
    phi_a = attrs["Axial phase [rad]"]
    n = len(phi_c)
    n_bins = 40
    expected = n / n_bins

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, data, name in [
        (axes[0], phi_c, "Cyclotron phase phi_c"),
        (axes[1], phi_a, "Axial phase phi_a"),
    ]:
        ax.hist(data, bins=n_bins, range=(0, 2 * np.pi),
                color="#3a7ca5", edgecolor="white")
        ax.axhline(expected, color="black", linestyle="--",
                   label=f"uniform expected ({expected:.0f})")
        ax.set_xlabel(f"{name} [rad]")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 2 * np.pi)
        ax.legend()

    axes[2].scatter(phi_c, phi_a, s=2, alpha=0.5, color="#3a7ca5")
    axes[2].set_xlabel("phi_c [rad]")
    axes[2].set_ylabel("phi_a [rad]")
    axes[2].set_xlim(0, 2 * np.pi)
    axes[2].set_ylim(0, 2 * np.pi)
    axes[2].set_aspect("equal")
    axes[2].set_title("phi_c vs phi_a (should look uniform)")

    fig.suptitle(f"Phase randomisation audit ({n} events)")
    fig.tight_layout()
    out_path = out_dir / "phase_histograms.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_attr_distributions(attrs: dict, out_dir: Path) -> None:
    pitch = attrs["Pitch angle [degrees]"]
    energy = attrs["Energy [eV]"]
    f_ax_mhz = attrs["Axial frequency [Hertz]"] / 1e6
    f_cyc_ghz = attrs["Cyclotron frequency [Hertz]"] / 1e9
    r_start_mm = starting_radius(attrs["Starting position [metres]"]) * 1e3

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, data, label in [
        (axes[0, 0], pitch, "Pitch angle [deg]"),
        (axes[0, 1], energy, "Energy [eV]"),
        (axes[0, 2], r_start_mm, "Starting radius [mm]"),
        (axes[1, 0], f_ax_mhz, "Axial frequency [MHz]"),
        (axes[1, 1], f_cyc_ghz, "Cyclotron frequency [GHz]"),
    ]:
        ax.hist(data, bins=40, color="#3a7ca5", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

    axes[1, 2].scatter(pitch, f_ax_mhz, s=2, alpha=0.5, color="#3a7ca5")
    axes[1, 2].set_xlabel("Pitch [deg]")
    axes[1, 2].set_ylabel("f_axial [MHz]")
    axes[1, 2].set_title("f_axial vs pitch (expect strong dependence)")

    fig.suptitle("Per-event truth attr distributions")
    fig.tight_layout()
    out_path = out_dir / "attr_distributions.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")




def alias_into_band(f_mhz: float, fs_mhz: float = 1000.0) -> float:
    """Wrap a frequency into [-fs/2, fs/2) via Nyquist aliasing."""
    return ((f_mhz + fs_mhz / 2) % fs_mhz) - fs_mhz / 2


def plot_fft_with_sidebands(
    h5_path: Path,
    attrs: dict,
    out_dir: Path,
    max_order: int,
    n_examples: int = 8,
) -> None:
    """Pick events evenly spaced across pitch, FFT, overlay expected sideband positions.

    The signal is complex IQ; we take the full complex FFT (scipy.fft.fft, NOT rfft)
    and plot the two-sided magnitude over [-fs/2, +fs/2]. Negative-frequency peaks
    correspond to negative-order sidebands at f_dm + n * f_ax < 0.
    """
    pitch = attrs["Pitch angle [degrees]"]
    sort_idx = np.argsort(pitch)
    spaced_idx = sort_idx[np.linspace(0, len(sort_idx) - 1, n_examples, dtype=int)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for i, evt_idx in enumerate(spaced_idx):
        evt_key = attrs["_event_keys"][evt_idx]
        sig, fs = load_signal(h5_path, evt_key)

        N = len(sig)
        # Complex FFT of the complex IQ time series. np.abs gives the magnitude.
        fft_complex = scipy.fft.fft(sig, norm="forward")
        freqs = scipy.fft.fftfreq(N, 1.0 / fs)
        order = np.argsort(freqs)
        freqs_mhz = freqs[order] / 1e6
        mag = np.abs(fft_complex)[order]

        ax = axes[i]
        ax.semilogy(freqs_mhz, mag, color="#3a7ca5", linewidth=0.5)

        # Anchor on the orbit-averaged carrier <f_c> - LO, NOT on f_c(z=0) - LO.
        # A phase-modulated tone parks at <f_inst>; for low pitch the electron
        # spends more time near mirror points where B is higher, so <f_c>
        # exceeds f_c(0) and the f_c(0)-anchored sidebands sit below the peaks.
        # Falls back to f_c(0)-LO if the simulator hasn't written the Mean attr.
        if "Mean downmixed cyclotron frequency [Hertz]" in attrs:
            f_dm_mhz = attrs["Mean downmixed cyclotron frequency [Hertz]"][evt_idx] / 1e6
            f_dm_z0_mhz = attrs["Downmixed cyclotron frequency [Hertz]"][evt_idx] / 1e6
            print(
                f"  evt {evt_idx}: pitch={attrs['Pitch angle [degrees]'][evt_idx]:.3f} deg, "
                f"f_dm(<f_c>)={f_dm_mhz:.3f} MHz, f_dm(z=0)={f_dm_z0_mhz:.3f} MHz, "
                f"shift={(f_dm_mhz - f_dm_z0_mhz):.3f} MHz"
            )
        else:
            print(
                "  WARNING: 'Mean downmixed cyclotron frequency [Hertz]' not in "
                "attrs. Falling back to f_c(z=0)-LO; sideband lines will be biased "
                "low for low-pitch events. Regenerate with the updated CRESWriter."
            )
            f_dm_mhz = attrs["Downmixed cyclotron frequency [Hertz]"][evt_idx] / 1e6
        f_ax_mhz = attrs["Axial frequency [Hertz]"][evt_idx] / 1e6
        fs_mhz = fs / 1e6

        for n in range(-max_order, max_order + 1):
            x_raw = f_dm_mhz + n * f_ax_mhz
            x_aliased = alias_into_band(x_raw, fs_mhz)
            in_band = -fs_mhz / 2 <= x_raw <= fs_mhz / 2

            if n == 0:
                colour = "#2ca02c"
                alpha = 0.8
                lw = 1.2
                zorder = 3
            elif in_band:
                colour = "red"
                alpha = 0.45
                lw = 0.8
                zorder = 2
            else:
                colour = "orange"
                alpha = 0.45
                lw = 0.8
                zorder = 2

            ax.axvline(x_aliased, color=colour, linestyle="--",
                       alpha=alpha, linewidth=lw, zorder=zorder)

        p = attrs["Pitch angle [degrees]"][evt_idx]
        phc = attrs["Cyclotron phase [rad]"][evt_idx]
        pha = attrs["Axial phase [rad]"][evt_idx]
        ax.set_title(
            f"pitch={p:.3f} deg, f_ax={f_ax_mhz:.1f} MHz\n"
            f"phi_c={phc:.2f}, phi_a={pha:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("|FFT|")
        ax.set_xlim(-fs_mhz / 2, fs_mhz / 2)

    fig.suptitle(
        "Per-event FFTs with expected sideband positions\n"
        f"Sidebands at f_dm + n * f_axial, n in [-{max_order}, {max_order}]; "
        f"green = carrier, red = in-band sideband, orange = aliased into band"
    )
    fig.tight_layout()
    out_path = out_dir / "fft_with_sidebands.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_phase_effect(h5_path: Path, attrs: dict, out_dir: Path) -> None:
    """Find a pair of events at near-identical physical params but different drawn phases,
    plot |FFT| side by side. The spectral envelope should look the same; the complex
    spectrum phases will differ (not shown directly here)."""
    pitch = attrs["Pitch angle [degrees]"]
    energy = attrs["Energy [eV]"]
    r_start = starting_radius(attrs["Starting position [metres]"])
    phi_c = attrs["Cyclotron phase [rad]"]

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    feat = np.stack([norm(pitch), norm(energy), norm(r_start)], axis=1)

    # Cap loop on big batches to keep this O(n) rather than O(n^2)
    n_search = min(len(pitch), 500)
    best = (0, 1, 0.0)
    for i in range(n_search):
        dists = np.linalg.norm(feat - feat[i], axis=1)
        dists[i] = np.inf
        j = int(dists.argmin())
        gap = abs(phi_c[i] - phi_c[j])
        if gap > best[2]:
            best = (i, j, float(gap))

    i, j, gap = best
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#3a7ca5", "#d57239"]
    labels = ["Event A", "Event B"]
    for ax, evt_idx, colour, label in zip(axes, [i, j], colours, labels):
        evt_key = attrs["_event_keys"][evt_idx]
        sig, fs = load_signal(h5_path, evt_key)

        N = len(sig)
        # Complex FFT of the complex IQ time series.
        fft_complex = scipy.fft.fft(sig, norm="forward")
        freqs = scipy.fft.fftfreq(N, 1.0 / fs)
        order = np.argsort(freqs)
        freqs_mhz = freqs[order] / 1e6
        mag = np.abs(fft_complex)[order]

        ax.semilogy(freqs_mhz, mag, color=colour, linewidth=0.5)
        ax.set_title(
            f"{label}: pitch={pitch[evt_idx]:.3f} deg, "
            f"r={r_start[evt_idx] * 1e3:.2f} mm\n"
            f"phi_c={phi_c[evt_idx]:.3f}, phi_a={attrs['Axial phase [rad]'][evt_idx]:.3f}"
        )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("|FFT|")
        ax.set_xlim(-fs / 2 / 1e6, fs / 2 / 1e6)

    fig.suptitle(
        "Two events at similar physical params, different drawn phases\n"
        f"Phase gap: {gap:.2f} rad. Spectral envelope should be near-identical."
    )
    fig.tight_layout()
    out_path = out_dir / "phase_effect_paired_events.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", required=True, type=Path,
        help="Path to CRESSignalStructure HDF5 output (run_NNN_signal.h5)",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for output figures (created if missing)",
    )
    p.add_argument(
        "--max-order", type=int, default=8,
        help="Max sideband order for overlay (default 8, matching BatchGen.py)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading attrs from {args.input}")
    attrs = load_all_attrs(args.input)

    print_summary(attrs)
    ks_test_phases(attrs)

    print(f"\nWriting figures to {args.output_dir}")
    plot_phase_histograms(attrs, args.output_dir)
    plot_attr_distributions(attrs, args.output_dir)
    plot_fft_with_sidebands(
        args.input, attrs, args.output_dir,
        max_order=args.max_order, n_examples=8,
    )
    plot_phase_effect(args.input, attrs, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()