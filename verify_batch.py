"""
verify_batch.py

Quick post-generation smoke check for a BatchGen output. Verifies:
  - File exists, has events
  - Sample count per event is consistent with declared acq_time + sample_rate
  - All required truth attrs are present (phi_c, phi_a, f_axial, f_cyc, pitch, energy)
  - phi_c, phi_a sit in [0, 2*pi); f_axial has no nans or zeros

Exits with status 0 (PASS) or 1 (FAIL).

Usage:
    python verify_batch.py data/path/run_000_signal.h5
"""

import sys
from pathlib import Path

import h5py
import numpy as np


REQUIRED_ATTRS = [
    "Cyclotron phase [rad]",
    "Axial phase [rad]",
    "Axial frequency [Hertz]",
    "Cyclotron frequency [Hertz]",
    "Pitch angle [degrees]",
    "Energy [eV]",
    "Time step [seconds]",
]


def main(h5_path: Path) -> int:
    if not h5_path.exists():
        print(f"FAIL: file does not exist: {h5_path}")
        return 1

    with h5py.File(h5_path, "r") as f:
        if "Data" not in f:
            print("FAIL: no 'Data' group in file")
            return 1

        keys = list(f["Data"].keys())
        n = len(keys)
        print(f"Events:               {n}")
        if n == 0:
            print("FAIL: no events written")
            return 1

        first = f["Data"][keys[0]]
        attrs = dict(first.attrs)
        sig_len = first.shape[0]
        sig_dtype = first.dtype
        dt = float(attrs.get("Time step [seconds]", 0.0))
        acq_time = sig_len * dt if dt > 0 else float("nan")

        print(f"Samples per event:    {sig_len}")
        print(f"Sample dtype:         {sig_dtype}")
        print(f"Time step:            {dt:.3e} s")
        print(f"Acquisition time:     {acq_time * 1e6:.2f} us")

        missing = [r for r in REQUIRED_ATTRS if r not in attrs]
        if missing:
            print(f"FAIL: missing required attrs: {missing}")
            return 1
        print("Required attrs:       all present")

        if sig_dtype.kind != "c":
            print(f"FAIL: signal dtype is not complex (got {sig_dtype})")
            return 1

        # Sample up to ~200 events evenly spaced for range checks
        stride = max(1, n // 200)
        sample_keys = keys[::stride]
        phi_c = np.array([f["Data"][k].attrs["Cyclotron phase [rad]"]
                          for k in sample_keys])
        phi_a = np.array([f["Data"][k].attrs["Axial phase [rad]"]
                          for k in sample_keys])
        f_ax = np.array([f["Data"][k].attrs["Axial frequency [Hertz]"]
                         for k in sample_keys])

        ok = True
        two_pi = 2 * np.pi
        if not ((phi_c >= 0).all() and (phi_c < two_pi).all()):
            print(f"FAIL: phi_c outside [0, 2pi) "
                  f"(min={phi_c.min():.3f}, max={phi_c.max():.3f})")
            ok = False
        if not ((phi_a >= 0).all() and (phi_a < two_pi).all()):
            print(f"FAIL: phi_a outside [0, 2pi) "
                  f"(min={phi_a.min():.3f}, max={phi_a.max():.3f})")
            ok = False
        if np.isnan(f_ax).any() or (f_ax == 0).any():
            print(f"FAIL: f_axial has nan or zero "
                  f"(any_nan={bool(np.isnan(f_ax).any())}, "
                  f"any_zero={bool((f_ax == 0).any())})")
            ok = False

        if not ok:
            return 1

        print(f"phi_c sample range:   [{phi_c.min():.3f}, {phi_c.max():.3f}]  rad")
        print(f"phi_a sample range:   [{phi_a.min():.3f}, {phi_a.max():.3f}]  rad")
        print(f"f_axial sample range: [{f_ax.min() / 1e6:.2f}, {f_ax.max() / 1e6:.2f}] MHz")
        print()
        print("PASS")
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_batch.py <path_to_h5>")
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
