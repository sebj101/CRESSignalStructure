"""
audit_split.py

Verify that a directory of split HDF5 files (produced by split_h5.py) is a
faithful chunked copy of a source BatchGen file AND that every event in the
splits looks physically sensible for U-Net training at 200us / 18.5 keV /
86.5-90 deg / r_wg=5 mm.

Tiered checks:
  Tier 1 - every event in every split file gets cheap attribute-level checks
           (required attrs, dtype, shape, phase/energy/pitch/freq/radius
           ranges, first-64-sample finite check).
  Tier 2 - structural integrity vs. the source: total event count matches,
           signal-name sets match (no drops, no duplicates), each source
           name appears in exactly one split, names within a split file are
           contiguous.
  Tier 3 - deep equality on a uniformly-sampled subset (default 200 events):
           full IQ byte-equality plus attribute-by-attribute equality.

Distribution sanity is reported at the end (min/max/mean of energy, pitch,
radius, |downmixed freq|; KS uniformity p-values for energy/pitch/phases).

Usage:
    python audit_split.py \\
        --source path/to/original_run_000_signal.h5 \\
        --split-dir path/to/split_outdir/

Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


SIGNAL_RE = re.compile(r"^signal(\d+)$")

REQUIRED_ATTRS = [
    "Cyclotron phase [rad]",
    "Axial phase [rad]",
    "Axial frequency [Hertz]",
    "Cyclotron frequency [Hertz]",
    "Pitch angle [degrees]",
    "Energy [eV]",
    "Time step [seconds]",
]

# Physical ranges expected for the unet_200us_5mm dataset.
EXPECTED_SAMPLES = 200_000
EXPECTED_DT = 1e-9
DT_TOLERANCE = 0.01  # 1%
FS = 1.0e9
NYQUIST = FS / 2
ENERGY_MIN, ENERGY_MAX = 18500.0, 18600.0
PITCH_MIN, PITCH_MAX = 86.5, 90.0
R_WG_M = 5e-3
TWO_PI = 2 * np.pi


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def sorted_signal_keys(group: h5py.Group) -> list[str]:
    out = []
    for k in group.keys():
        m = SIGNAL_RE.match(k)
        if m:
            out.append((int(m.group(1)), k))
    out.sort()
    return [k for _, k in out]


class Findings:
    """Accumulate per-file failures and warnings."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


# ----------------------------------------------------------------------------
# Tier 1 - per-event cheap checks
# ----------------------------------------------------------------------------

def check_event(split_file: Path, name: str, dset: h5py.Dataset,
                f: Findings) -> dict:
    """Run all cheap per-event checks; return collected stats for aggregation."""
    tag = f"{split_file.name}:{name}"

    # Dtype + shape
    if dset.dtype.kind != "c":
        f.fail(f"{tag}: dtype is not complex (got {dset.dtype})")
    if dset.shape != (EXPECTED_SAMPLES,):
        f.fail(f"{tag}: shape {dset.shape} != ({EXPECTED_SAMPLES},)")

    attrs = dset.attrs

    # Required attrs
    missing = [a for a in REQUIRED_ATTRS if a not in attrs]
    if missing:
        f.fail(f"{tag}: missing attrs {missing}")
        # Can't run further checks without attrs - return empty stats
        return {}

    # Time step
    dt = float(attrs["Time step [seconds]"])
    if abs(dt - EXPECTED_DT) / EXPECTED_DT > DT_TOLERANCE:
        f.fail(f"{tag}: Time step {dt:.3e} s differs from "
               f"{EXPECTED_DT:.3e} by >{DT_TOLERANCE*100:.0f}%")

    # Phases
    phi_c = float(attrs["Cyclotron phase [rad]"])
    phi_a = float(attrs["Axial phase [rad]"])
    if not (0 <= phi_c < TWO_PI):
        f.fail(f"{tag}: Cyclotron phase {phi_c:.4f} outside [0, 2pi)")
    if not (0 <= phi_a < TWO_PI):
        f.fail(f"{tag}: Axial phase {phi_a:.4f} outside [0, 2pi)")

    # Frequencies
    f_ax = float(attrs["Axial frequency [Hertz]"])
    if np.isnan(f_ax) or f_ax == 0:
        f.fail(f"{tag}: Axial frequency invalid (NaN or zero)")

    if "Downmixed cyclotron frequency [Hertz]" in attrs:
        f_dm = float(attrs["Downmixed cyclotron frequency [Hertz]"])
        if not (abs(f_dm) < NYQUIST):
            f.fail(f"{tag}: |Downmixed freq| {abs(f_dm):.2e} >= Nyquist "
                   f"{NYQUIST:.2e}")
    else:
        f_dm = float("nan")

    # Energy / pitch
    energy = float(attrs["Energy [eV]"])
    if not (ENERGY_MIN <= energy <= ENERGY_MAX):
        f.fail(f"{tag}: Energy {energy:.2f} outside "
               f"[{ENERGY_MIN}, {ENERGY_MAX}] eV")

    pitch = float(attrs["Pitch angle [degrees]"])
    if not (PITCH_MIN <= pitch <= PITCH_MAX):
        f.fail(f"{tag}: Pitch {pitch:.3f} outside [{PITCH_MIN}, {PITCH_MAX}] deg")

    # Radius
    radius = float("nan")
    if "Starting position [metres]" in attrs:
        pos = np.asarray(attrs["Starting position [metres]"])
        if pos.shape == (3,):
            radius = float(np.hypot(pos[0], pos[1]))
            if radius > R_WG_M + 1e-12:
                f.fail(f"{tag}: radius {radius*1e3:.3f} mm exceeds r_wg "
                       f"{R_WG_M*1e3:.3f} mm")

    # First 64 samples finite
    head = dset[:64]
    if not (np.all(np.isfinite(head.real)) and np.all(np.isfinite(head.imag))):
        f.fail(f"{tag}: first 64 samples contain NaN or Inf")

    return {
        "energy": energy,
        "pitch": pitch,
        "radius": radius,
        "phi_c": phi_c,
        "phi_a": phi_a,
        "f_dm_abs": abs(f_dm) if not np.isnan(f_dm) else np.nan,
    }


# ----------------------------------------------------------------------------
# Tier 2 - structural integrity vs. source
# ----------------------------------------------------------------------------

def check_structure(source_h5: Path, split_files: list[Path],
                    f: Findings) -> tuple[set[str], dict[str, Path]]:
    """Verify source <-> splits as sets and per-file contiguity.
    Returns (source_name_set, split_name_to_file_map) for use in Tier 3.
    """
    with h5py.File(source_h5, "r") as fs:
        if "Data" not in fs:
            f.fail(f"source {source_h5.name}: no /Data group")
            return set(), {}
        src_names = set(fs["Data"].keys())
        src_count = len(src_names)

    split_count = 0
    name_to_file: dict[str, Path] = {}
    dupes: list[str] = []
    union: set[str] = set()

    for sp in split_files:
        with h5py.File(sp, "r") as fp:
            if "Data" not in fp:
                f.fail(f"split {sp.name}: no /Data group")
                continue
            names_in_file = sorted_signal_keys(fp["Data"])
            split_count += len(names_in_file)

            # Within-file contiguity by trailing integer
            idxs = [int(SIGNAL_RE.match(n).group(1)) for n in names_in_file
                    if SIGNAL_RE.match(n)]
            if idxs and idxs != list(range(idxs[0], idxs[0] + len(idxs))):
                f.fail(f"split {sp.name}: signal indices not contiguous "
                       f"(first={idxs[0]}, last={idxs[-1]}, "
                       f"count={len(idxs)})")

            for n in names_in_file:
                if n in name_to_file:
                    dupes.append(n)
                else:
                    name_to_file[n] = sp
                union.add(n)

    if split_count != src_count:
        f.fail(f"total event count mismatch: source={src_count}, "
               f"sum_of_splits={split_count}")

    missing = src_names - union
    extra = union - src_names
    if missing:
        f.fail(f"{len(missing)} source signals missing from splits "
               f"(e.g. {sorted(missing)[:5]})")
    if extra:
        f.fail(f"{len(extra)} signals in splits not in source "
               f"(e.g. {sorted(extra)[:5]})")
    if dupes:
        f.fail(f"{len(dupes)} signals duplicated across splits "
               f"(e.g. {sorted(set(dupes))[:5]})")

    return src_names, name_to_file


# ----------------------------------------------------------------------------
# Tier 3 - deep equality on a sample
# ----------------------------------------------------------------------------

def _attrs_equal(a1, a2, name: str, tag: str, f: Findings) -> bool:
    keys1 = set(a1.keys())
    keys2 = set(a2.keys())
    if keys1 != keys2:
        f.fail(f"{tag}: attribute key set differs (source-only="
               f"{sorted(keys1-keys2)}, split-only={sorted(keys2-keys1)})")
        return False
    for k in keys1:
        v1, v2 = np.asarray(a1[k]), np.asarray(a2[k])
        if not np.array_equal(v1, v2):
            f.fail(f"{tag}: attribute '{k}' differs (src={v1!r}, split={v2!r})")
            return False
    return True


def check_deep_sample(source_h5: Path,
                      name_to_file: dict[str, Path],
                      src_names: set[str],
                      n_sample: int,
                      f: Findings) -> int:
    """Spot-check n_sample events for full IQ + attribute equality."""
    candidates = sorted(src_names & set(name_to_file.keys()),
                        key=lambda n: int(SIGNAL_RE.match(n).group(1)))
    if not candidates:
        f.fail("deep-sample: no overlap between source and split names")
        return 0
    n = min(n_sample, len(candidates))
    idxs = np.linspace(0, len(candidates) - 1, n, dtype=int)
    sample_names = [candidates[i] for i in idxs]

    # Group by split file so we don't reopen files for every event
    by_file: dict[Path, list[str]] = {}
    for nm in sample_names:
        by_file.setdefault(name_to_file[nm], []).append(nm)

    checked = 0
    with h5py.File(source_h5, "r") as fs:
        src_group = fs["Data"]
        for sp, names in by_file.items():
            with h5py.File(sp, "r") as fp:
                dst_group = fp["Data"]
                for nm in names:
                    tag = f"deep:{sp.name}:{nm}"
                    if nm not in dst_group:
                        f.fail(f"{tag}: missing from split file")
                        continue
                    src_dset = src_group[nm]
                    dst_dset = dst_group[nm]
                    if not _attrs_equal(src_dset.attrs, dst_dset.attrs,
                                        nm, tag, f):
                        continue
                    if src_dset.shape != dst_dset.shape:
                        f.fail(f"{tag}: shape differs "
                               f"({src_dset.shape} vs {dst_dset.shape})")
                        continue
                    src_data = src_dset[...]
                    dst_data = dst_dset[...]
                    if not np.array_equal(src_data, dst_data):
                        f.fail(f"{tag}: IQ samples differ")
                        continue
                    checked += 1
    return checked


# ----------------------------------------------------------------------------
# Distribution sanity
# ----------------------------------------------------------------------------

def print_distribution_summary(stats: dict[str, list[float]],
                               findings: Findings) -> None:
    print()
    print("Distribution summary (all events combined):")
    for name, lo, hi in [
        ("energy", ENERGY_MIN, ENERGY_MAX),
        ("pitch", PITCH_MIN, PITCH_MAX),
        ("radius", 0.0, R_WG_M),
        ("f_dm_abs", 0.0, NYQUIST),
    ]:
        arr = np.asarray(stats.get(name, []), dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"  {name:10s}: (no data)")
            continue
        units = {
            "energy": "eV",
            "pitch": "deg",
            "radius": "mm",
            "f_dm_abs": "MHz",
        }[name]
        scale = 1e3 if name == "radius" else (1e-6 if name == "f_dm_abs" else 1)
        a = arr * scale
        print(f"  {name:10s}: min={a.min():.4g}  max={a.max():.4g}  "
              f"mean={a.mean():.4g}  std={a.std():.4g}  ({units}, n={arr.size})")

    if _HAVE_SCIPY:
        print()
        print("KS-uniformity p-values (warn if p < 0.001):")
        for name, lo, hi in [
            ("energy", ENERGY_MIN, ENERGY_MAX),
            ("pitch", PITCH_MIN, PITCH_MAX),
            ("phi_c", 0.0, TWO_PI),
            ("phi_a", 0.0, TWO_PI),
        ]:
            arr = np.asarray(stats.get(name, []), dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size < 50:
                print(f"  {name:8s}: n={arr.size} too small for KS test")
                continue
            uniform = _scipy_stats.uniform(loc=lo, scale=hi - lo)
            _, p = _scipy_stats.kstest(arr, uniform.cdf)
            marker = "  (LOW)" if p < 1e-3 else ""
            print(f"  {name:8s}: p={p:.4f}{marker}")
            if p < 1e-3:
                findings.warn(f"{name} distribution KS p={p:.4f} < 0.001 "
                              f"(may indicate a dropped chunk)")
    else:
        print()
        print("(scipy not installed - skipping KS uniformity tests)")


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def find_split_files(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*_signal.h5"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="Original BatchGen .h5 file (the source of the split)")
    ap.add_argument("--split-dir", type=Path, required=True,
                    help="Directory containing the split *_signal.h5 files")
    ap.add_argument("--deep-sample", type=int, default=200,
                    help="Number of events to deep-check (full IQ + attrs) "
                         "vs. source (default: 200)")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}")
        return 1
    if not args.split_dir.is_dir():
        print(f"ERROR: split-dir not a directory: {args.split_dir}")
        return 1

    split_files = find_split_files(args.split_dir)
    if not split_files:
        print(f"ERROR: no *_signal.h5 files in {args.split_dir}")
        return 1

    print(f"Source:    {args.source}")
    print(f"Split dir: {args.split_dir}")
    print(f"Split files ({len(split_files)}):")
    for sp in split_files:
        print(f"  {sp.name}")

    findings = Findings()

    # Aggregator for distribution summary
    agg: dict[str, list[float]] = {
        "energy": [], "pitch": [], "radius": [],
        "phi_c": [], "phi_a": [], "f_dm_abs": [],
    }

    # Tier 1: every event in every split file
    print()
    print("=== Tier 1: per-event attribute/shape/range checks ===")
    total_events = 0
    for sp in split_files:
        with h5py.File(sp, "r") as fp:
            if "Data" not in fp:
                findings.fail(f"{sp.name}: no /Data group")
                print(f"  {sp.name}: FAIL (no /Data group)")
                continue
            group = fp["Data"]
            names = sorted_signal_keys(group)
            n_before = len(findings.failures)
            for nm in names:
                stats = check_event(sp, nm, group[nm], findings)
                for k, v in stats.items():
                    agg[k].append(v)
            n_new = len(findings.failures) - n_before
            total_events += len(names)
            status = "OK" if n_new == 0 else f"FAIL ({n_new} issues)"
            print(f"  {sp.name}: {len(names)} events  {status}")

    # Tier 2: structural integrity vs. source
    print()
    print("=== Tier 2: structural integrity vs. source ===")
    n_before = len(findings.failures)
    src_names, name_to_file = check_structure(args.source, split_files,
                                              findings)
    n_new = len(findings.failures) - n_before
    print(f"  source events: {len(src_names)}")
    print(f"  split events:  {total_events}")
    print(f"  result: {'OK' if n_new == 0 else f'FAIL ({n_new} issues)'}")

    # Tier 3: deep sample
    print()
    print(f"=== Tier 3: deep equality on {args.deep_sample} sampled events ===")
    n_before = len(findings.failures)
    checked = check_deep_sample(args.source, name_to_file, src_names,
                                args.deep_sample, findings)
    n_new = len(findings.failures) - n_before
    print(f"  events deep-checked: {checked}")
    print(f"  result: {'OK' if n_new == 0 else f'FAIL ({n_new} issues)'}")

    # Distribution summary
    print_distribution_summary(agg, findings)

    # Final
    print()
    print("=" * 60)
    if findings.warnings:
        print(f"WARNINGS ({len(findings.warnings)}):")
        for w in findings.warnings:
            print(f"  - {w}")
    if findings.failures:
        print(f"FAILURES ({len(findings.failures)}):")
        shown = findings.failures[:50]
        for x in shown:
            print(f"  - {x}")
        if len(findings.failures) > len(shown):
            print(f"  ... and {len(findings.failures) - len(shown)} more")
        print()
        print(f"AUDIT FAIL  ({len(findings.failures)} failures, "
              f"{len(findings.warnings)} warnings)")
        return 1

    print(f"AUDIT PASS  ({total_events} events across {len(split_files)} "
          f"files, {len(findings.warnings)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
