"""
split_h5.py

Split BatchGen-produced HDF5 files into smaller chunks (default 5000 events
each) while preserving every per-event dataset attribute byte-for-byte.

Dataset names are kept identical to the source (so signal27000 from the source
remains signal27000 in its split file) — this gives free traceability and
keeps phase reproducibility against (phase_seed, batch_idx, signal_index)
intact.

Two usage modes:

  # Single file
  python split_h5.py path/to/run_000_signal.h5 --outdir out_dir/

  # Walk a directory tree, splitting every *_signal.h5 in it
  python split_h5.py path/to/parent_dir --outdir out_dir/ --recursive

For each *source directory* visited (the dir containing the source .h5),
an annotated run_config.json is written into the matching output dir, with
underscore-prefixed fields documenting that this is a split copy. The
original keys (if a source run_config.json exists) are preserved verbatim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import h5py


SIGNAL_RE = re.compile(r"^signal(\d+)$")


def _sorted_signal_keys(group: h5py.Group) -> list[str]:
    """Return signal* dataset names sorted by their trailing integer."""
    out = []
    for k in group.keys():
        m = SIGNAL_RE.match(k)
        if m:
            out.append((int(m.group(1)), k))
    out.sort()
    return [k for _, k in out]


def _existing_split_event_count(path: Path) -> int | None:
    """How many events does an existing split file have? None if not openable."""
    try:
        with h5py.File(path, "r") as f:
            if "Data" not in f:
                return 0
            return len(f["Data"].keys())
    except Exception:
        return None


def _write_split_config(
    out_dir: Path,
    source_h5: Path,
    n_events: int,
    chunk: int,
    split_files: list[str],
) -> None:
    """Write an annotated run_config.json into out_dir."""
    src_config = source_h5.parent / "run_config.json"
    original: dict = {}
    if src_config.exists():
        try:
            with src_config.open("r") as f:
                original = json.load(f)
        except Exception as e:
            print(f"  warn: failed to read source run_config.json ({e}); "
                  f"writing split metadata only")

    annotated: dict = {
        "_split_note": (
            "This directory contains chunked outputs of an original BatchGen "
            "run. Per-event HDF5 attributes (and IQ data) are byte-identical "
            "to the source. The fields below '_split_files' (if any) are "
            "copied verbatim from the source run_config.json and describe "
            "the original run, NOT this split copy."
        ),
        "_split_source": str(source_h5.resolve()),
        "_original_events": n_events,
        "_chunk": chunk,
        "_split_files": split_files,
    }
    annotated.update(original)

    dst = out_dir / "run_config.json"
    with dst.open("w") as f:
        json.dump(annotated, f, indent=2)


def split_one_file(
    src_h5: Path,
    out_dir: Path,
    chunk: int,
    prefix: str,
    force: bool,
    dry_run: bool,
) -> int:
    """Split one source .h5 into chunks under out_dir. Returns 0 on success."""
    print(f"\n--- {src_h5} ---> {out_dir}/")
    if not src_h5.exists():
        print(f"  ERROR: source does not exist")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_h5, "r") as fsrc:
        if "Data" not in fsrc:
            print(f"  ERROR: source has no /Data group")
            return 1
        src_group = fsrc["Data"]
        keys = _sorted_signal_keys(src_group)
        n = len(keys)
        if n == 0:
            print(f"  ERROR: source has zero signal datasets")
            return 1

        n_chunks = math.ceil(n / chunk)
        print(f"  events:  {n}")
        print(f"  chunk:   {chunk}")
        print(f"  outputs: {n_chunks} files ({prefix}_000_signal.h5 .. "
              f"{prefix}_{n_chunks-1:03d}_signal.h5)")

        split_filenames: list[str] = []

        for k in range(n_chunks):
            chunk_keys = keys[k * chunk : (k + 1) * chunk]
            out_name = f"{prefix}_{k:03d}_signal.h5"
            out_path = out_dir / out_name
            split_filenames.append(out_name)

            if out_path.exists() and not force:
                existing = _existing_split_event_count(out_path)
                if existing == len(chunk_keys):
                    print(f"  [{k+1}/{n_chunks}] {out_name}: already exists "
                          f"with {existing} events, skipping (use --force "
                          f"to overwrite)")
                    continue
                else:
                    print(f"  [{k+1}/{n_chunks}] {out_name}: exists but has "
                          f"{existing} events (expected {len(chunk_keys)}); "
                          f"overwriting")

            if dry_run:
                print(f"  [{k+1}/{n_chunks}] {out_name}: DRY-RUN would write "
                      f"{len(chunk_keys)} events "
                      f"({chunk_keys[0]}..{chunk_keys[-1]})")
                continue

            t0 = time.time()
            tmp_path = out_path.with_suffix(".h5.partial")
            if tmp_path.exists():
                tmp_path.unlink()
            with h5py.File(tmp_path, "w") as fdst:
                dst_group = fdst.create_group("Data")
                bytes_written = 0
                for src_name in chunk_keys:
                    fsrc.copy(src_group[src_name], dst_group, name=src_name)
                    bytes_written += src_group[src_name].nbytes
            os.replace(tmp_path, out_path)
            elapsed = time.time() - t0
            mb = bytes_written / (1024 * 1024)
            print(f"  [{k+1}/{n_chunks}] {out_name}: {len(chunk_keys)} events, "
                  f"{mb:.0f} MB in {elapsed:.1f}s "
                  f"({chunk_keys[0]}..{chunk_keys[-1]})")

        if not dry_run:
            _write_split_config(out_dir, src_h5, n, chunk, split_filenames)
            print(f"  wrote {out_dir/'run_config.json'}")

    return 0


def walk_recursive(
    src_root: Path,
    out_root: Path,
    chunk: int,
    prefix: str,
    force: bool,
    dry_run: bool,
) -> int:
    """Find every *_signal.h5 under src_root, split each into mirrored dir."""
    src_root = src_root.resolve()
    sources = sorted(src_root.rglob("*_signal.h5"))
    if not sources:
        print(f"ERROR: no *_signal.h5 files found under {src_root}")
        return 1

    print(f"Found {len(sources)} source file(s) under {src_root}")
    rc_total = 0
    for src in sources:
        rel_subdir = src.parent.relative_to(src_root)
        out_dir = out_root / rel_subdir
        rc = split_one_file(src, out_dir, chunk, prefix, force, dry_run)
        if rc != 0:
            rc_total = 1
    return rc_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path,
                    help="Source .h5 file, or directory (with --recursive)")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Destination directory (will be created if needed)")
    ap.add_argument("--chunk", type=int, default=5000,
                    help="Events per output file (default: 5000)")
    ap.add_argument("--prefix", type=str, default="run",
                    help="Filename prefix for splits (default: 'run')")
    ap.add_argument("--recursive", action="store_true",
                    help="Treat src as a directory; split every *_signal.h5 "
                         "under it, mirroring the subdirectory structure into "
                         "--outdir")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing split files (default: skip if "
                         "they already exist with the right event count)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write anything; report what would happen")
    args = ap.parse_args()

    if args.chunk <= 0:
        print(f"ERROR: --chunk must be positive (got {args.chunk})")
        return 2

    if args.recursive:
        if not args.src.is_dir():
            print(f"ERROR: --recursive requires src to be a directory "
                  f"(got {args.src})")
            return 2
        return walk_recursive(args.src, args.outdir, args.chunk, args.prefix,
                              args.force, args.dry_run)
    else:
        if args.src.is_dir():
            print(f"ERROR: src is a directory; pass --recursive to walk it, "
                  f"or point at a single .h5 file")
            return 2
        return split_one_file(args.src, args.outdir, args.chunk, args.prefix,
                              args.force, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
