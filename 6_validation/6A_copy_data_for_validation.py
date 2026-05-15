#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_drive_trajectories.py

Scan data/results/ for drive run folders produced by 5D_dave2.py and copy
their trajectory.json into data/data_for_validation/GS_trajectories/ with
the flat naming expected by 6D_driving_quality_metrics.py.

Source layout (produced by 5D_dave2.py):
    data/results/
        splatfacto_run1/trajectory.json
        splatfacto_run2/trajectory.json
        splatfacto_run3/trajectory.json
        ...

Destination layout (read by 6D_driving_quality_metrics.py):
    data/data_for_validation/GS_trajectories/
        splatfacto_run1_trajectory.json
        splatfacto_run2_trajectory.json
        splatfacto_run3_trajectory.json

Behavior:
    - Source: any subdirectory of data/results/ matching <method>_run<N>/
      that contains a trajectory.json.
    - Destination filename: <source_folder_name>_trajectory.json
      (e.g. splatfacto_run1/  ->  splatfacto_run1_trajectory.json)
    - Existing files in the destination directory are left untouched.
      If a destination file with the same name already exists it is
      overwritten by the new copy.
    - Pass --dry_run to see what would happen without copying.

Usage (from project root):
    python data_management/collect_drive_trajectories.py
    python data_management/collect_drive_trajectories.py --dry_run
    python data_management/collect_drive_trajectories.py \
        --src data/results \
        --dst data/data_for_validation/GS_trajectories
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


# Default paths, relative to the project root (= parent of this script's dir).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SRC = PROJECT_ROOT / "data" / "results"
DEFAULT_DST = PROJECT_ROOT / "data" / "data_for_validation" / "GS_trajectories"

# Matches "<method>_run<N>", e.g. "splatfacto_run1", "splatfacto_run12".
RUN_DIR_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)_run(\d+)$")


def find_run_dirs(src_dir: Path):
    """
    Return a list of (run_dir_path, method, run_idx), sorted by (method, run_idx).
    Only directories whose name matches RUN_DIR_PATTERN and which contain
    trajectory.json are returned.
    """
    if not src_dir.is_dir():
        print(f"[ERROR] Source directory does not exist: {src_dir}")
        return []

    found = []
    for entry in sorted(src_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = RUN_DIR_PATTERN.match(entry.name)
        if not m:
            continue
        method = m.group(1)
        run_idx = int(m.group(2))
        traj_json = entry / "trajectory.json"
        if not traj_json.is_file():
            print(f"[WARN] Skipping {entry.name}: no trajectory.json inside")
            continue
        found.append((entry, method, run_idx))

    found.sort(key=lambda t: (t[1], t[2]))
    return found


def validate_trajectory_json(path: Path) -> int:
    """
    Quick sanity check on the JSON.
    Returns the number of points if valid, or -1 on failure.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not parse {path}: {e}")
        return -1

    if not isinstance(data, list) or not data:
        print(f"[WARN] {path} is not a non-empty JSON list")
        return -1

    first = data[0]
    if not isinstance(first, dict):
        print(f"[WARN] {path}: first entry is not a dict")
        return -1

    if "x" not in first or "y" not in first:
        print(f"[WARN] {path}: missing 'x'/'y' fields in first entry")
        return -1

    return len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectory.json files from drive runs into "
                    "the flat layout expected by 6D."
    )
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC),
                        help=f"Source directory containing <method>_run<N>/ "
                             f"folders (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", type=str, default=str(DEFAULT_DST),
                        help=f"Destination directory for flat "
                             f"<method>_run<N>_trajectory.json files "
                             f"(default: {DEFAULT_DST})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without writing.")
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()

    print("=" * 70)
    print("  COLLECT DRIVE TRAJECTORIES")
    print("=" * 70)
    print(f"[INFO] Source:      {src_dir}")
    print(f"[INFO] Destination: {dst_dir}")
    print(f"[INFO] Dry run:     {args.dry_run}")
    print("=" * 70)

    runs = find_run_dirs(src_dir)
    if not runs:
        print(f"[ERROR] No <method>_run<N>/ folders with trajectory.json "
              f"found under {src_dir}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(runs)} drive run(s):")
    for run_dir, method, run_idx in runs:
        n_pts = validate_trajectory_json(run_dir / "trajectory.json")
        if n_pts < 0:
            print(f"  ! {run_dir.name}  (invalid trajectory.json, will be skipped)")
        else:
            print(f"  - {run_dir.name}  ({n_pts} points, method={method}, run={run_idx})")

    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    print()
    for run_dir, method, run_idx in runs:
        src_json = run_dir / "trajectory.json"
        n_pts = validate_trajectory_json(src_json)
        if n_pts < 0:
            print(f"  [SKIP] {run_dir.name} (invalid JSON)")
            continue

        # Destination filename: "<source_dirname>_trajectory.json"
        # so splatfacto_run1/ -> splatfacto_run1_trajectory.json
        dst_name = f"{run_dir.name}_trajectory.json"
        dst_path = dst_dir / dst_name

        if args.dry_run:
            print(f"  [DRY] {src_json}  ->  {dst_path}")
        else:
            try:
                shutil.copy2(src_json, dst_path)
                print(f"  OK   {run_dir.name}/trajectory.json  ->  {dst_name}")
                n_copied += 1
            except Exception as e:
                print(f"  [ERR] {run_dir.name}: {e}")

    print()
    if args.dry_run:
        print(f"[INFO] Dry run: would have copied {len(runs)} file(s).")
    else:
        print(f"[INFO] Copied {n_copied}/{len(runs)} file(s) to {dst_dir}")
        print(f"[INFO] You can now run 6D_driving_quality_metrics.py:")
        print(f"       --sim_dirs GS={dst_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()