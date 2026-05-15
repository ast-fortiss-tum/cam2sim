#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6A_copy_data_for_validation.py

Collects everything Step 6 needs into data/data_for_validation/ so that
6B and 6C can read all their inputs from a single, self-contained directory.

What this script copies:

1) Simulated drive trajectories produced by Step 5D
   data/results/<method>_run<N>/trajectory.json
   ->  data/data_for_validation/GS_trajectories/<method>_run<N>_trajectory.json

2) Real-world semantic maps (SegFormer GT) produced by Step 2A
   data/processed_dataset/<BAG>/semantic_maps/*.png
   ->  data/data_for_validation/semantic/*.png

3) Simulated semantic maps (CARLA replay PRED) produced by Step 5A
   data/processed_dataset/<BAG>/carla_replay_dataset/semantic/*.png
   ->  data/data_for_validation/semantic_carla/*.png

Behavior:
    - Existing files in the destination directories are left untouched.
      A copy with the same name will overwrite the previous file.
    - Trajectory JSONs are validated before copying (must be a non-empty
      JSON list with x/y fields). Invalid files are reported and skipped.
    - Pass --dry_run to see what would happen without writing.
    - Pass --skip_trajectories / --skip_semantic_gt / --skip_semantic_carla
      to disable individual sub-copies.

Usage (from the project root):
    python 6_validation/6A_copy_data_for_validation.py
    python 6_validation/6A_copy_data_for_validation.py --dry_run
    python 6_validation/6A_copy_data_for_validation.py --skip_semantic_gt
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BAG_NAME = "reference_bag"

# Trajectory copy
DEFAULT_TRAJ_SRC = PROJECT_ROOT / "data" / "results"
DEFAULT_TRAJ_DST = PROJECT_ROOT / "data" / "data_for_validation" / "GS_trajectories"

# Semantic GT copy (SegFormer real, from Step 2A)
DEFAULT_SEM_GT_SRC = PROJECT_ROOT / "data" / "processed_dataset" / BAG_NAME / "semantic_maps"
DEFAULT_SEM_GT_DST = PROJECT_ROOT / "data" / "data_for_validation" / "semantic"

# Semantic PRED copy (CARLA replay, from Step 5A)
DEFAULT_SEM_CARLA_SRC = PROJECT_ROOT / "data" / "processed_dataset" / BAG_NAME / \
                        "carla_replay_dataset" / "semantic"
DEFAULT_SEM_CARLA_DST = PROJECT_ROOT / "data" / "data_for_validation" / "semantic_carla"

# Matches "<method>_run<N>", e.g. "splatfacto_run1", "splatfacto_run12".
RUN_DIR_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)_run(\d+)$")


# =============================================================================
# TRAJECTORY COPY
# =============================================================================

def find_run_dirs(src_dir: Path):
    """
    Return [(run_dir, method, run_idx)] for every direct subdirectory of
    src_dir whose name matches RUN_DIR_PATTERN and which contains a
    trajectory.json. Sorted by (method, run_idx).
    """
    if not src_dir.is_dir():
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
    """Return number of points if valid, -1 on failure."""
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


def copy_trajectories(src_dir: Path, dst_dir: Path, dry_run: bool):
    print("\n" + "=" * 70)
    print("  (1/3) DRIVE TRAJECTORIES")
    print("=" * 70)
    print(f"[INFO] Source:      {src_dir}")
    print(f"[INFO] Destination: {dst_dir}")

    runs = find_run_dirs(src_dir)
    if not runs:
        print(f"[INFO] No <method>_run<N>/ folders with trajectory.json found "
              f"under {src_dir}. Nothing to copy.")
        return 0

    print(f"\n[INFO] Found {len(runs)} drive run(s):")
    for run_dir, method, run_idx in runs:
        n_pts = validate_trajectory_json(run_dir / "trajectory.json")
        if n_pts < 0:
            print(f"  ! {run_dir.name}  (invalid trajectory.json, will be skipped)")
        else:
            print(f"  - {run_dir.name}  ({n_pts} points, "
                  f"method={method}, run={run_idx})")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    print()
    for run_dir, method, run_idx in runs:
        src_json = run_dir / "trajectory.json"
        n_pts = validate_trajectory_json(src_json)
        if n_pts < 0:
            print(f"  [SKIP] {run_dir.name} (invalid JSON)")
            continue

        dst_name = f"{run_dir.name}_trajectory.json"
        dst_path = dst_dir / dst_name

        if dry_run:
            print(f"  [DRY] {src_json}  ->  {dst_path}")
        else:
            try:
                shutil.copy2(src_json, dst_path)
                print(f"  OK   {run_dir.name}/trajectory.json  ->  {dst_name}")
                n_copied += 1
            except Exception as e:
                print(f"  [ERR] {run_dir.name}: {e}")

    if dry_run:
        print(f"\n[INFO] Dry run: would have copied {len(runs)} trajectory file(s).")
    else:
        print(f"\n[INFO] Copied {n_copied}/{len(runs)} trajectory file(s) to {dst_dir}")

    return n_copied


# =============================================================================
# SEMANTIC IMAGE COPY (used for both GT and PRED)
# =============================================================================

def copy_png_dir(src_dir: Path, dst_dir: Path, label: str, dry_run: bool):
    """
    Copy every *.png from src_dir directly into dst_dir (flat, same basename).
    """
    if not src_dir.is_dir():
        print(f"[WARN] {label}: source directory does not exist: {src_dir}")
        print(f"       Nothing to copy.")
        return 0

    pngs = sorted([p for p in src_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".png"])
    if not pngs:
        print(f"[WARN] {label}: no PNG files in {src_dir}. Nothing to copy.")
        return 0

    print(f"[INFO] {label}: found {len(pngs)} PNG file(s) in {src_dir}")
    print(f"       first: {pngs[0].name}, last: {pngs[-1].name}")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY] would copy {len(pngs)} PNG(s) to {dst_dir}")
        return 0

    n_copied = 0
    n_errors = 0
    for src_png in pngs:
        dst_png = dst_dir / src_png.name
        try:
            shutil.copy2(src_png, dst_png)
            n_copied += 1
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  [ERR] {src_png.name}: {e}")

    print(f"[INFO] {label}: copied {n_copied}/{len(pngs)} to {dst_dir}")
    if n_errors > 5:
        print(f"       ({n_errors} errors total, only the first 5 shown)")
    return n_copied


def copy_semantic_gt(src_dir: Path, dst_dir: Path, dry_run: bool):
    print("\n" + "=" * 70)
    print("  (2/3) SEMANTIC GT (SegFormer real-world)")
    print("=" * 70)
    print(f"[INFO] Source:      {src_dir}")
    print(f"[INFO] Destination: {dst_dir}")
    return copy_png_dir(src_dir, dst_dir, "semantic GT", dry_run)


def copy_semantic_carla(src_dir: Path, dst_dir: Path, dry_run: bool):
    print("\n" + "=" * 70)
    print("  (3/3) SEMANTIC PRED (CARLA replay)")
    print("=" * 70)
    print(f"[INFO] Source:      {src_dir}")
    print(f"[INFO] Destination: {dst_dir}")
    return copy_png_dir(src_dir, dst_dir, "semantic CARLA", dry_run)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect Step 5D drive trajectories and Step 2A/5A "
                    "semantic maps into data/data_for_validation/."
    )

    parser.add_argument("--traj_src", type=str, default=str(DEFAULT_TRAJ_SRC),
                        help=f"Source dir with <method>_run<N>/ folders "
                             f"(default: {DEFAULT_TRAJ_SRC})")
    parser.add_argument("--traj_dst", type=str, default=str(DEFAULT_TRAJ_DST),
                        help=f"Destination for flat *_trajectory.json files "
                             f"(default: {DEFAULT_TRAJ_DST})")

    parser.add_argument("--sem_gt_src", type=str, default=str(DEFAULT_SEM_GT_SRC),
                        help=f"Source dir with SegFormer GT PNGs "
                             f"(default: {DEFAULT_SEM_GT_SRC})")
    parser.add_argument("--sem_gt_dst", type=str, default=str(DEFAULT_SEM_GT_DST),
                        help=f"Destination for GT semantic PNGs "
                             f"(default: {DEFAULT_SEM_GT_DST})")

    parser.add_argument("--sem_carla_src", type=str, default=str(DEFAULT_SEM_CARLA_SRC),
                        help=f"Source dir with CARLA replay semantic PNGs "
                             f"(default: {DEFAULT_SEM_CARLA_SRC})")
    parser.add_argument("--sem_carla_dst", type=str, default=str(DEFAULT_SEM_CARLA_DST),
                        help=f"Destination for CARLA semantic PNGs "
                             f"(default: {DEFAULT_SEM_CARLA_DST})")

    parser.add_argument("--skip_trajectories", action="store_true",
                        help="Do not copy drive trajectories.")
    parser.add_argument("--skip_semantic_gt", action="store_true",
                        help="Do not copy SegFormer GT semantic maps.")
    parser.add_argument("--skip_semantic_carla", action="store_true",
                        help="Do not copy CARLA replay semantic maps.")

    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without writing.")
    args = parser.parse_args()

    print("=" * 70)
    print("  COPY DATA FOR VALIDATION")
    print("=" * 70)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Bag name:     {BAG_NAME}")
    print(f"[INFO] Dry run:      {args.dry_run}")
    print(f"[INFO] Sub-steps:")
    print(f"       trajectories : {'SKIP' if args.skip_trajectories  else 'ON'}")
    print(f"       semantic GT  : {'SKIP' if args.skip_semantic_gt   else 'ON'}")
    print(f"       semantic CARLA: {'SKIP' if args.skip_semantic_carla else 'ON'}")
    print("=" * 70)

    totals = {"trajectories": 0, "semantic_gt": 0, "semantic_carla": 0}

    if not args.skip_trajectories:
        totals["trajectories"] = copy_trajectories(
            Path(args.traj_src).resolve(),
            Path(args.traj_dst).resolve(),
            dry_run=args.dry_run,
        )

    if not args.skip_semantic_gt:
        totals["semantic_gt"] = copy_semantic_gt(
            Path(args.sem_gt_src).resolve(),
            Path(args.sem_gt_dst).resolve(),
            dry_run=args.dry_run,
        )

    if not args.skip_semantic_carla:
        totals["semantic_carla"] = copy_semantic_carla(
            Path(args.sem_carla_src).resolve(),
            Path(args.sem_carla_dst).resolve(),
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    if args.dry_run:
        print("[INFO] Dry run: nothing was written.")
    else:
        print(f"[INFO] Trajectories  copied: {totals['trajectories']}")
        print(f"[INFO] Semantic GT   copied: {totals['semantic_gt']}")
        print(f"[INFO] Semantic CARLA copied: {totals['semantic_carla']}")
    print(f"[INFO] data/data_for_validation/ is now ready for 6B and 6C.")


if __name__ == "__main__":
    main()