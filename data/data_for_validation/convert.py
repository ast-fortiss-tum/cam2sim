#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_old_trajectory_txt_to_csv.py

Convert legacy real-world trajectory files (from the thesis pipeline)
to the cam2sim CSV format.

OLD format (.txt, with `#` header):
    # FrameID, Timestamp, X, Y, Yaw
    0, 1771283861.205309, 692932.1277, 5339068.3913, 1.0879
    1, 1771283861.215191, 692932.1282, 5339068.3914, 1.0879
    ...

NEW format (.csv, with header row):
    timestamp,x,y,z,yaw
    1771283861.205309,692932.127700,5339068.391300,0.000000,1.087900
    ...

The z column is set to 0.0 (the legacy txt files don't have it).

Usage:
    # Convert a single file:
    python convert_old_trajectory_txt_to_csv.py input.txt output.csv

    # Convert a whole directory (writes <stem>.csv next to each <stem>.txt):
    python convert_old_trajectory_txt_to_csv.py /path/to/dir
"""

import argparse
import csv
import sys
from pathlib import Path


def convert_one(in_path: Path, out_path: Path) -> int:
    n_written = 0
    with open(in_path, "r") as fi, open(out_path, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["timestamp", "x", "y", "z", "yaw"])
        for line in fi:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            # Old layout: frame_id, timestamp, x, y, yaw  (5 columns)
            if len(parts) < 5:
                continue
            try:
                ts = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                yaw = float(parts[4])
            except ValueError:
                continue
            writer.writerow([
                f"{ts:.9f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{0.0:.6f}",   # z fixed to 0
                f"{yaw:.6f}",
            ])
            n_written += 1
    return n_written


def main():
    parser = argparse.ArgumentParser(description="Convert old trajectory txt to new csv")
    parser.add_argument("input", help="Input .txt file OR a directory containing .txt files")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .csv file (only if input is a single .txt). "
                             "Ignored if input is a directory.")
    args = parser.parse_args()

    in_path = Path(args.input)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if in_path.is_file():
        out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")
        n = convert_one(in_path, out_path)
        print(f"OK  {in_path.name} -> {out_path.name}  ({n} rows)")
        return

    if in_path.is_dir():
        if args.output is not None:
            print("WARN: --output ignored when input is a directory "
                  "(writing <stem>.csv next to each .txt)", file=sys.stderr)

        txt_files = sorted(in_path.glob("*.txt"))
        # only files that look like trajectory: contain "trajectory" in name
        # OR named like "<N>_trajectory.txt" / "trajectory<N>.txt"
        traj_files = [p for p in txt_files if "trajectory" in p.name.lower()]
        if not traj_files:
            print(f"No trajectory*.txt found in {in_path}")
            sys.exit(1)

        total = 0
        for p in traj_files:
            out = p.with_suffix(".csv")
            n = convert_one(p, out)
            print(f"OK  {p.name} -> {out.name}  ({n} rows)")
            total += n
        print(f"\nConverted {len(traj_files)} files, {total} total rows.")
        return


if __name__ == "__main__":
    main()