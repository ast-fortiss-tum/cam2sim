#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6B_aggregate_semantic_results.py

Aggregate per-frame IoU JSON results produced by compare_semantic_maps_iou.py
into a single summary (mean +/- std per semantic class).

MUST BE RUN FROM THE cam2sim PROJECT ROOT.

Layout:
    cam2sim/results/semantic/
        *_iou.json          <- one per frame, produced by compare_semantic_maps_iou.py
        summary.json        <- created by this script

Run:
    cd ~/Documents/cam2sim
    python 6_validation/6B_aggregate_semantic_results.py
"""

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results" / "semantic"


def aggregate_results(results_dir: Path):
    """Read all *_iou.json files in results_dir and compute stats."""
    json_files = sorted([
        p for p in results_dir.glob("*_iou.json")
        if p.name not in ("summary.json", "_summary.json")
    ])

    if not json_files:
        return None

    bg, car, road, miou = [], [], [], []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if data.get("iou_background") is not None: bg.append(data["iou_background"])
        if data.get("iou_car")        is not None: car.append(data["iou_car"])
        if data.get("iou_road")       is not None: road.append(data["iou_road"])
        if data.get("mean_iou")       is not None: miou.append(data["mean_iou"])

    return {
        "frames": len(json_files),
        "bg":   {"mean": float(np.mean(bg)),   "std": float(np.std(bg)),   "n": len(bg)},
        "car":  {"mean": float(np.mean(car)),  "std": float(np.std(car)),  "n": len(car)},
        "road": {"mean": float(np.mean(road)), "std": float(np.std(road)), "n": len(road)},
        "mIoU": {"mean": float(np.mean(miou)), "std": float(np.std(miou)), "n": len(miou)},
    }


def main():
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Results dir:  {RESULTS_DIR}")

    if not RESULTS_DIR.exists():
        print(f"[ERROR] Results directory not found: {RESULTS_DIR}")
        print(f"[HINT]  Run this from the cam2sim project root, and make sure "
              f"compare_semantic_maps_iou.py has produced per-frame JSONs first.")
        return

    stats = aggregate_results(RESULTS_DIR)
    if stats is None:
        print(f"[ERROR] No *_iou.json files found in {RESULTS_DIR}")
        return

    print(f"\nAggregated over {stats['frames']} frames")
    print(f"  {'class':<14} {'mean':>8} {'std':>8} {'n':>6}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 6}")
    for cls_name, key in [("background", "bg"), ("car", "car"),
                          ("road", "road"), ("mIoU", "mIoU")]:
        s = stats[key]
        print(f"  {cls_name:<14} {s['mean']:>8.4f} {s['std']:>8.4f} {s['n']:>6}")

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()