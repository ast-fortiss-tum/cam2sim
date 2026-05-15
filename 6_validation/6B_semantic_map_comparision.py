#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6B_semantic_map_comparision.py

Semantic IoU evaluation: per-frame metrics + aggregated summary.

Inputs are read from data/data_for_validation/, which is populated by
6A_copy_data_for_validation.py from the actual sources (Step 2A SegFormer
output for the GT, Step 5A CARLA replay output for the PRED).

  - GT   = SegFormer outputs from real-world images:
           data/data_for_validation/semantic/<framename>.png
  - PRED = CARLA-replay semantic outputs (cleaned, 512x512):
           data/data_for_validation/semantic_carla/<frame_id:06d>.png

For each matched (GT, PRED) pair:
  1. Load both, NEAREST-resize GT to 512x512 (PRED is already at that size)
  2. Map RGB -> class IDs using a fixed palette with small color tolerance
  3. Compute per-class IoU (background, car, road) and mIoU
  4. Save <frame>_iou.json + <frame>_vis.png (3-panel: GT | Diff | Pred)

After processing (or with --summary_only), aggregates all *_iou.json files
in the results dir into summary.json with mean+/-std per class.

Outputs (under PROJECT_ROOT/results/semantic/):
    <frame>_iou.json    one per frame
    <frame>_vis.png     one per frame
    summary.json        aggregated stats (mean/std/n per class)

Usage (from the cam2sim project root):
    # Process new frames + aggregate
    python 6_validation/6B_semantic_map_comparision.py

    # Only aggregate already-computed per-frame JSONs into summary.json
    python 6_validation/6B_semantic_map_comparision.py --summary_only

    # Rerun every frame, ignoring already-existing *_iou.json
    python 6_validation/6B_semantic_map_comparision.py --force
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Inputs come from data/data_for_validation/, populated by 6A.
GT_DIR = PROJECT_ROOT / "data" / "data_for_validation" / "semantic"
PRED_DIR = PROJECT_ROOT / "data" / "data_for_validation" / "semantic_carla"
RESULTS_DIR = PROJECT_ROOT / "results" / "semantic"

TARGET_SIZE = (512, 512)  # (W, H)
NUM_CLASSES = 3

# Fixed palette - must match what both pipelines produce.
#   SegFormer (step 2A) writes:
#     (0, 0, 0)       -> background
#     (128, 64, 128)  -> road
#     (0, 0, 142)     -> car
#   CARLA replay (cleaned) writes the same palette, with non-target
#   pixels zeroed.
COLOR_TO_CLASS = {
    (0, 0, 0):       0,   # background
    (0, 0, 142):     1,   # car
    (128, 64, 128):  2,   # road
}

# Tolerance for near-exact color matching (handles resize/compression noise)
COLOR_TOLERANCE = 15


# =============================================================================
# IMAGE / METRIC HELPERS
# =============================================================================

def load_image_array(path: Path, resize_to=None):
    """Load PNG as RGB numpy array. Optionally NEAREST-resize to (W, H)."""
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if resize_to is not None and img.size != resize_to:
            img = img.resize(resize_to, resample=Image.Resampling.NEAREST)
        return np.array(img)
    except Exception as e:
        print(f"[WARN] Could not open {path.name}: {e}")
        return None


def rgb_to_class_map(img_arr, color_to_class, tolerance=0):
    """Map an RGB image to class IDs via fixed-palette lookup with tolerance."""
    h, w, _ = img_arr.shape
    class_map = np.zeros((h, w), dtype=np.int32)
    img_int = img_arr.astype(int)
    for color, class_id in color_to_class.items():
        color_arr = np.array(color, dtype=int)
        if tolerance > 0:
            diff = np.abs(img_int - color_arr)
            mask = np.all(diff <= tolerance, axis=-1)
        else:
            mask = np.all(img_arr == np.array(color, dtype=np.uint8), axis=-1)
        class_map[mask] = class_id
    return class_map


def fast_hist(gt, pred, num_classes):
    mask = (gt >= 0) & (gt < num_classes)
    gt = gt[mask]
    pred = pred[mask]
    return np.bincount(
        num_classes * gt.astype(int) + pred.astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def compute_iou(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    gt_sum = confusion_matrix.sum(axis=1)
    pred_sum = confusion_matrix.sum(axis=0)
    union = gt_sum + pred_sum - intersection
    iou = intersection / np.maximum(union, 1)
    return iou


def save_diff_figure(gt_cls, pred_cls, num_classes, out_path):
    diff = (gt_cls != pred_cls)
    base_colors = ["black", "green", "red"]
    cmap = ListedColormap(base_colors[:num_classes])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax1, ax2, ax3 = axes

    ax1.imshow(gt_cls, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    ax1.set_title("GT (Real / SegFormer)")
    ax1.axis("off")

    ax2.imshow(diff, cmap="gray", interpolation="nearest")
    ax2.set_title("Diff (White = Mismatch)")
    ax2.axis("off")

    ax3.imshow(pred_cls, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    ax3.set_title("Pred (Sim / CARLA)")
    ax3.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# PER-FRAME PROCESSING
# =============================================================================

def match_gt_to_pred(gt_files, pred_dir):
    """
    For each GT file, derive the predicted filename (<frame_id:06d>.png).

    Returns (matched_pairs, missing_in_pred), where matched_pairs is a list
    of (gt_path, pred_path).
    """
    matched = []
    missing = []
    for gt_path in gt_files:
        stem = gt_path.stem
        number_part = stem.replace("seg_", "").replace("frame_", "")
        try:
            file_idx = int(number_part)
        except ValueError:
            # Filename without a numeric id, can't map - skip silently.
            continue
        sim_name = f"{file_idx:06d}.png"
        pred_path = pred_dir / sim_name
        if pred_path.exists():
            matched.append((gt_path, pred_path))
        else:
            missing.append((gt_path.name, sim_name))
    return matched, missing


def process_frame(gt_path, pred_path, results_dir):
    """Compute IoU for one (GT, PRED) pair and write its JSON + vis PNG."""
    gt_arr = load_image_array(gt_path, resize_to=TARGET_SIZE)
    pred_arr = load_image_array(pred_path, resize_to=None)

    if gt_arr is None or pred_arr is None:
        return None

    # Make sure pred is also 512x512 (defensive: it should already be).
    if pred_arr.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        pred_pil = Image.fromarray(pred_arr).resize(
            TARGET_SIZE, resample=Image.Resampling.NEAREST
        )
        pred_arr = np.array(pred_pil)

    if gt_arr.shape[:2] != pred_arr.shape[:2]:
        print(f"[WARN] Shape mismatch after resize: {gt_path.name} "
              f"GT={gt_arr.shape[:2]} PRED={pred_arr.shape[:2]}")
        return None

    gt_cls = rgb_to_class_map(gt_arr, COLOR_TO_CLASS, tolerance=COLOR_TOLERANCE)
    pred_cls = rgb_to_class_map(pred_arr, COLOR_TO_CLASS, tolerance=COLOR_TOLERANCE)

    confusion = fast_hist(gt_cls.ravel(), pred_cls.ravel(), NUM_CLASSES)
    iou = compute_iou(confusion)
    valid = confusion.sum(axis=1) > 0
    mean_iou = float(iou[valid].mean()) if valid.any() else None

    img_id = gt_path.stem
    out_json = results_dir / f"{img_id}_iou.json"
    vis_path = results_dir / f"{img_id}_vis.png"

    results = {
        "gt_file":        gt_path.name,
        "sim_file":       pred_path.name,
        "iou_background": float(iou[0]) if not np.isnan(iou[0]) else None,
        "iou_car":        float(iou[1]) if not np.isnan(iou[1]) else None,
        "iou_road":       float(iou[2]) if not np.isnan(iou[2]) else None,
        "mean_iou":       mean_iou,
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    save_diff_figure(gt_cls, pred_cls, NUM_CLASSES, vis_path)

    return results


# =============================================================================
# AGGREGATION (replaces 6B_aggregate_semantic_results.py)
# =============================================================================

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

    def _stats(arr):
        return {
            "mean": float(np.mean(arr)) if arr else None,
            "std":  float(np.std(arr))  if arr else None,
            "n":    len(arr),
        }

    return {
        "frames": len(json_files),
        "bg":   _stats(bg),
        "car":  _stats(car),
        "road": _stats(road),
        "mIoU": _stats(miou),
    }


def print_and_save_summary(results_dir: Path):
    stats = aggregate_results(results_dir)
    if stats is None:
        print(f"[INFO] No *_iou.json files in {results_dir} to aggregate.")
        return

    print("\n" + "=" * 80)
    print(f"Aggregated over {stats['frames']} frames")
    print(f"  {'class':<14} {'mean':>8} {'std':>8} {'n':>6}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 6}")
    for cls_name, key in [("background", "bg"), ("car", "car"),
                          ("road", "road"), ("mIoU", "mIoU")]:
        s = stats[key]
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "n/a"
        std = f"{s['std']:.4f}" if s["std"] is not None else "n/a"
        print(f"  {cls_name:<14} {mean:>8} {std:>8} {s['n']:>6}")
    print("=" * 80)

    out_path = results_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantic IoU: per-frame metrics + aggregated summary."
    )
    parser.add_argument("--summary_only", action="store_true",
                        help="Skip per-frame processing and only aggregate "
                             "existing *_iou.json into summary.json.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess every frame, ignoring already-existing "
                             "<frame>_iou.json files.")
    args = parser.parse_args()

    print("=" * 80)
    print("SEMANTIC IoU: SegFormer (real) vs CARLA replay (sim)")
    print("=" * 80)
    print(f"[INFO] GT  dir:     {GT_DIR}")
    print(f"[INFO] Pred dir:    {PRED_DIR}")
    print(f"[INFO] Results dir: {RESULTS_DIR}")
    print(f"[INFO] Target size: {TARGET_SIZE}")
    print(f"[INFO] Tolerance:   {COLOR_TOLERANCE}")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Summary-only mode: skip processing, just aggregate ---
    if args.summary_only:
        print_and_save_summary(RESULTS_DIR)
        return

    # --- Validate inputs ---
    if not GT_DIR.exists():
        print(f"[ERROR] GT directory does not exist: {GT_DIR}")
        print(f"[HINT]  Run 6A_copy_data_for_validation.py first to populate it.")
        sys.exit(1)
    if not PRED_DIR.exists():
        print(f"[ERROR] PRED directory does not exist: {PRED_DIR}")
        print(f"[HINT]  Run 6A_copy_data_for_validation.py first to populate it.")
        sys.exit(1)

    gt_files = sorted([
        p for p in GT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ])
    if not gt_files:
        print(f"[ERROR] No .png files in {GT_DIR}")
        sys.exit(1)

    # --- Match GT <-> PRED ---
    print(f"[INFO] Found {len(gt_files)} GT files. Matching with PRED...")
    matched, missing = match_gt_to_pred(gt_files, PRED_DIR)
    print(f"[INFO] Matched:  {len(matched)}/{len(gt_files)}")
    if missing:
        print(f"[WARN] Missing in PRED: {len(missing)} (first 5):")
        for gt_n, sim_n in missing[:5]:
            print(f"       {gt_n}  ->  {sim_n}")
    if not matched:
        print("[ERROR] No matching pairs. Aborting.")
        sys.exit(1)

    # --- Skip already-done frames unless --force ---
    if args.force:
        to_process = matched
        already_done = 0
    else:
        to_process = []
        already_done = 0
        for gt_path, pred_path in matched:
            out_json = RESULTS_DIR / f"{gt_path.stem}_iou.json"
            if out_json.exists():
                already_done += 1
            else:
                to_process.append((gt_path, pred_path))

    print(f"[INFO] Already processed: {already_done}")
    print(f"[INFO] To process:        {len(to_process)}")

    # --- Process per-frame ---
    batch_bg, batch_car, batch_road, batch_miou = [], [], [], []

    for gt_path, pred_path in to_process:
        r = process_frame(gt_path, pred_path, RESULTS_DIR)
        if r is None:
            continue
        batch_bg.append(r["iou_background"])
        batch_car.append(r["iou_car"])
        batch_road.append(r["iou_road"])
        batch_miou.append(r["mean_iou"])
        m = r["mean_iou"]
        m_str = f"{m:.4f}" if m is not None else "n/a"
        print(f"[OK] {gt_path.name} -> mIoU: {m_str}")

    # --- Batch summary (this run only) ---
    if to_process:
        print("-" * 80)
        print("Batch summary (this run only)")
        for name, vals in [("iou_background", batch_bg),
                           ("iou_car",        batch_car),
                           ("iou_road",       batch_road),
                           ("mean_iou",       batch_miou)]:
            clean = [v for v in vals if v is not None]
            if clean:
                print(f"  {name:16s}: {np.mean(clean):.4f}  (n={len(clean)})")
            else:
                print(f"  {name:16s}: n/a")
        print("-" * 80)

    # --- Aggregate everything in the results dir ---
    print_and_save_summary(RESULTS_DIR)


if __name__ == "__main__":
    main()