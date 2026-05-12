# -*- coding: utf-8 -*-
"""
compare_semantic_maps_iou.py

IoU comparison between:
  - GT  = SegFormer semantic maps from real-world images
          data/processed_dataset/<BAG>/semantic_maps/
          (filenames as produced by 2A-style SegFormer script: e.g. 000123.png
           or whatever name was in raw_dataset/.../images/)

  - PRED = CARLA-replay semantic maps (already 512x512, cleaned)
           data/processed_dataset/<BAG>/carla_replay_dataset/semantic/
           filenames: {frame_id:06d}.png

GT is resized to 512x512 (NEAREST) before comparison so the two grids match.
Aspect ratio mismatch is intentional and accepted.

Output per frame:
  - <frame>_iou.json
  - <frame>_vis.png (GT | Diff | Pred)
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# =======================
# HARDCODED CONFIG
# =======================

BAG_NAME = "reference_bag"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GT_DIR = PROJECT_ROOT / "data" / "processed_dataset" / BAG_NAME / "semantic_maps"
PRED_DIR = PROJECT_ROOT / "data" / "processed_dataset" / BAG_NAME / "carla_replay_dataset" / "semantic"
RESULTS_DIR = PROJECT_ROOT / "results" / "semantic"

TARGET_SIZE = (512, 512)  # (W, H)

NUM_CLASSES = 3

# Fixed color mapping (must match what BOTH scripts produce).
# SegFormer script (2A) writes:
#   (0, 0, 0)       -> background
#   (128, 64, 128)  -> road
#   (0, 0, 142)     -> car
# CARLA replay (cleaned) writes:
#   (0, 0, 0)       -> background   (cleaned non-target pixels)
#   (128, 64, 128)  -> road (BGR -> same in RGB since symmetric)
#   (0, 0, 142)     -> car  (in RGB after BGR<->RGB conversion in cleaner)
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # Background
    (0, 0, 142): 1,      # Car
    (128, 64, 128): 2,   # Road
}

# Tolerance for near-exact matching (handles resize/compression artifacts)
COLOR_TOLERANCE = 15


# =======================
# IMAGE / METRIC HELPERS
# =======================

def load_image_array(path: Path, resize_to=None):
    """Load PNG as RGB numpy array. Optionally NEAREST-resize to target (W, H)."""
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        if img.mode == "P":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if resize_to is not None and img.size != resize_to:
            img = img.resize(resize_to, resample=Image.Resampling.NEAREST)

        return np.array(img)
    except Exception as e:
        print(f"[WARN] Could not open {path.name}: {e}")
        return None


def rgb_to_class_map(img_arr, color_to_class, tolerance=0):
    """Map RGB image to class IDs using fixed color lookup."""
    h, w, _ = img_arr.shape
    class_map = np.zeros((h, w), dtype=np.int32)

    for color, class_id in color_to_class.items():
        color_arr = np.array(color, dtype=np.uint8)
        if tolerance > 0:
            diff = np.abs(img_arr.astype(int) - color_arr.astype(int))
            mask = np.all(diff <= tolerance, axis=-1)
        else:
            mask = np.all(img_arr == color_arr, axis=-1)
        class_map[mask] = class_id

    return class_map


def fast_hist(gt, pred, num_classes):
    mask = (gt >= 0) & (gt < num_classes)
    gt = gt[mask]
    pred = pred[mask]
    return np.bincount(
        num_classes * gt.astype(int) + pred.astype(int),
        minlength=num_classes**2
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


# =======================
# MAIN
# =======================

def main():
    print("=" * 80)
    print("SEMANTIC MAP IoU: SegFormer (real) vs CARLA replay (sim)")
    print("=" * 80)
    print(f"[INFO] GT  dir:     {GT_DIR}")
    print(f"[INFO] Pred dir:    {PRED_DIR}")
    print(f"[INFO] Results dir: {RESULTS_DIR}")
    print(f"[INFO] Target size: {TARGET_SIZE}")
    print(f"[INFO] Tolerance:   {COLOR_TOLERANCE}")
    print("=" * 80)

    if not GT_DIR.exists():
        print(f"[ERROR] GT directory does not exist: {GT_DIR}")
        return
    if not PRED_DIR.exists():
        print(f"[ERROR] PRED directory does not exist: {PRED_DIR}")
        return

    gt_files = sorted([
        p for p in GT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ])
    if not gt_files:
        print(f"[ERROR] No .png files found in GT directory.")
        return

    # --- PRE-CHECK ---
    print(f"[INFO] Found {len(gt_files)} GT files. Matching with PRED...")

    matched = []
    missing_sim = []

    for gt_path in gt_files:
        try:
            # GT filename may be like "000123.png", "frame_000123.png", "seg_000123.png"
            filename_no_ext = gt_path.stem
            number_part = (
                filename_no_ext
                .replace("seg_", "")
                .replace("frame_", "")
            )
            file_idx = int(number_part)
            sim_name = f"{file_idx:06d}.png"
            pred_path = PRED_DIR / sim_name

            if pred_path.exists():
                matched.append((gt_path, pred_path))
            else:
                missing_sim.append((gt_path.name, sim_name))
        except ValueError:
            # filename did not contain a valid number, skip
            pass

    print(f"[INFO] Matched:  {len(matched)}/{len(gt_files)}")
    if missing_sim:
        print(f"[WARN] Missing in PRED: {len(missing_sim)} (first 5 shown)")
        for gt_n, sim_n in missing_sim[:5]:
            print(f"       {gt_n}  ->  {sim_n}")

    if not matched:
        print("[ERROR] No matching pairs found. Aborting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- SKIP ALREADY DONE ---
    to_process = []
    already_done = 0
    for gt_path, pred_path in matched:
        out_json = RESULTS_DIR / f"{gt_path.stem}_iou.json"
        if out_json.exists():
            already_done += 1
        else:
            to_process.append((gt_path, pred_path))

    print(f"[INFO] Already processed: {already_done}")
    print(f"[INFO] Remaining:         {len(to_process)}")

    if not to_process:
        print("[INFO] All frames already processed.")
        print_summary(RESULTS_DIR)
        return

    # --- PROCESS ---
    iou_bg_list = []
    iou_car_list = []
    iou_road_list = []
    miou_list = []

    for gt_path, pred_path in to_process:
        gt_arr = load_image_array(gt_path, resize_to=TARGET_SIZE)
        pred_arr = load_image_array(pred_path, resize_to=None)

        if gt_arr is None or pred_arr is None:
            continue

        # If for any reason pred is not 512x512, resize it too (NEAREST).
        if pred_arr.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
            pred_pil = Image.fromarray(pred_arr).resize(
                TARGET_SIZE, resample=Image.Resampling.NEAREST
            )
            pred_arr = np.array(pred_pil)

        if gt_arr.shape[:2] != pred_arr.shape[:2]:
            print(f"[WARN] Shape mismatch after resize: {gt_path.name} "
                  f"GT={gt_arr.shape[:2]} PRED={pred_arr.shape[:2]}")
            continue

        # Direct color mapping - no clustering
        gt_cls = rgb_to_class_map(gt_arr, COLOR_TO_CLASS, tolerance=COLOR_TOLERANCE)
        pred_cls = rgb_to_class_map(pred_arr, COLOR_TO_CLASS, tolerance=COLOR_TOLERANCE)

        # --- METRICS ---
        confusion = fast_hist(gt_cls.ravel(), pred_cls.ravel(), NUM_CLASSES)
        iou = compute_iou(confusion)
        valid = confusion.sum(axis=1) > 0
        mean_iou = float(iou[valid].mean()) if valid.any() else None

        # --- SAVE ---
        img_id = gt_path.stem
        out_json = RESULTS_DIR / f"{img_id}_iou.json"
        vis_path = RESULTS_DIR / f"{img_id}_vis.png"

        results_data = {
            "gt_file":        gt_path.name,
            "sim_file":       pred_path.name,
            "iou_background": float(iou[0]) if not np.isnan(iou[0]) else None,
            "iou_car":        float(iou[1]) if not np.isnan(iou[1]) else None,
            "iou_road":       float(iou[2]) if not np.isnan(iou[2]) else None,
            "mean_iou":       mean_iou,
        }

        with open(out_json, "w") as f:
            json.dump(results_data, f, indent=2)

        save_diff_figure(gt_cls, pred_cls, NUM_CLASSES, vis_path)

        iou_bg_list.append(results_data["iou_background"])
        iou_car_list.append(results_data["iou_car"])
        iou_road_list.append(results_data["iou_road"])
        miou_list.append(mean_iou)

        miou_str = f"{mean_iou:.4f}" if mean_iou is not None else "n/a"
        print(f"[OK] {gt_path.name} -> mIoU: {miou_str}")

    # --- SUMMARY ---
    print("-" * 80)
    print("Per-batch summary (this run only)")
    print_mean("iou_background", iou_bg_list)
    print_mean("iou_car",        iou_car_list)
    print_mean("iou_road",       iou_road_list)
    print_mean("mean_iou",       miou_list)
    print("-" * 80)

    # full summary over RESULTS_DIR
    print_summary(RESULTS_DIR)


def print_mean(name, values):
    clean = [v for v in values if v is not None]
    if not clean:
        print(f"  {name:16s}: n/a")
        return
    print(f"  {name:16s}: {np.mean(clean):.4f}  (n={len(clean)})")


def print_summary(results_dir: Path):
    """Aggregate all *_iou.json files in results_dir."""
    json_files = sorted(results_dir.glob("*_iou.json"))
    if not json_files:
        return

    bg, car, road, m = [], [], [], []
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        if d.get("iou_background") is not None: bg.append(d["iou_background"])
        if d.get("iou_car")        is not None: car.append(d["iou_car"])
        if d.get("iou_road")       is not None: road.append(d["iou_road"])
        if d.get("mean_iou")       is not None: m.append(d["mean_iou"])

    print("=" * 80)
    print(f"Overall summary across {len(json_files)} frames")
    print(f"  iou_background : {np.mean(bg):.4f}   (n={len(bg)})")
    print(f"  iou_car        : {np.mean(car):.4f}   (n={len(car)})")
    print(f"  iou_road       : {np.mean(road):.4f}   (n={len(road)})")
    print(f"  mean_iou       : {np.mean(m):.4f}   (n={len(m)})")
    print("=" * 80)


if __name__ == "__main__":
    main()