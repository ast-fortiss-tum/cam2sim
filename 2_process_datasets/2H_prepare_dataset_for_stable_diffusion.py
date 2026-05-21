#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2H_prepare_dataset_for_stable_diffusion.py

Build a HuggingFace dataset (Arrow binary) for Stable Diffusion / ControlNet
training, using the cam2sim project layout.

Reads from (project root):
    data/raw_dataset/<BAG>/images/frame_XXXXXX.png
    data/processed_dataset/<BAG>/semantic_maps/frame_XXXXXX.png   (from 2F)
    data/processed_dataset/<BAG>/camera_detections/instance_maps/frame_XXXXXX.png  (from 2A_OPT)
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json

Writes to (project root):
    data/data_for_stable_diffusion/<BAG>/
        images/         (RGB, 512x512)
        segmentation/   (semantic, 512x512, NEAREST)
        instance/       (instance, 512x512, NEAREST)
        previous/       (previous-frame RGB, 512x512)
        hf_binary/      (Arrow dataset with columns:
                         image, segmentation, instance, previous, text, frame_id)

Dataset columns match what 5-train_split.py expects:
    image, segmentation, instance, previous, text
"""

import os
import re
import json
import shutil
import glob
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Image as HFImage


# ============================================================
# CONFIGURATION
# ============================================================

BAG_NAME = "reference_bag"

# Inputs (cam2sim layout)
RAW_IMAGES_DIR = f"data/raw_dataset/{BAG_NAME}/images"
SEG_MAPS_DIR = f"data/processed_dataset/{BAG_NAME}/semantic_maps"
INSTANCE_MAPS_DIR = f"data/processed_dataset/{BAG_NAME}/camera_detections/instance_maps"
COORD_JSON = f"data/data_for_carla/{BAG_NAME}/trajectory_positions_rear_odom_yaw.json"

# Output
OUTPUT_DIR = f"data/data_for_stable_diffusion/{BAG_NAME}"
HF_BINARY_DIR = os.path.join(OUTPUT_DIR, "hf_binary")

TARGET_SIZE = (512, 512)


# ============================================================
# HELPERS
# ============================================================

def get_frame_id(filename):
    """frame_000123.png -> 123"""
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def resize_and_save(src_path, dst_path, mode="rgb"):
    """
    Resize and save with appropriate filter:
    - rgb:  LANCZOS (smooth, for RGB)
    - mask: NEAREST (preserve class IDs / instance colors)
    """
    if os.path.exists(dst_path):
        return  # idempotent

    img = Image.open(src_path)
    if mode == "rgb":
        img = img.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    else:
        # NEAREST so class colors / instance IDs are preserved
        img = img.resize(TARGET_SIZE, Image.NEAREST)
    img.save(dst_path)


def add_temporal_links(entries):
    """
    Sort entries by frame_id. For each entry, set 'previous' to the previous
    frame's RGB path. The first frame gets itself as 'previous' (so the column
    is never None).
    """
    entries.sort(key=lambda e: e["frame_id"])
    for i, e in enumerate(entries):
        if i == 0:
            e["previous"] = e["image"]
        else:
            e["previous"] = entries[i - 1]["image"]
    return entries


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PREPARE DATASET FOR STABLE DIFFUSION TRAINING")
    print("=" * 70)

    # Validate inputs
    for path, label in [
        (RAW_IMAGES_DIR, "raw RGB images"),
        (SEG_MAPS_DIR, "semantic maps (from 2F)"),
        (INSTANCE_MAPS_DIR, "instance maps (from 2A_OPT)"),
        (COORD_JSON, "trajectory JSON"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[INPUT MISSING] {label}: {path}")

    print(f"\nBag:               {BAG_NAME}")
    print(f"RGB images:        {RAW_IMAGES_DIR}")
    print(f"Semantic maps:     {SEG_MAPS_DIR}")
    print(f"Instance maps:     {INSTANCE_MAPS_DIR}")
    print(f"Trajectory:        {COORD_JSON}")
    print(f"Output:            {OUTPUT_DIR}")
    print(f"Resolution:        {TARGET_SIZE}")

    # Create output directories
    out_img = os.path.join(OUTPUT_DIR, "images")
    out_seg = os.path.join(OUTPUT_DIR, "segmentation")
    out_inst = os.path.join(OUTPUT_DIR, "instance")
    out_prev = os.path.join(OUTPUT_DIR, "previous")  # symlinks, populated later

    for d in [out_img, out_seg, out_inst, out_prev]:
        os.makedirs(d, exist_ok=True)

    # ---------- Load coordinate captions ----------
    print(f"\n[1/5] Loading coordinates from trajectory JSON...")
    with open(COORD_JSON, "r") as f:
        traj = json.load(f)

    coords_map = {}
    for entry in traj:
        fid = entry["frame_id"]
        loc = entry["transform"]["location"]
        # Caption format compatible with the training script's expectation
        # (the training script doesn't require UTM specifically; it accepts
        # any string. Adding yaw helps the model condition on heading.)
        coords_map[fid] = f"pos x: {loc['x']:.2f}, y: {loc['y']:.2f}"

    print(f"   Found {len(coords_map)} coordinate entries.")

    # ---------- Collect frames that have ALL the required inputs ----------
    print(f"\n[2/5] Matching frames with all required inputs...")
    rgb_files = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, "*.png")))
    print(f"   RGB frames found: {len(rgb_files)}")

    entries = []
    n_skip_coords = 0
    n_skip_seg = 0
    n_skip_inst = 0

    for rgb_path in tqdm(rgb_files, desc="Matching"):
        fname = os.path.basename(rgb_path)
        fid = get_frame_id(fname)

        # Skip if any input is missing
        if fid not in coords_map:
            n_skip_coords += 1
            continue

        seg_path = os.path.join(SEG_MAPS_DIR, fname)
        if not os.path.exists(seg_path):
            n_skip_seg += 1
            continue

        inst_path = os.path.join(INSTANCE_MAPS_DIR, fname)
        if not os.path.exists(inst_path):
            n_skip_inst += 1
            continue

        entries.append({
            "frame_id": fid,
            "src_rgb": rgb_path,
            "src_seg": seg_path,
            "src_inst": inst_path,
            "fname": fname,
        })

    print(f"   Matched frames:   {len(entries)}")
    print(f"   Skipped (no coords):    {n_skip_coords}")
    print(f"   Skipped (no semantic):  {n_skip_seg}")
    print(f"   Skipped (no instance):  {n_skip_inst}")

    if not entries:
        raise RuntimeError("No valid frames found — check your input paths.")

    # ---------- Resize and copy to output folders ----------
    print(f"\n[3/5] Resizing to {TARGET_SIZE} and copying...")
    for e in tqdm(entries, desc="Resizing"):
        fname = e["fname"]
        dst_rgb = os.path.join(out_img, fname)
        dst_seg = os.path.join(out_seg, fname)
        dst_inst = os.path.join(out_inst, fname)

        resize_and_save(e["src_rgb"], dst_rgb, mode="rgb")
        resize_and_save(e["src_seg"], dst_seg, mode="mask")
        resize_and_save(e["src_inst"], dst_inst, mode="mask")

        e["image"] = os.path.abspath(dst_rgb)
        e["segmentation"] = os.path.abspath(dst_seg)
        e["instance"] = os.path.abspath(dst_inst)
        e["text"] = coords_map[e["frame_id"]]

    # ---------- Add temporal links (previous frame) ----------
    print(f"\n[4/5] Adding temporal links (previous-frame RGB)...")
    entries = add_temporal_links(entries)

    # ---------- Build HuggingFace dataset ----------
    print(f"\n[5/5] Building HuggingFace Arrow dataset...")
    final_list = []
    for e in entries:
        final_list.append({
            "image": e["image"],
            "segmentation": e["segmentation"],
            "instance": e["instance"],
            "previous": e["previous"],
            "text": e["text"],
            "frame_id": e["frame_id"],
        })

    ds = Dataset.from_list(final_list)

    # Cast file-path columns to HFImage so the binary embeds pixel data
    ds = ds.cast_column("image", HFImage())
    ds = ds.cast_column("segmentation", HFImage())
    ds = ds.cast_column("instance", HFImage())
    ds = ds.cast_column("previous", HFImage())

    print(f"   Dataset rows:    {len(ds)}")
    print(f"   Dataset columns: {ds.column_names}")

    # Wipe old binary if present, then save
    if os.path.exists(HF_BINARY_DIR):
        shutil.rmtree(HF_BINARY_DIR)
    ds.save_to_disk(HF_BINARY_DIR)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"   Binary dataset:  {HF_BINARY_DIR}")
    print(f"   Total samples:   {len(ds)}")
    print(f"\nUpdate your training script with:")
    print(f"   LOCAL_BINARY_PATH = \"{os.path.abspath(HF_BINARY_DIR)}\"")
    print(f"   then run with --local flag.")
    print("=" * 70)


if __name__ == "__main__":
    main()