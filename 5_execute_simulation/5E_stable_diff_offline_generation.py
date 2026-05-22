#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5E_stable_diff_offline_generation.py

Offline Stable Diffusion generation on top of the CARLA replay dataset
(produced by 5A_OPT_trajectory_only_carla_with_instance_mapping.py).

For each frame:
  - selects the model part (LoRA + 3 ControlNets) based on hero position
    along the trajectory (same logic as DAVE-2 with SD)
  - runs the SD pipeline with seg + inst + temp ControlNets
  - the temporal ControlNet uses the previous generated frame
  - saves the generated RGB image

Uses one fixed control schedule (the best config from the thesis):
    control_start = [0.0, 0.0, 0.35]   # [seg, inst, temp]
    control_end   = [1.0, 0.6, 0.55]

Reads from (project root):
    data/processed_dataset/<BAG>/carla_replay_dataset_sd/
        semantic/, instance/, data/all_frame_data.json
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json

Reads SD models from (external SSD):
    <EXTERNAL_DRIVE>/cam2sim_sd/<BAG>/SD_Training_Outputs_Split/part_<N>/
        config.json
        stable_diffusion/pytorch_lora_weights.safetensors
        controlnet_segmentation/
        controlnet_instance/
        controlnet_tempconsistency/

Writes to (project root):
    data/processed_dataset/<BAG>/sd_generated/
        rgb/frame_XXXXXX.png
        data/generation_info.json
"""

import os
import sys
import json
import time
from typing import List, Tuple, Dict

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# =======================
# PATH SETUP
# =======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )

if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)


# =======================
# LOCAL UTILS IMPORTS
# =======================

from utils.stable_diffusion import (
    load_pipeline_models,
    generate_image_realtime,
)


# =======================
# HARDCODED CONFIG
# =======================

BAG_NAME = "reference_bag"
NUM_PARTS = 3

# Best control schedule from the thesis
CONTROL_START = [0.0, 0.0, 0.35]
CONTROL_END = [1.0, 0.6, 0.55]
GUIDANCE_SCALE = 3.0

# Limit generation to first N frames (None = all)
MAX_FRAMES = None
# MAX_FRAMES = 100

# Skip frames whose output PNG already exists (idempotent)
SKIP_EXISTING = True


# =======================
# INPUT PATHS (project root)
# =======================

REPLAY_DATASET_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_dataset",
    BAG_NAME,
    "carla_replay_dataset_sd",
)

SEM_FOLDER = os.path.join(REPLAY_DATASET_FOLDER, "semantic")
INST_FOLDER = os.path.join(REPLAY_DATASET_FOLDER, "instance")
METADATA_PATH = os.path.join(REPLAY_DATASET_FOLDER, "data", "all_frame_data.json")

TRAJECTORY_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_carla",
    BAG_NAME,
    "trajectory_positions_rear_odom_yaw.json",
)


# =======================
# MODEL PATHS (external SSD)
# =======================

EXTERNAL_DRIVE = "/media/davidejannussi/ssd space"
CAM2SIM_SD_ROOT = os.path.join(EXTERNAL_DRIVE, "cam2sim_sd")

BAG_SD_DIR = os.path.join(CAM2SIM_SD_ROOT, BAG_NAME)
MODELS_BASE_DIR = os.path.join(BAG_SD_DIR, "SD_Training_Outputs_Split")

# Use the external SSD for HuggingFace cache too (so we don't re-download SD1.5)
os.environ["HF_HOME"] = os.path.join(CAM2SIM_SD_ROOT, "huggingface_cache")


# =======================
# OUTPUT PATHS (project root)
# =======================

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_dataset",
    BAG_NAME,
    "sd_generated",
)

OUTPUT_RGB_FOLDER = os.path.join(OUTPUT_FOLDER, "rgb")
OUTPUT_DATA_FOLDER = os.path.join(OUTPUT_FOLDER, "data")
OUTPUT_INFO_PATH = os.path.join(OUTPUT_DATA_FOLDER, "generation_info.json")


# =======================
# DEVICE
# =======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# TRAJECTORY-BASED MODEL SELECTION
# =======================

def split_trajectory_into_parts(trajectory_points, num_parts=3):
    """Split trajectory into equal chunks (same as training)."""
    total = len(trajectory_points)
    chunk_size = total // num_parts

    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = total if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(trajectory_points[start:end])

    return chunks


def find_closest_trajectory_point(frame_location, trajectory_chunk):
    """Find closest point in a trajectory chunk to (x, y) of frame_location."""
    min_dist = float("inf")
    closest_idx = 0

    for idx, point in enumerate(trajectory_chunk):
        traj_x = point["transform"]["location"]["x"]
        traj_y = point["transform"]["location"]["y"]

        dist = np.sqrt(
            (frame_location["x"] - traj_x) ** 2
            + (frame_location["y"] - traj_y) ** 2
        )

        if dist < min_dist:
            min_dist = dist
            closest_idx = idx

    return closest_idx, min_dist


def select_model_part(frame_location, trajectory_chunks):
    """Find which model part (chunk) the current frame belongs to."""
    best_part = 0
    best_distance = float("inf")

    for part_idx, chunk in enumerate(trajectory_chunks):
        _, dist = find_closest_trajectory_point(frame_location, chunk)

        if dist < best_distance:
            best_distance = dist
            best_part = part_idx

    return best_part, best_distance


# =======================
# DATA LOADING
# =======================

def load_json_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_replay_data(max_frames: int = None):
    """
    Load semantic + instance maps and metadata from the replay dataset.

    Returns:
        seg_list:       list of PIL.Image (RGB)
        inst_list:      list of PIL.Image (RGB)
        frame_data:     list of metadata dicts (location, rotation, caption)
        frame_indices:  list of frame_id ints
    """
    print(f"\n[INFO] Loading replay dataset from: {REPLAY_DATASET_FOLDER}")

    all_frame_data = load_json_file(METADATA_PATH)

    if max_frames is not None and max_frames > 0:
        all_frame_data = all_frame_data[:max_frames]

    seg_list = []
    inst_list = []
    frame_data = []
    frame_indices = []

    print(f"[INFO] Reading {len(all_frame_data)} frames...")

    n_missing = 0

    for item in tqdm(all_frame_data, desc="Reading frames"):
        frame_id = item["frame"]
        filename = f"{frame_id:06d}.png"

        seg_path = os.path.join(SEM_FOLDER, filename)
        inst_path = os.path.join(INST_FOLDER, filename)

        if not os.path.exists(seg_path) or not os.path.exists(inst_path):
            n_missing += 1
            continue

        seg_img = Image.open(seg_path).convert("RGB")
        inst_img = Image.open(inst_path).convert("RGB")

        seg_list.append(seg_img)
        inst_list.append(inst_img)
        frame_data.append(item)
        frame_indices.append(frame_id)

    if n_missing > 0:
        print(f"[WARN] Missing files for {n_missing} frames (skipped).")

    print(f"[INFO] Loaded {len(seg_list)} frames.")

    return seg_list, inst_list, frame_data, frame_indices


# =======================
# MAIN
# =======================

def main():
    print("=" * 80)
    print("OFFLINE STABLE DIFFUSION GENERATION (SD branch)")
    print("=" * 80)
    print(f"[INFO] Project root:        {PROJECT_ROOT}")
    print(f"[INFO] Bag name:            {BAG_NAME}")
    print(f"[INFO] Replay dataset:      {REPLAY_DATASET_FOLDER}")
    print(f"[INFO] Trajectory:          {TRAJECTORY_PATH}")
    print(f"[INFO] Models base:         {MODELS_BASE_DIR}")
    print(f"[INFO] Output folder:       {OUTPUT_FOLDER}")
    print(f"[INFO] Device:              {DEVICE}")
    print(f"[INFO] Num parts:           {NUM_PARTS}")
    print(f"[INFO] Control start:       {CONTROL_START}")
    print(f"[INFO] Control end:         {CONTROL_END}")
    print(f"[INFO] Guidance scale:      {GUIDANCE_SCALE}")
    print(f"[INFO] Skip existing PNGs:  {SKIP_EXISTING}")
    print("=" * 80)

    # ---------- Sanity checks ----------
    if not os.path.exists(REPLAY_DATASET_FOLDER):
        raise FileNotFoundError(
            f"Replay dataset not found: {REPLAY_DATASET_FOLDER}\n"
            f"Run 5A_OPT_trajectory_only_carla_with_instance_mapping.py first."
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}\n"
            f"The replay dataset is incomplete."
        )

    if not os.path.exists(TRAJECTORY_PATH):
        raise FileNotFoundError(
            f"Trajectory not found: {TRAJECTORY_PATH}"
        )

    if not os.path.isdir(MODELS_BASE_DIR):
        raise FileNotFoundError(
            f"SD models directory not found: {MODELS_BASE_DIR}\n"
            f"Run 4A_train_stable_diff.py first."
        )

    for part_idx in range(NUM_PARTS):
        part_dir = os.path.join(MODELS_BASE_DIR, f"part_{part_idx}")
        if not os.path.isdir(part_dir):
            raise FileNotFoundError(
                f"Missing model part directory: {part_dir}"
            )

    # ---------- Output folders ----------
    os.makedirs(OUTPUT_RGB_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)

    # ---------- Load trajectory and split ----------
    full_trajectory = load_json_file(TRAJECTORY_PATH)
    trajectory_chunks = split_trajectory_into_parts(full_trajectory, NUM_PARTS)
    print(f"\n[INFO] Trajectory: {len(full_trajectory)} points")
    for i, chunk in enumerate(trajectory_chunks):
        print(f"   Part {i}: {len(chunk)} frames")

    # ---------- Load replay data ----------
    seg_list, inst_list, frame_data, frame_indices = load_replay_data(
        max_frames=MAX_FRAMES,
    )

    total_frames = len(seg_list)
    if total_frames == 0:
        raise RuntimeError("No frames to process.")

    print(f"\n[INFO] Total frames to process: {total_frames}")
    print("=" * 80)

    # ---------- Generation loop ----------
    pipe = None
    model_data = None
    current_model_part = None
    prev_generated = None

    t_start = time.time()
    n_generated = 0
    n_skipped = 0

    for i, frame_id in enumerate(frame_indices):
        seg = seg_list[i]
        inst = inst_list[i]
        frame_info = frame_data[i]

        out_filename = f"frame_{frame_id:06d}.png"
        out_path = os.path.join(OUTPUT_RGB_FOLDER, out_filename)

        # ---- Skip if already generated ----
        if SKIP_EXISTING and os.path.exists(out_path):
            n_skipped += 1
            # Still need to keep prev_generated coherent: load existing image
            prev_generated = Image.open(out_path).convert("RGB")
            continue

        # ---- Model selection based on frame location ----
        frame_location = frame_info["location"]
        required_part, traj_distance = select_model_part(
            frame_location, trajectory_chunks
        )

        # ---- Switch model if needed ----
        if required_part != current_model_part:
            print("\n" + "=" * 60)
            print(
                f"MODEL SWITCH: Part {current_model_part} -> Part {required_part}"
            )
            print(
                f"   Frame: {frame_id} | "
                f"Position: ({frame_location['x']:.1f}, {frame_location['y']:.1f})"
            )
            print(f"   Trajectory distance: {traj_distance:.2f} m")
            print("=" * 60)

            # Preserve prev_generated across the switch
            previous_model_last_image = prev_generated

            # Unload old model
            if pipe is not None:
                del pipe
                del model_data
                torch.cuda.empty_cache()
                print("   Freed GPU memory.")

            # Load new model
            model_path = os.path.join(MODELS_BASE_DIR, f"part_{required_part}")
            print(f"   Loading model from: {model_path}")
            pipe, model_data = load_pipeline_models(model_path, DEVICE)

            current_model_part = required_part
            prev_generated = previous_model_last_image
            print(f"   Model Part {required_part} loaded.\n")

        # ---- Use caption from metadata ----
        current_prompt = frame_info.get("caption", "")
        if not current_prompt:
            current_prompt = (
                f"pos x: {frame_location['x']:.2f}, "
                f"y: {frame_location['y']:.2f}"
            )

        # ---- First frame has no previous image ----
        prev_img = None if i == 0 else prev_generated

        # ---- Generate ----
        out_img = generate_image_realtime(
            pipe=pipe,
            seg_image=seg,
            inst_image=inst,
            model_data=model_data,
            prev_image=prev_img,
            prompt=current_prompt,
            guidance=GUIDANCE_SCALE,
            control_start=CONTROL_START,
            control_end=CONTROL_END,
        )

        # ---- Save ----
        out_img.save(out_path)

        prev_generated = out_img
        n_generated += 1

        # ---- Progress log ----
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            avg_per_frame = elapsed / max(n_generated, 1)
            eta = avg_per_frame * (total_frames - (i + 1))
            print(
                f"[{i + 1}/{total_frames}] Frame: {frame_id} | "
                f"Part: {current_model_part} | "
                f"Avg: {avg_per_frame:.2f}s/frame | "
                f"ETA: {eta / 60:.1f} min"
            )

    # ---------- Final cleanup ----------
    if pipe is not None:
        del pipe
        del model_data
        torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start

    # ---------- Save generation info ----------
    info = {
        "bag_name": BAG_NAME,
        "num_parts": NUM_PARTS,
        "control_start": CONTROL_START,
        "control_end": CONTROL_END,
        "guidance_scale": GUIDANCE_SCALE,
        "total_frames_in_replay": total_frames,
        "frames_generated": n_generated,
        "frames_skipped_existing": n_skipped,
        "elapsed_seconds": round(t_elapsed, 2),
        "avg_seconds_per_generated_frame": (
            round(t_elapsed / max(n_generated, 1), 3)
        ),
        "output_folder": OUTPUT_RGB_FOLDER,
    }

    with open(OUTPUT_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE.")
    print(f"   Frames generated:  {n_generated}")
    print(f"   Frames skipped:    {n_skipped}")
    print(f"   Total elapsed:     {t_elapsed / 60:.1f} min")
    print(f"   Output folder:     {OUTPUT_RGB_FOLDER}")
    print(f"   Generation info:   {OUTPUT_INFO_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()