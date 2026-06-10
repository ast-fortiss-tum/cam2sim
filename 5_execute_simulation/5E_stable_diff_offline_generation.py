#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5E_stable_diff_offline_generation.py

Offline Stable Diffusion generation on top of the CARLA replay dataset
(produced by 5A_sd_trajectory_only_carla.py).

For each frame:
  - selects the model part (LoRA + 3 ControlNets) based on hero position
    along the trajectory (same logic as DAVE-2 with SD)
  - runs the SD pipeline with seg + inst + temp ControlNets
  - the temporal ControlNet uses the previous generated frame
  - saves the generated RGB image

Uses one fixed control schedule:
    control_start = [0.0, 0.0, 0.35]   # [seg, inst, temp]
    control_end   = [1.0, 0.6, 0.55]

Reads from (project root):
    data/processed_dataset/<BAG>/carla_replay_dataset_sd/
        semantic/, instance/, data/all_frame_data.json
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json

Reads SD models from:
    <SD_ROOT>/<BAG>/SD_Training_Outputs_Split/part_<N>/
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


# =======================
# PATH SETUP, script can be launched from any directory.
# (must come BEFORE any diffusers/HF import)
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
# HF CACHE SETUP (must come BEFORE diffusers import to take effect)
# Sets HF_HOME so Hugging Face libraries (diffusers, huggingface_hub) read
# and write models in our chosen SD storage root instead of the default
# ~/.cache/huggingface/. The env var is read at import time by
# huggingface_hub.constants, so it has to be set before the first
# diffusers/HF import — otherwise the library locks onto the default
# cache and our override is silently ignored.
# =======================
from utils.sd_paths import resolve_sd_root

CAM2SIM_SD_ROOT, _ = resolve_sd_root(PROJECT_ROOT)
os.environ["HF_HOME"] = os.path.join(CAM2SIM_SD_ROOT, "huggingface_cache")


# =======================
# STANDARD IMPORTS (safe now, HF_HOME is set)
# =======================

import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image

from utils.stable_diffusion import (
    load_pipeline_models,
    generate_image_realtime,
    split_trajectory_into_parts,
    select_model_part,
    load_replay_data,
    CONTROL_START,
    CONTROL_END,
    GUIDANCE_SCALE,
)
from utils.sd_paths import require_trained_parts


# =======================
# CONFIG
# =======================

# Bag name (with .bag extension): must match an existing bag from step 1.
DEFAULT_BAG_NAME = "reference_bag.bag"

# Limit generation to first N frames for testing. Set to None to disable.
MAX_FRAMES = 3

# Skip frames whose output PNG already exists, to resume interrupted generation without redoing completed frames.
SKIP_EXISTING = False


# =======================
# DEVICE
# =======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# MAIN
# =======================

def main():
    parser = argparse.ArgumentParser(
        description="Offline Stable Diffusion generation on CARLA replay dataset"
    )
    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help="Bag filename including .bag extension "
             "(default: env BAG_NAME or 'reference_bag.bag').",
    )
    args = parser.parse_args()

    bag_name = args.bag_name               # e.g. "reference_bag.bag"
    bag_stem = Path(bag_name).stem         # e.g. "reference_bag"

    # ---------- Build bag-dependent paths ----------
    replay_dataset_folder = os.path.join(
        PROJECT_ROOT, "data", "processed_dataset", bag_stem,
        "carla_replay_dataset_sd",
    )
    sem_folder = os.path.join(replay_dataset_folder, "semantic")
    inst_folder = os.path.join(replay_dataset_folder, "instance")
    metadata_path = os.path.join(replay_dataset_folder, "data", "all_frame_data.json")

    trajectory_path = os.path.join(
        PROJECT_ROOT, "data", "data_for_carla", bag_stem,
        "trajectory_positions_rear_odom_yaw.json",
    )

    # External paths (per-bag)
    bag_sd_dir = os.path.join(CAM2SIM_SD_ROOT, bag_stem)
    models_base_dir = os.path.join(bag_sd_dir, "SD_Training_Outputs_Split")

    # Output paths (per-bag, in project root)
    output_folder = os.path.join(
        PROJECT_ROOT, "data", "processed_dataset", bag_stem, "sd_generated",
    )
    output_rgb_folder = os.path.join(output_folder, "rgb")
    output_data_folder = os.path.join(output_folder, "data")
    output_info_path = os.path.join(output_data_folder, "generation_info.json")

    # ---------- Sanity checks on inputs ----------
    if not os.path.exists(replay_dataset_folder):
        raise FileNotFoundError(
            f"Replay dataset not found: {replay_dataset_folder}\n"
            f"Run 5A_sd_trajectory_only_carla.py --bag-name {bag_name} first."
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            f"The replay dataset is incomplete."
        )

    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(
            f"Trajectory not found: {trajectory_path}"
        )

    num_parts = require_trained_parts(models_base_dir, bag_name)

    # ---------- Banner ----------
    print("=" * 80)
    print("OFFLINE STABLE DIFFUSION GENERATION (SD branch)")
    print("=" * 80)
    print(f"[INFO] Project root:        {PROJECT_ROOT}")
    print(f"[INFO] Bag:                 {bag_name}")
    print(f"[INFO] Bag stem:            {bag_stem}")
    print(f"[INFO] Replay dataset:      {replay_dataset_folder}")
    print(f"[INFO] Trajectory:          {trajectory_path}")
    print(f"[INFO] Models base:         {models_base_dir}")
    print(f"[INFO] Output folder:       {output_folder}")
    print(f"[INFO] Device:              {DEVICE}")
    print(f"[INFO] Num parts:           {num_parts}")
    print(f"[INFO] Control start:       {CONTROL_START}")
    print(f"[INFO] Control end:         {CONTROL_END}")
    print(f"[INFO] Guidance scale:      {GUIDANCE_SCALE}")
    print(f"[INFO] Skip existing PNGs:  {SKIP_EXISTING}")
    print("=" * 80)

    # ---------- Output folders ----------
    os.makedirs(output_rgb_folder, exist_ok=True)
    os.makedirs(output_data_folder, exist_ok=True)

    # ---------- Load trajectory and split ----------
    with open(trajectory_path, "r") as f:
        full_trajectory = json.load(f)
    trajectory_chunks = split_trajectory_into_parts(full_trajectory, num_parts)
    print(f"\n[INFO] Trajectory: {len(full_trajectory)} points")
    for i, chunk in enumerate(trajectory_chunks):
        print(f"   Part {i}: {len(chunk)} frames")

    # ---------- Load replay data ----------
    seg_list, inst_list, frame_data, frame_indices = load_replay_data(
        sem_folder=sem_folder,
        inst_folder=inst_folder,
        metadata_path=metadata_path,
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
        out_path = os.path.join(output_rgb_folder, out_filename)

        # ---- Skip if already generated ----
        if SKIP_EXISTING and os.path.exists(out_path):
            n_skipped += 1
            # Still need to keep prev_generated coherent: load existing image
            prev_generated = Image.open(out_path).convert("RGB")
            continue

        # ---- Model selection based on frame location ----
        frame_location = frame_info["location"]
        required_part, traj_distance = select_model_part(
            (frame_location["x"], frame_location["y"]), trajectory_chunks
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
            model_path = os.path.join(models_base_dir, f"part_{required_part}")
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
        "bag_name": bag_stem,
        "num_parts": num_parts,
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
        "output_folder": output_rgb_folder,
    }

    with open(output_info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE.")
    print(f"   Frames generated:  {n_generated}")
    print(f"   Frames skipped:    {n_skipped}")
    print(f"   Total elapsed:     {t_elapsed / 60:.1f} min")
    print(f"   Output folder:     {output_rgb_folder}")
    print(f"   Generation info:   {output_info_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()