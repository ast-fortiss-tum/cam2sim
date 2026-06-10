#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5G_stable_diff_offline_generation_grid.py

Grid search version of 5E_stable_diff_offline_generation.py.

Generates the same replay dataset with MANY different SD control configurations,
one folder per config. This is the script used to explore ControlNet parameters
in the cam2sim research questions (RQ2): "which schedule/guess/seed combo
produces the best frames?"

Each configuration varies:
  - control_start = [seg_start, inst_start, temp_start]
  - control_end   = [seg_end,   inst_end,   temp_end]
  - guess_mode    = True / False (ControlNet "guess" mode)
  - use_fixed_seed= True (seed=FIXED_SEED) / False (random)

Reads from (project root):
    data/processed_dataset/<BAG>/carla_replay_dataset_sd/
        semantic/, instance/, data/all_frame_data.json
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json

Reads SD models from (external drive):
    <SD_ROOT>/<BAG>/SD_Training_Outputs_Split/part_<N>/

Writes to (external drive):
    <SD_ROOT>/<BAG>/sd_grid_search/
        <config_label>/frame_XXXXXX.png
        grid_search_info.json    (summary of all configs)

The output is laid out as SUBFOLDERS (one per config), which matches what
compute_metrics.py expects in its default (non-flat) mode. You can then run:

    python 6_validation/6D_image_quality_metrics.py \
        --gt-folder data/raw_dataset/<BAG>/images \
        --input-folder <SD_ROOT>/<BAG>/sd_grid_search \
        --output-folder <SD_ROOT>/<BAG>/sd_grid_search_METRICS \
        --crop-bottom 45

and then rank the results with 6E_stable_diff_eval_results.py.
"""

import os
import sys


# =======================
# PATH SETUP (must come BEFORE any diffusers/HF import)
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
# =======================

from utils.sd_paths import resolve_sd_root, require_trained_parts

CAM2SIM_SD_ROOT, _ = resolve_sd_root(PROJECT_ROOT)
os.environ["HF_HOME"] = os.path.join(CAM2SIM_SD_ROOT, "huggingface_cache")


# =======================
# STANDARD IMPORTS (safe now, HF_HOME is set)
# =======================

import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# =======================
# LOCAL UTILS IMPORTS
# =======================

from utils.stable_diffusion import (
    load_pipeline_models,
    split_trajectory_into_parts,
    select_model_part,
    load_replay_data,
    generate_image_realtime,
)

# We import NEGATIVE_PROMPT lazily so the script keeps working even on older
# utils versions that don't export it.
try:
    from utils.stable_diffusion import NEGATIVE_PROMPT
except ImportError:
    NEGATIVE_PROMPT = "blurry, distorted, street without street lines"


# =======================
# CONFIG (non bag-dependent)
# =======================

# Bag name (with .bag extension): must match an existing bag from step 1.
DEFAULT_BAG_NAME = "reference_bag.bag"

# Note: bag-dependent paths (REPLAY_DATASET_FOLDER, MODELS_BASE_DIR, ...) are
# built in main() after argparse parses --bag-name.

GUIDANCE_SCALE = 3.0
NUM_INFERENCE_STEPS = 50
FIXED_SEED = 50

DEFAULT_MAX_FRAMES = 10


# =======================
# DEVICE
# =======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# CONFIGURATION GENERATION
# =======================

ControlDict = Dict[str, Any]


def generate_grid_configs() -> List[ControlDict]:
    """
    Build the grid of configurations to evaluate.

    Default: 25 (start, end) schedules x 4 (guess_mode, fixed_seed) combos
    = 100 configurations total.

    To run a smaller experiment, slice the returned list, or use --max-configs.
    """

    # ---- 25 control schedules (start, end) ----
    # [seg, inst, temp] for both start and end (range 0.0 - 1.0)
    grid_schedules = [
        {'start': [0.00, 0.00, 0.00], 'end': [0.33, 0.66, 1.00]},
        {'start': [0.00, 0.00, 0.00], 'end': [0.33, 1.00, 0.33]},
        {'start': [0.00, 0.00, 0.00], 'end': [1.00, 0.33, 0.33]},
        {'start': [0.00, 0.00, 0.00], 'end': [1.00, 1.00, 1.00]},
        {'start': [0.00, 0.00, 0.33], 'end': [1.00, 0.33, 1.00]},
        {'start': [0.00, 0.00, 0.33], 'end': [1.00, 1.00, 0.66]},
        {'start': [0.00, 0.00, 0.66], 'end': [0.33, 0.33, 1.00]},
        {'start': [0.00, 0.00, 0.66], 'end': [0.66, 1.00, 1.00]},
        {'start': [0.00, 0.33, 0.00], 'end': [1.00, 1.00, 0.33]},
        {'start': [0.00, 0.33, 0.33], 'end': [0.33, 1.00, 0.66]},
        {'start': [0.00, 0.33, 0.33], 'end': [1.00, 0.66, 0.66]},
        {'start': [0.00, 0.66, 0.00], 'end': [0.33, 1.00, 0.33]},
        {'start': [0.00, 0.66, 0.00], 'end': [0.33, 1.00, 1.00]},
        {'start': [0.00, 0.66, 0.66], 'end': [0.33, 1.00, 1.00]},
        {'start': [0.00, 0.66, 0.66], 'end': [1.00, 1.00, 1.00]},
        {'start': [0.33, 0.00, 0.33], 'end': [0.66, 1.00, 0.66]},
        {'start': [0.33, 0.33, 0.00], 'end': [0.66, 0.66, 1.00]},
        {'start': [0.33, 0.66, 0.00], 'end': [0.66, 1.00, 0.66]},
        {'start': [0.33, 0.66, 0.00], 'end': [1.00, 1.00, 1.00]},
        {'start': [0.66, 0.00, 0.00], 'end': [1.00, 0.33, 1.00]},
        {'start': [0.66, 0.00, 0.00], 'end': [1.00, 0.66, 0.33]},
        {'start': [0.66, 0.00, 0.00], 'end': [1.00, 1.00, 1.00]},
        {'start': [0.66, 0.00, 0.66], 'end': [1.00, 0.33, 1.00]},
        {'start': [0.66, 0.00, 0.66], 'end': [1.00, 1.00, 1.00]},
        {'start': [0.66, 0.66, 0.00], 'end': [1.00, 1.00, 0.33]},
    ]

    # ---- 4 (guess_mode, fixed_seed) conditions ----
    conditions = [
        {'guess_mode': True,  'use_fixed_seed': True,  'label': 'A_guessT_seedFixed'},
        {'guess_mode': True,  'use_fixed_seed': False, 'label': 'B_guessT_seedRand'},
        {'guess_mode': False, 'use_fixed_seed': True,  'label': 'C_guessF_seedFixed'},
        {'guess_mode': False, 'use_fixed_seed': False, 'label': 'D_guessF_seedRand'},
    ]

    all_configs: List[ControlDict] = []
    for cond in conditions:
        for i, schedule in enumerate(grid_schedules):
            label = (
                f"{cond['label']}_cfg{i:02d}"
                f"__START_seg_{schedule['start'][0]}"
                f"_inst_{schedule['start'][1]}"
                f"_temp_{schedule['start'][2]}"
                f"__END_seg_{schedule['end'][0]}"
                f"_inst_{schedule['end'][1]}"
                f"_temp_{schedule['end'][2]}"
            )
            all_configs.append({
                'start': schedule['start'][:],
                'end': schedule['end'][:],
                'guess_mode': cond['guess_mode'],
                'use_fixed_seed': cond['use_fixed_seed'],
                'label': label,
            })

    return all_configs


# =======================
# CONFIG COMPLETENESS CHECK (for idempotency)
# =======================

def is_config_complete(config_dir, expected_frame_indices):
    """
    Returns True iff config_dir already contains a PNG for every frame_id
    in expected_frame_indices. Lets us skip already-done configs cleanly.
    """
    if not os.path.isdir(config_dir):
        return False
    for fid in expected_frame_indices:
        if not os.path.exists(os.path.join(config_dir, f"frame_{fid:06d}.png")):
            return False
    return True


# =======================
# MAIN
# =======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search over SD ControlNet schedules for cam2sim."
    )
    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help="Bag filename including .bag extension "
             "(default: env BAG_NAME or 'reference_bag.bag').",
    )
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help=f"How many replay frames to use per config (default: {DEFAULT_MAX_FRAMES}). "
             "Use a small number (~100) to iterate quickly, then scale up."
    )
    parser.add_argument(
        "--max-configs", type=int, default=None,
        help="Only run the first N grid configurations. Default: all 100."
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Where to write the grid search subfolders. "
             "Default: <SD_ROOT>/<bag>/sd_grid_search"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run configurations even if their output folder already looks complete."
    )
    return parser.parse_args()


def main():
    args = parse_args()

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

    bag_sd_dir = os.path.join(CAM2SIM_SD_ROOT, bag_stem)
    models_base_dir = os.path.join(bag_sd_dir, "SD_Training_Outputs_Split")

    # Output root: per-bag on external drive by default, or user override
    if args.output_root is None:
        output_root = os.path.join(bag_sd_dir, "sd_grid_search")
    else:
        output_root = args.output_root

    num_parts = require_trained_parts(models_base_dir, bag_name)

    print("=" * 80)
    print("STABLE DIFFUSION GRID SEARCH (cam2sim, SD branch)")
    print("=" * 80)
    print(f"[INFO] Project root:        {PROJECT_ROOT}")
    print(f"[INFO] Bag:                 {bag_name}")
    print(f"[INFO] Bag stem:            {bag_stem}")
    print(f"[INFO] Replay dataset:      {replay_dataset_folder}")
    print(f"[INFO] Trajectory:          {trajectory_path}")
    print(f"[INFO] Models base:         {models_base_dir}")
    print(f"[INFO] Output root:         {output_root}")
    print(f"[INFO] Device:              {DEVICE}")
    print(f"[INFO] Num parts:           {num_parts}")
    print(f"[INFO] Guidance scale:      {GUIDANCE_SCALE}")
    print(f"[INFO] Max frames / config: {args.max_frames}")
    print(f"[INFO] Force re-run:        {args.force}")
    print("=" * 80)

    # ---------- Sanity checks on remaining inputs ----------
    if not os.path.exists(replay_dataset_folder):
        raise FileNotFoundError(
            f"Replay dataset not found: {replay_dataset_folder}\n"
            f"Run 5A_sd_trajectory_only_carla.py --bag-name {bag_name} first."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

    # ---------- Output root ----------
    os.makedirs(output_root, exist_ok=True)

    # ---------- Load trajectory + chunks ----------
    with open(trajectory_path, "r") as f:
        full_trajectory = json.load(f)
    trajectory_chunks = split_trajectory_into_parts(full_trajectory, num_parts)
    print(f"\n[INFO] Trajectory: {len(full_trajectory)} points")
    for i, chunk in enumerate(trajectory_chunks):
        print(f"   Part {i}: {len(chunk)} frames")

    # ---------- Load replay data once (reused across all configs) ----------
    seg_list, inst_list, frame_data, frame_indices = load_replay_data(
        sem_folder=sem_folder,
        inst_folder=inst_folder,
        metadata_path=metadata_path,
        max_frames=args.max_frames,
    )
    if not seg_list:
        raise RuntimeError("No frames to process.")
    total_frames = len(seg_list)
    print(f"\n[INFO] Frames per config: {total_frames}")

    # ---------- Build config grid ----------
    all_configs = generate_grid_configs()
    if args.max_configs is not None:
        all_configs = all_configs[:args.max_configs]
    print(f"[INFO] Total configurations to run: {len(all_configs)}")
    print("=" * 80)

    # ---------- Iterate configs ----------
    grid_info = {
        "bag_name": bag_stem,
        "num_parts": num_parts,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "fixed_seed": FIXED_SEED,
        "frames_per_config": total_frames,
        "configurations": [],
    }

    grid_t_start = time.time()
    n_done = 0
    n_skipped = 0

    for config_idx, config in enumerate(all_configs):
        label = config["label"]
        config_dir = os.path.join(output_root, label)

        # Idempotency
        if not args.force and is_config_complete(config_dir, frame_indices):
            print(f"\n[{config_idx+1}/{len(all_configs)}] SKIP (already complete): {label}")
            n_skipped += 1
            grid_info["configurations"].append({**config, "status": "skipped_complete"})
            continue

        os.makedirs(config_dir, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"CONFIG {config_idx + 1}/{len(all_configs)}: {label}")
        print(f"  start={config['start']}  end={config['end']}")
        print(f"  guess_mode={config['guess_mode']}  use_fixed_seed={config['use_fixed_seed']}")
        print(f"  output: {config_dir}")
        print("#" * 80)

        # Pipeline state (reset per-config so each starts cleanly)
        pipe = None
        model_data = None
        current_model_part = None
        prev_generated = None
        config_t_start = time.time()
        n_generated = 0

        for i, frame_id in enumerate(frame_indices):
            seg = seg_list[i]
            inst = inst_list[i]
            frame_info = frame_data[i]

            out_path = os.path.join(config_dir, f"frame_{frame_id:06d}.png")
            if os.path.exists(out_path) and not args.force:
                # Per-frame idempotency: still update prev_generated for temporal coherence
                prev_generated = Image.open(out_path).convert("RGB")
                continue

            # ---- Model selection ----
            frame_location = frame_info["location"]
            required_part, traj_distance = select_model_part(
                (frame_location["x"], frame_location["y"]), trajectory_chunks
            )

            if required_part != current_model_part:
                print(f"\n  MODEL SWITCH: Part {current_model_part} -> Part {required_part} "
                      f"(frame {frame_id}, dist {traj_distance:.2f}m)")
                previous_model_last_image = prev_generated
                if pipe is not None:
                    del pipe
                    del model_data
                    torch.cuda.empty_cache()
                model_path = os.path.join(models_base_dir, f"part_{required_part}")
                pipe, model_data = load_pipeline_models(model_path, DEVICE)
                current_model_part = required_part
                prev_generated = previous_model_last_image

            # ---- Prompt ----
            prompt = frame_info.get("caption", "")
            if not prompt:
                prompt = (
                    f"pos x: {frame_location['x']:.2f}, "
                    f"y: {frame_location['y']:.2f}"
                )

            prev_img = None if i == 0 else prev_generated

            # ---- Generate with this config ----
            out_img = generate_image_realtime(
                pipe=pipe,
                seg_image=seg,
                inst_image=inst,
                model_data=model_data,
                prev_image=prev_img,
                prompt=prompt,
                guidance=GUIDANCE_SCALE,
                control_start=config["start"],
                control_end=config["end"],
                guess_mode=config["guess_mode"],
                use_fixed_seed=config["use_fixed_seed"],
                fixed_seed=FIXED_SEED,
            )

            out_img.save(out_path)
            prev_generated = out_img
            n_generated += 1

            if (i + 1) % 25 == 0:
                elapsed = time.time() - config_t_start
                avg = elapsed / max(n_generated, 1)
                eta = avg * (total_frames - (i + 1))
                print(f"  [{i + 1}/{total_frames}] avg {avg:.2f}s/frame  ETA {eta/60:.1f} min")

        # ---- Free pipeline before next config ----
        if pipe is not None:
            del pipe
            del model_data
            torch.cuda.empty_cache()

        elapsed = time.time() - config_t_start
        print(f"  Config done in {elapsed/60:.1f} min  ({n_generated} new frames)")
        n_done += 1

        grid_info["configurations"].append({
            **config,
            "status": "done",
            "frames_generated": n_generated,
            "elapsed_seconds": round(elapsed, 2),
            "output_dir": config_dir,
        })

        # Periodically flush the summary JSON in case the run is killed
        with open(os.path.join(output_root, "grid_search_info.json"), "w") as f:
            json.dump(grid_info, f, indent=2)

    grid_elapsed = time.time() - grid_t_start
    grid_info["total_elapsed_seconds"] = round(grid_elapsed, 2)
    grid_info["configs_done"] = n_done
    grid_info["configs_skipped"] = n_skipped

    with open(os.path.join(output_root, "grid_search_info.json"), "w") as f:
        json.dump(grid_info, f, indent=2)

    print("\n" + "=" * 80)
    print("GRID SEARCH DONE.")
    print(f"  Configurations done:    {n_done}")
    print(f"  Configurations skipped: {n_skipped}")
    print(f"  Total time:             {grid_elapsed/60:.1f} min")
    print(f"  Output root:            {output_root}")
    print("=" * 80)




if __name__ == "__main__":
    main()