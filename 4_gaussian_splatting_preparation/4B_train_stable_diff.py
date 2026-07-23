#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4B_train_stable_diff.py

Train per-part LoRA and ControlNet models for the Stable Diffusion branch.

Reads from:
    data/data_for_stable_diffusion/<BAG>/hf_binary/

Writes to:
    data/stable_diff_models/ by default, or --output-root when supplied.

Parameters:
    --bag-name <BAG>.bag
        Bag filename including .bag extension.
    --num-parts <N>
        Number of geographical training parts. Default: 3.
    --output-root <PATH>
        Optional model and cache storage root.
    --force
        Retrain models even when completion markers exist.

Usage:
    python 4_gaussian_splatting_preparation/4B_train_stable_diff.py --bag-name snowy.bag --num-parts 2
    python 4_gaussian_splatting_preparation/4B_train_stable_diff.py --bag-name sunny.bag --num-parts 3
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import snapshot_download


# ============================================================
# STEP 0: Arguments & Setup
# ============================================================
DEFAULT_BAG_NAME = "reference_bag.bag"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension "
         "(default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument(
    "--num-parts",
    type=int,
    default=3,
    help="Number of Stable Diffusion training parts/splits. Default: 3.",
)
parser.add_argument("--output-root", type=str, default=None,
                    help="Where to store trained SD models. "
                         "First time: optional (default: data/stable_diff_models). "
                         "Saved to data/.sd_root so you don't need to repeat it.")
parser.add_argument("--force", action="store_true",
                    help="Force retrain even if model markers exist")
args = parser.parse_args()

if args.num_parts <= 0:
    parser.error("--num-parts must be greater than zero")

# Project root is the parent of this script's folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)

# --- CONFIGURATION ---
BAG_NAME = args.bag_name                # e.g. "reference_bag.bag"
BAG_STEM = Path(BAG_NAME).stem          # e.g. "reference_bag"

NUM_PARTS = args.num_parts              
RESOLUTION = 512
PRETRAINED_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Base ControlNet (pretrained on Cityscapes, fine-tuned per split below)
CONTROLNET_HF_REPO = "doguilmak/cityscapes-controlnet-sd15"
CONTROLNET_HF_SUBFOLDER = "full_pipeline/controlnet"

# --- INPUT (inside project root, bag-dependent) ---
LOCAL_BINARY_PATH = os.path.join(
    PROJECT_ROOT,
    "data", "data_for_stable_diffusion", BAG_STEM, "hf_binary",
)

from utils.sd_paths import resolve_sd_root

CAM2SIM_SD_ROOT, sd_action = resolve_sd_root(PROJECT_ROOT, override=args.output_root)
print(f"[INFO] SD storage root: {CAM2SIM_SD_ROOT} (source: {sd_action})")

# Shared across bags
DIFFUSERS_DIR = os.path.join(CAM2SIM_SD_ROOT, "diffusers_repo")
CONTROLNET_DIR = os.path.join(CAM2SIM_SD_ROOT, "cityscapes-controlnet")
HF_CACHE_DIR = os.path.join(CAM2SIM_SD_ROOT, "huggingface_cache")

# Per-bag (training outputs and parquet shards)
BAG_SD_DIR = os.path.join(CAM2SIM_SD_ROOT, BAG_STEM)
OUTPUT_BASE = os.path.join(BAG_SD_DIR, "SD_Training_Outputs_Split")
DATA_CACHE_DIR = os.path.join(BAG_SD_DIR, "local_data_shards")

# Path that train_controlnet.py will load with ControlNetModel.from_pretrained()
# It needs to point to a folder containing config.json + diffusion_pytorch_model.safetensors
CONTROLNET_MODEL_PATH = os.path.join(CONTROLNET_DIR, CONTROLNET_HF_SUBFOLDER)


# --- ENVIRONMENT VARIABLES ---
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDNN_BENCHMARK"] = "True"

# Create directories
os.makedirs(CAM2SIM_SD_ROOT, exist_ok=True)
os.makedirs(BAG_SD_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)


# --- BANNER ---
print("=" * 60)
print("STABLE DIFFUSION TRAINING (LoRA + 3 ControlNets per split)")
print("=" * 60)
print(f"[INFO] Bag:                  {BAG_NAME}")
print(f"[INFO] Bag stem:             {BAG_STEM}")
print(f"[INFO] Project root:         {PROJECT_ROOT}")
print(f"[INFO] Local binary input:   {LOCAL_BINARY_PATH}")
print(f"[INFO] External drive root:  {CAM2SIM_SD_ROOT}")
print(f"[INFO] Bag SD dir:           {BAG_SD_DIR}")
print(f"[INFO] Output base:          {OUTPUT_BASE}")
print(f"[INFO] HF cache:             {HF_CACHE_DIR}")
print(f"[INFO] NUM_PARTS:            {NUM_PARTS}")
print(f"[INFO] Resolution:           {RESOLUTION}")
print(f"[INFO] Force retrain:        {args.force}")
print("=" * 60)


# ============================================================
# SKIP-CHECK HELPER
# ============================================================
def is_trained(output_dir, marker_files):
    """
    Check if a model is already trained by looking for marker files.
    Returns True if ANY of the marker files exist in output_dir.
    """
    if args.force:
        return False
    for marker in marker_files:
        if os.path.exists(os.path.join(output_dir, marker)):
            return True
    return False


# Marker files that indicate training is complete
LORA_MARKERS = ["pytorch_lora_weights.safetensors"]
CONTROLNET_MARKERS = [
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
]


# ============================================================
# STEP 1: Dependencies
# ============================================================
print("\n>>> CHECKING DEPENDENCIES...")

# 1.1 Clone diffusers repo (just Python source code, no LFS needed)
if not os.path.exists(DIFFUSERS_DIR):
    print(f"Cloning diffusers repository into {DIFFUSERS_DIR}...")
    subprocess.run(
        ["git", "clone", "https://github.com/huggingface/diffusers.git", DIFFUSERS_DIR],
        check=True,
    )
    subprocess.run(["git", "checkout", "v0.33.1"], cwd=DIFFUSERS_DIR, check=True)
else:
    print(f"diffusers repo already present: {DIFFUSERS_DIR}")

# 1.2 Download base ControlNet weights via huggingface_hub
# This avoids the git-lfs dependency. snapshot_download fetches actual binary
# blobs through the HF HTTP API and stores them as a normal directory tree
# under CONTROLNET_DIR. allow_patterns restricts the download to the subfolder
# we actually need (~1.4 GB) instead of the full repo.
def _has_real_weights(path):
    """True if the controlnet folder has a non-pointer safetensors file."""
    safetensors = os.path.join(path, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(safetensors):
        return False
    # LFS pointer files are < 1 KB; real weights are ~1.4 GB
    return os.path.getsize(safetensors) > 1_000_000

if _has_real_weights(CONTROLNET_MODEL_PATH):
    print(f"ControlNet weights already present at {CONTROLNET_MODEL_PATH}")
else:
    print(f"Downloading {CONTROLNET_HF_REPO} into {CONTROLNET_DIR}...")
    snapshot_download(
        repo_id=CONTROLNET_HF_REPO,
        local_dir=CONTROLNET_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=[
            f"{CONTROLNET_HF_SUBFOLDER}/*",
        ],
    )
    if not _has_real_weights(CONTROLNET_MODEL_PATH):
        raise RuntimeError(
            f"Download finished but no real weights found at "
            f"{CONTROLNET_MODEL_PATH}. Check the HF repo structure."
        )
    print(f"ControlNet downloaded successfully: {CONTROLNET_MODEL_PATH}")


# ============================================================
# STEP 2: LOAD & SPLIT DATASET (Parquet shards)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: PREPARING & SPLITTING DATASET")
print("=" * 60)

if not os.path.exists(LOCAL_BINARY_PATH):
    raise FileNotFoundError(
        f"Local dataset not found: {LOCAL_BINARY_PATH}\n"
        f"Run 2_process_datasets/2H_prepare_dataset_for_stable_diffusion.py "
        f"--bag-name {BAG_NAME} first."
    )

print(f"Loading local dataset: {LOCAL_BINARY_PATH}...")
full_ds = load_from_disk(LOCAL_BINARY_PATH)

# ---------- VALIDATE ----------
print("\n>>> VALIDATING DATASET...")
required_columns = ["image", "text", "segmentation", "instance", "previous"]
missing_columns = [c for c in required_columns if c not in full_ds.column_names]
if missing_columns:
    raise ValueError(f"Dataset missing required columns: {missing_columns}")
print(f"All required columns present: {required_columns}")

sample_text = full_ds[0]["text"]
print(f"First coordinate caption: {sample_text}")
if not sample_text:
    raise ValueError("Text column is empty! Coordinate captions are required.")

# ---------- SPLIT ----------
total_frames = len(full_ds)
chunk_size = total_frames // NUM_PARTS
print(f"\nTotal frames: {total_frames} | "
      f"Splitting into {NUM_PARTS} parts of ~{chunk_size} frames each.")

local_dataset_paths = []
for i in range(NUM_PARTS):
    start = i * chunk_size
    end = total_frames if i == NUM_PARTS - 1 else (i + 1) * chunk_size

    part_dir = os.path.join(DATA_CACHE_DIR, f"part_{i}")
    local_dataset_paths.append(part_dir)

    # Subfolder layout expected by load_dataset for parquet
    data_dir = os.path.join(part_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    parquet_file = os.path.join(data_dir, "train-00000-of-00001.parquet")
    if not os.path.exists(parquet_file):
        print(f"   > Saving Part {i} (Frames {start}-{end}) as Parquet...")
        ds_shard = full_ds.select(range(start, end))
        ds_shard.to_parquet(parquet_file)
    else:
        print(f"   > Part {i} parquet found, using cached version.")


# ============================================================
# STEP 3: TRAINING LOOP (per part)
# ============================================================
for i in range(NUM_PARTS):
    print("\n" + "#" * 60)
    print(f"STARTING TRAINING LOOP FOR PART {i} / {NUM_PARTS - 1}")
    print("#" * 60)

    CURRENT_DATA_DIR = local_dataset_paths[i]

    PART_OUT_DIR = os.path.join(OUTPUT_BASE, f"part_{i}")
    LORA_OUT = os.path.join(PART_OUT_DIR, "stable_diffusion")
    CN_SEG_OUT = os.path.join(PART_OUT_DIR, "controlnet_segmentation")
    CN_INST_OUT = os.path.join(PART_OUT_DIR, "controlnet_instance")
    CN_TEMP_OUT = os.path.join(PART_OUT_DIR, "controlnet_tempconsistency")

    # --------------------------------------------------------
    # 3.1 TRAIN LoRA
    # --------------------------------------------------------
    if is_trained(LORA_OUT, LORA_MARKERS):
        print(f"\n>>> [Part {i}] LoRA already trained, skipping.")
    else:
        print(f"\n>>> [Part {i}] Training LoRA...")
        lora_script = os.path.join(
            DIFFUSERS_DIR, "examples/text_to_image/train_text_to_image_lora.py"
        )
        lora_cmd = [
            "accelerate", "launch", "--mixed_precision=bf16",
            lora_script,
            f"--pretrained_model_name_or_path={PRETRAINED_SD}",
            f"--dataset_name={CURRENT_DATA_DIR}",
            "--dataloader_num_workers=4",
            f"--resolution={RESOLUTION}",
            "--train_batch_size=16",
            "--gradient_accumulation_steps=4",
            "--num_train_epochs=10",
            "--learning_rate=1e-4",
            "--max_grad_norm=1",
            "--lr_scheduler=cosine",
            "--lr_warmup_steps=0",
            f"--output_dir={LORA_OUT}",
            "--checkpointing_steps=500",
            "--caption_column=text",
            "--gradient_checkpointing",
        ]
        subprocess.run(lora_cmd, check=True)

    # --------------------------------------------------------
    # 3.2 TRAIN ControlNet Segmentation
    # --------------------------------------------------------
    cn_script = os.path.join(
        DIFFUSERS_DIR, "examples/controlnet/train_controlnet.py"
    )

    if is_trained(CN_SEG_OUT, CONTROLNET_MARKERS):
        print(f"\n>>> [Part {i}] Segmentation ControlNet already trained, skipping.")
    else:
        print(f"\n>>> [Part {i}] Training Segmentation ControlNet...")
        cn_seg_cmd = [
            "accelerate", "launch", "--mixed_precision=bf16",
            cn_script,
            f"--pretrained_model_name_or_path={PRETRAINED_SD}",
            f"--dataset_name={CURRENT_DATA_DIR}",
            f"--output_dir={CN_SEG_OUT}",
            f"--controlnet_model_name_or_path={CONTROLNET_MODEL_PATH}",
            "--conditioning_image_column=segmentation",
            "--image_column=image",
            "--caption_column=text",
            f"--resolution={RESOLUTION}",
            "--learning_rate=1e-5",
            "--train_batch_size=8",
            "--gradient_accumulation_steps=4",
            "--num_train_epochs=10",
            "--checkpointing_steps=1000",
        ]
        subprocess.run(cn_seg_cmd, check=True)

    # --------------------------------------------------------
    # 3.3 TRAIN ControlNet Instance
    # --------------------------------------------------------
    if is_trained(CN_INST_OUT, CONTROLNET_MARKERS):
        print(f"\n>>> [Part {i}] Instance ControlNet already trained, skipping.")
    else:
        print(f"\n>>> [Part {i}] Training Instance ControlNet...")
        cn_inst_cmd = [
            "accelerate", "launch", "--mixed_precision=bf16",
            cn_script,
            f"--pretrained_model_name_or_path={PRETRAINED_SD}",
            f"--dataset_name={CURRENT_DATA_DIR}",
            f"--output_dir={CN_INST_OUT}",
            "--conditioning_image_column=instance",
            "--image_column=image",
            "--caption_column=text",
            f"--resolution={RESOLUTION}",
            "--learning_rate=1e-5",
            "--train_batch_size=8",
            "--gradient_accumulation_steps=4",
            "--num_train_epochs=10",
            "--checkpointing_steps=1000",
        ]
        subprocess.run(cn_inst_cmd, check=True)

    # --------------------------------------------------------
    # 3.4 TRAIN ControlNet Temporal
    # --------------------------------------------------------
    if is_trained(CN_TEMP_OUT, CONTROLNET_MARKERS):
        print(f"\n>>> [Part {i}] Temporal ControlNet already trained, skipping.")
    else:
        print(f"\n>>> [Part {i}] Training Temporal ControlNet...")
        cn_temp_cmd = [
            "accelerate", "launch", "--mixed_precision=bf16",
            cn_script,
            f"--pretrained_model_name_or_path={PRETRAINED_SD}",
            f"--dataset_name={CURRENT_DATA_DIR}",
            f"--output_dir={CN_TEMP_OUT}",
            "--controlnet_model_name_or_path=CiaraRowles/TemporalNet",
            "--conditioning_image_column=previous",
            "--image_column=image",
            "--caption_column=text",
            f"--resolution={RESOLUTION}",
            "--learning_rate=1e-5",
            "--train_batch_size=8",
            "--gradient_accumulation_steps=4",
            "--num_train_epochs=10",
            "--checkpointing_steps=1000",
        ]
        subprocess.run(cn_temp_cmd, check=True)

    # --------------------------------------------------------
    # 3.5 CLEANUP intermediate checkpoints
    # --------------------------------------------------------
    print(f"\n>>> [Part {i}] Cleaning up intermediate checkpoints...")
    for folder in [LORA_OUT, CN_SEG_OUT, CN_INST_OUT, CN_TEMP_OUT]:
        if os.path.exists(folder):
            subprocess.run(
                f"mv {folder}/checkpoint-*/* {folder}/ 2>/dev/null",
                shell=True,
            )
            subprocess.run(f"rm -rf {folder}/checkpoint-*", shell=True)

    # --------------------------------------------------------
    # 3.6 SAVE PART CONFIG
    # --------------------------------------------------------
    print(f"\n>>> [Part {i}] Saving config.json...")
    config = {
        "part_index": i,
        "bag_name": BAG_STEM,
        "controlnet_segmentation": "controlnet_segmentation",
        "controlnet_instance": "controlnet_instance",
        "controlnet_tempconsistency": "controlnet_tempconsistency",
        "stable_diffusion_model": PRETRAINED_SD,
        "lora_weights": "stable_diffusion/pytorch_lora_weights.safetensors",
        "size": {"x": RESOLUTION, "y": RESOLUTION},
        "camera": {
            "position": {"x": 1.1, "y": 0.0, "z": 1.35},
            "fov": 55,
            "fps": 24,
            "pitch": 0.0,
        },
        "coordinates": "true",
    }
    config_path = os.path.join(PART_OUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Part {i} finished locally.")

print("\n" + "=" * 60)
print("ALL PARTS TRAINED SUCCESSFULLY")
print("=" * 60)
print(f"Outputs are in: {OUTPUT_BASE}")