#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare images and overlapping splits for Gaussian Splatting,
WITH sky masks (thesis-faithful).

Differences vs the original prepare_gs_splits_thesis.py:
  - Generates sky masks using SegFormer (nvidia/segformer-b1-finetuned-cityscapes-1024-1024)
  - Saves them in _tmp_sky_masks_gs_<skip>/ and sky_masks_gs_split_N_<skip>/
  - Convention (matches thesis):
        255 (white) = valid pixel (keep during training/COLMAP)
        0   (black) = masked-out pixel (= sky, ignore)

Everything else is IDENTICAL to the thesis-replicating script:
  - FRAME_SKIP = 2
  - NUM_SPLITS = 3
  - OVERLAP_FRAMES = 100 (in original frame_id units)
  - same hardcoded paths, same crop, same split partitioning logic
"""

import os
import sys
import shutil

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


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
# HARDCODED CONFIGURATION
# =======================

BAG_NAME = "reference_bag"

SOURCE_IMAGES_FOLDER = os.path.join(
    PROJECT_ROOT, "data", "raw_dataset", BAG_NAME, "images",
)
SOURCE_POSITIONS_FILE = os.path.join(
    PROJECT_ROOT, "data", "raw_dataset", BAG_NAME, "images_positions.txt",
)
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "data", "data_for_gaussian_splatting", BAG_NAME,
)

CROP_BOTTOM = 45

# THESIS PARAMETERS (do NOT change without rerunning COLMAP)
FRAME_SKIP = 2
NUM_SPLITS = 3
OVERLAP_FRAMES = 100   # in ORIGINAL frame_id units

OVERWRITE_EXISTING = True

# Sky-mask model (same as thesis)
SKY_MASK_MODEL_NAME = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"


# =======================
# SKY MASK
# =======================

def load_sky_model():
    """Load SegFormer model for sky segmentation."""
    print(f"[INFO] Loading SegFormer: {SKY_MASK_MODEL_NAME}...")
    processor = SegformerImageProcessor.from_pretrained(SKY_MASK_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(SKY_MASK_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Sky model loaded on {device}")

    # Find sky class ID
    id2label = model.config.id2label
    sky_id = None
    for class_id, label in id2label.items():
        if label.lower() == "sky":
            sky_id = class_id
            break

    if sky_id is None:
        raise RuntimeError("Could not find 'sky' class in model labels!")
    print(f"[INFO] Sky class ID: {sky_id}")

    return processor, model, device, sky_id


def generate_sky_mask(image, processor, model, device, sky_id):
    """
    Generate a binary sky mask for a single PIL image (cropped).

    Convention:
        255 = valid pixel  (keep)
        0   = sky pixel    (mask out)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    mask = torch.ones_like(pred_seg, dtype=torch.uint8) * 255
    mask[pred_seg == sky_id] = 0
    return Image.fromarray(mask.cpu().numpy())


# =======================
# HELPERS
# =======================

def ensure_input_paths():
    if not os.path.isdir(SOURCE_IMAGES_FOLDER):
        raise FileNotFoundError(
            f"Source image folder not found: {SOURCE_IMAGES_FOLDER}"
        )
    if not os.path.exists(SOURCE_POSITIONS_FILE):
        raise FileNotFoundError(
            f"Source positions file not found: {SOURCE_POSITIONS_FILE}"
        )
    os.makedirs(OUTPUT_ROOT, exist_ok=True)


def load_position_lines(path):
    header_lines = []
    data_lines = []
    with open(path, "r") as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(raw)
            else:
                data_lines.append(stripped)
    if not data_lines:
        raise RuntimeError(f"No valid data lines found in: {path}")
    return header_lines, data_lines


def get_image_filename_from_position_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        raise RuntimeError(f"Invalid position line: {line}")
    return parts[-1]


def link_or_copy(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file missing: {src}")
    if os.path.exists(dst):
        if OVERWRITE_EXISTING:
            os.remove(dst)
        else:
            return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split_positions_file(path, header_lines, split_frames, data_lines):
    with open(path, "w") as f:
        if header_lines:
            for header in header_lines:
                f.write(header + "\n")
        else:
            f.write(
                "# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, "
                "Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile\n"
            )
        for original_index, _filename in split_frames:
            f.write(data_lines[original_index] + "\n")


def create_empty_colmap_split_folders():
    colmap_root = os.path.join(OUTPUT_ROOT, "colmap")
    for split_index in range(1, NUM_SPLITS + 1):
        split_sparse_dir = os.path.join(
            colmap_root, f"split_{split_index}", "sparse", "0",
        )
        os.makedirs(split_sparse_dir, exist_ok=True)
    print(f"[INFO] Empty COLMAP split folders created in: {colmap_root}")


# =======================
# MAIN PROCESSING
# =======================

def process_frames():
    skip_label = f"1_of_{FRAME_SKIP}"

    tmp_image_folder = os.path.join(
        OUTPUT_ROOT, f"_tmp_images_gs_{skip_label}",
    )
    tmp_mask_folder = os.path.join(
        OUTPUT_ROOT, f"_tmp_sky_masks_gs_{skip_label}",
    )
    os.makedirs(tmp_image_folder, exist_ok=True)
    os.makedirs(tmp_mask_folder, exist_ok=True)

    header_lines, data_lines = load_position_lines(SOURCE_POSITIONS_FILE)
    total_frames = len(data_lines)

    # Subsampled indices (in original-frame units)
    indices = list(range(0, total_frames, FRAME_SKIP))

    print("=" * 80)
    print("GAUSSIAN SPLATTING IMAGE PREPARATION (thesis-replicating, WITH sky masks)")
    print("=" * 80)
    print(f"[INFO] Project root:            {PROJECT_ROOT}")
    print(f"[INFO] Bag name:                {BAG_NAME}")
    print(f"[INFO] Source images:           {SOURCE_IMAGES_FOLDER}")
    print(f"[INFO] Source positions:        {SOURCE_POSITIONS_FILE}")
    print(f"[INFO] Output root:             {OUTPUT_ROOT}")
    print(f"[INFO] Crop bottom:             {CROP_BOTTOM}")
    print(f"[INFO] Frame skip:              {FRAME_SKIP}")
    print(f"[INFO] Split count:             {NUM_SPLITS}")
    print(f"[INFO] Overlap frames (orig):   {OVERLAP_FRAMES}")
    print(f"[INFO] Total source frames:     {total_frames}")
    print(f"[INFO] Subsampled frame count:  {len(indices)}")
    print(f"[INFO] Sky mask model:          {SKY_MASK_MODEL_NAME}")
    print("=" * 80)

    # Load sky model BEFORE the loop
    processor, model, device, sky_id = load_sky_model()

    filenames_by_index = {}

    print("\n[INFO] Processing frames: crop + sky masks")
    for original_index in tqdm(indices, desc="Processing"):
        line = data_lines[original_index]
        filename = get_image_filename_from_position_line(line)

        input_path = os.path.join(SOURCE_IMAGES_FOLDER, filename)
        output_image_path = os.path.join(tmp_image_folder, filename)
        output_mask_path = os.path.join(tmp_mask_folder, filename)

        if not os.path.exists(input_path):
            print(f"[WARN] Image not found: {input_path}")
            continue

        # Skip work if both already exist and we don't overwrite
        if (not OVERWRITE_EXISTING
                and os.path.exists(output_image_path)
                and os.path.exists(output_mask_path)):
            filenames_by_index[original_index] = filename
            continue

        with Image.open(input_path) as img:
            width, height = img.size
            crop_height = height - CROP_BOTTOM
            if crop_height <= 0:
                raise RuntimeError(
                    f"CROP_BOTTOM={CROP_BOTTOM} too large for image: {input_path}"
                )
            crop_box = (0, 0, width, crop_height)
            cropped_img = img.crop(crop_box)
            cropped_img.save(output_image_path)

            # Generate sky mask on cropped image
            mask_img = generate_sky_mask(
                cropped_img.convert("RGB"),
                processor, model, device, sky_id,
            )
            mask_img.save(output_mask_path)

        filenames_by_index[original_index] = filename

    processed = sorted(filenames_by_index.items())
    num_processed = len(processed)

    print(f"\n[INFO] Processed frames: {num_processed}")
    print(f"[INFO] Temporary cropped images: {tmp_image_folder}")
    print(f"[INFO] Temporary sky masks:      {tmp_mask_folder}")

    if num_processed == 0:
        raise RuntimeError("No frames were processed.")

    # =======================
    # SPLIT CREATION
    # =======================
    split_step_orig = total_frames // NUM_SPLITS
    if split_step_orig <= 0:
        raise RuntimeError(
            f"NUM_SPLITS={NUM_SPLITS} is too large for {total_frames} total frames."
        )

    splits_orig = []
    for split_index in range(NUM_SPLITS):
        start_orig = split_index * split_step_orig
        if split_index < NUM_SPLITS - 1:
            end_orig = (split_index + 1) * split_step_orig + OVERLAP_FRAMES - 1
        else:
            end_orig = total_frames - 1
        end_orig = min(end_orig, total_frames - 1)
        splits_orig.append((start_orig, end_orig))

    print("\n[INFO] Split configuration:")
    print(f"       Overlap original frames:    {OVERLAP_FRAMES}")
    print(f"       Step (original frames):     {split_step_orig}")
    print(f"       Mode:                       overlap forward (in original-frame units)")

    for split_index, (start_orig, end_orig) in enumerate(splits_orig, start=1):
        split_frames = [
            (oi, fn) for (oi, fn) in processed
            if start_orig <= oi <= end_orig
        ]

        split_image_dir = os.path.join(
            OUTPUT_ROOT, f"images_gs_split_{split_index}_{skip_label}",
        )
        split_mask_dir = os.path.join(
            OUTPUT_ROOT, f"sky_masks_gs_split_{split_index}_{skip_label}",
        )
        split_positions_path = os.path.join(
            OUTPUT_ROOT, f"frame_positions_split_{split_index}_{skip_label}.txt",
        )
        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        for original_index, filename in split_frames:
            src_img = os.path.join(tmp_image_folder, filename)
            dst_img = os.path.join(split_image_dir, filename)
            link_or_copy(src_img, dst_img)

            src_mask = os.path.join(tmp_mask_folder, filename)
            dst_mask = os.path.join(split_mask_dir, filename)
            link_or_copy(src_mask, dst_mask)

        write_split_positions_file(
            split_positions_path,
            header_lines,
            split_frames,
            data_lines,
        )

        actual_first_orig = split_frames[0][0] if split_frames else None
        actual_last_orig  = split_frames[-1][0] if split_frames else None

        print(f"\n[INFO] Split {split_index}:")
        print(f"       Frames in split:       {len(split_frames)}")
        print(f"       Original range req'd:  [{start_orig}, {end_orig}]")
        print(f"       Actual orig range:     [{actual_first_orig}, {actual_last_orig}]")
        print(f"       Images:                {split_image_dir}")
        print(f"       Sky masks:             {split_mask_dir}")
        print(f"       Positions:             {split_positions_path}")

    # Overlap report
    for split_index in range(NUM_SPLITS - 1):
        a_start, a_end = splits_orig[split_index]
        b_start, b_end = splits_orig[split_index + 1]

        overlap_lo_orig = max(a_start, b_start)
        overlap_hi_orig = min(a_end, b_end)

        if overlap_hi_orig >= overlap_lo_orig:
            n_overlap_subsampled = sum(
                1 for (oi, _) in processed
                if overlap_lo_orig <= oi <= overlap_hi_orig
            )
            print(
                f"\n[INFO] Overlap split_{split_index + 1} ∩ "
                f"split_{split_index + 2}: orig [{overlap_lo_orig}, "
                f"{overlap_hi_orig}] = {n_overlap_subsampled} subsampled frames"
            )
        else:
            print(
                f"\n[WARN] No overlap between split_{split_index + 1} "
                f"and split_{split_index + 2}"
            )

    print("\n[INFO] Done.")
    print(f"[INFO] Gaussian Splatting data saved in: {OUTPUT_ROOT}")


# =======================
# MAIN
# =======================

def main():
    ensure_input_paths()
    create_empty_colmap_split_folders()
    process_frames()


if __name__ == "__main__":
    main()