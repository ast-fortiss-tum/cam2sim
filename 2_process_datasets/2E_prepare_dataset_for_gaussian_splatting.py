#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare images, sky masks, and overlapping splits for Gaussian Splatting.

Reads source data from:

    data/raw_dataset/<BAG_NAME>/images
    data/raw_dataset/<BAG_NAME>/images_positions.txt

Writes all Gaussian Splatting data to:

    data/data_for_gaussian_splatting/<BAG_NAME>/

Outputs include:

    _tmp_images_gs_1_of_<FRAME_SKIP>/
    # _tmp_sky_masks_gs_1_of_<FRAME_SKIP>/   (disabled)

    images_gs_split_1_1_of_<FRAME_SKIP>/
    # sky_masks_gs_split_1_1_of_<FRAME_SKIP>/   (disabled)
    frame_positions_split_1_1_of_<FRAME_SKIP>.txt

    images_gs_split_2_1_of_<FRAME_SKIP>/
    # sky_masks_gs_split_2_1_of_<FRAME_SKIP>/   (disabled)
    frame_positions_split_2_1_of_<FRAME_SKIP>.txt

    images_gs_split_3_1_of_<FRAME_SKIP>/
    # sky_masks_gs_split_3_1_of_<FRAME_SKIP>/   (disabled)
    frame_positions_split_3_1_of_<FRAME_SKIP>.txt

    colmap/split_1/sparse/0/   (empty, for the COLMAP reconstruction)
    colmap/split_2/sparse/0/   (empty, for the COLMAP reconstruction)
    colmap/split_3/sparse/0/   (empty, for the COLMAP reconstruction)

This script:
  - Uses hardcoded BAG_NAME, no command-line parameters
  - Imports local utils from the same folder as this script
  - Crops image bottoms
  # - Generates sky masks using SegFormer   (disabled, see commented blocks below)
  - Creates overlapping splits for Gaussian Splatting
  - Creates one empty COLMAP folder per split, ready to receive the
    sparse reconstruction exported from the COLMAP GUI

Overlap behavior:
  - Split 1 starts normally.
  - Split 2 starts earlier, overlapping the end of split 1.
  - Split 3 starts earlier, overlapping the end of split 2.
"""

import os
import sys
import shutil

# import torch
# import numpy as np
from PIL import Image
from tqdm import tqdm
# from transformers import (
#     SegformerImageProcessor,
#     SegformerForSemanticSegmentation,
# )


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

# Change this to select another bag.
# No command-line parameters are used.
BAG_NAME = "reference_bag"

# Source images.
SOURCE_IMAGES_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_dataset",
    BAG_NAME,
    "images",
)

# Source position file.
SOURCE_POSITIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_dataset",
    BAG_NAME,
    "images_positions.txt",
)

# Final Gaussian Splatting output root.
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_gaussian_splatting",
    BAG_NAME,
)

# Crop this many pixels from the bottom of every image.
CROP_BOTTOM = 45

# Keep one frame every FRAME_SKIP frames.
FRAME_SKIP = 3

# Split configuration.
OVERLAP_FRAMES = 400
NUM_SPLITS = 3

# Sky mask model. disabled
# MODEL_NAME = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"

# If True, replace existing processed files.
OVERWRITE_EXISTING = True


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
    """
    Load position file.

    Returns:
      header_lines: comment/header lines starting with '#'
      data_lines: valid non-empty non-comment lines
    """
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
    """
    The image file is expected to be the last comma-separated column.

    Example:
      FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z,
      Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile
    """
    parts = [p.strip() for p in line.split(",")]

    if len(parts) < 2:
        raise RuntimeError(f"Invalid position line: {line}")

    return parts[-1]


def link_or_copy(src, dst):
    """
    Prefer hard-link for speed and storage efficiency.
    Fall back to copy if hard-linking fails.
    """
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
    """
    Write position file for a split.
    """
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
    """
    Create one COLMAP folder per split, each with its own sparse/0 sub-folder.

    Layout produced with NUM_SPLITS=3:
      OUTPUT_ROOT/colmap/split_1/sparse/0
      OUTPUT_ROOT/colmap/split_2/sparse/0
      OUTPUT_ROOT/colmap/split_3/sparse/0

    Each split is an independent dataset for nerfstudio. The COLMAP GUI,
    when exporting a sparse model, will write cameras.bin / images.bin /
    points3D.bin directly into the chosen export folder; pointing it at
    `colmap/split_N/sparse/0` therefore produces the expected layout for
    nerfstudio without manual renames.
    """
    colmap_root = os.path.join(OUTPUT_ROOT, "colmap")

    for split_index in range(1, NUM_SPLITS + 1):
        split_sparse_dir = os.path.join(
            colmap_root,
            f"split_{split_index}",
            "sparse",
            "0",
        )
        os.makedirs(split_sparse_dir, exist_ok=True)

    print(f"[INFO] Empty COLMAP split folders created in: {colmap_root}")


# =======================
# SKY MASK MODEL  disabled
# =======================

# def load_sky_model():
#     """
#     Load SegFormer model for sky segmentation.
#     """
#     print(f"[INFO] Loading SegFormer model: {MODEL_NAME}")
#
#     processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
#     model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model.to(device)
#     model.eval()
#
#     print(f"[INFO] Model loaded on: {device}")
#
#     id2label = model.config.id2label
#
#     sky_id = None
#
#     for class_id, label in id2label.items():
#         if label.lower() == "sky":
#             sky_id = int(class_id)
#             break
#
#     if sky_id is None:
#         raise RuntimeError("Could not find 'sky' class in model labels.")
#
#     print(f"[INFO] Sky class ID: {sky_id}")
#
#     return processor, model, device, sky_id


# def generate_sky_mask(image, processor, model, device, sky_id):
#     """
#     Generate a binary sky mask for a single PIL image.
#
#     Nerfstudio convention:
#       255 white = valid data, train on this
#       0 black   = masked out, ignore this / sky
#     """
#     inputs = processor(
#         images=image,
#         return_tensors="pt",
#     ).to(device)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     logits = outputs.logits
#
#     upsampled_logits = torch.nn.functional.interpolate(
#         logits,
#         size=image.size[::-1],
#         mode="bilinear",
#         align_corners=False,
#     )
#
#     pred_seg = upsampled_logits.argmax(dim=1)[0]
#
#     mask = torch.ones_like(pred_seg, dtype=torch.uint8) * 255
#     mask[pred_seg == sky_id] = 0
#
#     return Image.fromarray(mask.cpu().numpy())


# =======================
# MAIN PROCESSING
# =======================

def process_frames():
    skip_label = f"1_of_{FRAME_SKIP}"

    tmp_image_folder = os.path.join(
        OUTPUT_ROOT,
        f"_tmp_images_gs_{skip_label}",
    )

    # tmp_mask_folder = os.path.join(
    #     OUTPUT_ROOT,
    #     f"_tmp_sky_masks_gs_{skip_label}",
    # )

    os.makedirs(tmp_image_folder, exist_ok=True)
    # os.makedirs(tmp_mask_folder, exist_ok=True)

    header_lines, data_lines = load_position_lines(SOURCE_POSITIONS_FILE)

    total_frames = len(data_lines)

    indices = list(range(0, total_frames, FRAME_SKIP))

    print("=" * 80)
    print("GAUSSIAN SPLATTING IMAGE PREPARATION")
    print("=" * 80)
    print(f"[INFO] Project root:            {PROJECT_ROOT}")
    print(f"[INFO] Script folder:           {SCRIPT_DIR}")
    print(f"[INFO] Local utils:             {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag name:                {BAG_NAME}")
    print(f"[INFO] Source images:           {SOURCE_IMAGES_FOLDER}")
    print(f"[INFO] Source positions:        {SOURCE_POSITIONS_FILE}")
    print(f"[INFO] Output root:             {OUTPUT_ROOT}")
    print(f"[INFO] Crop bottom:             {CROP_BOTTOM}")
    print(f"[INFO] Frame skip:              {FRAME_SKIP}")
    print(f"[INFO] Split count:             {NUM_SPLITS}")
    print(f"[INFO] Overlap frames:          {OVERLAP_FRAMES}")
    print(f"[INFO] Total source frames:     {total_frames}")
    print(f"[INFO] Subsampled frame count:  {len(indices)}")
    print("=" * 80)

    # processor, model, device, sky_id = load_sky_model()

    filenames_by_index = {}

    print("\n[INFO] Processing frames: crop images")

    for original_index in tqdm(indices, desc="Processing"):
        line = data_lines[original_index]
        filename = get_image_filename_from_position_line(line)

        input_path = os.path.join(SOURCE_IMAGES_FOLDER, filename)
        output_path = os.path.join(tmp_image_folder, filename)
        # mask_path = os.path.join(tmp_mask_folder, filename)

        if not os.path.exists(input_path):
            print(f"[WARN] Image not found: {input_path}")
            continue

        if (
            not OVERWRITE_EXISTING
            and os.path.exists(output_path)
            # and os.path.exists(mask_path)
        ):
            filenames_by_index[original_index] = filename
            continue

        with Image.open(input_path) as img:
            width, height = img.size

            crop_height = height - CROP_BOTTOM

            if crop_height <= 0:
                raise RuntimeError(
                    f"CROP_BOTTOM={CROP_BOTTOM} is too large for image: {input_path}"
                )

            crop_box = (
                0,
                0,
                width,
                crop_height,
            )

            cropped_img = img.crop(crop_box)
            cropped_img.save(output_path)

            # mask_img = generate_sky_mask(
            #     cropped_img.convert("RGB"),
            #     processor,
            #     model,
            #     device,
            #     sky_id,
            # )
            #
            # mask_img.save(mask_path)

        filenames_by_index[original_index] = filename

    processed = sorted(filenames_by_index.items())
    num_processed = len(processed)

    print(f"\n[INFO] Processed frames: {num_processed}")
    print(f"[INFO] Temporary cropped images: {tmp_image_folder}")
    # print(f"[INFO] Temporary sky masks:      {tmp_mask_folder}")

    if num_processed == 0:
        raise RuntimeError("No frames were processed. Check image paths and position file.")

    # =======================
    # SPLIT CREATION
    # =======================

    # Convert overlap from original frame units to subsampled frame units.
    #
    # Example:
    #   OVERLAP_FRAMES = 400
    #   FRAME_SKIP = 3
    #
    #   400 / 3 = 133.33...
    #
    # We use ceil-style division so the actual overlap is at least
    # OVERLAP_FRAMES in original-frame units.
    overlap_subsampled = max(
        1,
        (OVERLAP_FRAMES + FRAME_SKIP - 1) // FRAME_SKIP,
    )

    split_size = num_processed // NUM_SPLITS

    if split_size == 0:
        raise RuntimeError(
            f"NUM_SPLITS={NUM_SPLITS} is too large for {num_processed} processed frames."
        )

    if overlap_subsampled >= split_size:
        print(
            "[WARN] overlap_subsampled is greater than or equal to split_size. "
            "Splits will have very large overlap."
        )

    splits = []

    for split_index in range(NUM_SPLITS):
        base_start = split_index * split_size

        if split_index < NUM_SPLITS - 1:
            base_end = (split_index + 1) * split_size
        else:
            base_end = num_processed

        # Desired behavior:
        #   split 1: starts normally
        #   split 2: starts earlier, overlapping the end of split 1
        #   split 3: starts earlier, overlapping the end of split 2
        #
        # Therefore, the later split goes backward.
        if split_index == 0:
            start = base_start
        else:
            start = max(0, base_start - overlap_subsampled)

        end = base_end

        splits.append((start, end))

    print("\n[INFO] Split configuration:")
    print(f"       Overlap original frames:    {OVERLAP_FRAMES}")
    print(f"       Overlap subsampled frames:  {overlap_subsampled}")
    print(f"       Base split size:            {split_size}")
    print("       Overlap mode:               later split starts earlier")

    for split_index, (start, end) in enumerate(splits, start=1):
        split_frames = processed[start:end]

        split_image_dir = os.path.join(
            OUTPUT_ROOT,
            f"images_gs_split_{split_index}_{skip_label}",
        )

        # split_mask_dir = os.path.join(
        #     OUTPUT_ROOT,
        #     f"sky_masks_gs_split_{split_index}_{skip_label}",
        # )

        split_positions_path = os.path.join(
            OUTPUT_ROOT,
            f"frame_positions_split_{split_index}_{skip_label}.txt",
        )

        os.makedirs(split_image_dir, exist_ok=True)
        # os.makedirs(split_mask_dir, exist_ok=True)

        for original_index, filename in split_frames:
            src_img = os.path.join(tmp_image_folder, filename)
            dst_img = os.path.join(split_image_dir, filename)

            # src_mask = os.path.join(tmp_mask_folder, filename)
            # dst_mask = os.path.join(split_mask_dir, filename)

            link_or_copy(src_img, dst_img)
            # link_or_copy(src_mask, dst_mask)

        write_split_positions_file(
            split_positions_path,
            header_lines,
            split_frames,
            data_lines,
        )

        orig_start = split_frames[0][0] if split_frames else 0
        orig_end = split_frames[-1][0] if split_frames else 0

        print(f"\n[INFO] Split {split_index}:")
        print(f"       Frames:                {len(split_frames)}")
        print(f"       Subsampled range:      [{start}, {end})")
        print(f"       Original index range:  [{orig_start}, {orig_end}]")
        print(f"       Images:                {split_image_dir}")
        # print(f"       Sky masks:             {split_mask_dir}")
        print(f"       Positions:             {split_positions_path}")

    for split_index in range(NUM_SPLITS - 1):
        overlap_start = splits[split_index + 1][0]
        overlap_end = splits[split_index][1]

        if overlap_end > overlap_start:
            n_overlap = overlap_end - overlap_start

            print(
                f"\n[INFO] Overlap between split_{split_index + 1} "
                f"and split_{split_index + 2}: {n_overlap} subsampled frames"
            )
        else:
            print(
                f"\n[WARN] No overlap detected between split_{split_index + 1} "
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