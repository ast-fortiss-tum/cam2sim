#!/usr/bin/env python3
"""
2F_extract_semantic_maps.py

Extract reduced semantic segmentation maps (road, car, background) from RGB frames
using SegFormer (Cityscapes).

Reads from (project root):
    data/raw_dataset/<BAG>/images/

Writes to (project root):
    data/processed_dataset/<BAG>/
        semantic_maps/
"""

import os
import glob
import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# ================= CONFIG =================

# Bag name (with .bag extension): must match an existing bag from step 1.
DEFAULT_BAG_NAME = "reference_bag.bag"

parser = argparse.ArgumentParser(
    description="Extract reduced semantic segmentation maps (road, car, background) "
                "from RGB frames using SegFormer (Cityscapes)."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
args = parser.parse_args()

bag_name = args.bag_name                # e.g. "reference_bag.bag"
bag_stem = Path(bag_name).stem          # e.g. "reference_bag"

INPUT_DIR = f"data/raw_dataset/{bag_stem}/images"
OUTPUT_ROOT = f"data/processed_dataset/{bag_stem}"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "semantic_maps")

SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cityscapes IDs
ROAD_ID = 0
CAR_ID = 13

# Colors (you can change if needed)
COLOR_ROAD = (128, 64, 128)   # same as Cityscapes
COLOR_CAR  = (0, 0, 142)
COLOR_BG   = (0, 0, 0)

# ==========================================


def decode_reduced_mask(mask):
    """
    Keep only:
    - road
    - cars
    - background (black)
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Road
    out[mask == ROAD_ID] = COLOR_ROAD

    # Cars
    out[mask == CAR_ID] = COLOR_CAR

    return Image.fromarray(out)


def create_semantic_maps():
    print("=" * 60)
    print("REDUCED SEMANTIC MAP GENERATION")
    print("=" * 60)
    print(f"Bag:        {bag_name}")
    print(f"Bag stem:   {bag_stem}")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device:     {DEVICE}")
    print("=" * 60)

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

    if len(image_paths) == 0:
        raise RuntimeError("No images found!")

    print(f"Found {len(image_paths)} images")

    # Load model
    print("\nLoading SegFormer...")
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.to(DEVICE).eval()
    print(f"Using device: {DEVICE}")

    # Process
    for img_path in tqdm(image_paths, desc="Generating maps"):
        img_name = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, img_name)

        if os.path.exists(out_path):
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            pred = upsampled.argmax(1)[0].cpu().numpy()

        seg_img = decode_reduced_mask(pred)
        seg_img.save(out_path)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    create_semantic_maps()