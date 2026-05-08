#!/usr/bin/env python3

import os
import glob
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ================= CONFIG =================

DATASET_NAME = "reference_bag"

INPUT_DIR = f"data/raw_dataset/{DATASET_NAME}/images"

OUTPUT_ROOT = f"/home/davidejannussi/Documents/cam2sim/data/processed_dataset/{DATASET_NAME}"
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
                align_corners=False
            )

            pred = upsampled.argmax(1)[0].cpu().numpy()

        seg_img = decode_reduced_mask(pred)
        seg_img.save(out_path)

    print("\n" + "=" * 60)
    print("✅ DONE!")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    create_semantic_maps()