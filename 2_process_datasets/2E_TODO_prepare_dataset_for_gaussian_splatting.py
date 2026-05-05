import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Configuration
input_folder = 'datasets/snowy/images'
data_file = 'datasets/snowy/frame_positions.txt'
crop_bottom = 45

# Split configuration
OVERLAP_FRAMES = 100  # Number of original (pre-subsample) frames of overlap between splits
NUM_SPLITS = 2

# Sky mask model
MODEL_NAME = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"


def parse_args():
    parser = argparse.ArgumentParser(description="Process frames with configurable frame skip")
    parser.add_argument('--skip', type=int, default=3,
                        help='Keep 1 frame every N frames (default: 3)')
    return parser.parse_args()


def load_sky_model():
    """Load SegFormer model for sky segmentation."""
    print(f"[INFO] Loading SegFormer: {MODEL_NAME}...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Model loaded on {device}")

    # Find sky class ID
    id2label = model.config.id2label
    sky_id = None
    for id, label in id2label.items():
        if label.lower() == 'sky':
            sky_id = id
            break

    if sky_id is None:
        raise RuntimeError("Could not find 'sky' class in model labels!")
    print(f"[INFO] Sky Class ID: {sky_id}")

    return processor, model, device, sky_id


def generate_sky_mask(image, processor, model, device, sky_id):
    """
    Generate a binary sky mask for a single PIL image.

    Nerfstudio convention:
        255 (white) = valid data (train on this)
        0   (black) = masked out (ignore this / sky)
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


def process_frames(frame_skip):
    # Naming suffix: e.g. "1_of_3" means keeping 1 frame every 3
    skip_label = f"1_of_{frame_skip}"

    dataset_dir = os.path.dirname(input_folder)
    output_folder = os.path.join(dataset_dir, f"_tmp_images_gs_{skip_label}")
    mask_folder = os.path.join(dataset_dir, f"_tmp_sky_masks_gs_{skip_label}")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Load sky segmentation model
    processor, model, device, sky_id = load_sky_model()

    with open(data_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

    total_frames = len(lines)

    # Subsample based on frame_skip
    indices = list(range(0, total_frames, frame_skip))

    print(f"\nTotal frames in file: {total_frames}")
    print(f"Frame skip: keeping 1 of every {frame_skip} frames")
    print(f"Subsampled to {len(indices)} frames")
    print(f"Effective FPS: {30 / frame_skip:.1f} (assuming 30 fps input)")

    # --- Process all frames (crop + save + sky mask) ---
    filenames_by_index = {}  # index -> filename
    print(f"\nProcessing frames: crop + sky masks...")
    for i in tqdm(indices, desc="Processing"):
        line = lines[i]
        filename = line.split(',')[-1].strip()
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        if os.path.exists(input_path):
            with Image.open(input_path) as img:
                width, height = img.size
                crop_box = (0, 0, width, height - crop_bottom)
                cropped_img = img.crop(crop_box)
                cropped_img.save(output_path)

                # Generate sky mask on the cropped image
                mask_img = generate_sky_mask(cropped_img.convert("RGB"), processor, model, device, sky_id)
                mask_img.save(mask_path)

            filenames_by_index[i] = filename
        else:
            print(f"Warning: {filename} not found")

    # Ordered list of (original_index, filename) that were actually processed
    processed = sorted(filenames_by_index.items())
    num_processed = len(processed)
    print(f"\nProcessed {num_processed} frames total")
    print(f"Images: {output_folder}")
    print(f"Sky masks: {mask_folder}")

    # --- Split into overlapping folders ---
    overlap_subsampled = max(1, OVERLAP_FRAMES // frame_skip)
    split_size = num_processed // NUM_SPLITS

    splits = []
    for s in range(NUM_SPLITS):
        start = s * split_size
        if s < NUM_SPLITS - 1:
            end = (s + 1) * split_size + overlap_subsampled
        else:
            end = num_processed
        end = min(end, num_processed)
        splits.append((start, end))

    print(f"\nSplit configuration:")
    print(f"  Overlap: {OVERLAP_FRAMES} original frames ≈ {overlap_subsampled} subsampled frames")
    print(f"  Split size (base): {split_size} subsampled frames")

    for s, (start, end) in enumerate(splits):
        split_name = f"split_{s+1}"
        split_img_dir = os.path.join(dataset_dir, f"images_gs_split_{s+1}_{skip_label}")
        split_mask_dir = os.path.join(dataset_dir, f"sky_masks_gs_split_{s+1}_{skip_label}")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        split_frames = processed[start:end]

        # Frame positions file for this split
        split_positions_path = os.path.join(dataset_dir, f"frame_positions_split_{s+1}_{skip_label}.txt")

        with open(split_positions_path, 'w') as f:
            f.write("# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, ImageFile\n")
            for orig_idx, filename in split_frames:
                # Hard-link image to split folder
                src_img = os.path.join(output_folder, filename)
                dst_img = os.path.join(split_img_dir, filename)
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    os.link(src_img, dst_img)

                # Hard-link mask to split folder
                src_mask = os.path.join(mask_folder, filename)
                dst_mask = os.path.join(split_mask_dir, filename)
                if os.path.exists(src_mask) and not os.path.exists(dst_mask):
                    os.link(src_mask, dst_mask)

                f.write(lines[orig_idx] + "\n")

        orig_start = split_frames[0][0] if split_frames else 0
        orig_end = split_frames[-1][0] if split_frames else 0

        print(f"\n  {split_name}: {len(split_frames)} frames")
        print(f"    Subsampled range: [{start}, {end})")
        print(f"    Original index range: [{orig_start}, {orig_end}]")
        print(f"    Images: {split_img_dir}")
        print(f"    Sky masks: {split_mask_dir}")
        print(f"    Positions: {split_positions_path}")

    # Print overlap info
    for s in range(NUM_SPLITS - 1):
        overlap_start = splits[s+1][0]
        overlap_end = splits[s][1]
        if overlap_end > overlap_start:
            n_overlap = overlap_end - overlap_start
            print(f"\n  Overlap between split_{s+1} and split_{s+2}: {n_overlap} subsampled frames")


if __name__ == "__main__":
    args = parse_args()
    print(f"=== Frame processing with --skip={args.skip} (keeping 1 of every {args.skip}) ===")
    process_frames(args.skip)
    print("\nDone!")