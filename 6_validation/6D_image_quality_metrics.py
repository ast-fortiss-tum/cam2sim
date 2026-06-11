#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6D_image_quality_metrics.py

Offline image quality metrics pipeline.

Computes image-level, distribution-level, vehicle-consistency, and temporal
metrics between real-world reference images and generated/simulated images.

Pipeline mode:
    python 6_validation/6D_image_quality_metrics.py \
        --bag-name reference_bag.bag \
        --method stable_diffusion \
        --crop-bottom 45

Pipeline mode automatically evaluates one final image folder:
    GT:
        data/raw_dataset/<BAG>/images

    method=only_carla:
        data/replay_dataset/<BAG>/only_carla/rgb

    method=stable_diffusion:
        data/replay_dataset/<BAG>/stable_diffusion/rgb

    method in {splatfacto, splatfacto-big, nerfacto, nerfacto-big}:
        data/replay_dataset/<BAG>/<METHOD>/gs

    Output:
        results/<BAG>/image_quality/<METHOD>/

Manual flat mode:
    python 6_validation/6D_image_quality_metrics.py \
        --gt-folder data/raw_dataset/<BAG>/images \
        --input-folder data/replay_dataset/<BAG>/stable_diffusion/rgb \
        --output-folder results/<BAG>/image_quality/stable_diffusion \
        --flat \
        --job-name stable_diffusion \
        --crop-bottom 45

Manual subfolder mode, for Stable Diffusion grid search:
    python 6_validation/6D_image_quality_metrics.py \
        --gt-folder data/raw_dataset/<BAG>/images \
        --input-folder <SD_ROOT>/<BAG>/sd_grid_search \
        --output-folder <SD_ROOT>/<BAG>/sd_grid_search_METRICS \
        --crop-bottom 45

Important:
    Matching is done by numeric frame id, not exact filename.
    Therefore these names match each other:
        000123.png
        frame_000123.png
        seg_000123.png
"""

import argparse
import gc
import json
import os
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
from scipy import linalg
from shapely.geometry import box
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_BAG_NAME = "reference_bag.bag"

VALID_METHODS = (
    "only_carla",
    "splatfacto",
    "splatfacto-big",
    "nerfacto",
    "nerfacto-big",
    "stable_diffusion",
)

GS_METHODS = (
    "splatfacto",
    "splatfacto-big",
    "nerfacto",
    "nerfacto-big",
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}

MIN_VEHICLE_AREA = 600
IOU_THRESHOLD = 0.5
YOLO_MODEL_NAME = "yolov8n.pt"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

IMAGE_LEVEL_KEYS = [
    "MSE",
    "PSNR",
    "SSIM",
    "CPL",
    "SegScore",
    "Veh_Recall",
    "Veh_Precision",
    "Veh_AvgIoU",
]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline image metrics pipeline: image-level, distribution-level, "
            "vehicle-consistency, and temporal metrics."
        )
    )

    parser.add_argument(
        "--bag-name",
        type=str,
        default=None,
        help=(
            "Bag filename including .bag extension, or bag stem. "
            "Used with --method for pipeline mode."
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=VALID_METHODS,
        help=(
            "Pipeline method. If provided with --bag-name, paths are built "
            "automatically and the script evaluates one final image folder."
        ),
    )

    parser.add_argument(
        "--gt-folder",
        "-g",
        type=str,
        default=None,
        help=(
            "Manual mode: path to the ground-truth real-world images folder."
        ),
    )

    parser.add_argument(
        "--input-folder",
        "-i",
        type=str,
        default=None,
        help=(
            "Manual mode: path to generated images folder. With --flat, it "
            "contains images directly. Without --flat, it contains subfolders."
        ),
    )

    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        default=None,
        help="Manual mode: path where metric reports are saved.",
    )

    parser.add_argument(
        "--target-resolution",
        "-r",
        type=str,
        default=None,
        help=(
            "Optional target resolution as WxH, for example 1024x512. "
            "Default: original resolution."
        ),
    )

    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help=(
            "Number of pixels to crop from the bottom of both GT and generated "
            "images before comparison, for example 45 to remove hood."
        ),
    )

    parser.add_argument(
        "--flat",
        action="store_true",
        help=(
            "Manual mode only: input-folder contains images directly. "
            "Ignored in pipeline mode."
        ),
    )

    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help=(
            "Manual mode: name for the job/report. "
            "Default: input folder basename."
        ),
    )

    parser.add_argument(
        "--exclude-subfolders",
        type=str,
        nargs="*",
        default=["old", "depth", "canny", "seg", "baseline"],
        help=(
            "Manual subfolder mode: subfolder names to exclude. "
            "Default: old depth canny seg baseline."
        ),
    )

    parser.add_argument(
        "--skip-segformer",
        action="store_true",
        help="Skip SegFormer-based metrics: CPL, SegScore, Temp_CPL.",
    )

    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO-based vehicle consistency metrics.",
    )

    parser.add_argument(
        "--skip-distribution",
        action="store_true",
        help="Skip distribution-level metrics: FID, KID, IS, MMD, PRDC.",
    )

    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal consistency metrics.",
    )

    args = parser.parse_args()

    if args.target_resolution:
        try:
            width, height = args.target_resolution.lower().split("x")
            args.target_resolution = (int(width), int(height))
        except ValueError:
            parser.error(
                f"Invalid resolution format: {args.target_resolution}. "
                "Use WxH, for example 1024x512."
            )

    if args.crop_bottom < 0:
        parser.error("--crop-bottom must be >= 0.")

    return args


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def get_bag_stem(bag_name: str) -> str:
    return os.path.splitext(os.path.basename(bag_name))[0]


def build_pipeline_gt_folder(bag_stem: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "data",
        "raw_dataset",
        bag_stem,
        "images",
    )


def build_pipeline_input_folder(bag_stem: str, method: str) -> str:
    if method == "only_carla":
        return os.path.join(
            PROJECT_ROOT,
            "data",
            "replay_dataset",
            bag_stem,
            "only_carla",
            "rgb",
        )

    if method == "stable_diffusion":
        return os.path.join(
            PROJECT_ROOT,
            "data",
            "replay_dataset",
            bag_stem,
            "stable_diffusion",
            "rgb",
        )

    if method in GS_METHODS:
        return os.path.join(
            PROJECT_ROOT,
            "data",
            "replay_dataset",
            bag_stem,
            method,
            "gs",
        )

    raise ValueError(f"Unsupported method: {method}")


def build_pipeline_output_folder(bag_stem: str, method: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "results",
        bag_stem,
        "image_quality",
        method,
    )


def resolve_runtime_config(args):
    """
    Resolve paths and execution mode.

    Pipeline mode:
        --bag-name and --method are provided.
        The script evaluates one final method folder.
        --flat is not needed.

    Manual mode:
        --gt-folder, --input-folder, --output-folder are provided.
        --flat controls whether input-folder contains images directly.
        Without --flat, input-folder is treated as a grid/subfolder root.
    """
    pipeline_mode = args.bag_name is not None or args.method is not None

    if pipeline_mode:
        if args.bag_name is None or args.method is None:
            raise ValueError(
                "Pipeline mode requires both --bag-name and --method."
            )

        bag_stem = get_bag_stem(args.bag_name)

        gt_folder = (
            args.gt_folder
            if args.gt_folder is not None
            else build_pipeline_gt_folder(bag_stem)
        )

        input_folder = (
            args.input_folder
            if args.input_folder is not None
            else build_pipeline_input_folder(bag_stem, args.method)
        )

        output_folder = (
            args.output_folder
            if args.output_folder is not None
            else build_pipeline_output_folder(bag_stem, args.method)
        )

        job_name = args.job_name or args.method

        return {
            "pipeline_mode": True,
            "single_folder_mode": True,
            "bag_stem": bag_stem,
            "method": args.method,
            "gt_folder": os.path.abspath(os.path.expanduser(gt_folder)),
            "input_folder": os.path.abspath(os.path.expanduser(input_folder)),
            "output_folder": os.path.abspath(os.path.expanduser(output_folder)),
            "job_name": job_name,
        }

    required_manual = [args.gt_folder, args.input_folder, args.output_folder]
    if any(value is None for value in required_manual):
        raise ValueError(
            "Manual mode requires --gt-folder, --input-folder, "
            "and --output-folder."
        )

    input_folder = os.path.abspath(os.path.expanduser(args.input_folder))
    job_name = args.job_name or os.path.basename(os.path.normpath(input_folder))

    return {
        "pipeline_mode": False,
        "single_folder_mode": args.flat,
        "bag_stem": None,
        "method": None,
        "gt_folder": os.path.abspath(os.path.expanduser(args.gt_folder)),
        "input_folder": input_folder,
        "output_folder": os.path.abspath(os.path.expanduser(args.output_folder)),
        "job_name": job_name,
    }


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def get_image_filenames(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"[WARN] Folder does not exist: {folder}")
        return []

    return natsorted([
        filename
        for filename in os.listdir(folder)
        if is_image_file(filename)
    ])


def extract_frame_id(filename: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(\d+)$", stem)

    if match is None:
        return None

    return int(match.group(1))


def build_frame_index(folder: str) -> Tuple[Dict[int, str], List[str]]:
    index = {}
    skipped = []

    for filename in get_image_filenames(folder):
        frame_id = extract_frame_id(filename)

        if frame_id is None:
            skipped.append(filename)
            continue

        if frame_id not in index:
            index[frame_id] = filename

    return index, skipped


def get_matched_image_pairs(
    gt_folder: str,
    gen_folder: str,
) -> List[Tuple[str, str, int]]:
    gt_index, gt_skipped = build_frame_index(gt_folder)
    gen_index, gen_skipped = build_frame_index(gen_folder)

    matched_ids = sorted(set(gt_index.keys()) & set(gen_index.keys()))
    gt_only = sorted(set(gt_index.keys()) - set(gen_index.keys()))
    gen_only = sorted(set(gen_index.keys()) - set(gt_index.keys()))

    print(
        f"   GT images: {len(gt_index)} | "
        f"Generated images: {len(gen_index)} | "
        f"Matched: {len(matched_ids)}"
    )

    if gt_skipped:
        print(
            f"   [WARN] {len(gt_skipped)} GT image(s) without numeric "
            "frame id were skipped."
        )
        for filename in gt_skipped[:5]:
            print(f"          {filename}")

    if gen_skipped:
        print(
            f"   [WARN] {len(gen_skipped)} generated image(s) without "
            "numeric frame id were skipped."
        )
        for filename in gen_skipped[:5]:
            print(f"          {filename}")

    if gt_only:
        print(
            f"   [WARN] {len(gt_only)} GT frame id(s) have no generated "
            "match."
        )
        print(f"          First missing GT ids: {gt_only[:5]}")

    if gen_only:
        print(
            f"   [WARN] {len(gen_only)} generated frame id(s) have no GT "
            "match."
        )
        print(f"          First extra generated ids: {gen_only[:5]}")

    return [
        (gt_index[frame_id], gen_index[frame_id], frame_id)
        for frame_id in matched_ids
    ]


def safe_round(value, digits=4):
    if isinstance(value, (int, float)) and value != float("inf"):
        return round(float(value), digits)
    return value


# =============================================================================
# SEGFORMER HELPERS
# =============================================================================

def get_segformer_model():
    device = get_device()
    print(f"\n[INFO] Initializing SegFormer model on device: {device}")

    try:
        image_processor = SegformerImageProcessor.from_pretrained(
            SEGFORMER_MODEL
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_MODEL
        )
        model.eval().to(device)

        print("[OK] SegFormer model initialized.")
        return model, image_processor

    except Exception as exc:
        print(f"[WARN] Error initializing SegFormer model: {exc}")
        print("[WARN] CPL and SegScore metrics will be skipped.")
        return None, None


def decode_cityscapes_mask(predicted_mask):
    cityscapes_palette = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ]

    height, width = predicted_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in enumerate(cityscapes_palette):
        rgb_mask[predicted_mask == class_id] = color

    return rgb_mask


def encode_cityscapes_mask(rgb_img):
    rgb = np.array(rgb_img)
    height, width, _ = rgb.shape
    label_mask = np.full((height, width), fill_value=-1, dtype=np.int64)

    segmentation_colors = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ]

    for class_id, color in enumerate(segmentation_colors):
        mask = np.all(rgb == color, axis=-1)
        label_mask[mask] = class_id

    return torch.from_numpy(label_mask)


# =============================================================================
# YOLO HELPERS
# =============================================================================

class MockYoloResult:
    def __init__(self, data):
        self.boxes = self.MockBoxes(data)

    class MockBoxes:
        def __init__(self, data):
            self.data = torch.tensor(data)

        def cpu(self):
            return self


def load_yolo_model():
    try:
        from ultralytics import YOLO

        print(f"\n[INFO] Loading YOLO model: {YOLO_MODEL_NAME}")
        model = YOLO(YOLO_MODEL_NAME)
        print("[OK] YOLO model loaded.")
        return model

    except ImportError:
        print("[WARN] Ultralytics YOLO is not installed.")
        print("[WARN] Vehicle consistency metrics will be skipped.")
        return None

    except Exception as exc:
        print(f"[WARN] Error loading YOLO model: {exc}")
        print("[WARN] Vehicle consistency metrics will be skipped.")
        return None


def calculate_yolo_image(yolo_model, pil_image: Image.Image):
    if yolo_model is None:
        return [MockYoloResult([])]

    return yolo_model(pil_image, verbose=False)


image_transforms_512 = transforms.Compose([
    transforms.Resize(
        512,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.CenterCrop(512),
])


def box_iou(box1: List[float], box2: List[float]) -> float:
    b1 = box(box1[0], box1[1], box1[2], box1[3])
    b2 = box(box2[0], box2[1], box2[2], box2[3])

    union = b1.union(b2).area
    if union <= 0:
        return 0.0

    return b1.intersection(b2).area / union


def extract_vehicles(results) -> List[Dict[str, Union[List[float], float, str]]]:
    vehicles = []

    for result in results:
        for data_row in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = data_row
            cls = int(cls)

            if cls in VEHICLE_CLASSES:
                vehicles.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf),
                    "class": VEHICLE_CLASSES[cls],
                })

    return vehicles


def filter_large_vehicles(vehicles, min_area=MIN_VEHICLE_AREA):
    return [
        vehicle
        for vehicle in vehicles
        if (
            (vehicle["bbox"][2] - vehicle["bbox"][0])
            * (vehicle["bbox"][3] - vehicle["bbox"][1])
        ) >= min_area
    ]


def match_vehicles(real_vehicles, gen_vehicles, iou_thresh=IOU_THRESHOLD):
    matches = []
    used_gen_indices = set()

    for real_vehicle in real_vehicles:
        best_match = None
        best_iou = 0.0
        best_index = -1

        for index, gen_vehicle in enumerate(gen_vehicles):
            if index in used_gen_indices:
                continue

            score = box_iou(real_vehicle["bbox"], gen_vehicle["bbox"])

            if score > best_iou:
                best_iou = score
                best_match = (real_vehicle, gen_vehicle, score)
                best_index = index

        if best_match is not None and best_iou >= iou_thresh:
            matches.append(best_match)
            used_gen_indices.add(best_index)

    return matches


# =============================================================================
# IMAGE-LEVEL METRICS
# =============================================================================

def compute_ssim(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    image1_gray = cv2.cvtColor(image1_cv2, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_cv2, cv2.COLOR_BGR2GRAY)

    data_range = image1_gray.max() - image1_gray.min()
    if data_range <= 0:
        data_range = 255

    score, _ = ssim(
        image1_gray,
        image2_gray,
        data_range=data_range,
        full=True,
    )

    return float(score)


def compute_psnr(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    mse_value = np.mean(
        (image1_cv2.astype(np.float64) - image2_cv2.astype(np.float64)) ** 2
    )

    if mse_value == 0:
        return float("inf")

    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse_value)))


def compute_mse(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    return float(
        np.mean(
            (image1_cv2.astype(np.float64) - image2_cv2.astype(np.float64))
            ** 2
        )
    )


def compute_cpl(
    image1_rgb: np.ndarray,
    image2_rgb: np.ndarray,
    model_cpl,
    image_processor,
) -> float:
    img1_pil = Image.fromarray(image1_rgb)
    img2_pil = Image.fromarray(image2_rgb)

    inputs1 = image_processor(images=img1_pil, return_tensors="pt")
    inputs2 = image_processor(images=img2_pil, return_tensors="pt")

    device = model_cpl.device
    img1_tensor = inputs1["pixel_values"].to(device)
    img2_tensor = inputs2["pixel_values"].to(device)

    with torch.no_grad():
        outputs1 = model_cpl(img1_tensor, output_hidden_states=True)
        outputs2 = model_cpl(img2_tensor, output_hidden_states=True)

        features1 = outputs1.hidden_states[-1]
        features2 = outputs2.hidden_states[-1]

    return torch.nn.functional.mse_loss(features1, features2).item()


def calculate_semantic_segmentation_score(
    model_seg,
    image_processor,
    image_seg: np.ndarray,
    image_created: np.ndarray,
) -> Tuple[float, np.ndarray]:
    inputs = image_processor(images=image_created, return_tensors="pt")
    inputs = {
        key: value.to(model_seg.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model_seg(**inputs)
        logits = outputs.logits
        target_size = image_created.shape[:2]

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        predicted = upsampled.argmax(1)[0].cpu().numpy()

    segmentation_image = decode_cityscapes_mask(predicted)

    if image_seg is None:
        return -1.0, segmentation_image

    pred_ids = encode_cityscapes_mask(segmentation_image).float()
    gt_ids = encode_cityscapes_mask(image_seg).float()

    mse_score = torch.nn.functional.mse_loss(pred_ids, gt_ids).item()

    return mse_score, segmentation_image


def resize_if_needed(
    image_cv2: np.ndarray,
    target_resolution: Optional[Tuple[int, int]],
) -> np.ndarray:
    if target_resolution is None:
        return image_cv2

    width, height = target_resolution

    return cv2.resize(
        image_cv2,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def crop_bottom_if_needed(image_cv2: np.ndarray, crop_bottom: int) -> np.ndarray:
    if crop_bottom <= 0:
        return image_cv2

    if image_cv2.shape[0] <= crop_bottom:
        raise ValueError(
            f"Cannot crop {crop_bottom}px from image with height "
            f"{image_cv2.shape[0]}."
        )

    return image_cv2[:-crop_bottom, :, :]


def load_pair_for_comparison(
    gt_path: str,
    gen_path: str,
    target_resolution: Optional[Tuple[int, int]],
    crop_bottom: int,
):
    gt_cv2 = cv2.imread(gt_path, 1)
    gen_cv2 = cv2.imread(gen_path, 1)

    if gt_cv2 is None or gen_cv2 is None:
        return None, None

    gt_cv2 = resize_if_needed(gt_cv2, target_resolution)
    gen_cv2 = resize_if_needed(gen_cv2, target_resolution)

    if gt_cv2.shape != gen_cv2.shape:
        height, width = gt_cv2.shape[:2]
        gen_cv2 = cv2.resize(
            gen_cv2,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    gt_cv2 = crop_bottom_if_needed(gt_cv2, crop_bottom)
    gen_cv2 = crop_bottom_if_needed(gen_cv2, crop_bottom)

    return gt_cv2, gen_cv2


def calculate_single_metrics_all(
    gt_image_path: str,
    gen_image_path: str,
    segformer_model,
    image_processor,
    yolo_model,
    target_resolution: Optional[Tuple[int, int]] = None,
    crop_bottom: int = 0,
) -> dict:
    metrics = {
        "MSE": -3,
        "PSNR": -3,
        "SSIM": -3,
        "CPL": -3,
        "SegScore": -3,
        "Veh_Recall": -3,
        "Veh_Precision": -3,
        "Veh_AvgIoU": -3,
    }

    try:
        gt_cv2, gen_cv2 = load_pair_for_comparison(
            gt_image_path,
            gen_image_path,
            target_resolution,
            crop_bottom,
        )

        if gt_cv2 is None or gen_cv2 is None:
            return {key: -2 for key in metrics.keys()}

        gt_rgb = cv2.cvtColor(gt_cv2, cv2.COLOR_BGR2RGB)
        gen_rgb = cv2.cvtColor(gen_cv2, cv2.COLOR_BGR2RGB)

        metrics["MSE"] = round(compute_mse(gt_cv2, gen_cv2), 4)
        metrics["PSNR"] = round(compute_psnr(gt_cv2, gen_cv2), 4)
        metrics["SSIM"] = round(compute_ssim(gt_cv2, gen_cv2), 4)

        if segformer_model is not None and image_processor is not None:
            metrics["CPL"] = round(
                compute_cpl(gt_rgb, gen_rgb, segformer_model, image_processor),
                4,
            )

            seg_score_val, _ = calculate_semantic_segmentation_score(
                segformer_model,
                image_processor,
                image_seg=gt_rgb,
                image_created=gen_rgb,
            )

            metrics["SegScore"] = round(seg_score_val, 4)

        if yolo_model is not None:
            real_image_pil = Image.fromarray(gt_rgb).convert("RGB")
            gen_image_pil = Image.fromarray(gen_rgb).convert("RGB")

            real_image_t = image_transforms_512(real_image_pil)
            gen_image_t = image_transforms_512(gen_image_pil)

            yolo_results_real = calculate_yolo_image(yolo_model, real_image_t)
            yolo_results_gen = calculate_yolo_image(yolo_model, gen_image_t)

            real_vehicles = filter_large_vehicles(
                extract_vehicles(yolo_results_real),
                min_area=MIN_VEHICLE_AREA,
            )

            gen_vehicles = filter_large_vehicles(
                extract_vehicles(yolo_results_gen),
                min_area=MIN_VEHICLE_AREA,
            )

            matches = match_vehicles(
                real_vehicles,
                gen_vehicles,
                iou_thresh=IOU_THRESHOLD,
            )

            real_count = len(real_vehicles)
            gen_count = len(gen_vehicles)
            match_count = len(matches)

            recall = match_count / real_count if real_count else 0.0
            precision = match_count / gen_count if gen_count else 0.0
            ious = [match[2] for match in matches]
            avg_iou = float(np.mean(ious)) if ious else 0.0

            metrics["Veh_Recall"] = round(recall, 4)
            metrics["Veh_Precision"] = round(precision, 4)
            metrics["Veh_AvgIoU"] = round(avg_iou, 4)

        return metrics

    except Exception as exc:
        print(
            f"   [WARN] Error processing "
            f"{os.path.basename(gt_image_path)}: {exc}"
        )
        return {key: -4 for key in metrics.keys()}


def calculate_distribution_metrics_avg(all_results: dict) -> dict:
    distribution_metrics = {}

    for metric_name in IMAGE_LEVEL_KEYS:
        values = []

        for metrics in all_results.values():
            value = metrics.get(metric_name)

            if (
                isinstance(value, (int, float))
                and value >= 0
                and value != float("inf")
            ):
                values.append(value)

        if not values:
            distribution_metrics[f"stdev_{metric_name}"] = 0.0
        else:
            distribution_metrics[f"stdev_{metric_name}"] = round(
                float(np.std(values)),
                4,
            )

    return distribution_metrics


# =============================================================================
# TEMPORAL CONSISTENCY
# =============================================================================

def calculate_temporal_consistency(
    gen_folder: str,
    segformer_model,
    image_processor,
    target_resolution: Optional[Tuple[int, int]] = None,
    crop_bottom: int = 0,
) -> Dict[str, float]:
    gen_images = get_image_filenames(gen_folder)
    num_pairs = len(gen_images) - 1

    if num_pairs < 1:
        return {
            "Temp_SSIM": -1,
            "Temp_PSNR": -1,
            "Temp_MSE": -1,
            "Temp_CPL": -1,
            "Temp_num_pairs": 0,
        }

    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    cpl_scores = []

    for index in tqdm(
        range(num_pairs),
        desc="    Temporal pairs",
        leave=False,
    ):
        path_curr = os.path.join(gen_folder, gen_images[index])
        path_next = os.path.join(gen_folder, gen_images[index + 1])

        img_curr = cv2.imread(path_curr, 1)
        img_next = cv2.imread(path_next, 1)

        if img_curr is None or img_next is None:
            continue

        img_curr = resize_if_needed(img_curr, target_resolution)
        img_next = resize_if_needed(img_next, target_resolution)

        if img_curr.shape != img_next.shape:
            height, width = img_curr.shape[:2]
            img_next = cv2.resize(
                img_next,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        try:
            img_curr = crop_bottom_if_needed(img_curr, crop_bottom)
            img_next = crop_bottom_if_needed(img_next, crop_bottom)
        except ValueError as exc:
            print(f"    [WARN] Temporal crop failed: {exc}")
            continue

        ssim_val = compute_ssim(img_curr, img_next)
        psnr_val = compute_psnr(img_curr, img_next)
        mse_val = compute_mse(img_curr, img_next)

        if psnr_val != float("inf") and mse_val != 0.0:
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
            mse_scores.append(mse_val)

        if segformer_model is not None and image_processor is not None:
            img_curr_rgb = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)
            img_next_rgb = cv2.cvtColor(img_next, cv2.COLOR_BGR2RGB)

            cpl_val = compute_cpl(
                img_curr_rgb,
                img_next_rgb,
                segformer_model,
                image_processor,
            )
            cpl_scores.append(cpl_val)

        if index % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "Temp_SSIM": (
            round(float(np.mean(ssim_scores)), 4)
            if ssim_scores
            else -1
        ),
        "Temp_PSNR": (
            round(float(np.mean(psnr_scores)), 4)
            if psnr_scores
            else -1
        ),
        "Temp_MSE": (
            round(float(np.mean(mse_scores)), 4)
            if mse_scores
            else -1
        ),
        "Temp_CPL": (
            round(float(np.mean(cpl_scores)), 4)
            if cpl_scores
            else -1
        ),
        "Temp_num_pairs": len(ssim_scores),
    }


# =============================================================================
# DISTRIBUTION-LEVEL METRICS
# =============================================================================

image_transforms_inception = transforms.Compose([
    transforms.Resize(
        512,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.CenterCrop(512),
    transforms.Resize(
        299,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageFolderDataset(Dataset):
    def __init__(self, root, exts=IMAGE_EXTENSIONS, crop_bottom=0):
        self.paths = []

        for extension in exts:
            self.paths.extend(
                glob(
                    os.path.join(root, f"**/*{extension}"),
                    recursive=True,
                )
            )

        self.paths = natsorted(self.paths)

        if not self.paths:
            raise ValueError(f"No images found in {root}")

        self.transform = image_transforms_inception
        self.crop_bottom = crop_bottom

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            img = image.convert("RGB")

            if self.crop_bottom > 0:
                width, height = img.size

                if height <= self.crop_bottom:
                    raise ValueError(
                        f"Cannot crop {self.crop_bottom}px from image "
                        f"with height {height}: {self.paths[index]}"
                    )

                img = img.crop((0, 0, width, height - self.crop_bottom))

            return self.transform(img)


class InceptionPool3(nn.Module):
    def __init__(self):
        super().__init__()

        net = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        net.eval()

        for param in net.parameters():
            param.requires_grad = False

        layers = []

        for name, module in net.named_children():
            if name in ["AuxLogits", "dropout", "fc"]:
                continue
            layers.append(module)

        self.features = nn.Sequential(*layers[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = self.pool(x)
            return torch.flatten(x, 1)


def get_features(loader, model, device):
    features = []

    for batch in tqdm(loader, desc="  Extracting features"):
        features.append(model(batch.to(device)).cpu().numpy())

    return np.concatenate(features, 0)


def get_logits(loader, model, device):
    probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Calculating logits"):
            logits = model(batch.to(device))

            if isinstance(logits, tuple):
                logits = logits[0]

            probs.append(F.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probs, 0)


def compute_stats(features):
    return features.mean(0), np.cov(features, rowvar=False)


def sqrtm_product(cov1, cov2, eps=1e-6):
    cov1 = cov1.copy()
    cov2 = cov2.copy()

    cov1.flat[:: cov1.shape[0] + 1] += eps
    cov2.flat[:: cov2.shape[0] + 1] += eps

    cov, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)

    if np.iscomplexobj(cov):
        cov = cov.real

    return cov


def inception_score(probs, splits=10):
    num_samples = probs.shape[0]
    split_size = num_samples // splits

    if split_size == 0:
        splits = 1
        split_size = num_samples

    scores = []

    for index in range(splits):
        part = probs[index * split_size: (index + 1) * split_size]

        if len(part) == 0:
            continue

        py = part.mean(0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(1).mean()))

    if not scores:
        return -1.0, -1.0

    return float(np.mean(scores)), float(np.std(scores))


def fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    cov = sqrtm_product(sigma1, sigma2)

    return float(
        diff.dot(diff)
        + np.trace(sigma1 + sigma2 - 2 * cov)
    )


def kid_poly(
    real_features,
    fake_features,
    degree=3,
    gamma=None,
    coef0=1.0,
    subsets=100,
    subsize=1000,
):
    rng = np.random.default_rng(123)

    n = min(len(real_features), subsize)
    m = min(len(fake_features), subsize)

    if n < 2 or m < 2:
        return -1.0, -1.0

    if gamma is None:
        gamma = 1.0 / real_features.shape[1]

    values = []

    for _ in range(subsets):
        real_sample = real_features[
            rng.choice(len(real_features), n, False)
        ]
        fake_sample = fake_features[
            rng.choice(len(fake_features), m, False)
        ]

        k_real = (gamma * real_sample @ real_sample.T + coef0) ** degree
        k_fake = (gamma * fake_sample @ fake_sample.T + coef0) ** degree
        k_cross = (gamma * real_sample @ fake_sample.T + coef0) ** degree

        np.fill_diagonal(k_real, 0)
        np.fill_diagonal(k_fake, 0)

        values.append(
            k_real.sum() / (n * (n - 1))
            + k_fake.sum() / (m * (m - 1))
            - 2 * k_cross.mean()
        )

    values = np.array(values)

    return float(values.mean()), float(values.std())


def mmd_rbf(real_features, fake_features, sigma="median"):
    combined = np.vstack([real_features, fake_features])

    if sigma == "median":
        if len(combined) < 2000:
            sample = combined
        else:
            sample = combined[np.random.choice(len(combined), 2000, False)]

        distances = pairwise_distances(sample)
        positive = distances[distances > 0]

        if len(positive) == 0:
            return -1.0

        sigma = np.median(positive)

    gamma = 1 / (2 * sigma * sigma)

    k_real = np.exp(
        -gamma * pairwise_distances(real_features, real_features, squared=True)
    )
    k_fake = np.exp(
        -gamma * pairwise_distances(fake_features, fake_features, squared=True)
    )
    k_cross = np.exp(
        -gamma * pairwise_distances(real_features, fake_features, squared=True)
    )

    return float(k_real.mean() + k_fake.mean() - 2 * k_cross.mean())


def knn_radii(features, k=3, metric="euclidean", eps=1e-8):
    n = len(features)

    if n < 2:
        return np.full(n, eps, dtype=np.float64)

    k_eff = min(k, n - 1)
    distances = pairwise_distances(features, features, metric=metric)
    radii = np.partition(distances, kth=k_eff, axis=1)[:, k_eff]

    return np.maximum(radii, eps).astype(np.float64)


def precision_recall_density_coverage(
    real_features,
    fake_features,
    k=3,
    metric="euclidean",
    eps=1e-8,
):
    real_radii = knn_radii(real_features, k=k, metric=metric, eps=eps)
    fake_radii = knn_radii(fake_features, k=k, metric=metric, eps=eps)

    dist_fake_real = pairwise_distances(
        fake_features,
        real_features,
        metric=metric,
    )

    precision = (dist_fake_real <= real_radii).any(axis=1).mean().item()

    k_eff_real = min(k, max(1, len(real_features) - 1))
    kth_fake_real = np.partition(
        dist_fake_real,
        kth=k_eff_real,
        axis=1,
    )[:, k_eff_real]

    density = (
        (dist_fake_real <= kth_fake_real[:, None]).sum(axis=1)
        / max(1, k_eff_real)
    ).mean().item()

    dist_real_fake = pairwise_distances(
        real_features,
        fake_features,
        metric=metric,
    )

    recall = (dist_real_fake <= fake_radii).any(axis=1).mean().item()

    nearest_real_fake = dist_real_fake.min(axis=1)
    coverage = (nearest_real_fake <= real_radii).mean().item()

    return (
        float(precision),
        float(recall),
        float(density),
        float(coverage),
    )


def calculate_distribution_metrics_full(
    real_path: str,
    fake_path: str,
    device: torch.device,
    crop_bottom: int = 0,
) -> Dict[str, float]:
    batch_size = 64
    num_workers = 2

    try:
        real_dataset = ImageFolderDataset(real_path, crop_bottom=crop_bottom)
        fake_dataset = ImageFolderDataset(fake_path, crop_bottom=crop_bottom)
    except ValueError as exc:
        print(f"  [WARN] Distribution metrics skipped: {exc}")
        return {
            "IS_mean": -1,
            "IS_std": -1,
            "FID": -1,
            "KID_mean": -1,
            "KID_std": -1,
            "MMD_RBF": -1,
            "Precision": -1,
            "Recall": -1,
            "Density": -1,
            "Coverage": -1,
        }

    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    feature_net = InceptionPool3().to(device).eval()
    classifier_net = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1
    ).to(device).eval()

    real_features = get_features(real_loader, feature_net, device)
    fake_features = get_features(fake_loader, feature_net, device)

    fake_probs = get_logits(fake_loader, classifier_net, device)
    is_mean, is_std = inception_score(fake_probs)

    mu_real, sigma_real = compute_stats(real_features)
    mu_fake, sigma_fake = compute_stats(fake_features)

    fid_value = fid(mu_real, sigma_real, mu_fake, sigma_fake)
    kid_mean, kid_std = kid_poly(real_features, fake_features)
    mmd_value = mmd_rbf(real_features, fake_features)

    precision, recall, density, coverage = precision_recall_density_coverage(
        real_features,
        fake_features,
    )

    return {
        "IS_mean": is_mean,
        "IS_std": is_std,
        "FID": fid_value,
        "KID_mean": kid_mean,
        "KID_std": kid_std,
        "MMD_RBF": mmd_value,
        "Precision": precision,
        "Recall": recall,
        "Density": density,
        "Coverage": coverage,
    }


# =============================================================================
# REPORT ENCODING
# =============================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            value = float(obj)
            if value == float("inf"):
                return "Infinity"
            return value

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if obj == float("inf"):
            return "Infinity"

        return json.JSONEncoder.default(self, obj)


# =============================================================================
# PROCESSING
# =============================================================================

def process_single_folder(
    job_name,
    gt_folder,
    gen_folder,
    output_dir,
    device,
    segformer_model,
    image_processor,
    yolo_model,
    args,
):
    print(f"\n{'=' * 70}")
    print(f"Processing: {job_name}")
    print(f"  GT:  {gt_folder}")
    print(f"  Gen: {gen_folder}")

    json_output_path = os.path.join(
        output_dir,
        f"{job_name}_image_level_report.json",
    )

    if os.path.exists(json_output_path):
        print("[INFO] JSON report already exists. Skipping computation.")
        try:
            with open(json_output_path, "r") as file:
                existing_data = json.load(file)
            return existing_data.get("distribution_metrics", {})
        except Exception as exc:
            print(f"[WARN] Could not load existing JSON: {exc}")
            return {}

    matched_pairs = get_matched_image_pairs(gt_folder, gen_folder)

    if not matched_pairs:
        print("[WARN] No matching frame ids found. Skipping.")
        return {}

    folder_results = {}
    metric_sums = {key: 0.0 for key in IMAGE_LEVEL_KEYS}
    metric_counts = {key: 0 for key in IMAGE_LEVEL_KEYS}

    for gt_filename, gen_filename, frame_id in tqdm(
        matched_pairs,
        desc=f"  Image-level metrics ({job_name})",
    ):
        gt_path = os.path.join(gt_folder, gt_filename)
        gen_path = os.path.join(gen_folder, gen_filename)

        metrics = calculate_single_metrics_all(
            gt_path,
            gen_path,
            segformer_model,
            image_processor,
            yolo_model,
            target_resolution=args.target_resolution,
            crop_bottom=args.crop_bottom,
        )

        result_key = f"{frame_id:06d}"
        folder_results[result_key] = {
            "frame_id": frame_id,
            "gt_file": gt_filename,
            "gen_file": gen_filename,
            "metrics": metrics,
        }

        for key, value in metrics.items():
            if (
                isinstance(value, (int, float))
                and value >= 0
                and value != float("inf")
            ):
                metric_sums[key] += value
                metric_counts[key] += 1

    average_metrics = {}

    for key in IMAGE_LEVEL_KEYS:
        if metric_counts[key] > 0:
            average_metrics[key] = round(
                metric_sums[key] / metric_counts[key],
                4,
            )
        else:
            average_metrics[key] = 0.0

    distribution_stdev_metrics = calculate_distribution_metrics_avg({
        frame_id: item["metrics"]
        for frame_id, item in folder_results.items()
    })

    distribution_metrics = {}

    if not args.skip_distribution:
        print(f"  Running distribution-level metrics for {job_name}...")
        distribution_metrics = calculate_distribution_metrics_full(
            gt_folder,
            gen_folder,
            device,
            crop_bottom=args.crop_bottom,
        )

        average_metrics["FID"] = safe_round(
            distribution_metrics.get("FID", -1.0),
            4,
        )
        average_metrics["KID_mean"] = safe_round(
            distribution_metrics.get("KID_mean", -1.0),
            4,
        )
        average_metrics["IS_mean"] = safe_round(
            distribution_metrics.get("IS_mean", -1.0),
            4,
        )
        average_metrics["MMD_RBF"] = safe_round(
            distribution_metrics.get("MMD_RBF", -1.0),
            4,
        )
        average_metrics["PRDC_Precision"] = safe_round(
            distribution_metrics.get("Precision", -1.0),
            4,
        )
        average_metrics["PRDC_Recall"] = safe_round(
            distribution_metrics.get("Recall", -1.0),
            4,
        )
        average_metrics["PRDC_Density"] = safe_round(
            distribution_metrics.get("Density", -1.0),
            4,
        )
        average_metrics["PRDC_Coverage"] = safe_round(
            distribution_metrics.get("Coverage", -1.0),
            4,
        )

    temporal_metrics = {}

    if not args.skip_temporal:
        print(f"  Running temporal consistency metrics for {job_name}...")
        temporal_metrics = calculate_temporal_consistency(
            gen_folder,
            segformer_model,
            image_processor,
            target_resolution=args.target_resolution,
            crop_bottom=args.crop_bottom,
        )

        print(
            f"  Temporal: "
            f"SSIM={temporal_metrics['Temp_SSIM']} "
            f"PSNR={temporal_metrics['Temp_PSNR']} "
            f"MSE={temporal_metrics['Temp_MSE']} "
            f"CPL={temporal_metrics['Temp_CPL']} "
            f"pairs={temporal_metrics['Temp_num_pairs']}"
        )

        average_metrics["Temp_SSIM"] = temporal_metrics.get("Temp_SSIM", -1)
        average_metrics["Temp_PSNR"] = temporal_metrics.get("Temp_PSNR", -1)
        average_metrics["Temp_MSE"] = temporal_metrics.get("Temp_MSE", -1)
        average_metrics["Temp_CPL"] = temporal_metrics.get("Temp_CPL", -1)

    gt_index, _ = build_frame_index(gt_folder)
    gen_index, _ = build_frame_index(gen_folder)

    image_level_report = {
        "job_name": job_name,
        "gt_folder": gt_folder,
        "gen_folder": gen_folder,
        "total_gt_images_with_frame_id": len(gt_index),
        "total_gen_images_with_frame_id": len(gen_index),
        "matched_images_compared": len(matched_pairs),
        "target_resolution": args.target_resolution,
        "crop_bottom": args.crop_bottom,
        "average_metrics": average_metrics,
        "distribution_stdev_metrics": distribution_stdev_metrics,
        "distribution_metrics": distribution_metrics,
        "temporal_metrics": temporal_metrics,
        "image_metrics": folder_results,
    }

    with open(json_output_path, "w") as file:
        json.dump(image_level_report, file, indent=4, cls=NpEncoder)

    print(f"[OK] Report saved: {json_output_path}")
    print("-" * 70)

    return distribution_metrics


def validate_input_folders(gt_folder, input_folder):
    if not os.path.isdir(gt_folder):
        print(f"[ERROR] Ground-truth folder not found: {gt_folder}")
        return False

    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return False

    gt_images = get_image_filenames(gt_folder)

    if not gt_images:
        print(f"[ERROR] No images found in ground-truth folder: {gt_folder}")
        return False

    return True


def process_folders(args, config):
    gt_folder = config["gt_folder"]
    input_folder = config["input_folder"]
    output_dir = config["output_folder"]
    single_folder_mode = config["single_folder_mode"]

    if not validate_input_folders(gt_folder, input_folder):
        return 1

    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"[INFO] Using PyTorch device: {device}")

    gt_all_files = get_image_filenames(gt_folder)
    print(f"[INFO] Ground-truth folder: {gt_folder}")
    print(f"[INFO] Ground-truth images: {len(gt_all_files)}")

    segformer_model, image_processor = None, None

    if not args.skip_segformer:
        segformer_model, image_processor = get_segformer_model()

    yolo_model = None

    if not args.skip_yolo:
        yolo_model = load_yolo_model()

    print("-" * 70)

    distribution_results_list = []

    if single_folder_mode:
        distribution_metrics = process_single_folder(
            config["job_name"],
            gt_folder,
            input_folder,
            output_dir,
            device,
            segformer_model,
            image_processor,
            yolo_model,
            args,
        )

        if distribution_metrics:
            distribution_results_list.append({
                "Folder-Name": config["job_name"],
                **distribution_metrics,
            })

    else:
        exclude_list = args.exclude_subfolders or []

        subfolders = [
            name
            for name in os.listdir(input_folder)
            if os.path.isdir(os.path.join(input_folder, name))
            and name not in exclude_list
        ]

        subfolders = natsorted(subfolders)

        if not subfolders:
            print(f"[ERROR] No subfolders found in {input_folder}")
            return 1

        for subfolder_name in subfolders:
            subfolder_path = os.path.join(input_folder, subfolder_name)

            distribution_metrics = process_single_folder(
                subfolder_name,
                gt_folder,
                subfolder_path,
                output_dir,
                device,
                segformer_model,
                image_processor,
                yolo_model,
                args,
            )

            if distribution_metrics:
                distribution_results_list.append({
                    "Folder-Name": subfolder_name,
                    **distribution_metrics,
                })

    if distribution_results_list:
        df_distribution = pd.DataFrame(distribution_results_list)
        csv_path = os.path.join(output_dir, "distribution_summary.csv")
        df_distribution.to_csv(csv_path, index=False)
        print(f"\n[OK] Distribution summary saved to: {csv_path}")

    print("\n[OK] All processing complete.")
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    args = parse_args()

    try:
        config = resolve_runtime_config(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=" * 70)
    print("  OFFLINE IMAGE METRICS PIPELINE")
    print("=" * 70)
    print(f"  Project root:       {PROJECT_ROOT}")
    print(f"  Pipeline mode:      {config['pipeline_mode']}")
    print(f"  Single folder mode: {config['single_folder_mode']}")
    print(f"  Bag stem:           {config['bag_stem'] or '(manual)'}")
    print(f"  Method:             {config['method'] or '(manual)'}")
    print(f"  GT folder:          {config['gt_folder']}")
    print(f"  Input folder:       {config['input_folder']}")
    print(f"  Output reports:     {config['output_folder']}")
    print(f"  Job name:           {config['job_name']}")
    print(
        f"  Target resolution:  "
        f"{args.target_resolution or 'Original (no resize)'}"
    )
    if args.crop_bottom > 0:
        print(f"  Crop bottom:        {args.crop_bottom}px")
    else:
        print("  Crop bottom:        None")
    print(f"  Skip SegFormer:     {args.skip_segformer}")
    print(f"  Skip YOLO:          {args.skip_yolo}")
    print(f"  Skip Distribution:  {args.skip_distribution}")
    print(f"  Skip Temporal:      {args.skip_temporal}")
    print("=" * 70)

    return process_folders(args, config)


if __name__ == "__main__":
    sys.exit(main())