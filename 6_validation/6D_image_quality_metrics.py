import os
import sys
import json
import csv
import gc
import argparse
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any, Union

# --- Core Metric Imports ---
import cv2 
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from scipy import linalg
from sklearn.metrics import pairwise_distances
from glob import glob
import pandas as pd
from natsort import natsorted
from shapely.geometry import box
from collections import Counter
# ---------------------------

# ==============================================================================
# --- Configuration Constants (non-path defaults) ---
# ==============================================================================

TARGET_RESOLUTION = None  # Set to (W, H) tuple to force resize, or None for original

# PERCEPTUAL/SEMANTIC CONFIG
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" 

# VEHICLE CONSISTENCY CONFIG
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck", 1: "bicycle"}
MIN_VEHICLE_AREA = 600
IOU_THRESHOLD = 0.5
YOLO_MODEL_NAME = "yolov8n.pt" 

# IMAGE EXTENSIONS TO CONSIDER
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


# ==============================================================================
# --- CLI Argument Parsing ---
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Image Metrics Pipeline — compute image-level, distribution-level, and temporal metrics."
    )
    parser.add_argument(
        "--gt-folder", "-g",
        required=True,
        help="Path to the ground truth (real) images folder."
    )
    parser.add_argument(
        "--input-folder", "-i",
        required=True,
        help="Path to the generated images folder (contains subfolders to evaluate)."
    )
    parser.add_argument(
        "--output-folder", "-o",
        required=True,
        help="Path where metric reports (JSON + CSV) will be saved."
    )
    parser.add_argument(
        "--target-resolution", "-r",
        type=str,
        default=None,
        help="Optional target resolution as WxH (e.g. 1024x512). Default: original resolution."
    )
    parser.add_argument(
        "--skip-segformer",
        action="store_true",
        help="Skip SegFormer-based metrics (CPL, SegScore, Temp_CPL)."
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO-based vehicle consistency metrics."
    )
    parser.add_argument(
        "--skip-distribution",
        action="store_true",
        help="Skip distribution-level metrics (FID, KID, IS, MMD, PRDC)."
    )
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal consistency metrics (Temp_SSIM, Temp_PSNR, Temp_MSE, Temp_CPL)."
    )
    parser.add_argument(
        "--exclude-subfolders",
        type=str,
        nargs="*",
        default=["old", "depth", "canny", "seg", "baseline"],
        help="Subfolder names to exclude from evaluation. Default: old depth canny seg baseline"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Flat mode: input-folder contains images directly (no subfolder iteration)."
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Name for this job (used in report filenames). Default: input folder basename."
    )

    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="Number of pixels to crop from the bottom of both GT and generated images before comparison (e.g. 45 to remove hood)."
    )

    args = parser.parse_args()

    # Parse target resolution if provided
    if args.target_resolution:
        try:
            w, h = args.target_resolution.lower().split("x")
            args.target_resolution = (int(w), int(h))
        except ValueError:
            parser.error(f"Invalid resolution format: '{args.target_resolution}'. Use WxH (e.g. 1024x512).")
    
    return args


# ==============================================================================
# --- Utility & Setup Functions ---
# ==============================================================================

def get_device():
    """Returns the torch device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_image_filenames(folder: str) -> List[str]:
    """Returns sorted list of image filenames in a folder."""
    if not os.path.isdir(folder):
        print(f" Folder does not exist: {folder}")
        return []
    return sorted([
        f for f in os.listdir(folder) 
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])


def get_matched_filenames(gt_folder: str, gen_folder: str) -> List[str]:
    """
    Returns the sorted list of filenames that exist in BOTH the ground truth 
    and generated folders.
    """
    gt_files = set(get_image_filenames(gt_folder))
    gen_files = set(get_image_filenames(gen_folder))
    
    matched = sorted(gt_files & gen_files)
    
    gt_only = gt_files - gen_files
    gen_only = gen_files - gt_files
    
    print(f"   GT images: {len(gt_files)} | Generated images: {len(gen_files)} | Matched: {len(matched)}")
    if gt_only:
        print(f"    {len(gt_only)} GT images have no match in generated folder (skipped)")
    if gen_only:
        print(f"    {len(gen_only)} generated images have no match in GT folder (skipped)")
    
    return matched


# --- SegFormer Helpers (for CPL/SegScore) ---

def get_segformer_model():
    """Initializes and returns the SegFormer model and its image processor."""
    device = get_device()
    print(f"\n Initializing SegFormer model on device: {device}...")
    try:
        image_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
        model.eval().to(device)
        print(" SegFormer Model Initialized.")
        return model, image_processor
    except Exception as e:
        print(f"Error initializing SegFormer model ({SEGFORMER_MODEL}): {e}")
        print(" CPL and SegScore metrics will fail.")
        return None, None

def decode_cityscapes_mask(predicted_mask):
    """Decodes the predicted class IDs into Cityscapes color mask."""
    cityscapes_palette = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    h, w = predicted_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_palette):
        rgb_mask[predicted_mask == class_id] = color
    return rgb_mask

def encode_cityscapes_mask(rgb_img, carla_mode=False):
    """Encodes an RGB Cityscapes-like mask into class IDs (0-18)."""
    rgb = np.array(rgb_img)
    h, w, _ = rgb.shape
    label_mask = np.full((h, w), fill_value=-1, dtype=np.int64)

    segmentation_colors = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]

    for class_id, color in enumerate(segmentation_colors):
        mask = np.all(rgb == color, axis=-1)
        label_mask[mask] = class_id

    return torch.from_numpy(label_mask)


# --- YOLO Model Loading (for Vehicle Consistency) ---

class MockYoloResult:
    def __init__(self, data):
        self.boxes = self.MockBoxes(data)
    class MockBoxes:
        def __init__(self, data):
            self.data = torch.tensor(data)
        def cpu(self): return self

def load_yolo_model():
    """Loads the YOLO model (assuming Ultralytics dependency)."""
    try:
        from ultralytics import YOLO
        print(f"\n Loading YOLO model: {YOLO_MODEL_NAME}...")
        model = YOLO(YOLO_MODEL_NAME)
        print(" YOLO Model Loaded.")
        return model
    except ImportError:
        print(" Ultralytics YOLO not installed. Vehicle Consistency will be skipped.")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def calculate_yolo_image(yolo_model, pil_image: Image.Image):
    """Calculates YOLO results for a PIL image."""
    if yolo_model is None:
        return [MockYoloResult([])], 0.0
    results = yolo_model(pil_image, verbose=False)
    return results, 0.0

# Transforms for YOLO input (512x512)
image_transforms_512 = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ]
)

# --- Vehicle Consistency Core Logic ---

def iou(box1: List[float], box2: List[float]) -> float:
    """Calculates IoU between two bounding boxes."""
    b1 = box(box1[0], box1[1], box1[2], box1[3])
    b2 = box(box2[0], box2[1], box2[2], box2[3])
    return b1.intersection(b2).area / b1.union(b2).area if b1.union(b2).area > 0 else 0

def extract_vehicles(results) -> List[Dict[str, Union[List[float], float, str]]]:
    """Extracts vehicle bounding boxes, confidence, and class name from YOLO results."""
    vehicles = []
    for r in results:
        for data_row in r.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = data_row
            cls = int(cls)
            if cls in VEHICLE_CLASSES:
                vehicles.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class": VEHICLE_CLASSES[cls]
                })
    return vehicles

def match_vehicles(real_vehicles, gen_vehicles, iou_thresh=IOU_THRESHOLD):
    """Matches real vehicles to generated vehicles based on IoU."""
    matches = []
    used_gen_indices = set()

    for rv in real_vehicles:
        best_match = None
        best_iou = 0
        best_i = -1
        for i, gv in enumerate(gen_vehicles):
            if i in used_gen_indices:
                continue
            iou_score = iou(rv["bbox"], gv["bbox"])
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = (rv, gv, iou_score)
                best_i = i

        if best_match and best_iou >= iou_thresh:
            matches.append(best_match)
            used_gen_indices.add(best_i)

    return matches, used_gen_indices

def filter_large_vehicles(vehicles, min_area=MIN_VEHICLE_AREA):
    """Filters vehicles based on minimum bounding box area."""
    return [v for v in vehicles if (v["bbox"][2]-v["bbox"][0])*(v["bbox"][3]-v["bbox"][1]) >= min_area]


# ==============================================================================
# --- Image-Level Metric Calculation Functions ---
# ==============================================================================

def compute_ssim(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM)."""
    image1_gray = cv2.cvtColor(image1_cv2, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_cv2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(image1_gray, image2_gray, data_range=image1_gray.max() - image1_gray.min(), full=True)
    return float(score)

def compute_psnr(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((image1_cv2.astype(np.float64) - image2_cv2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)

def compute_mse(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    """Calculates Mean Squared Error (MSE)."""
    return float(np.mean((image1_cv2.astype(np.float64) - image2_cv2.astype(np.float64)) ** 2))

def compute_cpl(image1_rgb: np.ndarray, image2_rgb: np.ndarray, model_cpl, transform_cpl) -> float:
    """Calculates CLIP-Perceptual Loss (CPL) using a SegFormer model's features."""
    img1_pil = Image.fromarray(image1_rgb)
    img2_pil = Image.fromarray(image2_rgb)
    
    inputs1 = transform_cpl(images=img1_pil, return_tensors="pt")
    inputs2 = transform_cpl(images=img2_pil, return_tensors="pt")

    device = model_cpl.device
    img1_tensor = inputs1["pixel_values"].to(device)
    img2_tensor = inputs2["pixel_values"].to(device)
    
    with torch.no_grad():
        outputs1 = model_cpl(img1_tensor, output_hidden_states=True)
        outputs2 = model_cpl(img2_tensor, output_hidden_states=True)
        
        features1 = outputs1.hidden_states[-1] 
        features2 = outputs2.hidden_states[-1]

    cpl_loss = torch.nn.functional.mse_loss(features1, features2).item()
    return cpl_loss

def calculate_semantic_segmentation_score(model_seg, image_extractor, image_seg: np.ndarray, image_created: np.ndarray, carla_mode=False) -> Tuple[float, np.ndarray]:
    """Calculates the MSE of class IDs between predicted and ground truth segmentation."""
    inputs = image_extractor(images=image_created, return_tensors="pt")
    inputs = {k: v.to(model_seg.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_seg(**inputs)
        logits = outputs.logits
        target_size = image_created.shape[:2]

        upsampled = torch.nn.functional.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
        predicted = upsampled.argmax(1)[0].cpu().numpy()

    segmentation_image = decode_cityscapes_mask(predicted)

    if image_seg is None:
        return -1.0, segmentation_image 

    pred_ids = encode_cityscapes_mask(segmentation_image, carla_mode=carla_mode)
    gt_ids   = encode_cityscapes_mask(image_seg, carla_mode=carla_mode)

    pred_ids = pred_ids.float()
    gt_ids   = gt_ids.float()

    mse_score = torch.nn.functional.mse_loss(pred_ids, gt_ids).item()
    return mse_score, segmentation_image


def calculate_single_metrics_all(
    image_a_path: str, 
    image_b_path: str,
    segformer_model,
    image_processor,
    yolo_model,
    target_resolution: Tuple[int, int] = None,
    crop_bottom: int = 0
) -> dict:
    """
    Calculates all metrics for a single pair of images.
    Optionally resizes both images to target_resolution before comparison.
    Optionally crops crop_bottom pixels from the bottom of both images.
    """
    metrics = {
        'MSE': -3, 'PSNR': -3, 'SSIM': -3, 'CPL': -3, 'SegScore': -3,
        'Veh_Recall': -3, 'Veh_Precision': -3, 'Veh_AvgIoU': -3
    }
    
    try:
        img_a_cv2 = cv2.imread(image_a_path, 1) 
        img_b_cv2 = cv2.imread(image_b_path, 1)

        if img_a_cv2 is None or img_b_cv2 is None:
             return {k: -2 for k in metrics.keys()}

        if target_resolution is not None:
            w, h = target_resolution
            img_a_cv2 = cv2.resize(img_a_cv2, (w, h), interpolation=cv2.INTER_LINEAR)
            img_b_cv2 = cv2.resize(img_b_cv2, (w, h), interpolation=cv2.INTER_LINEAR)

        if img_a_cv2.shape != img_b_cv2.shape:
            h, w = img_a_cv2.shape[:2]
            img_b_cv2 = cv2.resize(img_b_cv2, (w, h), interpolation=cv2.INTER_LINEAR)

        # Crop bottom pixels (e.g. to remove vehicle hood)
        if crop_bottom > 0:
            img_a_cv2 = img_a_cv2[:-crop_bottom, :, :]
            img_b_cv2 = img_b_cv2[:-crop_bottom, :, :]

        img_a_rgb = cv2.cvtColor(img_a_cv2, cv2.COLOR_BGR2RGB)
        img_b_rgb = cv2.cvtColor(img_b_cv2, cv2.COLOR_BGR2RGB)

        # 1. Pixel-based Metrics
        metrics['MSE'] = round(compute_mse(img_a_cv2, img_b_cv2), 4)
        metrics['PSNR'] = round(compute_psnr(img_a_cv2, img_b_cv2), 4)
        metrics['SSIM'] = round(compute_ssim(img_a_cv2, img_b_cv2), 4)
        
        # 2. Perceptual/Semantic Metrics
        if segformer_model is not None:
            metrics['CPL'] = round(compute_cpl(img_a_rgb, img_b_rgb, segformer_model, image_processor), 4)
            seg_score_val, _ = calculate_semantic_segmentation_score(
                segformer_model, image_processor, 
                image_seg=img_a_rgb, 
                image_created=img_b_rgb 
            )
            metrics['SegScore'] = round(seg_score_val, 4)
        
        # 3. Vehicle Consistency Metrics
        if yolo_model is not None:
            real_image_pil = Image.fromarray(img_a_rgb).convert("RGB")
            gen_image_pil = Image.fromarray(img_b_rgb).convert("RGB")
            
            real_image_t = image_transforms_512(real_image_pil)
            gen_image_t = image_transforms_512(gen_image_pil)

            yolo_results_real, _ = calculate_yolo_image(yolo_model, real_image_t)
            yolo_results_sim, _ = calculate_yolo_image(yolo_model, gen_image_t)

            real_vehicles = extract_vehicles(yolo_results_real)
            gen_vehicles = extract_vehicles(yolo_results_sim)

            real_vehicles = filter_large_vehicles(real_vehicles, min_area=MIN_VEHICLE_AREA)
            gen_vehicles = filter_large_vehicles(gen_vehicles, min_area=MIN_VEHICLE_AREA)

            matches, _ = match_vehicles(real_vehicles, gen_vehicles, iou_thresh=IOU_THRESHOLD)
            
            real_count = len(real_vehicles)
            gen_count = len(gen_vehicles)
            match_count = len(matches)

            recall = match_count / real_count if real_count else 0
            precision = match_count / gen_count if gen_count else 0
            ious = [m[2] for m in matches]
            avg_iou = np.mean(ious) if ious else 0

            metrics['Veh_Recall'] = round(recall, 4)
            metrics['Veh_Precision'] = round(precision, 4)
            metrics['Veh_AvgIoU'] = round(avg_iou, 4)
        
        return metrics
    
    except Exception as e:
        print(f"   Error processing {os.path.basename(image_a_path)}: {e}")
        return {k: -4 for k in metrics.keys()}


def calculate_distribution_metrics_avg(all_results: dict) -> dict:
    """Calculates standard deviation across all image-level metrics."""
    metrics_to_track = ['MSE', 'PSNR', 'SSIM', 'CPL', 'SegScore', 'Veh_Recall', 'Veh_Precision', 'Veh_AvgIoU']
    distribution_metrics = {}
    
    for metric_name in metrics_to_track:
        values = []
        for metrics in all_results.values():
            val = metrics.get(metric_name)
            if isinstance(val, (int, float)) and val >= 0 and val != float('inf'):
                values.append(val)

        if not values:
            distribution_metrics[f'stdev_{metric_name}'] = 0.0
            continue
        
        stdev = np.std(values)
        distribution_metrics[f'stdev_{metric_name}'] = round(stdev, 4)

    return distribution_metrics


# ==============================================================================
# --- Temporal Consistency Metrics ---
# ==============================================================================

def calculate_temporal_consistency(
    gen_folder: str,
    segformer_model,
    image_processor,
    target_resolution: Tuple[int, int] = None,
    crop_bottom: int = 0,
) -> Dict[str, float]:
    """
    For each consecutive pair (frame_i, frame_{i+1}) inside gen_folder, compute
    SSIM, PSNR, MSE, and (if SegFormer is loaded) CPL between the two frames.
    Returns the mean over all valid pairs.

    Pairs with MSE == 0 (duplicate / identical frames) are skipped from the
    pixel-level averages, since they typically indicate generation artefacts
    rather than meaningful temporal stability.
    """
    gen_images = natsorted([
        f for f in os.listdir(gen_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])
    num_pairs = len(gen_images) - 1
    if num_pairs < 1:
        return {
            'Temp_SSIM': -1, 'Temp_PSNR': -1, 'Temp_MSE': -1, 'Temp_CPL': -1,
            'Temp_num_pairs': 0,
        }

    ssim_scores, psnr_scores, mse_scores, cpl_scores = [], [], [], []

    for i in tqdm(range(num_pairs), desc="    Temporal pairs", leave=False):
        path_curr = os.path.join(gen_folder, gen_images[i])
        path_next = os.path.join(gen_folder, gen_images[i + 1])
        img_curr = cv2.imread(path_curr, 1)
        img_next = cv2.imread(path_next, 1)
        if img_curr is None or img_next is None:
            continue

        if target_resolution is not None:
            w, h = target_resolution
            img_curr = cv2.resize(img_curr, (w, h), interpolation=cv2.INTER_LINEAR)
            img_next = cv2.resize(img_next, (w, h), interpolation=cv2.INTER_LINEAR)
        if img_curr.shape != img_next.shape:
            h, w = img_curr.shape[:2]
            img_next = cv2.resize(img_next, (w, h), interpolation=cv2.INTER_LINEAR)

        if crop_bottom > 0:
            img_curr = img_curr[:-crop_bottom, :, :]
            img_next = img_next[:-crop_bottom, :, :]

        ssim_val = compute_ssim(img_curr, img_next)
        psnr_val = compute_psnr(img_curr, img_next)
        mse_val = compute_mse(img_curr, img_next)

        if psnr_val != float('inf') and mse_val != 0.0:
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
            mse_scores.append(mse_val)

        if segformer_model is not None:
            img_curr_rgb = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)
            img_next_rgb = cv2.cvtColor(img_next, cv2.COLOR_BGR2RGB)
            cpl_val = compute_cpl(img_curr_rgb, img_next_rgb, segformer_model, image_processor)
            cpl_scores.append(cpl_val)

        if i % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        'Temp_SSIM': round(float(np.mean(ssim_scores)), 4) if ssim_scores else -1,
        'Temp_PSNR': round(float(np.mean(psnr_scores)), 4) if psnr_scores else -1,
        'Temp_MSE':  round(float(np.mean(mse_scores)), 4) if mse_scores else -1,
        'Temp_CPL':  round(float(np.mean(cpl_scores)), 4) if cpl_scores else -1,
        'Temp_num_pairs': len(ssim_scores),
    }


# ==============================================================================
# --- Distribution-Level Metric Calculation Functions ---
# ==============================================================================

image_transforms_inception = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]
)

class ImageFolderDataset(Dataset):
    def __init__(self, root, exts=IMAGE_EXTENSIONS, crop_bottom=0):
        self.paths = []
        for e in exts:
            self.paths.extend(glob(os.path.join(root, f"**/*{e}"), recursive=True))
        if not self.paths:
             raise ValueError(f"No images found in {root}")
        self.tf = image_transforms_inception
        self.crop_bottom = crop_bottom

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            img = im.convert("RGB")
            if self.crop_bottom > 0:
                w, h = img.size
                img = img.crop((0, 0, w, h - self.crop_bottom))
            return self.tf(img)

class InceptionPool3(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        net.eval()
        for p in net.parameters(): 
            p.requires_grad = False
            
        layers = []
        for name, module in net.named_children():
            if name in ['AuxLogits', 'dropout', 'fc']:
                continue
            layers.append(module)

        self.features = nn.Sequential(*layers[:-1]) 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,x):
        with torch.no_grad():
            x = self.features(x)
            x = self.pool(x)
            return torch.flatten(x, 1)

def get_features(loader, model, device):
    feats=[]
    for x in tqdm(loader, desc="  Extracting Features"):
        feats.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(feats,0)

def get_logits(loader, model, device):
    probs=[]
    with torch.no_grad():
        for x in tqdm(loader, desc="  Calculating Logits"):
            logits = model(x.to(device))
            if isinstance(logits,tuple): logits=logits[0]
            probs.append(F.softmax(logits,dim=1).cpu().numpy())
    return np.concatenate(probs,0)

def compute_stats(feats): return feats.mean(0), np.cov(feats,rowvar=False)

def _sqrtm(c1,c2,eps=1e-6):
    c1=c1.copy(); c2=c2.copy()
    c1.flat[::c1.shape[0]+1]+=eps
    c2.flat[::c2.shape[0]+1]+=eps
    cov,info = linalg.sqrtm(c1.dot(c2),disp=False)
    cov = cov.real if np.iscomplexobj(cov) else cov
    return cov

def inception_score(probs,splits=10):
    N=probs.shape[0]; split=N//splits; scores=[]
    for i in range(splits):
        part=probs[i*split:(i+1)*split]
        py=part.mean(0,keepdims=True)
        kl=part*(np.log(part+1e-10)-np.log(py+1e-10))
        scores.append(np.exp(kl.sum(1).mean()))
    return float(np.mean(scores)), float(np.std(scores))

def fid(mu1,s1,mu2,s2):
    diff=mu1-mu2; cov=_sqrtm(s1,s2)
    return float(diff.dot(diff)+np.trace(s1+s2-2*cov))

def kid_poly(X,Y,deg=3,gamma=None,coef0=1.0,subsets=100,subsize=1000):
    rng=np.random.default_rng(123); n=min(len(X),subsize); m=min(len(Y),subsize)
    if gamma is None: gamma=1.0/X.shape[1]
    vals=[]
    for _ in range(subsets):
        Xs=X[rng.choice(len(X),n,False)]; Ys=Y[rng.choice(len(Y),m,False)]
        Kxx=(gamma*Xs@Xs.T+coef0)**deg; Kyy=(gamma*Ys@Ys.T+coef0)**deg
        Kxy=(gamma*Xs@Ys.T+coef0)**deg
        np.fill_diagonal(Kxx,0); np.fill_diagonal(Kyy,0)
        vals.append(Kxx.sum()/(n*(n-1))+Kyy.sum()/(m*(m-1))-2*Kxy.mean())
    vals=np.array(vals)
    return float(vals.mean()),float(vals.std())

def mmd_rbf(X,Y,sigma='median'):
    Z=np.vstack([X,Y])
    if sigma=='median':
        sample=Z if len(Z)<2000 else Z[np.random.choice(len(Z),2000,False)]
        D=pairwise_distances(sample); sigma=np.median(D[D>0])
    gamma=1/(2*sigma*sigma)
    Kxx=np.exp(-gamma*pairwise_distances(X,X,squared=True))
    Kyy=np.exp(-gamma*pairwise_distances(Y,Y,squared=True))
    Kxy=np.exp(-gamma*pairwise_distances(X,Y,squared=True))
    return float(Kxx.mean()+Kyy.mean()-2*Kxy.mean())

def _knn_radii(X, k=3, metric='euclidean', eps=1e-8):
    n = len(X)
    if n < 2: return np.full(n, eps, dtype=np.float64)
    k_eff = min(k, n - 1)
    D = pairwise_distances(X, X, metric=metric)
    radii = np.partition(D, kth=k_eff, axis=1)[:, k_eff]
    radii = np.maximum(radii, eps).astype(np.float64)
    return radii

def precision_recall_density_coverage(real_feats, fake_feats, k=3, metric='euclidean', eps=1e-8):
    r_r = _knn_radii(real_feats, k=k, metric=metric, eps=eps)
    f_r = _knn_radii(fake_feats, k=k, metric=metric, eps=eps)

    D_fr = pairwise_distances(fake_feats, real_feats, metric=metric)
    precision = (D_fr <= r_r).any(axis=1).mean().item()

    k_eff_r = min(k, max(1, len(real_feats) - 1))
    kth_fr = np.partition(D_fr, kth=k_eff_r, axis=1)[:, k_eff_r]
    density = ((D_fr <= kth_fr[:, None]).sum(axis=1) / max(1, k_eff_r)).mean().item()

    D_rf = pairwise_distances(real_feats, fake_feats, metric=metric)
    recall = (D_rf <= f_r).any(axis=1).mean().item()

    nearest_rf = D_rf.min(axis=1)
    coverage = (nearest_rf <= r_r).mean().item()

    return float(precision), float(recall), float(density), float(coverage)


def calculate_distribution_metrics_full(real_path: str, fake_path: str, device: torch.device, crop_bottom: int = 0) -> Dict[str, float]:
    """Calculates all distribution-level metrics for one fake set."""
    batch_size = 64
    num_workers = 2

    try:
        real_ds=ImageFolderDataset(real_path, crop_bottom=crop_bottom)
        fake_ds=ImageFolderDataset(fake_path, crop_bottom=crop_bottom)
    except ValueError as e:
        return {
            "IS_mean": -1, "IS_std": -1, "FID": -1, "KID_mean": -1, 
            "KID_std": -1, "MMD_RBF": -1, "Precision": -1, "Recall": -1, 
            "Density": -1, "Coverage": -1
        }
    
    real_loader=DataLoader(real_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    fake_loader=DataLoader(fake_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    feat_net=InceptionPool3().to(device).eval()
    cls_net=models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()

    real_feats=get_features(real_loader,feat_net,device)
    fake_feats=get_features(fake_loader,feat_net,device)

    fake_probs=get_logits(fake_loader,cls_net,device)
    is_mean,is_std=inception_score(fake_probs)

    mu_r,s_r=compute_stats(real_feats); mu_f,s_f=compute_stats(fake_feats)
    fid_val=fid(mu_r,s_r,mu_f,s_f)

    kid_mean,kid_std=kid_poly(real_feats,fake_feats)

    mmd_val=mmd_rbf(real_feats,fake_feats)

    prec,rec,dens,cov=precision_recall_density_coverage(real_feats,fake_feats)

    return {
        "IS_mean": is_mean, "IS_std": is_std, "FID": fid_val, 
        "KID_mean": kid_mean, "KID_std": kid_std, "MMD_RBF": mmd_val, 
        "Precision": prec, "Recall": rec, "Density": dens, "Coverage": cov,
    }


# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types and float('inf')."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float('inf') if obj == float('inf') else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "Infinity"
        return json.JSONEncoder.default(self, obj)


def process_single_folder(
    job_name, gt_folder, gen_folder, output_dir, device,
    segformer_model, image_processor, yolo_model, args
):
    """
    Runs all metrics for a single GT-vs-Generated folder pair.
    Returns the distribution metrics dict (for the summary CSV).
    """
    target_resolution = args.target_resolution
    crop_bottom = args.crop_bottom
    all_metric_keys = ['MSE', 'PSNR', 'SSIM', 'CPL', 'SegScore', 'Veh_Recall', 'Veh_Precision', 'Veh_AvgIoU']

    print(f"\n{'='*50}")
    print(f"Processing: **{job_name}**")
    print(f"  GT:  {gt_folder}")
    print(f"  Gen: {gen_folder}")

    # --- Check if already computed ---
    json_output_path = os.path.join(output_dir, f"{job_name}_image_level_report.json")

    if os.path.exists(json_output_path):
        print(f"   JSON report already exists. Skipping computation.")
        try:
            with open(json_output_path, 'r') as f:
                existing_data = json.load(f)
                return existing_data.get('distribution_metrics', {})
        except Exception as e:
            print(f"   Could not load existing JSON: {e}")
            return {}

    # --- Match filenames ---
    matched_filenames = get_matched_filenames(gt_folder, gen_folder)
    if not matched_filenames:
        print(f"   No matching filenames found. Skipping.")
        return {}

    # --- A. Image-Level Metrics ---
    folder_results = {}
    metric_sums = {key: 0.0 for key in all_metric_keys}
    metric_counts = {key: 0 for key in all_metric_keys}
    image_count = 0

    for image_name in tqdm(matched_filenames, desc=f"  Image-Level ({job_name})"):
        path_a = os.path.join(gt_folder, image_name)
        path_b = os.path.join(gen_folder, image_name)

        metrics = calculate_single_metrics_all(
            path_a, path_b,
            segformer_model, image_processor, yolo_model,
            target_resolution=target_resolution,
            crop_bottom=crop_bottom
        )
        folder_results[image_name] = metrics
        image_count += 1

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value >= 0 and value != float('inf'):
                metric_sums[key] += value
                metric_counts[key] += 1

    # Compute averages
    average_metrics = {}
    for k in all_metric_keys:
        if metric_counts[k] > 0:
            average_metrics[k] = round(metric_sums[k] / metric_counts[k], 4)
        else:
            average_metrics[k] = 0.0

    distribution_avg_metrics = calculate_distribution_metrics_avg(folder_results)

    # --- B. Distribution-Level Metrics ---
    dist_metrics = {}
    if not args.skip_distribution:
        print(f"  Running Distribution-Level Metrics for {job_name}...")
        dist_metrics = calculate_distribution_metrics_full(gt_folder, gen_folder, device, crop_bottom=crop_bottom)

        average_metrics['FID'] = round(dist_metrics.get('FID', -1.0), 4)
        average_metrics['KID_mean'] = round(dist_metrics.get('KID_mean', -1.0), 4)
        average_metrics['IS_mean'] = round(dist_metrics.get('IS_mean', -1.0), 4)
        average_metrics['MMD_RBF'] = round(dist_metrics.get('MMD_RBF', -1.0), 4)
        average_metrics['PRDC_Precision'] = round(dist_metrics.get('Precision', -1.0), 4)
        average_metrics['PRDC_Recall'] = round(dist_metrics.get('Recall', -1.0), 4)
        average_metrics['PRDC_Density'] = round(dist_metrics.get('Density', -1.0), 4)
        average_metrics['PRDC_Coverage'] = round(dist_metrics.get('Coverage', -1.0), 4)

    # --- C. Temporal Consistency Metrics ---
    temporal_metrics = {}
    if not args.skip_temporal:
        print(f"  Running Temporal Consistency Metrics for {job_name}...")
        temporal_metrics = calculate_temporal_consistency(
            gen_folder,
            segformer_model,
            image_processor,
            target_resolution=target_resolution,
            crop_bottom=crop_bottom,
        )
        print(f"  Temporal: SSIM={temporal_metrics['Temp_SSIM']}  "
              f"PSNR={temporal_metrics['Temp_PSNR']}  "
              f"MSE={temporal_metrics['Temp_MSE']}  "
              f"CPL={temporal_metrics['Temp_CPL']}  "
              f"pairs={temporal_metrics['Temp_num_pairs']}")

        average_metrics['Temp_SSIM'] = temporal_metrics.get('Temp_SSIM', -1)
        average_metrics['Temp_PSNR'] = temporal_metrics.get('Temp_PSNR', -1)
        average_metrics['Temp_MSE']  = temporal_metrics.get('Temp_MSE', -1)
        average_metrics['Temp_CPL']  = temporal_metrics.get('Temp_CPL', -1)

    gt_all_files = get_image_filenames(gt_folder)

    image_level_report = {
        'job_name': job_name,
        'total_gt_images': len(gt_all_files),
        'total_gen_images': len(get_image_filenames(gen_folder)),
        'matched_images_compared': image_count,
        'average_metrics': average_metrics,
        'distribution_stdev_metrics': distribution_avg_metrics,
        'distribution_metrics': dist_metrics,
        'temporal_metrics': temporal_metrics,
        'image_metrics': folder_results
    }

    with open(json_output_path, 'w') as f:
        json.dump(image_level_report, f, indent=4, cls=NpEncoder)
    print(f"   Report saved: {json_output_path}")
    print("-" * 50)

    return dist_metrics


def process_folders(args):
    """
    Main function. Supports two modes:
      --flat:  input-folder contains images directly (single GT vs single Gen comparison)
      default: input-folder contains subfolders, each compared against GT
    """
    gt_folder = args.gt_folder
    input_folder = args.input_folder
    output_dir = args.output_folder
    exclude_list = args.exclude_subfolders or []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    device = get_device()
    print(f"Using device for PyTorch: {device}")

    # --- Validate GT folder ---
    if not os.path.isdir(gt_folder):
        print(f" Ground truth folder not found: {gt_folder}")
        return 1

    gt_all_files = get_image_filenames(gt_folder)
    print(f" Ground truth folder: {gt_folder} ({len(gt_all_files)} images)")
    if not gt_all_files:
        print(f" No images found in ground truth folder. Stopping.")
        return 1

    # Initialize models (conditionally)
    segformer_model, image_processor = (None, None)
    if not args.skip_segformer:
        segformer_model, image_processor = get_segformer_model()

    yolo_model = None
    if not args.skip_yolo:
        yolo_model = load_yolo_model()

    print("-" * 50)

    distribution_results_list = []

    if args.flat:
        # ── FLAT MODE: input_folder IS the generated images folder directly ──
        if not os.path.isdir(input_folder):
            print(f" Input folder not found: {input_folder}")
            return 1

        job_name = args.job_name or os.path.basename(os.path.normpath(input_folder))

        dist_metrics = process_single_folder(
            job_name, gt_folder, input_folder, output_dir, device,
            segformer_model, image_processor, yolo_model, args
        )
        if dist_metrics:
            distribution_results_list.append({"Folder-Name": job_name, **dist_metrics})

    else:
        # ── SUBFOLDER MODE: iterate over subfolders inside input_folder ──
        subfolders_b = [
            d for d in os.listdir(input_folder)
            if os.path.isdir(os.path.join(input_folder, d))
            and d not in exclude_list
        ]

        if not subfolders_b:
            print(f" No subfolders found in {input_folder}")
            return 1

        for subfolder_name in natsorted(subfolders_b):
            subfolder_b_path = os.path.join(input_folder, subfolder_name)

            dist_metrics = process_single_folder(
                subfolder_name, gt_folder, subfolder_b_path, output_dir, device,
                segformer_model, image_processor, yolo_model, args
            )
            if dist_metrics:
                distribution_results_list.append({"Folder-Name": subfolder_name, **dist_metrics})

    # Save Final Distribution Summary CSV
    if distribution_results_list:
        df_dist = pd.DataFrame(distribution_results_list)
        csv_path_dist = os.path.join(output_dir, "distribution_summary.csv")
        df_dist.to_csv(csv_path_dist, index=False)
        print(f"\n Distribution Summary saved to {csv_path_dist}")

    print("\n All processing complete.")
    return 0


if __name__ == '__main__':
    args = parse_args()

    print("="*60)
    print("  OFFLINE IMAGE METRICS PIPELINE")
    print("="*60)
    print(f"  GT Folder:         {args.gt_folder}")
    print(f"  Input Folder:      {args.input_folder}")
    print(f"  Output Reports:    {args.output_folder}")
    print(f"  Flat Mode:         {args.flat}")
    print(f"  Job Name:          {args.job_name or '(auto)'}")
    print(f"  Target Resolution: {args.target_resolution or 'Original (no resize)'}")
    print(f"  Crop Bottom:       {args.crop_bottom}px" if args.crop_bottom > 0 else f"  Crop Bottom:       None")
    print(f"  Skip SegFormer:    {args.skip_segformer}")
    print(f"  Skip YOLO:         {args.skip_yolo}")
    print(f"  Skip Distribution: {args.skip_distribution}")
    print(f"  Skip Temporal:     {args.skip_temporal}")
    print("="*60)

    exit_code = process_folders(args)
    sys.exit(exit_code or 0)