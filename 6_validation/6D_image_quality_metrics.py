#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6D_image_quality_metrics.py

Compute visual-gap metrics for camera-ready evaluation.

The script compares real reference frames against:
    - CARLA replay frames
    - Gaussian Splatting replay frames

Computed metrics:
    - PSNR
    - SSIM
    - LPIPS
    - FID
    - KID
    - MMD

Input folders:
    data/raw_dataset/<BAG>/images
    data/data_for_carla/<BAG>/replay_results/<BAG>_replay/carla
    data/data_for_carla/<BAG>/replay_results/<BAG>_replay/gs

Output folder:
    results/<BAG>/image_quality

Usage:
    python 6_validation/6D_image_quality_metrics.py

    python 6_validation/6D_image_quality_metrics.py \
        --bag-name reference_bag

    python 6_validation/6D_image_quality_metrics.py \
        --bag-name reference_bag \
        --crop-bottom 45
"""

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_BAG_NAME = "reference_bag"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, FID, KID, and MMD."
    )

    parser.add_argument(
        "--bag-name",
        default=DEFAULT_BAG_NAME,
        help="Bag stem or bag filename. Default: reference_bag.",
    )

    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="Pixels cropped from the bottom of every image before comparison.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of matched frames to evaluate.",
    )

    parser.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone. Default: alex.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for Inception feature extraction. Default: 32.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers. Default: 2.",
    )

    args = parser.parse_args()

    if args.crop_bottom < 0:
        parser.error("--crop-bottom must be >= 0.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")

    return args


def get_bag_stem(bag_name: str) -> str:
    return os.path.splitext(os.path.basename(bag_name))[0]


def get_paths(bag_stem: str) -> Dict[str, str]:
    replay_root = os.path.join(
        PROJECT_ROOT,
        "data",
        "data_for_carla",
        bag_stem,
        "replay_results",
        f"{bag_stem}_replay",
    )

    return {
        "real": os.path.join(
            PROJECT_ROOT,
            "data",
            "raw_dataset",
            bag_stem,
            "images",
        ),
        "carla": os.path.join(replay_root, "carla"),
        "gs": os.path.join(replay_root, "gs"),
        "output": os.path.join(
            PROJECT_ROOT,
            "results",
            bag_stem,
            "image_quality",
        ),
    }


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


def build_frame_index(folder: str) -> Dict[int, str]:
    index = {}

    for filename in get_image_filenames(folder):
        frame_id = extract_frame_id(filename)

        if frame_id is None:
            continue

        if frame_id not in index:
            index[frame_id] = filename

    return index


def get_matched_pairs(
    real_folder: str,
    candidate_folder: str,
    max_frames: Optional[int],
) -> List[Tuple[str, str, int]]:
    real_index = build_frame_index(real_folder)
    candidate_index = build_frame_index(candidate_folder)

    matched_ids = sorted(set(real_index.keys()) & set(candidate_index.keys()))

    if max_frames is not None:
        matched_ids = matched_ids[:max_frames]

    print(
        f"    Real frames: {len(real_index)} | "
        f"Candidate frames: {len(candidate_index)} | "
        f"Matched: {len(matched_ids)}"
    )

    real_only = sorted(set(real_index.keys()) - set(candidate_index.keys()))
    candidate_only = sorted(set(candidate_index.keys()) - set(real_index.keys()))

    if real_only:
        print(f"    [WARN] Real-only frame ids: {real_only[:5]} ... total={len(real_only)}")

    if candidate_only:
        print(
            f"    [WARN] Candidate-only frame ids: "
            f"{candidate_only[:5]} ... total={len(candidate_only)}"
        )

    return [
        (
            os.path.join(real_folder, real_index[frame_id]),
            os.path.join(candidate_folder, candidate_index[frame_id]),
            frame_id,
        )
        for frame_id in matched_ids
    ]


def validate_folder(path: str, label: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} folder not found: {path}")

    if not get_image_filenames(path):
        raise FileNotFoundError(f"No images found in {label} folder: {path}")


def crop_bottom_cv2(image: np.ndarray, crop_bottom: int) -> np.ndarray:
    if crop_bottom <= 0:
        return image

    if image.shape[0] <= crop_bottom:
        raise ValueError(
            f"Cannot crop {crop_bottom}px from image with height {image.shape[0]}."
        )

    return image[:-crop_bottom, :, :]


def crop_bottom_pil(image: Image.Image, crop_bottom: int) -> Image.Image:
    if crop_bottom <= 0:
        return image

    width, height = image.size

    if height <= crop_bottom:
        raise ValueError(
            f"Cannot crop {crop_bottom}px from image with height {height}."
        )

    return image.crop((0, 0, width, height - crop_bottom))


def load_pair(
    real_path: str,
    candidate_path: str,
    crop_bottom: int,
) -> Tuple[np.ndarray, np.ndarray]:
    real_bgr = cv2.imread(real_path, cv2.IMREAD_COLOR)
    candidate_bgr = cv2.imread(candidate_path, cv2.IMREAD_COLOR)

    if real_bgr is None:
        raise ValueError(f"Could not read image: {real_path}")

    if candidate_bgr is None:
        raise ValueError(f"Could not read image: {candidate_path}")

    if real_bgr.shape != candidate_bgr.shape:
        height, width = real_bgr.shape[:2]
        candidate_bgr = cv2.resize(
            candidate_bgr,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    real_bgr = crop_bottom_cv2(real_bgr, crop_bottom)
    candidate_bgr = crop_bottom_cv2(candidate_bgr, crop_bottom)

    return real_bgr, candidate_bgr


def compute_psnr(real_bgr: np.ndarray, candidate_bgr: np.ndarray) -> float:
    mse = np.mean(
        (real_bgr.astype(np.float64) - candidate_bgr.astype(np.float64)) ** 2
    )

    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(real_bgr: np.ndarray, candidate_bgr: np.ndarray) -> float:
    real_gray = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)

    return float(
        structural_similarity(
            real_gray,
            candidate_gray,
            data_range=255,
        )
    )


def bgr_to_lpips_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
    tensor = tensor / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def compute_lpips(
    real_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    lpips_model,
    device: torch.device,
) -> float:
    real_tensor = bgr_to_lpips_tensor(real_bgr).to(device)
    candidate_tensor = bgr_to_lpips_tensor(candidate_bgr).to(device)

    with torch.no_grad():
        value = lpips_model(real_tensor, candidate_tensor)

    return float(value.item())


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return -1.0, -1.0

    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def compute_frame_metrics(
    pairs: List[Tuple[str, str, int]],
    lpips_model,
    device: torch.device,
    crop_bottom: int,
) -> Tuple[List[Dict], Dict[str, float]]:
    rows = []

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for real_path, candidate_path, frame_id in tqdm(pairs, desc="    Frame metrics"):
        real_bgr, candidate_bgr = load_pair(
            real_path,
            candidate_path,
            crop_bottom,
        )

        psnr = compute_psnr(real_bgr, candidate_bgr)
        ssim = compute_ssim(real_bgr, candidate_bgr)
        lpips_value = compute_lpips(
            real_bgr,
            candidate_bgr,
            lpips_model,
            device,
        )

        if not math.isinf(psnr):
            psnr_values.append(psnr)

        ssim_values.append(ssim)
        lpips_values.append(lpips_value)

        rows.append({
            "frame_id": frame_id,
            "real_file": os.path.basename(real_path),
            "candidate_file": os.path.basename(candidate_path),
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_value,
        })

    psnr_mean, psnr_std = mean_std(psnr_values)
    ssim_mean, ssim_std = mean_std(ssim_values)
    lpips_mean, lpips_std = mean_std(lpips_values)

    summary = {
        "PSNR_mean": psnr_mean,
        "PSNR_std": psnr_std,
        "SSIM_mean": ssim_mean,
        "SSIM_std": ssim_std,
        "LPIPS_mean": lpips_mean,
        "LPIPS_std": lpips_std,
    }

    return rows, summary


inception_transform = transforms.Compose([
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


class MetricImageDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        side: str,
        crop_bottom: int,
    ):
        if side not in {"real", "candidate"}:
            raise ValueError("side must be either 'real' or 'candidate'.")

        self.pairs = pairs
        self.side = side
        self.crop_bottom = crop_bottom

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        real_path, candidate_path, _ = self.pairs[index]
        path = real_path if self.side == "real" else candidate_path

        image = Image.open(path).convert("RGB")
        image = crop_bottom_pil(image, self.crop_bottom)

        return inception_transform(image)


def build_inception_model(device: torch.device):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(
        weights=weights,
        aux_logits=True,
        transform_input=False,
    )

    model.fc = nn.Identity()
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)


def extract_features(
    dataset: Dataset,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="    Inception features"):
            batch = batch.to(device)
            output = model(batch)

            if isinstance(output, tuple):
                output = output[0]

            features.append(output.detach().cpu().numpy())

    return np.concatenate(features, axis=0)


def feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def sqrtm_product(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    sigma_a = sigma_a.copy()
    sigma_b = sigma_b.copy()

    sigma_a.flat[:: sigma_a.shape[0] + 1] += eps
    sigma_b.flat[:: sigma_b.shape[0] + 1] += eps

    product, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)

    if np.iscomplexobj(product):
        product = product.real

    return product


def compute_fid(
    real_features: np.ndarray,
    candidate_features: np.ndarray,
) -> float:
    mu_real, sigma_real = feature_stats(real_features)
    mu_candidate, sigma_candidate = feature_stats(candidate_features)

    diff = mu_real - mu_candidate
    covmean = sqrtm_product(sigma_real, sigma_candidate)

    return float(
        diff.dot(diff)
        + np.trace(sigma_real + sigma_candidate - 2.0 * covmean)
    )


def compute_kid(
    real_features: np.ndarray,
    candidate_features: np.ndarray,
    subsets: int = 100,
    subset_size: int = 1000,
) -> Tuple[float, float]:
    rng = np.random.default_rng(123)

    n = min(len(real_features), subset_size)
    m = min(len(candidate_features), subset_size)

    if n < 2 or m < 2:
        return -1.0, -1.0

    gamma = 1.0 / real_features.shape[1]
    degree = 3
    coef0 = 1.0

    values = []

    for _ in range(subsets):
        real_sample = real_features[
            rng.choice(len(real_features), n, replace=False)
        ]

        candidate_sample = candidate_features[
            rng.choice(len(candidate_features), m, replace=False)
        ]

        kernel_real = (gamma * real_sample @ real_sample.T + coef0) ** degree
        kernel_candidate = (
            gamma * candidate_sample @ candidate_sample.T + coef0
        ) ** degree
        kernel_cross = (
            gamma * real_sample @ candidate_sample.T + coef0
        ) ** degree

        np.fill_diagonal(kernel_real, 0.0)
        np.fill_diagonal(kernel_candidate, 0.0)

        value = (
            kernel_real.sum() / (n * (n - 1))
            + kernel_candidate.sum() / (m * (m - 1))
            - 2.0 * kernel_cross.mean()
        )

        values.append(value)

    values = np.asarray(values, dtype=np.float64)

    return float(np.mean(values)), float(np.std(values))


def compute_mmd(
    real_features: np.ndarray,
    candidate_features: np.ndarray,
) -> float:
    combined = np.vstack([real_features, candidate_features])

    if len(combined) > 2000:
        rng = np.random.default_rng(123)
        sample = combined[rng.choice(len(combined), 2000, replace=False)]
    else:
        sample = combined

    distances = pairwise_distances(sample)
    positive = distances[distances > 0.0]

    if len(positive) == 0:
        return -1.0

    sigma = float(np.median(positive))
    gamma = 1.0 / (2.0 * sigma * sigma)

    kernel_real = np.exp(
        -gamma * pairwise_distances(
            real_features,
            real_features,
            squared=True,
        )
    )

    kernel_candidate = np.exp(
        -gamma * pairwise_distances(
            candidate_features,
            candidate_features,
            squared=True,
        )
    )

    kernel_cross = np.exp(
        -gamma * pairwise_distances(
            real_features,
            candidate_features,
            squared=True,
        )
    )

    return float(
        kernel_real.mean()
        + kernel_candidate.mean()
        - 2.0 * kernel_cross.mean()
    )


def compute_distribution_metrics(
    pairs: List[Tuple[str, str, int]],
    inception_model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    crop_bottom: int,
) -> Dict[str, float]:
    real_dataset = MetricImageDataset(
        pairs=pairs,
        side="real",
        crop_bottom=crop_bottom,
    )

    candidate_dataset = MetricImageDataset(
        pairs=pairs,
        side="candidate",
        crop_bottom=crop_bottom,
    )

    print("    Real distribution features")
    real_features = extract_features(
        real_dataset,
        inception_model,
        device,
        batch_size,
        num_workers,
    )

    print("    Candidate distribution features")
    candidate_features = extract_features(
        candidate_dataset,
        inception_model,
        device,
        batch_size,
        num_workers,
    )

    fid = compute_fid(real_features, candidate_features)
    kid_mean, kid_std = compute_kid(real_features, candidate_features)
    mmd = compute_mmd(real_features, candidate_features)

    return {
        "FID": fid,
        "KID_mean": kid_mean,
        "KID_std": kid_std,
        "MMD": mmd,
    }


def round_value(value, digits: int = 6):
    if isinstance(value, float) and math.isinf(value):
        return "Infinity"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), digits)

    return value


def round_dict(data: Dict, digits: int = 6) -> Dict:
    return {
        key: round_value(value, digits)
        for key, value in data.items()
    }


def save_frame_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "frame_id",
        "real_file",
        "candidate_file",
        "PSNR",
        "SSIM",
        "LPIPS",
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: round_value(row[key])
                for key in fieldnames
            })


def save_summary_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "candidate",
        "num_frames",
        "PSNR_mean",
        "PSNR_std",
        "SSIM_mean",
        "SSIM_std",
        "LPIPS_mean",
        "LPIPS_std",
        "FID",
        "KID_mean",
        "KID_std",
        "MMD",
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: round_value(row.get(key, ""))
                for key in fieldnames
            })


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def evaluate_candidate(
    candidate_name: str,
    real_folder: str,
    candidate_folder: str,
    output_folder: str,
    lpips_model,
    inception_model,
    device: torch.device,
    args,
) -> Dict:
    print("\n" + "=" * 80)
    print(f"Evaluating: {candidate_name}")
    print("=" * 80)
    print(f"  Real:      {real_folder}")
    print(f"  Candidate: {candidate_folder}")

    validate_folder(candidate_folder, candidate_name)

    pairs = get_matched_pairs(
        real_folder,
        candidate_folder,
        max_frames=args.max_frames,
    )

    if not pairs:
        raise RuntimeError(f"No matched frames for candidate: {candidate_name}")

    candidate_output = os.path.join(output_folder, candidate_name)
    os.makedirs(candidate_output, exist_ok=True)

    frame_rows, frame_summary = compute_frame_metrics(
        pairs=pairs,
        lpips_model=lpips_model,
        device=device,
        crop_bottom=args.crop_bottom,
    )

    distribution_summary = compute_distribution_metrics(
        pairs=pairs,
        inception_model=inception_model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_bottom=args.crop_bottom,
    )

    summary = {
        "candidate": candidate_name,
        "num_frames": len(pairs),
        **frame_summary,
        **distribution_summary,
    }

    summary = round_dict(summary)

    frame_csv = os.path.join(candidate_output, f"{candidate_name}_frame_metrics.csv")
    report_json = os.path.join(candidate_output, f"{candidate_name}_report.json")

    save_frame_csv(frame_csv, frame_rows)

    report = {
        "candidate": candidate_name,
        "real_folder": real_folder,
        "candidate_folder": candidate_folder,
        "num_frames": len(pairs),
        "crop_bottom": args.crop_bottom,
        "lpips_net": args.lpips_net,
        "summary": summary,
        "frame_metrics": [
            round_dict(row)
            for row in frame_rows
        ],
    }

    with open(report_json, "w") as file:
        json.dump(report, file, indent=4, cls=JsonEncoder)

    print(f"  [OK] Frame CSV: {frame_csv}")
    print(f"  [OK] Report:    {report_json}")
    print("  Summary:")
    print(f"    PSNR:  {summary['PSNR_mean']}")
    print(f"    SSIM:  {summary['SSIM_mean']}")
    print(f"    LPIPS: {summary['LPIPS_mean']}")
    print(f"    FID:   {summary['FID']}")
    print(f"    KID:   {summary['KID_mean']}")
    print(f"    MMD:   {summary['MMD']}")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    args = parse_args()
    bag_stem = get_bag_stem(args.bag_name)
    paths = get_paths(bag_stem)

    real_folder = paths["real"]
    carla_folder = paths["carla"]
    gs_folder = paths["gs"]
    output_folder = paths["output"]

    validate_folder(real_folder, "real")

    os.makedirs(output_folder, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("VISUAL-GAP IMAGE METRICS")
    print("=" * 80)
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Bag:            {bag_stem}")
    print(f"Real folder:    {real_folder}")
    print(f"CARLA folder:   {carla_folder}")
    print(f"GS folder:      {gs_folder}")
    print(f"Output folder:  {output_folder}")
    print(f"Crop bottom:    {args.crop_bottom}px")
    print(f"Max frames:     {args.max_frames or 'all matched'}")
    print(f"LPIPS net:      {args.lpips_net}")
    print(f"Device:         {device}")
    print("=" * 80)

    print("\n[INFO] Loading LPIPS")
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_model.eval()

    print("[INFO] Loading InceptionV3")
    inception_model = build_inception_model(device)

    summary_rows = []

    for candidate_name, candidate_folder in [
        ("carla", carla_folder),
        ("gs", gs_folder),
    ]:
        summary = evaluate_candidate(
            candidate_name=candidate_name,
            real_folder=real_folder,
            candidate_folder=candidate_folder,
            output_folder=output_folder,
            lpips_model=lpips_model,
            inception_model=inception_model,
            device=device,
            args=args,
        )

        summary_rows.append(summary)

    summary_csv = os.path.join(output_folder, "summary_metrics.csv")
    summary_json = os.path.join(output_folder, "summary_metrics.json")

    save_summary_csv(summary_csv, summary_rows)

    with open(summary_json, "w") as file:
        json.dump(summary_rows, file, indent=4, cls=JsonEncoder)

    print("\n" + "=" * 80)
    print("[OK] Completed")
    print("=" * 80)
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())