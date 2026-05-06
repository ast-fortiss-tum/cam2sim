#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute similarity transform from UTM coordinates to NERFSTUDIO coordinate system.

IMPORTANT: This is different from raw COLMAP coordinates!

Nerfstudio applies internal transforms to COLMAP data:
- Centering (moves centroid to origin)
- Scaling (normalizes scene size)
- Axis reordering (may swap/flip axes)

So we need to compute the transform to nerfstudio's FINAL coordinate system,
not the raw COLMAP positions from images.bin.

This script:
1. Loads the TRAINED nerfstudio model
2. Extracts the camera positions AND orientations that nerfstudio actually uses
3. Matches them with UTM positions + yaw by frame number
4. Computes Umeyama similarity transform: P_nerfstudio = s * R @ P_utm + t
5. Computes yaw offset directly from matched orientations
6. Saves the transform for use in inference

Usage:
    python 3v-utm_to_nerfstudio.py \
        --gs_config path/to/outputs/.../config.yml \
        --utm_file path/to/frame_positions.txt \
        --data_root path/to/dataset \
        --output utm_to_nerfstudio_transform.json

Supported UTM file formats (auto-detected by number of columns):

    5+ cols:  FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, ImageFile
    11 cols:  FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z,
              Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile

In both cases, Odom_X / Odom_Y are interpreted as UTM Easting / Northing.
"""

import os
import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

import torch
from nerfstudio.utils.eval_utils import eval_setup


# ==============================================================================
# UTM File Reader (auto-detects column layout)
# ==============================================================================

def read_utm_positions(filepath):
    """
    Read UTM positions from frame_positions.txt.

    Auto-detects two layouts:

    Short (>=5 columns):
        FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, ImageFile
        index:    0       1         2       3        4         5

    Long (11 columns, with quaternion):
        FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z,
        Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile
        index:    0       1         2       3       4
                  5    6    7    8       9         10

    The yaw column index is detected per line:
      - 11 columns -> yaw at index 9
      - otherwise  -> yaw at index 4 (best effort)

    Returns:
        dict: frame_id -> {'easting': float, 'northing': float, 'yaw': float}
    """
    positions = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue

            try:
                frame_id = int(parts[0])
                easting = float(parts[2])     # Odom_X
                northing = float(parts[3])    # Odom_Y

                if len(parts) >= 11:
                    yaw = float(parts[9])     # long format: yaw is index 9
                else:
                    yaw = float(parts[4])     # short format: yaw is index 4

                positions[frame_id] = {
                    "easting": easting,
                    "northing": northing,
                    "yaw": yaw,
                }
            except (ValueError, IndexError):
                continue

    return positions


def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_000123.png'"""
    name = os.path.splitext(str(filename))[0]
    name = os.path.basename(name)
    numbers = re.findall(r"\d+", name)
    if numbers:
        return int(numbers[-1])
    return None


# ==============================================================================
# YPR Extraction from c2w (same convention as replay script)
# ==============================================================================

def extract_ypr_from_c2w(c2w):
    """
    Extract yaw, pitch, roll from a nerfstudio c2w matrix.

    Convention (matching build_nerfstudio_c2w in replay script):
        col0 = right, col1 = up, col2 = -forward
        R_world = R_yaw @ R_pitch @ R_roll
        forward = R_world @ [0,1,0]
        right   = R_world @ [1,0,0]
        up      = R_world @ [0,0,1]

    Key insight: forward = R_yaw @ R_pitch @ [0,1,0] regardless of roll
    (because R_roll @ [0,1,0] = [0,1,0])

    So: forward = [-sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)]

    Returns (yaw_rad, pitch_rad, roll_rad)
    """
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    forward = -c2w[:3, 2]

    pitch = np.arcsin(np.clip(forward[2], -1.0, 1.0))

    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(-forward[0] / cos_pitch, forward[1] / cos_pitch)
    else:
        yaw = np.arctan2(right[1], right[0])

    if abs(cos_pitch) > 1e-6:
        roll = np.arctan2(-right[2] / cos_pitch, up[2] / cos_pitch)
    else:
        roll = 0.0

    return yaw, pitch, roll


# ==============================================================================
# Umeyama Alignment (Similarity Transform)
# ==============================================================================

def umeyama_alignment(source, target, with_scale=True):
    """
    Compute similarity transform using Umeyama's method.

    Finds s, R, t such that: target ~ s * R @ source + t

    Args:
        source: (N, D) array - source points (UTM coordinates)
        target: (N, D) array - target points (Nerfstudio coordinates)
        with_scale: if True, estimate scale; if False, fix scale=1

    Returns:
        s: scale factor (scalar)
        R: (D, D) rotation matrix
        t: (D,) translation vector
        error: RMS alignment error
    """
    assert source.shape == target.shape
    n, dim = source.shape

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)

    source_centered = source - mu_source
    target_centered = target - mu_target

    var_source = np.sum(source_centered ** 2) / n

    cov = (target_centered.T @ source_centered) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    R = U @ S @ Vt

    if with_scale:
        s = np.trace(np.diag(D) @ S) / var_source
    else:
        s = 1.0

    t = mu_target - s * R @ mu_source

    transformed = s * (source @ R.T) + t
    error = np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1)))

    return s, R, t, error


# ==============================================================================
# Yaw Offset Computation
# ==============================================================================

def compute_yaw_offset(utm_yaws, ns_yaws):
    """
    Compute the constant offset between UTM yaw and Nerfstudio yaw.

    For each matched frame we have:
        utm_yaw  (from odometry, in UTM frame)
        ns_yaw   (extracted from nerfstudio c2w matrix)

    We try multiple hypotheses for the yaw mapping and pick the one
    with smallest residual variance:
        H1: ns_yaw = utm_yaw + offset
        H2: ns_yaw = -utm_yaw + offset

    Returns:
        dict with:
            'yaw_sign': +1 or -1
            'yaw_offset': float (radians)
            'residual_std': float (radians)
            'per_frame_residuals': array of per-frame residuals
    """
    utm_yaws = np.array(utm_yaws)
    ns_yaws = np.array(ns_yaws)

    best_result = None
    best_std = float("inf")

    for sign in [+1, -1]:
        diffs = ns_yaws - sign * utm_yaws
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

        mean_sin = np.mean(np.sin(diffs))
        mean_cos = np.mean(np.cos(diffs))
        circular_mean = np.arctan2(mean_sin, mean_cos)

        residuals = diffs - circular_mean
        residuals = (residuals + np.pi) % (2 * np.pi) - np.pi

        std = np.std(residuals)

        if std < best_std:
            best_std = std
            best_result = {
                "yaw_sign": sign,
                "yaw_offset": float(circular_mean),
                "residual_std": float(std),
                "per_frame_residuals": residuals,
            }

    return best_result


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute similarity transform from UTM to Nerfstudio coordinates"
    )
    parser.add_argument(
        "--gs_config",
        required=True,
        help="Path to nerfstudio config.yml (e.g., outputs/.../config.yml)",
    )
    parser.add_argument(
        "--utm_file",
        required=True,
        help="Path to frame_positions.txt with UTM coordinates "
             "(short or long format, auto-detected)",
    )
    parser.add_argument(
        "--data_root",
        default=".",
        help="Data root directory (where nerfstudio expects to find data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. If omitted, the file is saved next to the "
             "config.yml as utm_to_nerfstudio_transform.json",
    )
    args = parser.parse_args()

    config_path = Path(args.gs_config).resolve()
    data_root = Path(args.data_root).resolve()

    if args.output is None:
        output_path = config_path.parent / "utm_to_nerfstudio_transform.json"
    else:
        output_path = Path(args.output).resolve()

    # =========================================================================
    # Step 1: Load the trained Nerfstudio model
    # =========================================================================
    print("=" * 70)
    print("STEP 1: Loading Nerfstudio model")
    print("=" * 70)
    print(f"[INFO] Config:    {config_path}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Output:    {output_path}")

    original_cwd = os.getcwd()
    os.chdir(data_root)

    try:
        _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")

        dp = pipeline.datamanager.train_dataparser_outputs
        c2w_matrices = dp.cameras.camera_to_worlds.cpu().numpy()
        image_filenames = dp.image_filenames

        print(f"[INFO] Found {len(image_filenames)} training cameras")
        print(f"[INFO] c2w matrix shape: {c2w_matrices.shape}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
        sys.exit(1)

    os.chdir(original_cwd)

    nerfstudio_data = {}
    for i, img_path in enumerate(image_filenames):
        frame_num = extract_frame_number(img_path)
        if frame_num is not None:
            pos = c2w_matrices[i, :, 3]
            yaw, pitch, roll = extract_ypr_from_c2w(c2w_matrices[i])
            nerfstudio_data[frame_num] = {
                "position": pos,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "filename": str(img_path),
            }

    print(f"[INFO] Extracted {len(nerfstudio_data)} nerfstudio positions+orientations")

    sorted_frames = sorted(nerfstudio_data.keys())[:5]
    print(f"\n[INFO] First 5 nerfstudio cameras:")
    for f in sorted_frames:
        d = nerfstudio_data[f]
        pos = d["position"]
        print(f"       Frame {f}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
              f"yaw={np.degrees(d['yaw']):.2f}deg pitch={np.degrees(d['pitch']):.2f}deg "
              f"roll={np.degrees(d['roll']):.2f}deg")

    # =========================================================================
    # Step 2: Load UTM positions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Loading UTM positions (ground truth)")
    print("=" * 70)

    utm_positions = read_utm_positions(args.utm_file)
    print(f"[INFO] Loaded {len(utm_positions)} UTM positions from: {args.utm_file}")

    sorted_utm_frames = sorted(utm_positions.keys())[:5]
    print(f"\n[INFO] First 5 UTM positions:")
    for f in sorted_utm_frames:
        pos = utm_positions[f]
        print(f"       Frame {f}: E={pos['easting']:.2f}, N={pos['northing']:.2f}, "
              f"yaw={pos['yaw']:.4f} rad ({np.degrees(pos['yaw']):.2f}deg)")

    # =========================================================================
    # Step 3: Match frames
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Matching frames between Nerfstudio and UTM")
    print("=" * 70)

    matched_frames = sorted(set(nerfstudio_data.keys()) & set(utm_positions.keys()))
    print(f"[INFO] Matched {len(matched_frames)} frames")

    if len(matched_frames) < 3:
        print("[ERROR] Need at least 3 matched frames!")
        sys.exit(1)

    # =========================================================================
    # Step 4: Build point arrays & analyze ranges
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Analyzing coordinate ranges")
    print("=" * 70)

    utm_pts = np.array([
        [utm_positions[f]["easting"], utm_positions[f]["northing"]]
        for f in matched_frames
    ])

    ns_pts = np.array([nerfstudio_data[f]["position"] for f in matched_frames])

    print(f"\n[INFO] UTM coordinate ranges:")
    print(f"       Easting:  [{utm_pts[:, 0].min():.2f}, {utm_pts[:, 0].max():.2f}]  "
          f"(range: {utm_pts[:, 0].max() - utm_pts[:, 0].min():.2f} m)")
    print(f"       Northing: [{utm_pts[:, 1].min():.2f}, {utm_pts[:, 1].max():.2f}]  "
          f"(range: {utm_pts[:, 1].max() - utm_pts[:, 1].min():.2f} m)")

    print(f"\n[INFO] Nerfstudio coordinate ranges:")
    print(f"       X: [{ns_pts[:, 0].min():.4f}, {ns_pts[:, 0].max():.4f}]  "
          f"(range: {ns_pts[:, 0].max() - ns_pts[:, 0].min():.4f})")
    print(f"       Y: [{ns_pts[:, 1].min():.4f}, {ns_pts[:, 1].max():.4f}]  "
          f"(range: {ns_pts[:, 1].max() - ns_pts[:, 1].min():.4f})")
    print(f"       Z: [{ns_pts[:, 2].min():.4f}, {ns_pts[:, 2].max():.4f}]  "
          f"(range: {ns_pts[:, 2].max() - ns_pts[:, 2].min():.4f})")

    # =========================================================================
    # Step 5: Compute 2D position alignment (Umeyama)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Computing 2D similarity transform (Umeyama) - POSITIONS")
    print("=" * 70)

    ns_pts_2d = ns_pts[:, :2]

    s, R_2d, t_2d, error = umeyama_alignment(utm_pts, ns_pts_2d, with_scale=True)

    print(f"\n[INFO] 2D Position Alignment Results:")
    print(f"       Scale: {s:.10f}")
    print(f"       Rotation (2x2):")
    print(f"          [{R_2d[0, 0]:>10.6f}, {R_2d[0, 1]:>10.6f}]")
    print(f"          [{R_2d[1, 0]:>10.6f}, {R_2d[1, 1]:>10.6f}]")
    print(f"       Translation: [{t_2d[0]:.6f}, {t_2d[1]:.6f}]")
    print(f"       RMS Error: {error:.6f} (in nerfstudio units)")

    position_rotation_angle = np.arctan2(R_2d[1, 0], R_2d[0, 0])
    print(f"       Position rotation angle: {np.degrees(position_rotation_angle):.2f}deg")

    R_3d = np.eye(3)
    R_3d[:2, :2] = R_2d

    t_3d = np.zeros(3)
    t_3d[:2] = t_2d
    t_3d[2] = ns_pts[:, 2].mean()

    print(f"\n[INFO] Extended to 3D:")
    print(f"       Z offset (mean nerfstudio Z): {t_3d[2]:.6f}")

    # =========================================================================
    # Step 6: Compute YAW OFFSET from matched orientations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Computing yaw offset from matched ORIENTATIONS")
    print("=" * 70)

    utm_yaws = np.array([utm_positions[f]["yaw"] for f in matched_frames])
    ns_yaws = np.array([nerfstudio_data[f]["yaw"] for f in matched_frames])

    ns_pitches = np.array([nerfstudio_data[f]["pitch"] for f in matched_frames])
    ns_rolls = np.array([nerfstudio_data[f]["roll"] for f in matched_frames])

    print(f"\n[INFO] UTM yaw range: [{np.degrees(utm_yaws.min()):.2f}deg, "
          f"{np.degrees(utm_yaws.max()):.2f}deg]")
    print(f"[INFO] NS yaw range:  [{np.degrees(ns_yaws.min()):.2f}deg, "
          f"{np.degrees(ns_yaws.max()):.2f}deg]")
    print(f"[INFO] NS pitch: median={np.degrees(np.median(ns_pitches)):.2f}deg, "
          f"std={np.degrees(np.std(ns_pitches)):.2f}deg")
    print(f"[INFO] NS roll:  median={np.degrees(np.median(ns_rolls)):.2f}deg, "
          f"std={np.degrees(np.std(ns_rolls)):.2f}deg")

    yaw_result = compute_yaw_offset(utm_yaws, ns_yaws)

    yaw_sign = yaw_result["yaw_sign"]
    yaw_offset = yaw_result["yaw_offset"]
    yaw_residual_std = yaw_result["residual_std"]

    print(f"\n[INFO] Yaw Alignment Results:")
    print(f"       Best mapping: ns_yaw = "
          f"{'+' if yaw_sign > 0 else '-'}utm_yaw + offset")
    print(f"       Yaw sign: {yaw_sign:+d}")
    print(f"       Yaw offset: {yaw_offset:.6f} rad ({np.degrees(yaw_offset):.2f}deg)")
    print(f"       Residual std: {yaw_residual_std:.6f} rad "
          f"({np.degrees(yaw_residual_std):.2f}deg)")

    old_yaw_formula_offset = position_rotation_angle - np.pi / 2
    print(f"\n[INFO] Comparison with old method (position_rotation_angle - pi/2):")
    print(f"       Old offset: {old_yaw_formula_offset:.6f} rad "
          f"({np.degrees(old_yaw_formula_offset):.2f}deg)")
    print(f"       New offset: {yaw_offset:.6f} rad ({np.degrees(yaw_offset):.2f}deg)")
    print(f"       Difference: "
          f"{np.degrees(yaw_offset - old_yaw_formula_offset):.2f}deg")

    print(f"\n[INFO] First 10 frames - yaw comparison:")
    print(f"       {'Frame':>6s}  {'UTM yaw':>9s}  {'NS yaw':>9s}  "
          f"{'Predicted':>9s}  {'Error':>8s}")
    for i in range(min(10, len(matched_frames))):
        f = matched_frames[i]
        utm_y = utm_positions[f]["yaw"]
        ns_y = nerfstudio_data[f]["yaw"]
        pred_y = yaw_sign * utm_y + yaw_offset
        err = ns_y - pred_y
        err = (err + np.pi) % (2 * np.pi) - np.pi
        print(f"       {f:>6d}  {np.degrees(utm_y):>8.2f}deg  "
              f"{np.degrees(ns_y):>8.2f}deg  "
              f"{np.degrees(pred_y):>8.2f}deg  {np.degrees(err):>7.2f}deg")

    # =========================================================================
    # Step 7: Verify the full transform
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Verification (position + yaw)")
    print("=" * 70)

    print("\n[INFO] First 5 points:")
    all_pos_errors = []
    all_yaw_errors = []
    for i, f in enumerate(matched_frames):
        utm = np.array([
            utm_positions[f]["easting"],
            utm_positions[f]["northing"],
            0.0,
        ])
        ns_true = nerfstudio_data[f]["position"]
        ns_pred = s * R_3d @ utm + t_3d

        err_xy = np.linalg.norm(ns_true[:2] - ns_pred[:2])
        all_pos_errors.append(err_xy)

        utm_y = utm_positions[f]["yaw"]
        ns_y_true = nerfstudio_data[f]["yaw"]
        ns_y_pred = yaw_sign * utm_y + yaw_offset
        yaw_err = ns_y_true - ns_y_pred
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        all_yaw_errors.append(yaw_err)

        if i < 5:
            print(f"       Frame {f}:")
            print(f"          UTM:     ({utm[0]:.2f}, {utm[1]:.2f}) "
                  f"yaw={np.degrees(utm_y):.2f}deg")
            print(f"          NS true: ({ns_true[0]:.4f}, {ns_true[1]:.4f}, "
                  f"{ns_true[2]:.4f}) yaw={np.degrees(ns_y_true):.2f}deg")
            print(f"          NS pred: ({ns_pred[0]:.4f}, {ns_pred[1]:.4f}, "
                  f"{ns_pred[2]:.4f}) yaw={np.degrees(ns_y_pred):.2f}deg")
            print(f"          XY Error: {err_xy:.6f}  "
                  f"Yaw Error: {np.degrees(yaw_err):.2f}deg")

    all_pos_errors = np.array(all_pos_errors)
    all_yaw_errors = np.degrees(np.array(all_yaw_errors))
    print(f"\n[INFO] Error statistics (over {len(all_pos_errors)} frames):")
    print(f"       Position - Mean: {all_pos_errors.mean():.6f}  "
          f"Max: {all_pos_errors.max():.6f}  "
          f"RMS: {np.sqrt(np.mean(all_pos_errors**2)):.6f}")
    print(f"       Yaw      - Mean: {np.mean(np.abs(all_yaw_errors)):.2f}deg  "
          f"Max: {np.max(np.abs(all_yaw_errors)):.2f}deg  "
          f"Std: {np.std(all_yaw_errors):.2f}deg")

    # =========================================================================
    # Step 8: Save transform
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Saving transform")
    print("=" * 70)

    transform_data = {
        "description": "Similarity transform from UTM to Nerfstudio: "
                       "P_ns = scale * R @ P_utm + translation",
        "note": "This maps UTM coordinates to Nerfstudio's internal coordinate "
                "system (after its transforms)",
        "mode": "2D (XY only, Z is mean offset)",
        "scale": float(s),
        "rotation": R_3d.tolist(),
        "translation": t_3d.tolist(),
        "rotation_angle_deg": float(np.degrees(position_rotation_angle)),
        "rms_error": float(error),
        "num_matched_points": len(matched_frames),

        "yaw_alignment": {
            "description": "ns_yaw = yaw_sign * utm_yaw + yaw_offset_rad",
            "yaw_sign": int(yaw_sign),
            "yaw_offset_rad": float(yaw_offset),
            "yaw_offset_deg": float(np.degrees(yaw_offset)),
            "residual_std_deg": float(np.degrees(yaw_residual_std)),
            "old_method_offset_deg": float(np.degrees(old_yaw_formula_offset)),
            "difference_from_old_deg": float(
                np.degrees(yaw_offset - old_yaw_formula_offset)
            ),
        },

        "camera_mount_angles": {
            "description": "Median pitch and roll from COLMAP training cameras "
                           "(physical mount angle)",
            "avg_pitch_rad": float(np.median(ns_pitches)),
            "avg_pitch_deg": float(np.degrees(np.median(ns_pitches))),
            "avg_roll_rad": float(np.median(ns_rolls)),
            "avg_roll_deg": float(np.degrees(np.median(ns_rolls))),
            "pitch_std_deg": float(np.degrees(np.std(ns_pitches))),
            "roll_std_deg": float(np.degrees(np.std(ns_rolls))),
        },

        "utm_centroid": [
            float(utm_pts[:, 0].mean()),
            float(utm_pts[:, 1].mean()),
        ],
        "nerfstudio_centroid": [
            float(ns_pts[:, 0].mean()),
            float(ns_pts[:, 1].mean()),
            float(ns_pts[:, 2].mean()),
        ],
        "coordinate_ranges": {
            "utm_easting": [
                float(utm_pts[:, 0].min()),
                float(utm_pts[:, 0].max()),
            ],
            "utm_northing": [
                float(utm_pts[:, 1].min()),
                float(utm_pts[:, 1].max()),
            ],
            "nerfstudio_x": [
                float(ns_pts[:, 0].min()),
                float(ns_pts[:, 0].max()),
            ],
            "nerfstudio_y": [
                float(ns_pts[:, 1].min()),
                float(ns_pts[:, 1].max()),
            ],
            "nerfstudio_z": [
                float(ns_pts[:, 2].min()),
                float(ns_pts[:, 2].max()),
            ],
        },
        "source_files": {
            "gs_config": str(config_path),
            "utm_file": os.path.abspath(args.utm_file),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transform_data, f, indent=2)

    print(f"\nTransform saved to: {output_path}")


if __name__ == "__main__":
    main()