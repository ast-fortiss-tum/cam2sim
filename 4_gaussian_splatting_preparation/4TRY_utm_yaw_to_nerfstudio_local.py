#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4C_utm_yaw_to_nerfstudio_knn.py

Compute KNN LOCAL SIMILARITY transform from UTM coordinates to Nerfstudio
coordinate system.

Difference from 4C_utm_yaw_to_nerfstudio.py:
    - Old: fits a single global Umeyama similarity (4 params in 2D).
           Bad on long splits with strong curves because COLMAP + Nerfstudio
           introduce non-rigid drift along the sequence.
    - New: saves all training points (UTM and NS). At runtime, for each query
           the coordinate transformer fits a local Umeyama on the k=20 nearest
           training points. Sub-decimeter alignment everywhere inside the
           training support.

The yaw alignment is left global (the yaw drift in COLMAP+NS is much smaller
than the position drift).

Output JSON has a "mode" field:
    - "2D" (old format)               -> read scale, rotation, translation
    - "local_similarity_knn" (new)    -> read training_points and build a
                                         BallTree at runtime

CoordinateTransformer in the drive/replay scripts supports both formats
(see utils/coordinate_transformer.py).

Usage:
    python 4C_utm_yaw_to_nerfstudio_knn.py \
        --gs_config path/to/outputs/.../config.yml \
        --utm_file path/to/frame_positions.txt \
        --data_root path/to/dataset \
        [--k 20]                # default k=20 neighbors
"""

import os
import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from nerfstudio.utils.eval_utils import eval_setup


# ==============================================================================
# UTM File Reader (auto-detects column layout)
# ==============================================================================

def read_utm_positions(filepath):
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
                easting = float(parts[2])
                northing = float(parts[3])
                if len(parts) >= 11:
                    yaw = float(parts[9])
                else:
                    yaw = float(parts[4])
                positions[frame_id] = {
                    "easting": easting,
                    "northing": northing,
                    "yaw": yaw,
                }
            except (ValueError, IndexError):
                continue
    return positions


def extract_frame_number(filename):
    name = os.path.splitext(str(filename))[0]
    name = os.path.basename(name)
    numbers = re.findall(r"\d+", name)
    if numbers:
        return int(numbers[-1])
    return None


# ==============================================================================
# YPR Extraction from c2w
# ==============================================================================

def extract_ypr_from_c2w(c2w):
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
# Umeyama Alignment (used for local fits + global verification)
# ==============================================================================

def umeyama_alignment(source, target, with_scale=True):
    assert source.shape == target.shape
    n, dim = source.shape

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)

    source_centered = source - mu_source
    target_centered = target - mu_target

    var_source = np.sum(source_centered ** 2) / n
    if var_source < 1e-12:
        return 1.0, np.eye(dim), mu_target - mu_source, 0.0

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
# Local KNN Transform (verification at training points)
# ==============================================================================

def transform_point_local(query_xy, source, target, nn, k):
    """
    Single-point KNN local Umeyama transform.

    query_xy:  (2,) UTM coordinate
    source:    (N, 2) UTM training points
    target:    (N, 2) NS training points
    nn:        fitted NearestNeighbors instance on source
    k:         number of neighbors to use for the local fit
    """
    _, idx = nn.kneighbors(query_xy.reshape(1, -1), n_neighbors=k)
    local_src = source[idx[0]]
    local_tgt = target[idx[0]]

    s, R, t, _ = umeyama_alignment(local_src, local_tgt, with_scale=True)
    return s * (R @ query_xy) + t


# ==============================================================================
# Yaw Offset Computation (kept global)
# ==============================================================================

def compute_yaw_offset(utm_yaws, ns_yaws):
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
            }

    return best_result


# ==============================================================================
# 2D Plot helpers
# ==============================================================================

def build_knn_runtime_points(utm_pts, ns_pts, nn, k):
    """
    Apply runtime-style KNN local Umeyama (no leave-one-out) on each training
    UTM point to produce its predicted NS position. Z is interpolated from
    neighbors' ns_z via inverse-distance weighting, matching the runtime
    CoordinateTransformer behavior.
    """
    out = np.zeros((len(utm_pts), 3), dtype=float)
    for i, q in enumerate(utm_pts):
        dists, idx = nn.kneighbors(q.reshape(1, -1), n_neighbors=k)
        idx = idx[0]
        dists = dists[0]

        local_src = utm_pts[idx]
        local_tgt = ns_pts[idx, :2]
        s_loc, R_loc, t_loc, _ = umeyama_alignment(
            local_src, local_tgt, with_scale=True
        )
        out[i, :2] = s_loc * (R_loc @ q) + t_loc

        # Z: inverse-distance weighted from neighbors' ns_z
        if dists[0] < 1e-12:
            out[i, 2] = float(ns_pts[idx[0], 2])
        else:
            weights = 1.0 / np.maximum(dists, 1e-12)
            weights /= weights.sum()
            out[i, 2] = float(np.sum(weights * ns_pts[idx, 2]))
    return out


def plot_alignment_3d(ns_pts, utm_as_ns_knn, utm_as_ns_global,
                      matched_frames, title):
    """
    Plot 3 trajectories together in 2D (XY only):
      - blue:    nerfstudio training cameras (target)
      - orange:  UTM transformed via KNN local similarity (should overlap blue)
      - green:   UTM transformed via global Umeyama (for reference)
    Plus sparse error lines between blue and orange for visual sanity check.

    The plot is only displayed, never saved.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.plot(
        ns_pts[:, 0], ns_pts[:, 1],
        marker="o", markersize=2, linewidth=1.5,
        label="Nerfstudio cameras (target)",
    )
    ax.plot(
        utm_as_ns_knn[:, 0], utm_as_ns_knn[:, 1],
        marker=".", markersize=2, linewidth=1.5,
        label="UTM via KNN local similarity",
    )
    ax.plot(
        utm_as_ns_global[:, 0], utm_as_ns_global[:, 1],
        linewidth=1.0, linestyle="--",
        label="UTM via global Umeyama (reference)",
    )

    # Sparse correspondence lines: KNN prediction <-> NS truth
    step = max(1, len(ns_pts) // 60)
    for i in range(0, len(ns_pts), step):
        a = utm_as_ns_knn[i]
        b = ns_pts[i]
        ax.plot([a[0], b[0]], [a[1], b[1]],
                linewidth=0.5, alpha=0.35)

    # Annotate start and end frame IDs
    if len(matched_frames) > 0:
        for idx, label in [(0, "start"), (-1, "end")]:
            p = ns_pts[idx]
            ax.text(p[0], p[1], f" {label} f={matched_frames[idx]}")

    ax.set_title(title)
    ax.set_xlabel("Nerfstudio X")
    ax.set_ylabel("Nerfstudio Y")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute KNN local similarity transform from UTM to "
                    "Nerfstudio coordinates (better than global Umeyama "
                    "for long splits with strong curves)."
    )
    parser.add_argument("--gs_config", required=True)
    parser.add_argument("--utm_file", required=True)
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. If omitted, saves next to "
                             "config.yml as "
                             "utm_to_nerfstudio_transform_knn.json. "
                             "Does not overwrite the original "
                             "utm_to_nerfstudio_transform.json (global "
                             "Umeyama), so the two can be compared.")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of neighbors for local fit (default 20)")
    args = parser.parse_args()

    config_path = Path(args.gs_config).resolve()
    data_root = Path(args.data_root).resolve()

    if args.output is None:
        output_path = config_path.parent / "utm_to_nerfstudio_transform_knn.json"
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
    print(f"[INFO] k:         {args.k}")

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

    print(f"[INFO] Extracted {len(nerfstudio_data)} nerfstudio positions")

    # =========================================================================
    # Step 2: Load UTM positions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Loading UTM positions (ground truth)")
    print("=" * 70)

    utm_positions = read_utm_positions(args.utm_file)
    print(f"[INFO] Loaded {len(utm_positions)} UTM positions from: {args.utm_file}")

    # =========================================================================
    # Step 3: Match frames
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Matching frames between Nerfstudio and UTM")
    print("=" * 70)

    matched_frames = sorted(set(nerfstudio_data.keys()) & set(utm_positions.keys()))
    print(f"[INFO] Matched {len(matched_frames)} frames")

    if len(matched_frames) < args.k:
        print(f"[ERROR] Need at least k={args.k} matched frames!")
        sys.exit(1)

    # =========================================================================
    # Step 4: Build arrays
    # =========================================================================
    utm_pts = np.array([
        [utm_positions[f]["easting"], utm_positions[f]["northing"]]
        for f in matched_frames
    ])
    ns_pts = np.array([nerfstudio_data[f]["position"] for f in matched_frames])

    utm_yaws = np.array([utm_positions[f]["yaw"] for f in matched_frames])
    ns_yaws = np.array([nerfstudio_data[f]["yaw"] for f in matched_frames])
    ns_pitches = np.array([nerfstudio_data[f]["pitch"] for f in matched_frames])
    ns_rolls = np.array([nerfstudio_data[f]["roll"] for f in matched_frames])

    print(f"\n[INFO] UTM ranges: "
          f"E [{utm_pts[:, 0].min():.2f}, {utm_pts[:, 0].max():.2f}], "
          f"N [{utm_pts[:, 1].min():.2f}, {utm_pts[:, 1].max():.2f}]")
    print(f"[INFO] NS ranges:  "
          f"X [{ns_pts[:, 0].min():.4f}, {ns_pts[:, 0].max():.4f}], "
          f"Y [{ns_pts[:, 1].min():.4f}, {ns_pts[:, 1].max():.4f}], "
          f"Z [{ns_pts[:, 2].min():.4f}, {ns_pts[:, 2].max():.4f}]")

    # =========================================================================
    # Step 5: Global Umeyama (for reference/comparison only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Global Umeyama (for reference only)")
    print("=" * 70)

    s_global, R_global, t_global, error_global = umeyama_alignment(
        utm_pts, ns_pts[:, :2], with_scale=True
    )
    print(f"[INFO] Global scale: {s_global:.10f}")
    print(f"[INFO] Global rotation angle: "
          f"{np.degrees(np.arctan2(R_global[1, 0], R_global[0, 0])):.2f} deg")
    print(f"[INFO] Global RMS error: {error_global:.6f} NS units "
          f"(= {error_global / s_global:.4f} m)")

    # =========================================================================
    # Step 6: KNN Local Umeyama
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 6: KNN local similarity (k={args.k})")
    print("=" * 70)

    nn = NearestNeighbors(n_neighbors=args.k).fit(utm_pts)

    knn_errors = []
    for i, f in enumerate(matched_frames):
        # Predict NS_xy for this UTM point using KNN local fit on its
        # k nearest neighbors (excluding itself by using k+1 then dropping
        # the closest; for verification we want a fair leave-one-out).
        _, idx = nn.kneighbors(utm_pts[i].reshape(1, -1),
                                n_neighbors=min(args.k + 1, len(utm_pts)))
        # Exclude the query itself (index 0 usually) - safer to filter by id
        neighbor_idx = [j for j in idx[0] if j != i][:args.k]
        local_src = utm_pts[neighbor_idx]
        local_tgt = ns_pts[neighbor_idx, :2]
        s_loc, R_loc, t_loc, _ = umeyama_alignment(
            local_src, local_tgt, with_scale=True
        )
        pred_xy = s_loc * (R_loc @ utm_pts[i]) + t_loc
        true_xy = ns_pts[i, :2]
        knn_errors.append(np.linalg.norm(pred_xy - true_xy))

    knn_errors = np.array(knn_errors)
    knn_mean = float(knn_errors.mean())
    knn_max = float(knn_errors.max())
    knn_rms = float(np.sqrt(np.mean(knn_errors ** 2)))

    # Mean local scale = average scale across all local fits.
    # Used only to express NS errors in meters in the print.
    mean_local_scale = s_global

    print(f"[INFO] KNN local Umeyama (leave-one-out verification):")
    print(f"       Mean error: {knn_mean:.6f} NS units "
          f"(~ {knn_mean / mean_local_scale * 100.0:.1f} cm)")
    print(f"       Max  error: {knn_max:.6f} NS units "
          f"(~ {knn_max / mean_local_scale * 100.0:.1f} cm)")
    print(f"       RMS  error: {knn_rms:.6f} NS units "
          f"(~ {knn_rms / mean_local_scale * 100.0:.1f} cm)")

    if error_global > 1e-9:
        improvement = error_global / max(knn_rms, 1e-9)
        print(f"[INFO] Improvement over global Umeyama: {improvement:.2f}x "
              f"({error_global:.4f} -> {knn_rms:.4f} NS units)")

    # =========================================================================
    # Step 7: Yaw offset (kept global)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Yaw offset (global)")
    print("=" * 70)

    yaw_result = compute_yaw_offset(utm_yaws, ns_yaws)
    yaw_sign = yaw_result["yaw_sign"]
    yaw_offset = yaw_result["yaw_offset"]
    yaw_residual_std = yaw_result["residual_std"]

    print(f"[INFO] ns_yaw = {'+' if yaw_sign > 0 else '-'}utm_yaw + offset")
    print(f"[INFO] yaw_offset: {np.degrees(yaw_offset):.2f} deg")
    print(f"[INFO] residual std: {np.degrees(yaw_residual_std):.2f} deg")

    # =========================================================================
    # Step 8: Save new-format JSON
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Saving KNN transform JSON")
    print("=" * 70)

    # Note: the default output filename is utm_to_nerfstudio_transform_knn.json
    # (different from the global-Umeyama file utm_to_nerfstudio_transform.json)
    # so the two transforms can coexist for comparison.
    if output_path.exists():
        print(f"[INFO] Overwriting existing file: {output_path.name}")

    transform_data = {
        "description": ("KNN local similarity transform from UTM to "
                        "Nerfstudio. For each query the runtime fits an "
                        "Umeyama similarity on the k nearest training "
                        "points (in UTM)."),
        "mode": "local_similarity_knn",
        "k": int(args.k),
        "num_matched_points": len(matched_frames),

        "training_points": {
            "utm_easting":  utm_pts[:, 0].tolist(),
            "utm_northing": utm_pts[:, 1].tolist(),
            "ns_x":         ns_pts[:, 0].tolist(),
            "ns_y":         ns_pts[:, 1].tolist(),
            "ns_z":         ns_pts[:, 2].tolist(),
        },

        "yaw_alignment": {
            "description": "ns_yaw = yaw_sign * utm_yaw + yaw_offset_rad",
            "yaw_sign": int(yaw_sign),
            "yaw_offset_rad": float(yaw_offset),
            "yaw_offset_deg": float(np.degrees(yaw_offset)),
            "residual_std_deg": float(np.degrees(yaw_residual_std)),
        },

        "camera_mount_angles": {
            "description": "Median pitch and roll from training cameras",
            "avg_pitch_rad": float(np.median(ns_pitches)),
            "avg_pitch_deg": float(np.degrees(np.median(ns_pitches))),
            "avg_roll_rad":  float(np.median(ns_rolls)),
            "avg_roll_deg":  float(np.degrees(np.median(ns_rolls))),
            "pitch_std_deg": float(np.degrees(np.std(ns_pitches))),
            "roll_std_deg":  float(np.degrees(np.std(ns_rolls))),
        },

        "verification_errors_leave_one_out": {
            "knn_mean_ns_units": knn_mean,
            "knn_max_ns_units":  knn_max,
            "knn_rms_ns_units":  knn_rms,
            "global_umeyama_rms_ns_units": float(error_global),
            "improvement_factor": float(error_global / max(knn_rms, 1e-9)),
        },

        # For backward compatibility: also save global Umeyama params
        # so old CoordinateTransformer (without KNN support) can still read
        # the file. The "mode" field is what triggers the new behavior.
        "scale": float(s_global),
        "rotation": np.eye(3).tolist(),
        "translation": [float(t_global[0]), float(t_global[1]),
                        float(ns_pts[:, 2].mean())],

        "source_files": {
            "gs_config": str(config_path),
            "utm_file": os.path.abspath(args.utm_file),
        },
    }

    R3 = np.eye(3)
    R3[:2, :2] = R_global
    transform_data["rotation"] = R3.tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transform_data, f, indent=2)

    print(f"[INFO] Saved: {output_path}")
    print(f"[INFO] Mode:  local_similarity_knn (k={args.k})")

    # =========================================================================
    # Step 9: 2D Plot
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: 2D alignment plot (XY)")
    print("=" * 70)

    # Runtime-style KNN (includes the query in its own neighborhood) - this is
    # what the CoordinateTransformer actually does at runtime.
    utm_as_ns_knn = build_knn_runtime_points(utm_pts, ns_pts, nn, args.k)

    # Global Umeyama for visual reference (the old behavior).
    global_xy = s_global * (utm_pts @ R_global.T) + t_global
    global_z = np.full((len(global_xy), 1), float(ns_pts[:, 2].mean()))
    utm_as_ns_global = np.hstack([global_xy, global_z])

    title = (f"UTM -> Nerfstudio KNN local similarity, k={args.k}\n"
             f"LOO RMS XY error: {knn_rms:.4f} NS units | "
             f"Global RMS: {error_global:.4f}")

    plot_alignment_3d(
        ns_pts=ns_pts,
        utm_as_ns_knn=utm_as_ns_knn,
        utm_as_ns_global=utm_as_ns_global,
        matched_frames=matched_frames,
        title=title,
    )

    print(f"[INFO] Done.")


if __name__ == "__main__":
    main()