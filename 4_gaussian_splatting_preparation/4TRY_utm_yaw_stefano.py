#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4C_utm_yaw_to_nerfstudio_knn_plot.py

Compute a KNN LOCAL SIMILARITY transform from UTM coordinates to the
Nerfstudio coordinate system, using the same input/output style as your
4C_utm_yaw_to_nerfstudio_knn.py script, but with a 3D alignment plot kept.

Main outputs:
    1. JSON transform file, default:
       <config_dir>/utm_to_nerfstudio_transform_knn.json

    2. Optional 3D plot, default:
       <config_dir>/utm_to_nerfstudio_transform_knn_plot.png

Runtime transform model:
    For each query UTM point, use the k nearest matched training UTM points,
    fit a local 2D Umeyama similarity UTM(E,N) -> NS(X,Y), and transform the
    query. Z is not observable from UTM(E,N) alone, so this script stores NS Z
    training values and uses local inverse-distance interpolation for plotting.

The yaw alignment is global:
    ns_yaw = yaw_sign * utm_yaw + yaw_offset_rad

Usage:
    python 4C_utm_yaw_to_nerfstudio_knn_plot.py \
        --gs_config path/to/outputs/.../config.yml \
        --utm_file path/to/frame_positions.txt \
        --data_root path/to/dataset \
        --k 20 \
        --show_plot

Dependencies:
    pip install numpy matplotlib scikit-learn torch
    plus Nerfstudio in the active environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401  # kept because Nerfstudio/eval_setup expects torch env
from sklearn.neighbors import NearestNeighbors

from nerfstudio.utils.eval_utils import eval_setup


# ==============================================================================
# UTM File Reader
# ==============================================================================


def read_utm_positions(filepath: str | Path) -> dict[int, dict[str, float]]:
    """
    Read frame positions from a comma-separated text file.

    Expected compatible layouts:
        frame_id, ..., easting, northing, yaw, ...

    Matching your original script:
        - frame_id = parts[0]
        - easting  = parts[2]
        - northing = parts[3]
        - yaw      = parts[9] if len(parts) >= 11 else parts[4]
    """
    positions: dict[int, dict[str, float]] = {}
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
                yaw = float(parts[9]) if len(parts) >= 11 else float(parts[4])
                positions[frame_id] = {
                    "easting": easting,
                    "northing": northing,
                    "yaw": yaw,
                }
            except (ValueError, IndexError):
                continue

    return positions


def extract_frame_number(filename: str | Path) -> Optional[int]:
    name = os.path.splitext(str(filename))[0]
    name = os.path.basename(name)
    numbers = re.findall(r"\d+", name)
    if numbers:
        return int(numbers[-1])
    return None


# ==============================================================================
# YPR Extraction from Nerfstudio c2w
# ==============================================================================


def extract_ypr_from_c2w(c2w: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw, pitch, roll from a Nerfstudio camera-to-world matrix."""
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    forward = -c2w[:3, 2]

    pitch = np.arcsin(np.clip(forward[2], -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(-forward[0] / cos_pitch, forward[1] / cos_pitch)
        roll = np.arctan2(-right[2] / cos_pitch, up[2] / cos_pitch)
    else:
        yaw = np.arctan2(right[1], right[0])
        roll = 0.0

    return float(yaw), float(pitch), float(roll)


# ==============================================================================
# Umeyama Alignment
# ==============================================================================


def umeyama_alignment(
    source: np.ndarray,
    target: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Estimate target ~= s * R @ source + t.

    source: (N, D)
    target: (N, D)

    Returns:
        scale, rotation_matrix, translation, rms_error
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.shape != target.shape:
        raise ValueError(f"source and target shape mismatch: {source.shape} vs {target.shape}")

    n, dim = source.shape
    if n < dim + 1:
        raise ValueError(f"Need at least {dim + 1} points for a stable {dim}D similarity fit")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)

    source_centered = source - mu_source
    target_centered = target - mu_target

    var_source = np.sum(source_centered**2) / n
    if var_source < 1e-12:
        return 1.0, np.eye(dim), mu_target - mu_source, 0.0

    cov = (target_centered.T @ source_centered) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1.0

    R = U @ S @ Vt

    if with_scale:
        scale = float(np.trace(np.diag(D) @ S) / var_source)
    else:
        scale = 1.0

    translation = mu_target - scale * (R @ mu_source)
    transformed = scale * (source @ R.T) + translation
    error = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))

    return scale, R, translation, error


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


# ==============================================================================
# Local KNN Transform Utilities
# ==============================================================================


def get_neighbor_indices(
    query_xy: np.ndarray,
    nn: NearestNeighbors,
    k: int,
    exclude_index: Optional[int] = None,
    total_points: Optional[int] = None,
) -> np.ndarray:
    """Return k neighbor indices, optionally excluding one training index."""
    query_xy = np.asarray(query_xy, dtype=float).reshape(1, -1)

    if exclude_index is None:
        n_neighbors = k
    else:
        if total_points is None:
            raise ValueError("total_points is required when exclude_index is used")
        n_neighbors = min(k + 1, total_points)

    _, idx = nn.kneighbors(query_xy, n_neighbors=n_neighbors)
    idx = idx[0]

    if exclude_index is not None:
        idx = np.array([j for j in idx if j != exclude_index], dtype=int)[:k]

    if len(idx) < 3:
        raise ValueError("Need at least 3 local neighbors for 2D similarity")

    return idx


def transform_point_local_xy(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    nn: NearestNeighbors,
    k: int,
    exclude_index: Optional[int] = None,
) -> np.ndarray:
    """
    Transform one UTM query point to Nerfstudio XY using local KNN Umeyama.

    This is the runtime behavior your CoordinateTransformer should reproduce:
        1. find k nearest source points in UTM space
        2. fit local UTM -> NS XY similarity
        3. transform the query
    """
    idx = get_neighbor_indices(
        query_xy=query_xy,
        nn=nn,
        k=k,
        exclude_index=exclude_index,
        total_points=len(source_xy),
    )

    local_src = source_xy[idx]
    local_tgt = target_xy[idx]

    s, R, t, _ = umeyama_alignment(local_src, local_tgt, with_scale=True)
    return s * (R @ query_xy) + t


def interpolate_local_z(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    target_z: np.ndarray,
    nn: NearestNeighbors,
    k: int,
) -> float:
    """
    Estimate NS Z for plotting from nearby training cameras.

    UTM only has E/N, so there is no true similarity-derived Z. For the 3D plot,
    use inverse-distance weighting of neighboring Nerfstudio Z values. If the
    query exactly equals a training point, return that point's Z.
    """
    distances, idx = nn.kneighbors(np.asarray(query_xy).reshape(1, -1), n_neighbors=k)
    distances = distances[0]
    idx = idx[0]

    if distances[0] < 1e-12:
        return float(target_z[idx[0]])

    weights = 1.0 / np.maximum(distances, 1e-12)
    weights = weights / weights.sum()
    return float(np.sum(weights * target_z[idx]))


def transform_points_local_3d_for_plot(
    source_xy: np.ndarray,
    target_ns: np.ndarray,
    nn: NearestNeighbors,
    k: int,
) -> np.ndarray:
    """Transform all UTM training points into NS XYZ for visualization."""
    out = np.zeros((len(source_xy), 3), dtype=float)
    for i, q in enumerate(source_xy):
        out[i, :2] = transform_point_local_xy(q, source_xy, target_ns[:, :2], nn, k)
        out[i, 2] = interpolate_local_z(q, source_xy, target_ns[:, 2], nn, k)
    return out


# ==============================================================================
# Yaw Offset Computation
# ==============================================================================


def compute_yaw_offset(utm_yaws: np.ndarray, ns_yaws: np.ndarray) -> dict[str, float | int]:
    utm_yaws = np.asarray(utm_yaws, dtype=float)
    ns_yaws = np.asarray(ns_yaws, dtype=float)

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
        std = float(np.std(residuals))

        if std < best_std:
            best_std = std
            best_result = {
                "yaw_sign": int(sign),
                "yaw_offset": float(circular_mean),
                "residual_std": std,
            }

    assert best_result is not None
    return best_result


# ==============================================================================
# Plotting
# ==============================================================================


def set_axes_equal(ax) -> None:
    """Make a 3D axis have equal visual scale on X/Y/Z."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_alignment_3d(
    ns_pts: np.ndarray,
    utm_as_ns_knn: np.ndarray,
    utm_as_ns_global: np.ndarray,
    matched_frames: list[int],
    plot_path: Optional[Path],
    show_plot: bool,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        ns_pts[:, 0], ns_pts[:, 1], ns_pts[:, 2],
        marker="o", markersize=2, linewidth=1.5,
        label="Nerfstudio cameras / target",
    )
    ax.plot(
        utm_as_ns_knn[:, 0], utm_as_ns_knn[:, 1], utm_as_ns_knn[:, 2],
        marker=".", markersize=2, linewidth=1.5,
        label="UTM transformed by KNN local similarity",
    )
    ax.plot(
        utm_as_ns_global[:, 0], utm_as_ns_global[:, 1], utm_as_ns_global[:, 2],
        linewidth=1.0, linestyle="--",
        label="UTM transformed by global Umeyama reference",
    )

    # Sparse correspondence/error lines for visual sanity check.
    step = max(1, len(ns_pts) // 60)
    for i in range(0, len(ns_pts), step):
        a = utm_as_ns_knn[i]
        b = ns_pts[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=0.5, alpha=0.35)

    # Annotate start/end frame IDs.
    if len(matched_frames) > 0:
        for idx, label in [(0, "start"), (-1, "end")]:
            p = ns_pts[idx]
            ax.text(p[0], p[1], p[2], f" {label} f={matched_frames[idx]}")

    ax.set_title(title)
    ax.set_xlabel("Nerfstudio X")
    ax.set_ylabel("Nerfstudio Y")
    ax.set_zlabel("Nerfstudio Z")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180)
        print(f"[INFO] Saved plot: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute KNN local similarity transform from UTM to Nerfstudio "
            "coordinates and plot the 3D match."
        )
    )
    parser.add_argument("--gs_config", required=True, help="Path to Nerfstudio config.yml")
    parser.add_argument("--utm_file", required=True, help="Path to frame_positions.txt / UTM CSV-like file")
    parser.add_argument("--data_root", default=".", help="Dataset root used as cwd while loading Nerfstudio")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. If omitted, saves next to config.yml as "
            "utm_to_nerfstudio_transform_knn.json."
        ),
    )
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for local fit")
    parser.add_argument(
        "--plot_out",
        default=None,
        help=(
            "Output PNG path. If omitted, saves next to config.yml as "
            "utm_to_nerfstudio_transform_knn_plot.png. Use --no_save_plot to disable."
        ),
    )
    parser.add_argument("--no_save_plot", action="store_true", help="Do not save the 3D plot PNG")
    parser.add_argument("--show_plot", action="store_true", help="Display the matplotlib plot window")
    args = parser.parse_args()

    config_path = Path(args.gs_config).resolve()
    data_root = Path(args.data_root).resolve()

    if args.output is None:
        output_path = config_path.parent / "utm_to_nerfstudio_transform_knn.json"
    else:
        output_path = Path(args.output).resolve()

    if args.no_save_plot:
        plot_path = None
    elif args.plot_out is None:
        plot_path = config_path.parent / "utm_to_nerfstudio_transform_knn_plot.png"
    else:
        plot_path = Path(args.plot_out).resolve()

    print("=" * 70)
    print("STEP 1: Loading Nerfstudio model")
    print("=" * 70)
    print(f"[INFO] Config:    {config_path}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Output:    {output_path}")
    print(f"[INFO] Plot:      {plot_path if plot_path is not None else 'disabled'}")
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

    nerfstudio_data: dict[int, dict[str, object]] = {}
    for i, img_path in enumerate(image_filenames):
        frame_num = extract_frame_number(img_path)
        if frame_num is not None:
            pos = np.asarray(c2w_matrices[i, :, 3], dtype=float)
            yaw, pitch, roll = extract_ypr_from_c2w(c2w_matrices[i])
            nerfstudio_data[frame_num] = {
                "position": pos,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "filename": str(img_path),
            }

    print(f"[INFO] Extracted {len(nerfstudio_data)} Nerfstudio positions")

    print("\n" + "=" * 70)
    print("STEP 2: Loading UTM positions")
    print("=" * 70)
    utm_positions = read_utm_positions(args.utm_file)
    print(f"[INFO] Loaded {len(utm_positions)} UTM positions from: {args.utm_file}")

    print("\n" + "=" * 70)
    print("STEP 3: Matching frames")
    print("=" * 70)
    matched_frames = sorted(set(nerfstudio_data.keys()) & set(utm_positions.keys()))
    print(f"[INFO] Matched {len(matched_frames)} frames")

    if len(matched_frames) < max(args.k + 1, 4):
        print(f"[ERROR] Need at least max(k+1, 4) matched frames for leave-one-out verification. Got {len(matched_frames)}.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 4: Building arrays")
    print("=" * 70)

    utm_pts = np.array(
        [[utm_positions[f]["easting"], utm_positions[f]["northing"]] for f in matched_frames],
        dtype=float,
    )
    ns_pts = np.array([nerfstudio_data[f]["position"] for f in matched_frames], dtype=float)

    utm_yaws = np.array([utm_positions[f]["yaw"] for f in matched_frames], dtype=float)
    ns_yaws = np.array([nerfstudio_data[f]["yaw"] for f in matched_frames], dtype=float)
    ns_pitches = np.array([nerfstudio_data[f]["pitch"] for f in matched_frames], dtype=float)
    ns_rolls = np.array([nerfstudio_data[f]["roll"] for f in matched_frames], dtype=float)

    print(
        f"[INFO] UTM ranges: "
        f"E [{utm_pts[:, 0].min():.2f}, {utm_pts[:, 0].max():.2f}], "
        f"N [{utm_pts[:, 1].min():.2f}, {utm_pts[:, 1].max():.2f}]"
    )
    print(
        f"[INFO] NS ranges:  "
        f"X [{ns_pts[:, 0].min():.4f}, {ns_pts[:, 0].max():.4f}], "
        f"Y [{ns_pts[:, 1].min():.4f}, {ns_pts[:, 1].max():.4f}], "
        f"Z [{ns_pts[:, 2].min():.4f}, {ns_pts[:, 2].max():.4f}]"
    )

    print("\n" + "=" * 70)
    print("STEP 5: Global Umeyama reference")
    print("=" * 70)

    s_global, R_global, t_global, error_global = umeyama_alignment(
        utm_pts,
        ns_pts[:, :2],
        with_scale=True,
    )
    print(f"[INFO] Global scale: {s_global:.10f}")
    print(
        f"[INFO] Global rotation angle: "
        f"{np.degrees(np.arctan2(R_global[1, 0], R_global[0, 0])):.2f} deg"
    )
    print(
        f"[INFO] Global RMS error: {error_global:.6f} NS units "
        f"(= {error_global / max(s_global, 1e-12):.4f} m approx)"
    )

    print("\n" + "=" * 70)
    print(f"STEP 6: KNN local similarity verification, k={args.k}")
    print("=" * 70)

    nn = NearestNeighbors(n_neighbors=args.k).fit(utm_pts)

    knn_errors = []
    knn_predictions_xy_loo = np.zeros((len(utm_pts), 2), dtype=float)
    for i, q in enumerate(utm_pts):
        pred_xy = transform_point_local_xy(
            query_xy=q,
            source_xy=utm_pts,
            target_xy=ns_pts[:, :2],
            nn=nn,
            k=args.k,
            exclude_index=i,
        )
        knn_predictions_xy_loo[i] = pred_xy
        knn_errors.append(np.linalg.norm(pred_xy - ns_pts[i, :2]))

    knn_errors = np.asarray(knn_errors, dtype=float)
    knn_mean = float(knn_errors.mean())
    knn_max = float(knn_errors.max())
    knn_rms = float(np.sqrt(np.mean(knn_errors**2)))

    mean_local_scale = max(s_global, 1e-12)
    print("[INFO] KNN local Umeyama, leave-one-out verification:")
    print(f"       Mean error: {knn_mean:.6f} NS units (~ {knn_mean / mean_local_scale * 100.0:.1f} cm)")
    print(f"       Max  error: {knn_max:.6f} NS units (~ {knn_max / mean_local_scale * 100.0:.1f} cm)")
    print(f"       RMS  error: {knn_rms:.6f} NS units (~ {knn_rms / mean_local_scale * 100.0:.1f} cm)")

    improvement = float(error_global / max(knn_rms, 1e-9))
    print(f"[INFO] Improvement over global Umeyama: {improvement:.2f}x ({error_global:.4f} -> {knn_rms:.4f} NS units)")

    print("\n" + "=" * 70)
    print("STEP 7: Yaw offset")
    print("=" * 70)

    yaw_result = compute_yaw_offset(utm_yaws, ns_yaws)
    yaw_sign = int(yaw_result["yaw_sign"])
    yaw_offset = float(yaw_result["yaw_offset"])
    yaw_residual_std = float(yaw_result["residual_std"])

    print(f"[INFO] ns_yaw = {'+' if yaw_sign > 0 else '-'}utm_yaw + offset")
    print(f"[INFO] yaw_offset: {np.degrees(yaw_offset):.2f} deg")
    print(f"[INFO] residual std: {np.degrees(yaw_residual_std):.2f} deg")

    print("\n" + "=" * 70)
    print("STEP 8: Building 3D plot data")
    print("=" * 70)

    # Runtime-style local fit for plotting. This includes the query in its own
    # neighborhood, matching the normal CoordinateTransformer runtime behavior.
    utm_as_ns_knn = transform_points_local_3d_for_plot(utm_pts, ns_pts, nn, args.k)

    global_xy = apply_similarity(utm_pts, s_global, R_global, t_global)
    global_z = np.full((len(global_xy), 1), float(ns_pts[:, 2].mean()))
    utm_as_ns_global = np.hstack([global_xy, global_z])

    plot_xy_errors = np.linalg.norm(utm_as_ns_knn[:, :2] - ns_pts[:, :2], axis=1)
    print(f"[INFO] Runtime-style in-support KNN plot RMS XY error: {np.sqrt(np.mean(plot_xy_errors**2)):.6f} NS units")

    print("\n" + "=" * 70)
    print("STEP 9: Saving KNN transform JSON")
    print("=" * 70)

    if output_path.exists():
        print(f"[INFO] Overwriting existing file: {output_path.name}")

    R3 = np.eye(3)
    R3[:2, :2] = R_global

    # Approximate global inverse reference for debugging only. The local KNN
    # model itself is bidirectional only if your runtime also builds a reverse
    # NN model using NS_xy as source and UTM_xy as target.
    R_global_inv = R_global.T
    s_global_inv = 1.0 / max(s_global, 1e-12)
    t_global_inv = -s_global_inv * (R_global_inv @ t_global)

    transform_data = {
        "description": (
            "KNN local similarity transform from UTM to Nerfstudio. For each query, "
            "the runtime fits an Umeyama similarity on the k nearest training points in UTM. "
            "This file also includes global reference parameters and plotting diagnostics."
        ),
        "mode": "local_similarity_knn",
        "k": int(args.k),
        "num_matched_points": int(len(matched_frames)),
        "matched_frames": [int(f) for f in matched_frames],

        "training_points": {
            "utm_easting": utm_pts[:, 0].tolist(),
            "utm_northing": utm_pts[:, 1].tolist(),
            "ns_x": ns_pts[:, 0].tolist(),
            "ns_y": ns_pts[:, 1].tolist(),
            "ns_z": ns_pts[:, 2].tolist(),
        },

        "forward_transform": {
            "description": "Runtime local model: UTM(easting,northing) -> Nerfstudio(x,y). Z is stored from training NS cameras.",
            "source": "utm_easting_northing",
            "target": "nerfstudio_x_y",
            "mode": "local_similarity_knn",
            "k": int(args.k),
        },

        "inverse_transform_reference": {
            "description": (
                "Approximate global inverse reference only: Nerfstudio(x,y) -> UTM(easting,northing). "
                "For a true local inverse, build another KNN model with source=NS_xy and target=UTM_en."
            ),
            "scale": float(s_global_inv),
            "rotation_2x2": R_global_inv.tolist(),
            "translation_2d": t_global_inv.tolist(),
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
            "avg_roll_rad": float(np.median(ns_rolls)),
            "avg_roll_deg": float(np.degrees(np.median(ns_rolls))),
            "pitch_std_deg": float(np.degrees(np.std(ns_pitches))),
            "roll_std_deg": float(np.degrees(np.std(ns_rolls))),
        },

        "verification_errors_leave_one_out": {
            "knn_mean_ns_units": float(knn_mean),
            "knn_max_ns_units": float(knn_max),
            "knn_rms_ns_units": float(knn_rms),
            "global_umeyama_rms_ns_units": float(error_global),
            "improvement_factor": float(improvement),
        },

        "plot_diagnostics": {
            "plot_path": str(plot_path) if plot_path is not None else None,
            "runtime_style_knn_plot_rms_xy_error_ns_units": float(np.sqrt(np.mean(plot_xy_errors**2))),
            "note": "Plot KNN errors include the query point in its own neighborhood; verification errors above are leave-one-out.",
        },

        # Backward compatibility fields matching your existing JSON shape.
        "scale": float(s_global),
        "rotation": R3.tolist(),
        "translation": [float(t_global[0]), float(t_global[1]), float(ns_pts[:, 2].mean())],

        "source_files": {
            "gs_config": str(config_path),
            "utm_file": os.path.abspath(args.utm_file),
            "data_root": str(data_root),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transform_data, f, indent=2)

    print(f"[INFO] Saved: {output_path}")
    print(f"[INFO] Mode:  local_similarity_knn (k={args.k})")

    print("\n" + "=" * 70)
    print("STEP 10: Plotting 3D alignment")
    print("=" * 70)

    title = (
        f"UTM -> Nerfstudio KNN local similarity, k={args.k}\n"
        f"LOO RMS XY error: {knn_rms:.4f} NS units | Global RMS: {error_global:.4f}"
    )
    plot_alignment_3d(
        ns_pts=ns_pts,
        utm_as_ns_knn=utm_as_ns_knn,
        utm_as_ns_global=utm_as_ns_global,
        matched_frames=matched_frames,
        plot_path=plot_path,
        show_plot=args.show_plot,
        title=title,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
