#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_colmap_old_vs_new.py

Aligns two COLMAP reconstructions (e.g. OLD/thesis vs NEW/cam2sim) to UTM
coordinates via Umeyama, and plots both on an OpenStreetMap basemap so you
can visually compare drift / jitter / coverage between them.

Per-split:
  1. Reads camera poses from images.bin (or images.txt) in <colmap>/sparse/0/
  2. Loads matching frame_positions.txt for UTM coords
  3. Matches camera filename -> frame_id -> (easting, northing)
  4. Computes Umeyama similarity transform (COLMAP -> UTM)
  5. Reports scale, rotation angle, RMS error, max error
  6. Transforms all COLMAP poses to UTM

Final step:
  - Plots OLD and NEW (all splits stacked) on OSM tiles via contextily.

Usage:
    python compare_colmap_old_vs_new.py \\
        --old_root /home/davidejannussi/Desktop/cam2sim/data/data_for_gaussian_splatting/reference_bag_old \\
        --new_root /home/davidejannussi/Desktop/cam2sim/data/data_for_gaussian_splatting/reference_bag_new \\
        --num_splits 3 \\
        --frame_skip 2 \\
        --output_dir /tmp/colmap_compare

Optional:
    --plot_only_split 1     # plot only one split
    --no_basemap            # skip OSM tiles (useful if no internet)
"""

import os
import re
import sys
import argparse
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False


# =============================================================================
# COLMAP BINARY READER (minimal, no colmap dependency)
# =============================================================================

def read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


def read_images_binary(path):
    """
    Read COLMAP images.bin. Returns dict: image_id -> {qvec, tvec, name}
    qvec is (qw, qx, qy, qz), tvec is camera-from-world translation.
    """
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])  # qw, qx, qy, qz
            tvec = np.array(binary_image_properties[5:8])
            current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = b""
            while current_char != b"\x00":
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            # skip points2D data
            read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "name": image_name.decode("utf-8"),
            }
    return images


def read_images_text(path):
    """Read COLMAP images.txt (fallback if .bin not available)."""
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array([float(x) for x in elems[1:5]])
            tvec = np.array([float(x) for x in elems[5:8]])
            image_name = elems[9]
            # skip the 2D points line
            fid.readline()
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "name": image_name,
            }
    return images


def qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def camera_center_from_qt(qvec, tvec):
    """COLMAP stores camera-from-world. Camera center = -R^T t."""
    R = qvec_to_rotmat(qvec)
    return -R.T @ tvec


# =============================================================================
# FRAME POSITIONS READER
# =============================================================================

def read_frame_positions(filepath):
    """
    Read frame_positions.txt. Returns dict: frame_id -> (easting, northing).
    Auto-detects short vs long format (see 4C_utm_yaw_to_nerfstudio.py).
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
                easting = float(parts[2])
                northing = float(parts[3])
                positions[frame_id] = (easting, northing)
            except (ValueError, IndexError):
                continue
    return positions


def extract_frame_number(filename):
    """Extract the last integer from a filename like 'frame_000123.png'."""
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    numbers = re.findall(r"\d+", name)
    if numbers:
        return int(numbers[-1])
    return None


# =============================================================================
# UMEYAMA
# =============================================================================

def umeyama(source, target, with_scale=True):
    """
    Find (s, R, t) such that target ≈ s * R @ source + t.
    Returns scale, R (2x2), t (2,), rms_error (in target units).
    """
    assert source.shape == target.shape
    n, dim = source.shape

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    sc = source - mu_s
    tc = target - mu_t

    var_s = np.sum(sc ** 2) / n
    cov = (tc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_t - s * R @ mu_s

    transformed = s * (source @ R.T) + t
    errs = np.linalg.norm(transformed - target, axis=1)
    return s, R, t, errs


# =============================================================================
# CORE: load one COLMAP split, return camera centers in UTM
# =============================================================================

def load_colmap_split_to_utm(colmap_root, split_idx, fp_file_path, label):
    """
    Returns:
        utm_pts:    (N, 2) camera centers in UTM, after Umeyama alignment
        gt_utm:     (N, 2) ground-truth UTM positions for the matched frames
        info:       dict with scale, rotation angle, rms, max_err, n_matched
    """
    sparse_dir = Path(colmap_root) / "colmap" / f"split_{split_idx}" / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"COLMAP sparse dir not found: {sparse_dir}")

    images_bin = sparse_dir / "images.bin"
    images_txt = sparse_dir / "images.txt"
    if images_bin.exists():
        images = read_images_binary(images_bin)
    elif images_txt.exists():
        images = read_images_text(images_txt)
    else:
        raise FileNotFoundError(f"No images.bin/.txt in {sparse_dir}")

    print(f"  [{label} split {split_idx}] read {len(images)} COLMAP poses")

    # Frame positions (UTM)
    if not Path(fp_file_path).exists():
        raise FileNotFoundError(f"frame_positions not found: {fp_file_path}")
    fp = read_frame_positions(fp_file_path)
    print(f"  [{label} split {split_idx}] read {len(fp)} UTM positions")

    # Match by frame number
    matched_colmap = []
    matched_utm = []
    for img in images.values():
        fnum = extract_frame_number(img["name"])
        if fnum is None or fnum not in fp:
            continue
        center = camera_center_from_qt(img["qvec"], img["tvec"])
        matched_colmap.append(center[:2])  # XY only
        matched_utm.append(fp[fnum])

    matched_colmap = np.array(matched_colmap)
    matched_utm = np.array(matched_utm)
    print(f"  [{label} split {split_idx}] matched {len(matched_colmap)} frames")

    if len(matched_colmap) < 3:
        raise RuntimeError(
            f"Too few matched frames in {label} split {split_idx}: "
            f"{len(matched_colmap)}"
        )

    s, R, t, errs = umeyama(matched_colmap, matched_utm)

    aligned = s * (matched_colmap @ R.T) + t

    rot_angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    info = {
        "n_matched": len(matched_colmap),
        "scale": float(s),
        "rotation_deg": float(rot_angle_deg),
        "rms": float(np.sqrt(np.mean(errs ** 2))),
        "mean_err": float(np.mean(errs)),
        "max_err": float(np.max(errs)),
        "median_err": float(np.median(errs)),
    }
    return aligned, matched_utm, info


# =============================================================================
# UTM -> WebMercator for plotting on OSM
# =============================================================================

def utm_array_to_mercator(easting, northing):
    """EPSG:25832 (UTM 32N) -> EPSG:3857 (Web Mercator) via WGS84."""
    tf_utm_to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    tf_wgs_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon, lat = tf_utm_to_wgs.transform(easting, northing)
    mx, my = tf_wgs_to_merc.transform(lon, lat)
    return np.array(mx), np.array(my)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_root", required=True,
                        help="Path to old COLMAP root (e.g. .../reference_bag_old)")
    parser.add_argument("--new_root", required=True,
                        help="Path to new COLMAP root (e.g. .../reference_bag_new)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--frame_skip", type=int, default=2,
                        help="frame_skip used to build frame_positions filename "
                             "(e.g. 2 -> '1_of_2'). Default 2.")
    parser.add_argument("--fp_pattern", default=None,
                        help="Optional custom frame_positions filename pattern. "
                             "Use {split} placeholder. "
                             "Default: frame_positions_split_{split}_1_of_<skip>.txt")
    parser.add_argument("--output_dir", default="/tmp/colmap_compare")
    parser.add_argument("--plot_only_split", type=int, default=None)
    parser.add_argument("--no_basemap", action="store_true")
    parser.add_argument("--buffer", type=float, default=80.0,
                        help="Map padding in meters")
    args = parser.parse_args()

    if args.fp_pattern is None:
        args.fp_pattern = (
            f"frame_positions_split_{{split}}_1_of_{args.frame_skip}.txt"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("COMPARE COLMAP OLD vs NEW (Umeyama -> UTM -> OSM plot)")
    print("=" * 70)
    print(f"OLD root: {args.old_root}")
    print(f"NEW root: {args.new_root}")
    print(f"Splits:   {args.num_splits}")
    print(f"FP file pattern: {args.fp_pattern}")
    print("=" * 70)

    # Decide which splits
    split_list = (
        [args.plot_only_split]
        if args.plot_only_split is not None
        else list(range(1, args.num_splits + 1))
    )

    old_per_split = {}
    new_per_split = {}
    gt_per_split = {}
    summary_rows = []

    for split in split_list:
        print(f"\n--- Split {split} ---")
        fp_filename = args.fp_pattern.format(split=split)

        # OLD
        fp_old = Path(args.old_root) / fp_filename
        try:
            old_pts, gt_old, info_old = load_colmap_split_to_utm(
                args.old_root, split, fp_old, "OLD"
            )
            old_per_split[split] = old_pts
            gt_per_split[split] = gt_old
            summary_rows.append((f"OLD split {split}", info_old))
            print(f"  [OLD split {split}] scale={info_old['scale']:.6g}, "
                  f"rot={info_old['rotation_deg']:.2f}°, "
                  f"RMS={info_old['rms']:.3f} m, "
                  f"median={info_old['median_err']:.3f} m, "
                  f"max={info_old['max_err']:.3f} m")
        except Exception as e:
            print(f"  [OLD split {split}] FAILED: {e}")
            old_per_split[split] = None

        # NEW
        fp_new = Path(args.new_root) / fp_filename
        try:
            new_pts, gt_new, info_new = load_colmap_split_to_utm(
                args.new_root, split, fp_new, "NEW"
            )
            new_per_split[split] = new_pts
            # GT preferably from old; fall back to new
            if split not in gt_per_split:
                gt_per_split[split] = gt_new
            summary_rows.append((f"NEW split {split}", info_new))
            print(f"  [NEW split {split}] scale={info_new['scale']:.6g}, "
                  f"rot={info_new['rotation_deg']:.2f}°, "
                  f"RMS={info_new['rms']:.3f} m, "
                  f"median={info_new['median_err']:.3f} m, "
                  f"max={info_new['max_err']:.3f} m")
        except Exception as e:
            print(f"  [NEW split {split}] FAILED: {e}")
            new_per_split[split] = None

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Umeyama fit COLMAP -> UTM (all in METERS)")
    print("=" * 70)
    print(f"  {'label':<20} {'N':>6} {'RMS':>9} {'median':>9} {'max':>9}  scale")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*9} {'-'*9}  {'-'*10}")
    for label, info in summary_rows:
        print(f"  {label:<20} {info['n_matched']:>6} "
              f"{info['rms']:>9.3f} {info['median_err']:>9.3f} "
              f"{info['max_err']:>9.3f}  {info['scale']:.6g}")

    print()
    print("INTERPRETATION:")
    print("  - Lower RMS/median = COLMAP poses fit the UTM trajectory better")
    print("  - If NEW has visibly higher RMS than OLD, NEW COLMAP is the culprit")
    print("  - 'max' tells you whether the noise is uniform or spike-driven")

    # ============================================================
    # PLOT (per-split + combined)
    # ============================================================
    print("\n" + "=" * 70)
    print("PLOTTING")
    print("=" * 70)

    # Combined plot
    fig, ax = plt.subplots(figsize=(14, 10))

    all_mx_for_bounds = []
    all_my_for_bounds = []

    colors_old = ["#1B5E20", "#2E7D32", "#388E3C"]   # greens
    colors_new = ["#B71C1C", "#C62828", "#D32F2F"]   # reds
    colors_gt  = ["#0D47A1", "#1565C0", "#1976D2"]   # blues

    for i, split in enumerate(split_list):
        # Plot GT
        if split in gt_per_split:
            gt = gt_per_split[split]
            gtmx, gtmy = utm_array_to_mercator(gt[:, 0], gt[:, 1])
            ax.plot(gtmx, gtmy,
                    color=colors_gt[i % len(colors_gt)],
                    linewidth=4, alpha=0.35, zorder=5,
                    label=f"GT UTM split {split}")
            all_mx_for_bounds.append(gtmx)
            all_my_for_bounds.append(gtmy)

        # Plot OLD
        if old_per_split.get(split) is not None:
            old = old_per_split[split]
            omx, omy = utm_array_to_mercator(old[:, 0], old[:, 1])
            ax.plot(omx, omy,
                    color=colors_old[i % len(colors_old)],
                    linewidth=1.3, alpha=0.9, zorder=10,
                    label=f"OLD split {split}")
            ax.scatter(omx, omy, s=4,
                       color=colors_old[i % len(colors_old)],
                       alpha=0.6, zorder=11)
            all_mx_for_bounds.append(omx)
            all_my_for_bounds.append(omy)

        # Plot NEW
        if new_per_split.get(split) is not None:
            new = new_per_split[split]
            nmx, nmy = utm_array_to_mercator(new[:, 0], new[:, 1])
            ax.plot(nmx, nmy,
                    color=colors_new[i % len(colors_new)],
                    linewidth=1.3, alpha=0.9, zorder=12,
                    label=f"NEW split {split}")
            ax.scatter(nmx, nmy, s=4,
                       color=colors_new[i % len(colors_new)],
                       alpha=0.6, zorder=13)
            all_mx_for_bounds.append(nmx)
            all_my_for_bounds.append(nmy)

    if all_mx_for_bounds:
        all_mx = np.concatenate(all_mx_for_bounds)
        all_my = np.concatenate(all_my_for_bounds)
        xmin, xmax = all_mx.min() - args.buffer, all_mx.max() + args.buffer
        ymin, ymax = all_my.min() - args.buffer, all_my.max() + args.buffer
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if HAS_CONTEXTILY and not args.no_basemap:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857",
                            source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            print(f"  [WARN] Could not load OSM tiles: {e}")

    ax.set_title("COLMAP poses aligned to UTM (Umeyama) — OLD vs NEW", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_axis_off()
    plt.tight_layout()

    out_combined = Path(args.output_dir) / "colmap_old_vs_new_combined.png"
    plt.savefig(out_combined, dpi=160, bbox_inches="tight")
    print(f"  Saved: {out_combined}")

    # Per-split plots
    for split in split_list:
        if old_per_split.get(split) is None and new_per_split.get(split) is None:
            continue

        fig2, ax2 = plt.subplots(figsize=(12, 9))

        bx, by = [], []
        if split in gt_per_split:
            gt = gt_per_split[split]
            gtmx, gtmy = utm_array_to_mercator(gt[:, 0], gt[:, 1])
            ax2.plot(gtmx, gtmy, color="#1565C0", linewidth=5,
                     alpha=0.35, zorder=5, label="GT UTM")
            bx.append(gtmx); by.append(gtmy)

        if old_per_split.get(split) is not None:
            old = old_per_split[split]
            omx, omy = utm_array_to_mercator(old[:, 0], old[:, 1])
            ax2.plot(omx, omy, color="#2E7D32", linewidth=1.5,
                     alpha=0.9, zorder=10, label="OLD COLMAP")
            ax2.scatter(omx, omy, s=6, color="#2E7D32",
                        alpha=0.7, zorder=11)
            bx.append(omx); by.append(omy)

        if new_per_split.get(split) is not None:
            new = new_per_split[split]
            nmx, nmy = utm_array_to_mercator(new[:, 0], new[:, 1])
            ax2.plot(nmx, nmy, color="#C62828", linewidth=1.5,
                     alpha=0.9, zorder=12, label="NEW COLMAP")
            ax2.scatter(nmx, nmy, s=6, color="#C62828",
                        alpha=0.7, zorder=13)
            bx.append(nmx); by.append(nmy)

        if bx:
            all_x = np.concatenate(bx); all_y = np.concatenate(by)
            ax2.set_xlim(all_x.min() - args.buffer, all_x.max() + args.buffer)
            ax2.set_ylim(all_y.min() - args.buffer, all_y.max() + args.buffer)

        if HAS_CONTEXTILY and not args.no_basemap:
            try:
                ctx.add_basemap(ax2, crs="EPSG:3857",
                                source=ctx.providers.CartoDB.Positron)
            except Exception:
                pass

        ax2.set_title(f"Split {split} — OLD vs NEW COLMAP (Umeyama -> UTM)",
                      fontsize=13)
        ax2.legend(loc="upper right", fontsize=10, framealpha=0.85)
        ax2.set_axis_off()
        plt.tight_layout()

        out_split = Path(args.output_dir) / f"colmap_old_vs_new_split_{split}.png"
        plt.savefig(out_split, dpi=160, bbox_inches="tight")
        print(f"  Saved: {out_split}")
        plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()