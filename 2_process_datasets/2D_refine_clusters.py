#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D_refine_clusters.py

Interactive GUI to clean parked-car centroids over an OSM basemap.

The script refines either camera or LiDAR centroid detections depending on
--source.

When --source camera and --match are enabled, the script aligns camera
detections to LiDAR detections using greedy global nearest-neighbor matching:

    - camera detections keep their RGB color
    - matched camera detections are moved to LiDAR coordinates
    - matched camera detections copy LiDAR orientation and side
    - unmatched camera detections are dropped
    - unmatched LiDAR detections are inserted with neutral RGB color 128-128-128

When matching is enabled, the GUI displays:
    - pre-match camera detections as blue empty circles
    - LiDAR detections as yellow squares
    - accepted match links as black dotted lines
    - editable final detections as colored points and orientation bars

After optional matching, the GUI opens and allows manual edits.
Use --no-window to save the automatic matching result without opening the GUI.

Reads from:
    data/raw_dataset/<BAG>/trajectory.csv
    data/raw_dataset/<BAG>/shift.txt (optional)
    data/processed_dataset/<BAG>/<SOURCE>_detections/unified_clusters.txt

Optionally reads, when using:
    --source camera --match

    data/processed_dataset/<BAG>/lidar_detections/unified_clusters.txt

Writes to:
    data/processed_dataset/<BAG>/<SOURCE>_detections/
        unified_clusters_filtered.txt

Parameters:
    --bag-name <BAG>.bag
        Bag filename including .bag extension.
        Default: env BAG_NAME or reference_bag.bag.

    --source {camera,lidar}
        Detection source to refine.
        Default: camera.

    --match
        Only valid with --source camera.
        Greedily match camera detections to LiDAR detections before opening
        the GUI.

    --match-threshold <METERS>
        Maximum camera-LiDAR distance for greedy matching.
        Default: 4.0.

    --no-window
        Save the refinement result without opening the interactive GUI.
        The GUI opens by default.

Usage:
    python 2_process_datasets/2D_refine_clusters.py \
        --bag-name snowy.bag \
        --source camera

    python 2_process_datasets/2D_refine_clusters.py \
        --bag-name snowy.bag \
        --source lidar

    python 2_process_datasets/2D_refine_clusters.py \
        --bag-name snowy.bag \
        --source camera \
        --match

    python 2_process_datasets/2D_refine_clusters.py \
        --bag-name snowy.bag \
        --source camera \
        --match \
        --match-threshold 4.0

    python 2_process_datasets/2D_refine_clusters.py \
        --bag-name snowy.bag --source camera --match --no-window
"""

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import contextily as cx
from pyproj import Transformer


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_UTILS_DIR = SCRIPT_DIR / "utils"

if not LOCAL_UTILS_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

sys.path.insert(0, str(SCRIPT_DIR))


from utils.coordinates import (
    ODOM0_X,
    ODOM0_Y,
    get_projected_coords,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_SOURCE = "camera"
DEFAULT_MATCH_THRESHOLD = 4.0

BUFFER_M = 150
BAR_LENGTH = 4.0

NEUTRAL_RGB_COLOR = "128-128-128"

TRANSFORMER_TO_4326 = Transformer.from_crs(
    "EPSG:3857",
    "EPSG:4326",
    always_xy=True,
)

TRANSFORMER_WGS84_TO_UTM = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:25832",
    always_xy=True,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive GUI to clean parked-car centroid detections. "
            "Supports camera, LiDAR, and optional camera-to-LiDAR matching."
        )
    )

    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help=(
            "Bag filename including .bag extension "
            "(default: env BAG_NAME or 'reference_bag.bag')."
        ),
    )

    parser.add_argument(
        "--source",
        choices=["camera", "lidar"],
        default=DEFAULT_SOURCE,
        help=f"Detection source to refine. Default: {DEFAULT_SOURCE}.",
    )

    parser.add_argument(
        "--match",
        action="store_true",
        help=(
            "Only valid with --source camera. Greedily match camera detections "
            "to LiDAR detections before opening the GUI. Camera RGB colors are "
            "preserved; LiDAR position/orientation/side are used."
        ),
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help=(
            "Maximum camera-LiDAR distance in meters for matching. "
            f"Default: {DEFAULT_MATCH_THRESHOLD}."
        ),
    )

    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Save the result without opening the GUI. The GUI opens by default.",
    )

    return parser.parse_args()


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def get_inverse_coords(proj_x, proj_y, shift_x, shift_y, yaw_offset):
    """
    Convert Web Mercator coordinates back to odometry / UTM coordinates.

    Input:
        proj_x, proj_y:
            EPSG:3857 map coordinates.

    Output:
        raw_x, raw_y:
            Odometry / UTM coordinates.
    """
    lon, lat = TRANSFORMER_TO_4326.transform(proj_x, proj_y)
    utm_x, utm_y = TRANSFORMER_WGS84_TO_UTM.transform(lon, lat)

    if abs(yaw_offset) > 1e-9:
        dx = utm_x - ODOM0_X
        dy = utm_y - ODOM0_Y

        c = math.cos(-yaw_offset)
        s = math.sin(-yaw_offset)

        dx_rot = c * dx - s * dy
        dy_rot = s * dx + c * dy

        utm_x = ODOM0_X + dx_rot
        utm_y = ODOM0_Y + dy_rot

    raw_x = utm_x - shift_x
    raw_y = utm_y - shift_y

    return raw_x, raw_y


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory_from_txt(path):
    """
    Load trajectory file.

    Supports both:

        timestamp,x,y,z,yaw

    and headerless format:

        timestamp, x, y, z, yaw
    """
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    first_data_line = None

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            first_data_line = line
            break

    if first_data_line is None:
        raise RuntimeError(f"Trajectory file is empty: {path}")

    first_token = first_data_line.split(",")[0].strip()

    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        raw = np.genfromtxt(
            path,
            delimiter=",",
            comments="#",
            names=True,
            dtype=None,
            encoding="utf-8",
        )

        if raw.ndim == 0:
            raw = np.array([raw])

        name_map = {
            name.strip().lower(): name
            for name in raw.dtype.names
        }

        if "x" not in name_map or "y" not in name_map:
            raise KeyError(
                f"Trajectory CSV must contain x and y columns. Found: {raw.dtype.names}"
            )

        x = np.asarray(raw[name_map["x"]], dtype=float)
        y = np.asarray(raw[name_map["y"]], dtype=float)

    else:
        data = np.loadtxt(
            path,
            delimiter=",",
            comments="#",
            usecols=(1, 2),
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        x = data[:, 0]
        y = data[:, 1]

    return x, y


def load_calibration(path):
    """
    Load optional shift/yaw calibration file.

    Expected keys:
        SHIFT_X=
        SHIFT_Y=
        YAW_OFFSET=
    """
    shift_x = 0.0
    shift_y = 0.0
    yaw_offset = 0.0

    if not path.is_file():
        print(f"[INFO] Calibration file not found, using zero shift: {path}")
        return shift_x, shift_y, yaw_offset

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("SHIFT_X="):
                shift_x = float(line.split("=", 1)[1])
            elif line.startswith("SHIFT_Y="):
                shift_y = float(line.split("=", 1)[1])
            elif line.startswith("YAW_OFFSET="):
                yaw_offset = float(line.split("=", 1)[1])

    return shift_x, shift_y, yaw_offset


def color_from_side(side):
    if side == "left":
        return "limegreen"

    if side == "right":
        return "hotpink"

    return "gray"


def parse_rgb_color(row):
    """
    Parse optional rgb_color column.

    Expected format:
        R-G-B

    Returns:
        string or None
    """
    if len(row) < 9:
        return None

    value = str(row[8]).strip()

    if value == "" or value.lower() in {"none", "unknown", "nan"}:
        return None

    return value


def load_centroids(path):
    """
    Load detected centroid file.

    Supports both:

        cluster_id, x, y, z, count, conf, orientation, side

    and:

        cluster_id, x, y, z, count, conf, orientation, side, rgb_color
    """
    centroid_list = []

    if not path.is_file():
        raise FileNotFoundError(f"Centroid file not found: {path}")

    raw = np.genfromtxt(
        path,
        delimiter=",",
        comments="#",
        dtype=None,
        encoding="utf-8",
    )

    if raw.ndim == 0:
        raw = np.array([raw])

    for row in raw:
        centroid = {
            "id": int(float(row[0])),
            "x": float(row[1]),
            "y": float(row[2]),
            "z": float(row[3]),
            "count": int(float(row[4])),
            "conf": float(row[5]),
            "orient": str(row[6]).lower().strip(),
            "side": str(row[7]).lower().strip(),
            "rgb_color": parse_rgb_color(row),
        }

        centroid["color"] = color_from_side(centroid["side"])

        centroid["pre_match_x"] = centroid["x"]
        centroid["pre_match_y"] = centroid["y"]
        centroid["pre_match_z"] = centroid["z"]

        centroid_list.append(centroid)

    return centroid_list


# =============================================================================
# MATCHING
# =============================================================================

def greedy_camera_lidar_match(camera_list, lidar_gt_list, threshold):
    """
    Greedily align camera detections to LiDAR detections.

    Matching rule:
        1. compute all camera-LiDAR distances below threshold
        2. sort globally by distance
        3. scan from shortest to longest
        4. accept pair only if both camera and LiDAR are still unmatched

    Matched camera detections:
        - keep camera RGB color
        - receive LiDAR x, y, z
        - receive LiDAR orient
        - receive LiDAR side

    Unmatched camera detections:
        - dropped

    Unmatched LiDAR detections:
        - inserted using LiDAR x, y, z, orient, side
        - assigned neutral RGB color 128-128-128
    """
    pairs = []

    for cam_idx, cam in enumerate(camera_list):
        for gt_idx, gt in enumerate(lidar_gt_list):
            dist = math.hypot(
                cam["x"] - gt["x"],
                cam["y"] - gt["y"],
            )

            if dist <= threshold:
                pairs.append({
                    "cam_idx": cam_idx,
                    "gt_idx": gt_idx,
                    "distance": dist,
                })

    pairs.sort(key=lambda item: item["distance"])

    used_cam = set()
    used_gt = set()
    accepted_pairs = []

    for pair in pairs:
        cam_idx = pair["cam_idx"]
        gt_idx = pair["gt_idx"]

        if cam_idx in used_cam or gt_idx in used_gt:
            continue

        used_cam.add(cam_idx)
        used_gt.add(gt_idx)
        accepted_pairs.append(pair)

    refined = []
    debug_links = []

    for pair in accepted_pairs:
        cam = dict(camera_list[pair["cam_idx"]])
        gt = lidar_gt_list[pair["gt_idx"]]

        debug_links.append({
            "cam_id": cam["id"],
            "gt_id": gt["id"],
            "distance": pair["distance"],
            "cam_x": cam["pre_match_x"],
            "cam_y": cam["pre_match_y"],
            "gt_x": gt["x"],
            "gt_y": gt["y"],
        })

        rgb_color = cam.get("rgb_color")

        cam["x"] = gt["x"]
        cam["y"] = gt["y"]
        cam["z"] = gt["z"]
        cam["orient"] = gt["orient"]
        cam["side"] = gt["side"]
        cam["color"] = color_from_side(gt["side"])

        cam["rgb_color"] = rgb_color

        refined.append(cam)

    next_id = max([item["id"] for item in camera_list], default=0) + 1

    for gt_idx, gt in enumerate(lidar_gt_list):
        if gt_idx in used_gt:
            continue

        inserted = {
            "id": next_id,
            "x": gt["x"],
            "y": gt["y"],
            "z": gt["z"],
            "count": gt["count"],
            "conf": gt["conf"],
            "orient": gt["orient"],
            "side": gt["side"],
            "rgb_color": NEUTRAL_RGB_COLOR,
            "color": color_from_side(gt["side"]),
            "pre_match_x": gt["x"],
            "pre_match_y": gt["y"],
            "pre_match_z": gt["z"],
            "inserted_from_lidar": True,
        }

        refined.append(inserted)
        next_id += 1

    refined.sort(key=lambda item: item["id"])

    unmatched_camera = [
        camera_list[idx]
        for idx in range(len(camera_list))
        if idx not in used_cam
    ]

    unmatched_lidar = [
        lidar_gt_list[idx]
        for idx in range(len(lidar_gt_list))
        if idx not in used_gt
    ]

    print("\n[INFO] Camera-to-LiDAR matching")
    print(f"  Camera detections:     {len(camera_list)}")
    print(f"  LiDAR GT detections:   {len(lidar_gt_list)}")
    print(f"  Candidate pairs:       {len(pairs)}")
    print(f"  Accepted matches:      {len(accepted_pairs)}")
    print(f"  Dropped camera-only:   {len(unmatched_camera)}")
    print(f"  Inserted LiDAR-only:   {len(unmatched_lidar)}")
    print(f"  Match threshold:       {threshold:.2f} m")

    for link in debug_links:
        print(
            f"  MATCH cam {link['cam_id']} -> lidar {link['gt_id']} "
            f"dist={link['distance']:.2f} m"
        )

    return refined, debug_links, lidar_gt_list


# =============================================================================
# PLOT HELPERS
# =============================================================================

def calculate_bar_segments(cx, cy, tx, ty, orientations, length):
    segments = []

    for i in range(len(cx)):
        dists = (tx - cx[i]) ** 2 + (ty - cy[i]) ** 2
        idx = np.argmin(dists)

        if idx < len(tx) - 1:
            angle = np.arctan2(
                ty[idx + 1] - ty[idx],
                tx[idx + 1] - tx[idx],
            )
        else:
            angle = np.arctan2(
                ty[idx] - ty[idx - 1],
                tx[idx] - tx[idx - 1],
            )

        if orientations[i] == "perpendicular":
            angle += np.pi / 2.0

        dx = (length / 2.0) * np.cos(angle)
        dy = (length / 2.0) * np.sin(angle)

        segments.append([
            (cx[i] - dx, cy[i] - dy),
            (cx[i] + dx, cy[i] + dy),
        ])

    return segments


def project_debug_links(debug_links):
    if not debug_links:
        return []

    segments = []

    for link in debug_links:
        proj_cam_x, proj_cam_y = get_projected_coords(
            np.array([link["cam_x"]]),
            np.array([link["cam_y"]]),
        )

        proj_gt_x, proj_gt_y = get_projected_coords(
            np.array([link["gt_x"]]),
            np.array([link["gt_y"]]),
        )

        segments.append([
            (proj_cam_x[0], proj_cam_y[0]),
            (proj_gt_x[0], proj_gt_y[0]),
        ])

    return segments


def plot_pre_match_camera(ax, pre_match_camera_list):
    if not pre_match_camera_list:
        return

    pre_match_x = np.array([c["x"] for c in pre_match_camera_list])
    pre_match_y = np.array([c["y"] for c in pre_match_camera_list])

    proj_pre_match_x, proj_pre_match_y = get_projected_coords(
        pre_match_x,
        pre_match_y,
    )

    ax.scatter(
        proj_pre_match_x,
        proj_pre_match_y,
        facecolors="none",
        edgecolors="dodgerblue",
        s=90,
        marker="o",
        linewidth=1.5,
        zorder=4,
        label="Pre-match camera",
    )

    for i, cam in enumerate(pre_match_camera_list):
        ax.annotate(
            f"C{cam['id']}",
            (proj_pre_match_x[i], proj_pre_match_y[i]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=7,
            color="dodgerblue",
        )


def plot_lidar_gt(ax, lidar_gt_list):
    if not lidar_gt_list:
        return

    gt_x = np.array([g["x"] for g in lidar_gt_list])
    gt_y = np.array([g["y"] for g in lidar_gt_list])

    proj_gt_x, proj_gt_y = get_projected_coords(gt_x, gt_y)

    ax.scatter(
        proj_gt_x,
        proj_gt_y,
        c="gold",
        s=100,
        marker="s",
        edgecolor="black",
        linewidth=1,
        zorder=4,
        label="LiDAR GT",
    )

    for i, gt in enumerate(lidar_gt_list):
        ax.annotate(
            f"GT{gt['id']}",
            (proj_gt_x[i], proj_gt_y[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            color="darkgoldenrod",
        )


# =============================================================================
# SAVE
# =============================================================================

def save_filtered_centroids(output_file, centroid_data_list):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        file.write("# cluster_id, x, y, z, count, last_conf, orientation, side, rgb_color\n")

        for c in centroid_data_list:
            rgb_color = c.get("rgb_color") or "unknown"

            file.write(
                f"{c['id']}, "
                f"{c['x']:.3f}, "
                f"{c['y']:.3f}, "
                f"{c['z']:.3f}, "
                f"{c['count']}, "
                f"{c['conf']:.3f}, "
                f"{c['orient']}, "
                f"{c['side']}, "
                f"{rgb_color}\n"
            )

    print(f"[OK] Saved {len(centroid_data_list)} cars to {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem
    source = args.source

    if args.match and source != "camera":
        raise ValueError("--match is only valid with --source camera.")

    raw_dataset_dir = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem
    processed_dataset_dir = PROJECT_ROOT / "data" / "processed_dataset" / bag_stem

    trajectory_path = raw_dataset_dir / "trajectory.csv"
    shift_file = raw_dataset_dir / "shift.txt"

    source_dir = processed_dataset_dir / f"{source}_detections"
    centroid_file = source_dir / "unified_clusters.txt"
    output_filtered_file = source_dir / "unified_clusters_filtered.txt"

    lidar_gt_file = processed_dataset_dir / "lidar_detections" / "unified_clusters.txt"

    print("=" * 70)
    print("CLUSTER CLEANER")
    print("=" * 70)
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Bag:               {bag_name}")
    print(f"Bag stem:          {bag_stem}")
    print(f"Source:            {source}")
    print(f"Trajectory:        {trajectory_path}")
    print(f"Centroids:         {centroid_file}")
    print(f"LiDAR GT:          {lidar_gt_file if lidar_gt_file.is_file() else 'not found'}")
    print(f"Shift file:        {shift_file}")
    print(f"Output:            {output_filtered_file}")
    print(f"Match enabled:     {args.match}")
    print(f"Match threshold:   {args.match_threshold}")
    print(f"Interactive window: {not args.no_window}")
    print("=" * 70)

    if not centroid_file.is_file():
        raise FileNotFoundError(
            f"Centroid file not found: {centroid_file}\n"
            f"Run the {source} detection step before this refinement script."
        )

    odom_x, odom_y = load_trajectory_from_txt(trajectory_path)
    init_sx, init_sy, init_yaw = load_calibration(shift_file)

    centroid_data_list = load_centroids(centroid_file)
    print(f"[INFO] Loaded {len(centroid_data_list)} {source} centroids.")

    pre_match_camera_list = []

    if source == "camera":
        pre_match_camera_list = [
            dict(item)
            for item in centroid_data_list
        ]

    lidar_gt_list = []
    debug_links = []

    if args.match:
        if not lidar_gt_file.is_file():
            raise FileNotFoundError(
                f"--match requested, but LiDAR detections were not found: {lidar_gt_file}"
            )

        lidar_gt_list = load_centroids(lidar_gt_file)

        centroid_data_list, debug_links, lidar_gt_list = greedy_camera_lidar_match(
            camera_list=centroid_data_list,
            lidar_gt_list=lidar_gt_list,
            threshold=args.match_threshold,
        )

    if args.no_window:
        save_filtered_centroids(output_filtered_file, centroid_data_list)
        print("[OK] Non-interactive cluster refinement completed.")
        return

    proj_tx, proj_ty = get_projected_coords(odom_x, odom_y)

    print(
        f"[INFO] Projected trajectory x range: "
        f"[{np.min(proj_tx):.2f}, {np.max(proj_tx):.2f}]"
    )
    print(
        f"[INFO] Projected trajectory y range: "
        f"[{np.min(proj_ty):.2f}, {np.max(proj_ty):.2f}]"
    )

    # -------------------------------------------------------------------------
    # Figure setup
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.20)

    ax.set_xlim(np.min(proj_tx) - BUFFER_M, np.max(proj_tx) + BUFFER_M)
    ax.set_ylim(np.min(proj_ty) - BUFFER_M, np.max(proj_ty) + BUFFER_M)

    try:
        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=cx.providers.CartoDB.Positron,
        )
    except Exception as exc:
        print(f"[WARN] Basemap failed: {type(exc).__name__}: {exc}")

    ax.plot(
        proj_tx,
        proj_ty,
        "r-",
        alpha=0.5,
        label="Trajectory",
    )

    if args.match:
        plot_pre_match_camera(
            ax=ax,
            pre_match_camera_list=pre_match_camera_list,
        )

    if args.match:
        plot_lidar_gt(
            ax=ax,
            lidar_gt_list=lidar_gt_list,
        )

    debug_segments = project_debug_links(debug_links)

    if debug_segments:
        debug_coll = LineCollection(
            debug_segments,
            colors="black",
            linewidths=1.2,
            linestyles="dotted",
            alpha=0.8,
            zorder=5,
        )
        ax.add_collection(debug_coll)

    cent_bars_coll = LineCollection(
        [],
        linewidths=2.5,
        zorder=10,
    )
    ax.add_collection(cent_bars_coll)

    cent_scat = ax.scatter(
        [],
        [],
        s=40,
        edgecolor="k",
        linewidth=0.5,
        zorder=11,
        picker=5,
    )

    def update_plot_data():
        if not centroid_data_list:
            cent_scat.set_offsets(np.zeros((0, 2)))
            cent_bars_coll.set_segments([])
            fig.canvas.draw_idle()
            return

        cx_curr = np.array([c["x"] for c in centroid_data_list])
        cy_curr = np.array([c["y"] for c in centroid_data_list])

        colors_curr = [c["color"] for c in centroid_data_list]
        orients_curr = [c["orient"] for c in centroid_data_list]

        proj_cx, proj_cy = get_projected_coords(cx_curr, cy_curr)

        new_segments = calculate_bar_segments(
            proj_cx,
            proj_cy,
            proj_tx,
            proj_ty,
            orients_curr,
            BAR_LENGTH,
        )

        cent_scat.set_offsets(np.c_[proj_cx, proj_cy])
        cent_scat.set_facecolors(colors_curr)

        cent_bars_coll.set_segments(new_segments)
        cent_bars_coll.set_color(colors_curr)

        fig.canvas.draw_idle()

    update_plot_data()

    # -------------------------------------------------------------------------
    # Interaction state
    # -------------------------------------------------------------------------
    current_mode = "DELETE"
    dragging_idx = None
    next_insert_id = max([c["id"] for c in centroid_data_list], default=0) + 1

    def get_next_id():
        nonlocal next_insert_id

        new_id = next_insert_id
        next_insert_id += 1

        return new_id

    def on_pick(event):
        nonlocal dragging_idx

        if current_mode == "INSERT":
            return

        if event.artist != cent_scat:
            return

        if event.mouseevent.button != 1:
            return

        ind = event.ind[0]
        car_id = centroid_data_list[ind]["id"]

        if current_mode == "DELETE":
            removed = centroid_data_list.pop(ind)
            print(f"DELETE: ID {removed['id']}")
            update_plot_data()

        elif current_mode == "ROTATE":
            current_orientation = centroid_data_list[ind]["orient"]

            new_orientation = (
                "perpendicular"
                if current_orientation == "parallel"
                else "parallel"
            )

            centroid_data_list[ind]["orient"] = new_orientation

            print(f"ROTATE: ID {car_id} -> {new_orientation.upper()}")
            update_plot_data()

        elif current_mode == "SWITCH_SIDE":
            current_side = centroid_data_list[ind]["side"]

            if current_side == "left":
                centroid_data_list[ind]["side"] = "right"
                centroid_data_list[ind]["color"] = "hotpink"
            else:
                centroid_data_list[ind]["side"] = "left"
                centroid_data_list[ind]["color"] = "limegreen"

            print(f"SIDE: ID {car_id} -> {centroid_data_list[ind]['side'].upper()}")
            update_plot_data()

        elif current_mode == "MOVE":
            dragging_idx = ind
            print(f"GRABBED: ID {car_id}. Drag to move.")

    def on_click(event):
        if current_mode != "INSERT":
            return

        if event.inaxes != ax:
            return

        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        raw_x, raw_y = get_inverse_coords(
            event.xdata,
            event.ydata,
            init_sx,
            init_sy,
            init_yaw,
        )

        new_id = get_next_id()

        if source == "camera":
            rgb_color = NEUTRAL_RGB_COLOR
        else:
            rgb_color = "0-0-0"

        new_car = {
            "id": new_id,
            "x": raw_x,
            "y": raw_y,
            "z": 0.0,
            "count": 1,
            "conf": 1.0,
            "orient": "parallel",
            "side": "right",
            "rgb_color": rgb_color,
            "color": "hotpink",
            "pre_match_x": raw_x,
            "pre_match_y": raw_y,
            "pre_match_z": 0.0,
        }

        centroid_data_list.append(new_car)

        print(
            f"INSERT: New car ID {new_id} "
            f"at ({raw_x:.2f}, {raw_y:.2f}) - right/parallel"
        )

        update_plot_data()

    def on_release(event):
        nonlocal dragging_idx

        if dragging_idx is not None and current_mode == "MOVE":
            if event.xdata is not None and event.ydata is not None:
                raw_x, raw_y = get_inverse_coords(
                    event.xdata,
                    event.ydata,
                    init_sx,
                    init_sy,
                    init_yaw,
                )

                centroid_data_list[dragging_idx]["x"] = raw_x
                centroid_data_list[dragging_idx]["y"] = raw_y

                print(
                    f"MOVED: ID {centroid_data_list[dragging_idx]['id']} "
                    f"to ({raw_x:.2f}, {raw_y:.2f})"
                )

                update_plot_data()

            dragging_idx = None

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # -------------------------------------------------------------------------
    # GUI buttons
    # -------------------------------------------------------------------------
    ax_btn_del = plt.axes([0.03, 0.05, 0.12, 0.05])
    ax_btn_rot = plt.axes([0.16, 0.05, 0.12, 0.05])
    ax_btn_swi = plt.axes([0.29, 0.05, 0.12, 0.05])
    ax_btn_mov = plt.axes([0.42, 0.05, 0.12, 0.05])
    ax_btn_ins = plt.axes([0.55, 0.05, 0.12, 0.05])
    ax_btn_sav = plt.axes([0.75, 0.05, 0.20, 0.05])

    btn_del = Button(ax_btn_del, "DELETE")
    btn_rot = Button(ax_btn_rot, "ROTATE")
    btn_swi = Button(ax_btn_swi, "SIDE")
    btn_mov = Button(ax_btn_mov, "MOVE")
    btn_ins = Button(ax_btn_ins, "INSERT")
    btn_sav = Button(ax_btn_sav, "Save Filtered")

    active_color = "gold"
    inactive_color = "0.95"

    def set_mode(mode):
        nonlocal current_mode, dragging_idx

        current_mode = mode
        dragging_idx = None

        btn_del.color = active_color if mode == "DELETE" else inactive_color
        btn_rot.color = active_color if mode == "ROTATE" else inactive_color
        btn_swi.color = active_color if mode == "SWITCH_SIDE" else inactive_color
        btn_mov.color = active_color if mode == "MOVE" else inactive_color
        btn_ins.color = active_color if mode == "INSERT" else inactive_color

        btn_del.ax.set_facecolor(btn_del.color)
        btn_rot.ax.set_facecolor(btn_rot.color)
        btn_swi.ax.set_facecolor(btn_swi.color)
        btn_mov.ax.set_facecolor(btn_mov.color)
        btn_ins.ax.set_facecolor(btn_ins.color)

        fig.canvas.draw_idle()

        mode_hints = {
            "DELETE": "Click points to delete",
            "ROTATE": "Click points to toggle parallel/perpendicular",
            "SWITCH_SIDE": "Click points to toggle left/right",
            "MOVE": "Click and drag to move points",
            "INSERT": "Click on map to add new car, default right/parallel",
        }

        ax.set_title(f"MODE: {mode} - {mode_hints.get(mode, '')}")

    def cb_del(event):
        set_mode("DELETE")

    def cb_rot(event):
        set_mode("ROTATE")

    def cb_swi(event):
        set_mode("SWITCH_SIDE")

    def cb_mov(event):
        set_mode("MOVE")

    def cb_ins(event):
        set_mode("INSERT")

    def cb_save(event):
        print(f"Saving to {output_filtered_file}")

        try:
            save_filtered_centroids(
                output_filtered_file,
                centroid_data_list,
            )

            btn_sav.label.set_text("Saved")
            fig.canvas.draw_idle()

        except Exception as exc:
            print(f"[ERROR] Error saving filtered file: {exc}")

    btn_del.on_clicked(cb_del)
    btn_rot.on_clicked(cb_rot)
    btn_swi.on_clicked(cb_swi)
    btn_mov.on_clicked(cb_mov)
    btn_ins.on_clicked(cb_ins)
    btn_sav.on_clicked(cb_save)

    set_mode("DELETE")

    custom_legend = [
        Line2D([0], [0], color="limegreen", lw=3, label="Left"),
        Line2D([0], [0], color="hotpink", lw=3, label="Right"),
        Line2D([0], [0], color="black", lw=1, label="Orientation"),
    ]

    if args.match and pre_match_camera_list:
        custom_legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="dodgerblue",
                markerfacecolor="none",
                markersize=9,
                linestyle="None",
                label="Pre-match camera",
            )
        )

    if args.match and lidar_gt_list:
        custom_legend.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="gold",
                markersize=10,
                markeredgecolor="black",
                label="LiDAR GT",
            )
        )

        custom_legend.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="dotted",
                lw=1,
                label="Accepted match",
            )
        )

    ax.legend(handles=custom_legend, loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()