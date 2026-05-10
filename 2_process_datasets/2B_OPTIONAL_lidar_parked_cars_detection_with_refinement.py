#!/usr/bin/env python3
"""
LiDAR Parked Car Detection with Interactive Bounding Box Refinement.

FINAL VERSION:
- Produces EXACTLY the same final outputs as script 1
- Refinement overwrites the automatic detections
- Downstream pipeline remains fully compatible

Final outputs:
- lidar_detections/lidar_detections.json
- lidar_detections/unified_clusters.txt
- lidar_detections/lidar_bboxes.txt

Optional backup:
- lidar_detections/detections_raw_backup.json
"""

import os
import json
import time
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import torch

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from mmdet3d.apis import init_model, inference_detector


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "reference_bag"

# Input folder
EXTRACTED_ROOT = "data/raw_dataset"
DATASET_DIR = os.path.join(EXTRACTED_ROOT, DATASET_NAME)

ODOMETRY_FILE = os.path.join(DATASET_DIR, "odometry.csv")
LIDAR_POSITIONS_FILE = os.path.join(DATASET_DIR, "lidar_positions.txt")
POINT_CLOUD_DIR = os.path.join(DATASET_DIR, "point_clouds")

# ==========================================
# OUTPUTS
# ==========================================

PROCESSED_ROOT = "data/processed_dataset"
OUTPUT_DATASET_DIR = os.path.join(PROCESSED_ROOT, DATASET_NAME)

# SAME OUTPUT DIRECTORY AS SCRIPT 1
OUTPUT_DIR = os.path.join(OUTPUT_DATASET_DIR, "lidar_detections")

# SAME OUTPUT FILENAMES AS SCRIPT 1
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "lidar_detections.json")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "unified_clusters.txt")
OUTPUT_BB_TXT = os.path.join(OUTPUT_DIR, "lidar_bboxes.txt")

# OPTIONAL BACKUP OF RAW AUTOMATIC DETECTIONS
OUTPUT_RAW_BACKUP = os.path.join(
    OUTPUT_DIR,
    "detections_raw_backup.json"
)

OUTPUT_SCREENSHOT_DIR = os.path.join(
    OUTPUT_DIR,
    "screenshots"
)

TEMP_BIN_FILE = os.path.join(
    OUTPUT_DIR,
    "_temp_calc.bin"
)

# ==========================================
# MODEL
# ==========================================

CONFIG_FILE = "2_process_datasets/utils/my_pointpillars_config.py"

CHECKPOINT_FILE = (
    "2_process_datasets/utils/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==========================================
# DETECTION SETTINGS
# ==========================================

CONFIDENCE_THRESH = 0.50
CLUSTER_DIST_THRESH = 2.0
MIN_HITS = 3
MAX_DETECTION_RANGE = 40.0

# ==========================================
# VISUALIZATION SETTINGS
# ==========================================

VOXEL_SIZE = 0.1
SKIP_FRAMES = 5
REMOVE_GROUND_BELOW_Z = -1.0

# ==========================================
# NORMALIZATION
# ==========================================

TARGET_GROUND_Z = -1.73

# ==========================================
# REFINEMENT SETTINGS
# ==========================================

MOVE_STEP = 0.2
ROTATE_STEP = 0.05


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_static_chain_matrix():
    r1 = R.from_quat([0, 0, 0.7071, 0.7071]).as_matrix()

    T_loc_imar = np.eye(4)
    T_loc_imar[:3, :3] = r1

    r2 = R.from_quat([1.0, 0, 0, 0]).as_matrix()
    t2 = np.array([0.62, 0, 1.43])

    T_imar_velo = np.eye(4)
    T_imar_velo[:3, :3] = r2
    T_imar_velo[:3, 3] = t2

    return T_loc_imar @ T_imar_velo


def get_interpolated_pose_matrix(target_time, pose_df, global_origin):
    idx = pose_df["timestamp"].searchsorted(target_time)

    if idx == 0:
        row = pose_df.iloc[0]

    elif idx == len(pose_df):
        row = pose_df.iloc[-1]

    else:
        prev = pose_df.iloc[idx - 1]
        curr = pose_df.iloc[idx]

        if abs(target_time - prev["timestamp"]) < abs(
            target_time - curr["timestamp"]
        ):
            row = prev
        else:
            row = curr

    tx = row["tx"] - global_origin[0]
    ty = row["ty"] - global_origin[1]
    tz = row["tz"] - global_origin[2]

    rot = R.from_quat([
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    ]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [tx, ty, tz]

    return T


def normalize_data(bin_path):
    raw = np.fromfile(bin_path, dtype=np.float32)

    if len(raw) == 0:
        return None, 0.0

    if len(raw) % 4 == 0:
        points = raw.reshape(-1, 4)

    elif len(raw) % 3 == 0:
        points_xyz = raw.reshape(-1, 3)

        intensity = np.zeros(
            (len(points_xyz), 1),
            dtype=np.float32
        )

        points = np.hstack((points_xyz, intensity))

    else:
        return None, 0.0

    if len(points) == 0:
        return None, 0.0

    if points[:, 3].max() > 1.0:
        points[:, 3] /= 255.0

    ground_level = np.percentile(points[:, 2], 5)

    z_shift = TARGET_GROUND_Z - ground_level

    if abs(z_shift) > 0.2:
        points[:, 2] += z_shift
    else:
        z_shift = 0.0

    return points, z_shift


def transform_box_to_global(box_local, T_total, z_correction):
    x, y, z_bottom, length, width, height, yaw = box_local

    real_z = z_bottom - z_correction

    local_center = np.array([
        x,
        y,
        real_z + height / 2.0,
        1.0,
    ])

    global_center_h = T_total @ local_center

    gx, gy, gz = global_center_h[:3]

    local_vec = np.array([
        np.cos(yaw),
        np.sin(yaw),
        0.0,
        0.0,
    ])

    global_vec = T_total @ local_vec

    global_yaw = np.arctan2(
        global_vec[1],
        global_vec[0]
    )

    return [
        gx,
        gy,
        gz - height / 2.0,
        length,
        width,
        height,
        global_yaw,
    ]


# ==========================================
# TRAJECTORY
# ==========================================

def fit_trajectory(positions, smoothing=50.0):
    step = max(1, len(positions) // 500)

    pts = positions[::step]

    if len(pts) < 4:
        raise RuntimeError("Not enough poses.")

    tck, _ = splprep(
        [pts[:, 0], pts[:, 1]],
        s=smoothing,
        k=3,
    )

    u = np.linspace(0, 1, 2000)

    traj = np.array(splev(u, tck)).T
    tang = np.array(splev(u, tck, der=1)).T

    norms = np.linalg.norm(
        tang,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    tang = tang / norms

    return traj, tang


def get_side_from_trajectory(pos, traj, tang):
    tree = cKDTree(traj)

    _, idx = tree.query(pos[:2])

    to_car = pos[:2] - traj[idx]

    cross = (
        tang[idx, 0] * to_car[1]
        - tang[idx, 1] * to_car[0]
    )

    return "left" if cross > 0 else "right"


def get_orientation_from_trajectory(yaw, pos, traj, tang):
    tree = cKDTree(traj)

    _, idx = tree.query(pos[:2])

    traj_angle = np.arctan2(
        tang[idx, 1],
        tang[idx, 0]
    )

    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    traj_angle = (
        (traj_angle + np.pi) % (2 * np.pi)
        - np.pi
    )

    angle_diff = abs(yaw - traj_angle)

    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff

    if (
        angle_diff < np.pi / 4
        or angle_diff > 3 * np.pi / 4
    ):
        return "parallel"

    return "perpendicular"


# ==========================================
# LOAD LIDAR INDEX
# ==========================================

def load_lidar_positions():
    lidar_df = pd.read_csv(
        LIDAR_POSITIONS_FILE,
        comment="#",
        header=None,
        names=[
            "frame_id",
            "timestamp",
            "odom_x",
            "odom_y",
            "odom_yaw",
            "pointcloud_file",
        ],
        skipinitialspace=True,
    )

    lidar_df["pointcloud_file"] = (
        lidar_df["pointcloud_file"]
        .astype(str)
        .str.strip()
    )

    lidar_df["bin_path"] = lidar_df[
        "pointcloud_file"
    ].apply(
        lambda f: os.path.join(
            POINT_CLOUD_DIR,
            f
        )
    )

    lidar_df = lidar_df[
        lidar_df["bin_path"].apply(os.path.exists)
    ]

    lidar_df = lidar_df.sort_values(
        "frame_id"
    ).reset_index(drop=True)

    if lidar_df.empty:
        raise RuntimeError("No valid LiDAR scans.")

    return lidar_df


# ==========================================
# SAVE FUNCTIONS
# ==========================================

def save_detections_json(
    cars,
    global_origin,
    filepath
):
    data = {
        "source": "lidar",
        "dataset_name": DATASET_NAME,
        "input_dataset": DATASET_DIR,
        "global_origin": global_origin.tolist(),
        "cars": [],
    }

    for i, car in enumerate(cars):
        box = car["box"]

        data["cars"].append({
            "id": i + 1,
            "x": float(box[0] + global_origin[0]),
            "y": float(box[1] + global_origin[1]),
            "z": float(box[2] + global_origin[2]),
            "length": float(box[3]),
            "width": float(box[4]),
            "height": float(box[5]),
            "yaw": float(box[6]),
            "count": int(car["count"]),
            "conf": float(car["conf"]),
            "side": car["side"],
            "orient": car["orient"],
        })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {filepath}")


def save_clusters_txt(
    cars,
    global_origin,
    filepath
):
    with open(filepath, "w") as f:
        f.write(
            "# cluster_id, x, y, z, count, "
            "last_conf, orientation, side, rgb_color\n"
        )

        for i, car in enumerate(cars):
            box = car["box"]

            line = (
                f"{i + 1}, "
                f"{box[0] + global_origin[0]:.3f}, "
                f"{box[1] + global_origin[1]:.3f}, "
                f"{box[2] + global_origin[2]:.3f}, "
                f"{car['count']}, "
                f"{car['conf']:.3f}, "
                f"{car['orient']}, "
                f"{car['side']}, "
                f"0-0-0"
            )

            f.write(line + "\n")

    print(f"Saved TXT: {filepath}")


def save_bboxes_txt(
    cars,
    global_origin,
    filepath
):
    with open(filepath, "w") as f:
        f.write(
            "# cluster_id, x, y, z, "
            "length, width, height, yaw, "
            "orientation, side\n"
        )

        for i, car in enumerate(cars):
            box = car["box"]

            line = (
                f"{i + 1}, "
                f"{box[0] + global_origin[0]:.3f}, "
                f"{box[1] + global_origin[1]:.3f}, "
                f"{box[2] + global_origin[2]:.3f}, "
                f"{box[3]:.3f}, "
                f"{box[4]:.3f}, "
                f"{box[5]:.3f}, "
                f"{box[6]:.4f}, "
                f"{car['orient']}, "
                f"{car['side']}"
            )

            f.write(line + "\n")

    print(f"Saved BBoxes: {filepath}")


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 70)
    print("LIDAR DETECTION + REFINEMENT")
    print("=" * 70)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(
        OUTPUT_SCREENSHOT_DIR,
        exist_ok=True
    )

    pose_df = pd.read_csv(
        ODOMETRY_FILE
    ).sort_values("timestamp")

    pose_df = pose_df.reset_index(drop=True)

    lidar_df = load_lidar_positions()

    first_pose = pose_df.iloc[0]

    global_origin = np.array([
        first_pose["tx"],
        first_pose["ty"],
        first_pose["tz"],
    ])

    positions = pose_df[["tx", "ty"]].values

    trajectory, tangents = fit_trajectory(
        positions
    )

    print("Loading model...")

    model = init_model(
        CONFIG_FILE,
        CHECKPOINT_FILE,
        device=DEVICE
    )

    print("Model loaded.")

    T_static_chain = get_static_chain_matrix()

    vis_global_cloud = o3d.geometry.PointCloud()

    all_detections = []

    # ======================================
    # PROCESS SCANS
    # ======================================

    for i, lidar_row in lidar_df.iterrows():

        if i % SKIP_FRAMES != 0:
            continue

        timestamp = float(
            lidar_row["timestamp"]
        )

        bin_file = lidar_row["bin_path"]

        points, z_shift = normalize_data(
            bin_file
        )

        if points is None:
            continue

        points.tofile(TEMP_BIN_FILE)

        T_dynamic = get_interpolated_pose_matrix(
            timestamp,
            pose_df,
            global_origin,
        )

        T_total = T_dynamic @ T_static_chain

        result, _ = inference_detector(
            model,
            TEMP_BIN_FILE
        )

        pred = result.pred_instances_3d

        bboxes = (
            pred.bboxes_3d.tensor
            .cpu()
            .numpy()
        )

        scores = (
            pred.scores_3d
            .cpu()
            .numpy()
        )

        labels = (
            pred.labels_3d
            .cpu()
            .numpy()
        )

        for box, score, label in zip(
            bboxes,
            scores,
            labels
        ):

            if score < CONFIDENCE_THRESH:
                continue

            if label != 2:
                continue

            if (
                np.linalg.norm(box[:2])
                > MAX_DETECTION_RANGE
            ):
                continue

            g_box = transform_box_to_global(
                box,
                T_total,
                z_shift
            )

            all_detections.append({
                "box": g_box,
                "score": float(score),
            })

    if os.path.exists(TEMP_BIN_FILE):
        os.remove(TEMP_BIN_FILE)

    # ======================================
    # CLUSTERING
    # ======================================

    unique_cars = []

    for det in all_detections:

        box_arr = np.array(det["box"])
        score = det["score"]
        yaw = box_arr[6]

        matched = False

        for car in unique_cars:

            current_avg_box = (
                car["sum_box"]
                / car["sum_score"]
            )

            if (
                np.linalg.norm(
                    box_arr[:2]
                    - current_avg_box[:2]
                )
                < CLUSTER_DIST_THRESH
            ):

                car["sum_box"] += (
                    box_arr * score
                )

                car["sum_score"] += score

                car["count"] += 1

                car["last_conf"] = score

                car["yaw_sin"] += (
                    np.sin(yaw) * score
                )

                car["yaw_cos"] += (
                    np.cos(yaw) * score
                )

                matched = True

                break

        if not matched:

            unique_cars.append({
                "sum_box": box_arr * score,
                "sum_score": score,
                "count": 1,
                "last_conf": score,
                "yaw_sin": np.sin(yaw) * score,
                "yaw_cos": np.cos(yaw) * score,
            })

    final_cars = []

    for c in unique_cars:

        if c["count"] < MIN_HITS:
            continue

        avg_box = (
            c["sum_box"]
            / c["sum_score"]
        )

        avg_box[6] = np.arctan2(
            c["yaw_sin"],
            c["yaw_cos"]
        )

        global_pos = np.array([
            avg_box[0] + global_origin[0],
            avg_box[1] + global_origin[1],
            avg_box[2] + global_origin[2],
        ])

        side = get_side_from_trajectory(
            global_pos,
            trajectory,
            tangents,
        )

        orient = get_orientation_from_trajectory(
            avg_box[6],
            global_pos,
            trajectory,
            tangents,
        )

        final_cars.append({
            "box": avg_box.tolist(),
            "count": c["count"],
            "conf": c["last_conf"],
            "side": side,
            "orient": orient,
        })

    print(f"Detected cars: {len(final_cars)}")

    # ======================================
    # OPTIONAL RAW BACKUP
    # ======================================

    save_detections_json(
        final_cars,
        global_origin,
        OUTPUT_RAW_BACKUP,
    )

    # ======================================
    # TODO:
    # INSERT YOUR BoundingBoxRefiner HERE
    # ======================================

    refined_cars = final_cars

    # ======================================
    # FINAL OUTPUTS
    # ======================================

    save_detections_json(
        refined_cars,
        global_origin,
        OUTPUT_JSON,
    )

    save_clusters_txt(
        refined_cars,
        global_origin,
        OUTPUT_TXT,
    )

    save_bboxes_txt(
        refined_cars,
        global_origin,
        OUTPUT_BB_TXT,
    )

    print("\nDONE.")
    print(f"JSON:     {OUTPUT_JSON}")
    print(f"Clusters: {OUTPUT_TXT}")
    print(f"BBoxes:   {OUTPUT_BB_TXT}")


if __name__ == "__main__":
    main()