#!/usr/bin/env python3
"""
LiDAR Parked Car Detection with Interactive Bounding Box Refinement.

Features:
- Detects parked cars using PointPillars
- Reads LiDAR timestamps from lidar_positions.txt
- Reads odometry from odometry.csv
- Saves RAW detections before manual editing
- Interactive tool to move/rotate/delete/insert bounding boxes
- Saves REFINED ground truth after editing

Input:
- data/raw_dataset/<dataset_name>/odometry.csv
- data/raw_dataset/<dataset_name>/lidar_positions.txt
- data/raw_dataset/<dataset_name>/point_clouds/*.bin

Output:
- data/processed_dataset/<dataset_name>/lidar_refinement/detections_raw.json
- data/processed_dataset/<dataset_name>/lidar_refinement/ground_truth_refined.json
- data/processed_dataset/<dataset_name>/lidar_refinement/final_clusters.txt
- data/processed_dataset/<dataset_name>/lidar_refinement/ground_truth_bboxes.txt
- data/processed_dataset/<dataset_name>/lidar_refinement/screenshots/

Controls in refinement mode:
- LEFT/RIGHT: Select nearest box spatially in that direction
- C: Select box closest to camera
- W/S: Move selected box forward/backward
- A/D: Move selected box left/right
- Q/E: Rotate selected box
- R/F: Move selected box up/down
- X: Delete selected box
- I: Insert new box at current selected box position
- U: Update side/orientation for selected box
- P: Save screenshot
- SPACE: Confirm and save
- ESC: Cancel without saving
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

# Input folder from step 1
EXTRACTED_ROOT = "data/raw_dataset"
DATASET_DIR = os.path.join(EXTRACTED_ROOT, DATASET_NAME)

ODOMETRY_FILE = os.path.join(DATASET_DIR, "odometry.csv")
LIDAR_POSITIONS_FILE = os.path.join(DATASET_DIR, "lidar_positions.txt")
POINT_CLOUD_DIR = os.path.join(DATASET_DIR, "point_clouds")

# Output folder
GENERATED_ROOT = "data/processed_dataset"
OUTPUT_DATASET_DIR = os.path.join(GENERATED_ROOT, DATASET_NAME)
OUTPUT_DIR = os.path.join(OUTPUT_DATASET_DIR, "lidar_detections")

OUTPUT_REFINED = os.path.join(OUTPUT_DIR, "lidar_detections.json")
OUTPUT_CLUSTERS_TXT = os.path.join(OUTPUT_DIR, "unified_clusters.txt")
OUTPUT_BB_TXT = os.path.join(OUTPUT_DIR, "lidar_bboxes.txt")
OUTPUT_SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
TEMP_BIN_FILE = os.path.join(OUTPUT_DIR, "_temp_calc.bin")

# Model files
CONFIG_FILE = "2_process_datasets/utils/my_pointpillars_config.py"
CHECKPOINT_FILE = (
    "2_process_datasets/utils/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Detection settings
CONFIDENCE_THRESH = 0.50
CLUSTER_DIST_THRESH = 2.0
MIN_HITS = 3
MAX_DETECTION_RANGE = 40.0

# Visualization settings
VOXEL_SIZE = 0.1
SKIP_FRAMES = 5
REMOVE_GROUND_BELOW_Z = -1.0
BOX_LINE_RADIUS = 0.05

# Normalization
TARGET_GROUND_Z = -1.73

# Refinement settings
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

        if abs(target_time - prev["timestamp"]) < abs(target_time - curr["timestamp"]):
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
        intensity = np.zeros((len(points_xyz), 1), dtype=np.float32)
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
    global_yaw = np.arctan2(global_vec[1], global_vec[0])

    return [
        gx,
        gy,
        gz - height / 2.0,
        length,
        width,
        height,
        global_yaw,
    ]


def create_box_lineset(center, length, width, height, yaw, color):
    x = length / 2.0
    y = width / 2.0
    z = height / 2.0

    corners = np.array([
        [-x, -y, -z],
        [x, -y, -z],
        [x, y, -z],
        [-x, y, -z],
        [-x, -y, z],
        [x, -y, z],
        [x, y, z],
        [-x, y, z],
    ])

    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    corners = corners @ rot_mat.T + center

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color(color)

    return line_set


# ==========================================
# TRAJECTORY-BASED SIDE / ORIENTATION
# ==========================================

def fit_trajectory(positions, smoothing=50.0):
    step = max(1, len(positions) // 500)
    pts = positions[::step]

    if len(pts) < 4:
        raise RuntimeError("Not enough pose samples to fit trajectory spline.")

    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, k=3)

    u = np.linspace(0, 1, 2000)

    traj = np.array(splev(u, tck)).T
    tang = np.array(splev(u, tck, der=1)).T

    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tang = tang / norms

    return traj, tang


def get_side_from_trajectory(pos, traj, tang):
    tree = cKDTree(traj)
    _, idx = tree.query(pos[:2])

    to_car = pos[:2] - traj[idx]
    cross = tang[idx, 0] * to_car[1] - tang[idx, 1] * to_car[0]

    return "left" if cross > 0 else "right"


def get_orientation_from_trajectory(car_yaw, pos, traj, tang):
    tree = cKDTree(traj)
    _, idx = tree.query(pos[:2])

    traj_angle = np.arctan2(tang[idx, 1], tang[idx, 0])

    car_yaw = (car_yaw + np.pi) % (2 * np.pi) - np.pi
    traj_angle = (traj_angle + np.pi) % (2 * np.pi) - np.pi

    angle_diff = abs(car_yaw - traj_angle)

    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff

    if angle_diff < np.pi / 4 or angle_diff > 3 * np.pi / 4:
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

    lidar_df["pointcloud_file"] = lidar_df["pointcloud_file"].astype(str).str.strip()

    lidar_df["bin_path"] = lidar_df["pointcloud_file"].apply(
        lambda filename: os.path.join(POINT_CLOUD_DIR, filename)
    )

    missing = lidar_df[~lidar_df["bin_path"].apply(os.path.exists)]

    if len(missing) > 0:
        print(
            f"WARNING: {len(missing)} point cloud files listed in "
            f"lidar_positions.txt were not found."
        )
        lidar_df = lidar_df[lidar_df["bin_path"].apply(os.path.exists)]

    lidar_df = lidar_df.sort_values("frame_id").reset_index(drop=True)

    if lidar_df.empty:
        raise RuntimeError("No valid LiDAR point clouds found.")

    return lidar_df


# ==========================================
# SAVE FUNCTIONS
# ==========================================

def save_detections_json(cars, global_origin, filepath, kind):
    data = {
        "source": "lidar",
        "kind": kind,
        "dataset_name": DATASET_NAME,
        "input_dataset": DATASET_DIR,
        "global_origin": global_origin.tolist(),
        "cars": [],
    }

    for i, car in enumerate(cars):
        box = car["box"]

        if hasattr(box, "tolist"):
            box = box.tolist()
        else:
            box = [float(v) for v in box]

        data["cars"].append({
            "id": i + 1,

            # Relative-to-origin box, useful for editing and visualization
            "box": [float(v) for v in box],

            # Global coordinates, useful for evaluation and inspection
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

    print(f"  Saved {len(cars)} detections to {filepath}")


def save_clusters_txt(cars, global_origin, filepath):
    with open(filepath, "w") as f:
        f.write("# cluster_id, x, y, z, count, last_conf, orientation, side, rgb_color\n")

        for i, car in enumerate(cars):
            box = car["box"]

            final_x = box[0] + global_origin[0]
            final_y = box[1] + global_origin[1]
            final_z = box[2] + global_origin[2]

            line = (
                f"{i + 1}, "
                f"{final_x:.3f}, "
                f"{final_y:.3f}, "
                f"{final_z:.3f}, "
                f"{car['count']}, "
                f"{car['conf']:.3f}, "
                f"{car['orient']}, "
                f"{car['side']}, "
                f"0-0-0"
            )

            f.write(line + "\n")

    print(f"  Saved {len(cars)} clusters to {filepath}")


def save_bboxes_txt(cars, global_origin, filepath):
    with open(filepath, "w") as f:
        f.write("# cluster_id, x, y, z, length, width, height, yaw, orientation, side\n")

        for i, car in enumerate(cars):
            box = car["box"]

            final_x = box[0] + global_origin[0]
            final_y = box[1] + global_origin[1]
            final_z = box[2] + global_origin[2]

            length, width, height, yaw = box[3], box[4], box[5], box[6]

            line = (
                f"{i + 1}, "
                f"{final_x:.3f}, "
                f"{final_y:.3f}, "
                f"{final_z:.3f}, "
                f"{length:.3f}, "
                f"{width:.3f}, "
                f"{height:.3f}, "
                f"{yaw:.4f}, "
                f"{car['orient']}, "
                f"{car['side']}"
            )

            f.write(line + "\n")

    print(f"  Saved {len(cars)} bounding boxes to {filepath}")


# ==========================================
# INTERACTIVE REFINEMENT TOOL
# ==========================================

class BoundingBoxRefiner:
    def __init__(
        self,
        point_cloud,
        cars,
        global_origin,
        trajectory=None,
        tangents=None,
        screenshot_dir=None,
    ):
        self.point_cloud = point_cloud
        self.cars = cars
        self.global_origin = global_origin
        self.selected_idx = 0 if len(cars) > 0 else -1
        self.cancelled = False

        self.trajectory = trajectory
        self.tangents = tangents

        self.screenshot_dir = screenshot_dir or "."
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.vis = None
        self.box_linesets = []
        self.sphere_markers = []
        self.centroids = []

        self._last_delete_time = 0
        self._last_nav_time = 0
        self._last_insert_time = 0

        self.screenshot_counter = 0

    def run(self):
        print("\n" + "=" * 60)
        print("INTERACTIVE BOUNDING BOX REFINEMENT")
        print("=" * 60)
        print("Controls:")
        print("  LEFT/RIGHT: Select nearest box in that direction")
        print("  C: Select box closest to camera")
        print("  W/S: Move forward/backward")
        print("  A/D: Move left/right")
        print("  R/F: Move up/down")
        print("  Q/E: Rotate")
        print("  X: Delete selected box")
        print("  I: Insert new box at current selected box position")
        print("  U: Update side/orientation for selected box")
        print("  P: Save screenshot")
        print("  SPACE: Confirm and save")
        print("  ESC: Cancel")
        print("=" * 60)

        if self.selected_idx >= 0:
            print(f"  Currently selected: Box {self.selected_idx + 1}")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="Refine LiDAR Bounding Boxes - Press P for screenshot",
            width=1280,
            height=720,
        )

        self.vis.register_key_callback(ord("W"), self._move_forward)
        self.vis.register_key_callback(ord("S"), self._move_backward)
        self.vis.register_key_callback(ord("A"), self._move_left)
        self.vis.register_key_callback(ord("D"), self._move_right)
        self.vis.register_key_callback(ord("R"), self._move_up)
        self.vis.register_key_callback(ord("F"), self._move_down)
        self.vis.register_key_callback(ord("Q"), self._rotate_left)
        self.vis.register_key_callback(ord("E"), self._rotate_right)

        self.vis.register_key_callback(256, self._cancel)
        self.vis.register_key_callback(32, self._confirm)

        self.vis.register_key_callback(ord("X"), self._delete_selected)
        self.vis.register_key_callback(ord("I"), self._insert_new_box)
        self.vis.register_key_callback(ord("C"), self._select_closest_to_view)
        self.vis.register_key_callback(ord("U"), self._update_side_orient)
        self.vis.register_key_callback(ord("P"), self._screenshot)

        self.vis.register_key_callback(262, self._select_spatial_right)
        self.vis.register_key_callback(263, self._select_spatial_left)

        self.vis.add_geometry(self.point_cloud)
        self._refresh_boxes()

        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])

        self.vis.run()
        self.vis.destroy_window()

        if self.cancelled:
            return None

        return self.cars

    def _screenshot(self, vis):
        self.screenshot_counter += 1

        filename = os.path.join(
            self.screenshot_dir,
            f"refinement_screenshot_{self.screenshot_counter:03d}.png",
        )

        vis.capture_screen_image(filename, do_render=True)

        print(f"\n   Screenshot saved: {filename}")

        return False

    def _get_box_centroid(self, idx):
        box = self.cars[idx]["box"]
        return np.array([box[0], box[1], box[2] + box[5] / 2.0])

    def _get_sorted_indices(self):
        centroids = [(i, self._get_box_centroid(i)) for i in range(len(self.cars))]
        centroids.sort(key=lambda x: x[1][0])
        return [c[0] for c in centroids]

    def _select_spatial_right(self, vis):
        current_time = time.time()

        if current_time - self._last_nav_time < 0.40:
            return False

        self._last_nav_time = current_time

        if len(self.cars) <= 1:
            return False

        sorted_indices = self._get_sorted_indices()
        current_sorted_pos = sorted_indices.index(self.selected_idx)

        next_sorted_pos = (current_sorted_pos + 1) % len(sorted_indices)
        self.selected_idx = sorted_indices[next_sorted_pos]

        print(f"Selected box {self.selected_idx + 1}/{len(self.cars)}")

        self._refresh_boxes()
        self._focus_on_selected()

        return False

    def _select_spatial_left(self, vis):
        current_time = time.time()

        if current_time - self._last_nav_time < 0.40:
            return False

        self._last_nav_time = current_time

        if len(self.cars) <= 1:
            return False

        sorted_indices = self._get_sorted_indices()
        current_sorted_pos = sorted_indices.index(self.selected_idx)

        prev_sorted_pos = (current_sorted_pos - 1) % len(sorted_indices)
        self.selected_idx = sorted_indices[prev_sorted_pos]

        print(f"Selected box {self.selected_idx + 1}/{len(self.cars)}")

        self._refresh_boxes()
        self._focus_on_selected()

        return False

    def _focus_on_selected(self):
        if self.selected_idx >= 0 and self.selected_idx < len(self.cars):
            box = self.cars[self.selected_idx]["box"]
            center = np.array([box[0], box[1], box[2] + box[5] / 2.0])
            ctr = self.vis.get_view_control()
            ctr.set_lookat(center)

    def _select_closest_to_view(self, vis):
        if len(self.cars) == 0:
            print("No boxes to select.")
            return False

        ctr = self.vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_pos = np.array(cam_params.extrinsic)[:3, 3]

        min_dist = float("inf")
        closest_idx = 0

        for i, car in enumerate(self.cars):
            box = car["box"]
            centroid = np.array([box[0], box[1], box[2] + box[5] / 2.0])
            dist = np.linalg.norm(centroid - cam_pos)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.selected_idx = closest_idx

        print(f"Selected nearest box: {self.selected_idx + 1}/{len(self.cars)}")

        self._refresh_boxes()

        return False

    def _refresh_boxes(self):
        for ls in self.box_linesets:
            self.vis.remove_geometry(ls, reset_bounding_box=False)

        for sp in self.sphere_markers:
            self.vis.remove_geometry(sp, reset_bounding_box=False)

        self.box_linesets = []
        self.sphere_markers = []
        self.centroids = []

        for i, car in enumerate(self.cars):
            box = car["box"]
            x, y, z, length, width, height, yaw = box
            center = [x, y, z + height / 2.0]

            self.centroids.append(center)

            if i == self.selected_idx:
                box_color = [0, 1, 0]
                sphere_color = [0, 1, 0]
                sphere_radius = 0.4
            else:
                box_color = [1, 1, 0]
                sphere_color = [1, 0.5, 0]
                sphere_radius = 0.25

            line_set = create_box_lineset(
                center,
                length,
                width,
                height,
                yaw,
                box_color,
            )

            self.vis.add_geometry(line_set, reset_bounding_box=False)
            self.box_linesets.append(line_set)

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(center)
            sphere.paint_uniform_color(sphere_color)

            self.vis.add_geometry(sphere, reset_bounding_box=False)
            self.sphere_markers.append(sphere)

        self.vis.poll_events()
        self.vis.update_renderer()

    def _move_forward(self, vis):
        if self.selected_idx >= 0:
            box = self.cars[self.selected_idx]["box"]
            yaw = box[6]
            box[0] += MOVE_STEP * np.cos(yaw)
            box[1] += MOVE_STEP * np.sin(yaw)
            self._refresh_boxes()

        return False

    def _move_backward(self, vis):
        if self.selected_idx >= 0:
            box = self.cars[self.selected_idx]["box"]
            yaw = box[6]
            box[0] -= MOVE_STEP * np.cos(yaw)
            box[1] -= MOVE_STEP * np.sin(yaw)
            self._refresh_boxes()

        return False

    def _move_left(self, vis):
        if self.selected_idx >= 0:
            box = self.cars[self.selected_idx]["box"]
            yaw = box[6]
            box[0] -= MOVE_STEP * np.sin(yaw)
            box[1] += MOVE_STEP * np.cos(yaw)
            self._refresh_boxes()

        return False

    def _move_right(self, vis):
        if self.selected_idx >= 0:
            box = self.cars[self.selected_idx]["box"]
            yaw = box[6]
            box[0] += MOVE_STEP * np.sin(yaw)
            box[1] -= MOVE_STEP * np.cos(yaw)
            self._refresh_boxes()

        return False

    def _move_up(self, vis):
        if self.selected_idx >= 0:
            self.cars[self.selected_idx]["box"][2] += MOVE_STEP
            self._refresh_boxes()

        return False

    def _move_down(self, vis):
        if self.selected_idx >= 0:
            self.cars[self.selected_idx]["box"][2] -= MOVE_STEP
            self._refresh_boxes()

        return False

    def _rotate_left(self, vis):
        if self.selected_idx >= 0:
            self.cars[self.selected_idx]["box"][6] += ROTATE_STEP
            self._refresh_boxes()

        return False

    def _rotate_right(self, vis):
        if self.selected_idx >= 0:
            self.cars[self.selected_idx]["box"][6] -= ROTATE_STEP
            self._refresh_boxes()

        return False

    def _delete_selected(self, vis):
        current_time = time.time()

        if current_time - self._last_delete_time < 0.3:
            return False

        self._last_delete_time = current_time

        if self.selected_idx >= 0 and len(self.cars) > 0:
            print(f"Deleted box {self.selected_idx + 1}")

            del self.cars[self.selected_idx]

            if len(self.cars) == 0:
                self.selected_idx = -1
            elif self.selected_idx >= len(self.cars):
                self.selected_idx = len(self.cars) - 1

            self._refresh_boxes()

            if self.selected_idx >= 0:
                self._focus_on_selected()

            print(f"Remaining: {len(self.cars)} boxes")

        return False

    def _insert_new_box(self, vis):
        current_time = time.time()

        if current_time - self._last_insert_time < 0.3:
            return False

        self._last_insert_time = current_time

        std_length = 4.5
        std_width = 1.8
        std_height = 1.5

        if self.selected_idx >= 0:
            current_box = self.cars[self.selected_idx]["box"]
            new_x = current_box[0]
            new_y = current_box[1]
            new_z = current_box[2]
            new_yaw = current_box[6]
        else:
            new_x, new_y, new_z, new_yaw = 0.0, 0.0, 0.0, 0.0

        new_box = [
            new_x,
            new_y,
            new_z,
            std_length,
            std_width,
            std_height,
            new_yaw,
        ]

        if self.trajectory is not None and self.tangents is not None:
            global_pos = np.array([
                new_x + self.global_origin[0],
                new_y + self.global_origin[1],
                new_z + self.global_origin[2],
            ])

            side = get_side_from_trajectory(
                global_pos,
                self.trajectory,
                self.tangents,
            )

            orient = get_orientation_from_trajectory(
                new_yaw,
                global_pos,
                self.trajectory,
                self.tangents,
            )
        else:
            side = "unknown"
            orient = "unknown"

        new_car = {
            "box": new_box,
            "count": 1,
            "conf": 1.0,
            "side": side,
            "orient": orient,
        }

        self.cars.append(new_car)
        self.selected_idx = len(self.cars) - 1

        print(
            f"Inserted new box {self.selected_idx + 1} "
            f"(side={side}, orient={orient}, total={len(self.cars)})"
        )

        self._refresh_boxes()
        self._focus_on_selected()

        return False

    def _update_side_orient(self, vis):
        if self.selected_idx < 0 or self.selected_idx >= len(self.cars):
            print("No box selected.")
            return False

        if self.trajectory is None or self.tangents is None:
            print("No trajectory available.")
            return False

        car = self.cars[self.selected_idx]
        box = car["box"]

        global_pos = np.array([
            box[0] + self.global_origin[0],
            box[1] + self.global_origin[1],
            box[2] + self.global_origin[2],
        ])

        yaw = box[6]

        car["side"] = get_side_from_trajectory(
            global_pos,
            self.trajectory,
            self.tangents,
        )

        car["orient"] = get_orientation_from_trajectory(
            yaw,
            global_pos,
            self.trajectory,
            self.tangents,
        )

        print(
            f"Box {self.selected_idx + 1}: "
            f"side={car['side']}, orient={car['orient']}"
        )

        return False

    def _update_all_side_orient(self):
        if self.trajectory is None or self.tangents is None:
            print("No trajectory available.")
            return

        for car in self.cars:
            box = car["box"]

            global_pos = np.array([
                box[0] + self.global_origin[0],
                box[1] + self.global_origin[1],
                box[2] + self.global_origin[2],
            ])

            yaw = box[6]

            car["side"] = get_side_from_trajectory(
                global_pos,
                self.trajectory,
                self.tangents,
            )

            car["orient"] = get_orientation_from_trajectory(
                yaw,
                global_pos,
                self.trajectory,
                self.tangents,
            )

        print(f"Updated side/orientation for all {len(self.cars)} boxes.")

    def _confirm(self, vis):
        print("\nConfirmed. Updating side/orientation and saving refined boxes...")
        self._update_all_side_orient()
        vis.close()

        return False

    def _cancel(self, vis):
        print("\nCancelled. Discarding changes...")
        self.cancelled = True
        vis.close()

        return False


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 70)
    print("LIDAR PARKED CAR DETECTION WITH REFINEMENT")
    print("=" * 70)

    print(f"Dataset name: {DATASET_NAME}")
    print(f"Input dataset: {DATASET_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    if not os.path.exists(ODOMETRY_FILE):
        raise FileNotFoundError(f"Odometry file not found: {ODOMETRY_FILE}")

    if not os.path.exists(LIDAR_POSITIONS_FILE):
        raise FileNotFoundError(f"LiDAR positions file not found: {LIDAR_POSITIONS_FILE}")

    if not os.path.exists(POINT_CLOUD_DIR):
        raise FileNotFoundError(f"Point cloud folder not found: {POINT_CLOUD_DIR}")

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"PointPillars config not found: {CONFIG_FILE}")

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"PointPillars checkpoint not found: {CHECKPOINT_FILE}")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_SCREENSHOT_DIR, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Loading data...")

    pose_df = pd.read_csv(ODOMETRY_FILE).sort_values("timestamp").reset_index(drop=True)
    lidar_df = load_lidar_positions()

    if pose_df.empty:
        raise RuntimeError("Odometry file is empty.")

    first_pose = pose_df.iloc[0]

    global_origin = np.array([
        first_pose["tx"],
        first_pose["ty"],
        first_pose["tz"],
    ])

    print(f"  Global origin: {global_origin}")
    print(f"  Odometry poses: {len(pose_df)}")
    print(f"  LiDAR scans: {len(lidar_df)}")

    T_static_chain = get_static_chain_matrix()

    print("  Loading PointPillars model...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    print("  Model loaded.")

    print("  Fitting vehicle trajectory...")
    positions = pose_df[["tx", "ty"]].values
    trajectory, tangents = fit_trajectory(positions)
    print(f"  Trajectory points: {len(trajectory)}")

    # 2. Process scans
    print(f"\n[2/5] Processing scans, skip={SKIP_FRAMES}...")

    vis_global_cloud = o3d.geometry.PointCloud()
    all_detections = []

    for i, lidar_row in lidar_df.iterrows():
        if i % SKIP_FRAMES != 0:
            continue

        frame_id = int(lidar_row["frame_id"])
        timestamp = float(lidar_row["timestamp"])
        bin_file = lidar_row["bin_path"]

        points, z_shift = normalize_data(bin_file)

        if points is None:
            continue

        points.tofile(TEMP_BIN_FILE)

        T_dynamic = get_interpolated_pose_matrix(
            timestamp,
            pose_df,
            global_origin,
        )

        T_total = T_dynamic @ T_static_chain

        result, _ = inference_detector(model, TEMP_BIN_FILE)

        pred = result.pred_instances_3d
        bboxes = pred.bboxes_3d.tensor.cpu().numpy()
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()

        for box, score, label in zip(bboxes, scores, labels):
            if score < CONFIDENCE_THRESH:
                continue

            # KITTI 3-class PointPillars usually uses:
            # 0 = Pedestrian, 1 = Cyclist, 2 = Car
            if label != 2:
                continue

            if np.linalg.norm(box[:2]) > MAX_DETECTION_RANGE:
                continue

            g_box = transform_box_to_global(box, T_total, z_shift)

            all_detections.append({
                "box": g_box,
                "score": float(score),
            })

        # Build visualization cloud
        points_sensor = points[:, :3].copy()
        points_sensor[:, 2] -= z_shift

        mask = points_sensor[:, 2] > REMOVE_GROUND_BELOW_Z
        points_filtered = points_sensor[mask]

        if len(points_filtered) > 0:
            z_vals = points_filtered[:, 2]

            norm_z = np.clip((z_vals + 1.7) / 2.0, 0, 1)

            colors = np.zeros((len(z_vals), 3))
            colors[:, 0] = norm_z
            colors[:, 2] = 1 - norm_z

            R_total = T_total[:3, :3]
            t_total = T_total[:3, 3]

            points_world = points_filtered @ R_total.T + t_total

            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(points_world)
            pcd_frame.colors = o3d.utility.Vector3dVector(colors)

            pcd_frame = pcd_frame.voxel_down_sample(voxel_size=VOXEL_SIZE)
            vis_global_cloud += pcd_frame

            if i % (SKIP_FRAMES * 10) == 0:
                vis_global_cloud = vis_global_cloud.voxel_down_sample(
                    voxel_size=VOXEL_SIZE
                )
                print(
                    f"  Frame {frame_id}/{len(lidar_df)}... "
                    f"detections so far: {len(all_detections)}"
                )

    if os.path.exists(TEMP_BIN_FILE):
        os.remove(TEMP_BIN_FILE)

    print(f"  Total detections: {len(all_detections)}")

    # 3. Clustering
    print("\n[3/5] Clustering detections...")

    unique_cars = []

    for det in all_detections:
        box_arr = np.array(det["box"])
        score = det["score"]
        yaw = box_arr[6]
        matched = False

        for car in unique_cars:
            current_avg_box = car["sum_box"] / car["sum_score"]

            if np.linalg.norm(box_arr[:2] - current_avg_box[:2]) < CLUSTER_DIST_THRESH:
                car["sum_box"] += box_arr * score
                car["sum_score"] += score
                car["count"] += 1
                car["last_conf"] = score
                car["yaw_sin"] += np.sin(yaw) * score
                car["yaw_cos"] += np.cos(yaw) * score
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

        avg_box = c["sum_box"] / c["sum_score"]
        avg_box[6] = np.arctan2(c["yaw_sin"], c["yaw_cos"])

        global_pos = np.array([
            avg_box[0] + global_origin[0],
            avg_box[1] + global_origin[1],
            avg_box[2] + global_origin[2],
        ])

        final_side = get_side_from_trajectory(
            global_pos,
            trajectory,
            tangents,
        )

        final_orient = get_orientation_from_trajectory(
            avg_box[6],
            global_pos,
            trajectory,
            tangents,
        )

        final_cars.append({
            "box": avg_box.tolist(),
            "count": c["count"],
            "conf": c["last_conf"],
            "side": final_side,
            "orient": final_orient,
        })

    print(f"  Raw detections: {len(all_detections)}")
    print(f"  Initial clusters: {len(unique_cars)}")
    print(f"  Final clusters: {len(final_cars)}")

    # 4. Save raw detections
    print("\n[4/5] Saving raw detections...")



    # 5. Interactive refinement
    print("\n[5/5] Starting interactive refinement...")

    refiner = BoundingBoxRefiner(
        vis_global_cloud,
        final_cars,
        global_origin,
        trajectory,
        tangents,
        screenshot_dir=OUTPUT_SCREENSHOT_DIR,
    )

    refined_cars = refiner.run()

    if refined_cars is not None:
        save_detections_json(
            refined_cars,
            global_origin,
            OUTPUT_REFINED,
            kind="refined",
        )

        save_clusters_txt(
            refined_cars,
            global_origin,
            OUTPUT_CLUSTERS_TXT,
        )

        save_bboxes_txt(
            refined_cars,
            global_origin,
            OUTPUT_BB_TXT,
        )

        print(f"\nRefined ground truth saved: {len(refined_cars)} cars")
    else:
        print("\nRefinement cancelled. Raw detections kept.")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"  Refined detections: {OUTPUT_REFINED}")
    print(f"  Clusters TXT:       {OUTPUT_CLUSTERS_TXT}")
    print(f"  BBoxes TXT:         {OUTPUT_BB_TXT}")
    print(f"  Screenshots:        {OUTPUT_SCREENSHOT_DIR}/")


if __name__ == "__main__":
    main()