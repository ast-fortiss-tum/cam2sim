#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4-multiple_log_gs_new_knn.py

Variant of 4-multiple_log_gs_new.py that uses the KNN local similarity
transform (utm_to_nerfstudio_transform_knn.json) produced by
4C_utm_yaw_to_nerfstudio_knn.py.

Differences from 4-multiple_log_gs_new.py:
  - Looks for utm_to_nerfstudio_transform_knn.json (not _transform.json)
  - CoordinateTransformer is updated to load the new format and to fit
    a local Umeyama on the k nearest training points at every query.

Everything else is identical, so the calibration GUI, the replay loop,
and the CARLA / hero / sensor handling are the same.

Reads from (project root):
    data/processed_dataset/<BAG>/maps/map.xodr
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json
    data/data_for_carla/<BAG>/camera.json
    data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_N/splatfacto/<TS>/config.yml
    data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_N/splatfacto/<TS>/utm_to_nerfstudio_transform_knn.json
    data/data_for_gaussian_splatting/<BAG>/frame_positions_split_N_1_of_K.txt
"""

import os
import sys
import json
import math
import re
import argparse
from pathlib import Path
from queue import Empty

import carla
import numpy as np
import pygame
import torch
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.neighbors import NearestNeighbors

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType


# =============================================================================
#  PATH SETUP
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")
if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )

if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.config import (
    CARLA_IP,
    CARLA_PORT,
    HERO_VEHICLE_TYPE,
)
from utils.carla_simulator import (
    update_synchronous_mode,
    cleanup_old_sensors,
    spawn_sensor,
    get_xodr_projection_params,
)


# =============================================================================
#  CONFIG
# =============================================================================

BAG_NAME = "reference_bag"

XODR_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed_dataset", BAG_NAME, "maps", "map.xodr"
)
TRAJECTORY_FILE = os.path.join(
    PROJECT_ROOT, "data", "data_for_carla", BAG_NAME,
    "trajectory_positions_rear_odom_yaw.json"
)
CAMERA_CONFIG_FILE = os.path.join(
    PROJECT_ROOT, "data", "data_for_carla", BAG_NAME, "camera.json"
)
GS_DATA_ROOT = os.path.join(
    PROJECT_ROOT, "data", "data_for_gaussian_splatting", BAG_NAME
)
GS_OUTPUTS_DIR = os.path.join(GS_DATA_ROOT, "outputs")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "data_for_carla", BAG_NAME, "replay_results_knn"
)

# KNN transform filename (different from the global Umeyama one)
UTM_TRANSFORM_FILENAME = "utm_to_nerfstudio_transform_knn.json"

IM_WIDTH = 800
IM_HEIGHT = 503


# =============================================================================
#  SLIDER (calibration GUI)
# =============================================================================

class Slider:
    def __init__(self, name, x, y, w, h, min_val, max_val, start_val):
        self.name = name
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = start_val
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging:
            mouse_x = pygame.mouse.get_pos()[0]
            rel_x = max(self.rect.left, min(mouse_x, self.rect.right)) - self.rect.left
            ratio = rel_x / self.rect.width
            self.val = self.min_val + (self.max_val - self.min_val) * ratio

    def draw(self, screen, font):
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.left + (self.rect.width * ratio)
        handle_rect = pygame.Rect(handle_x - 5, self.rect.top, 10, self.rect.height)
        pygame.draw.rect(screen, (200, 200, 200), handle_rect)
        label = font.render(f"{self.name}: {self.val:.2f}", True, (255, 50, 50))
        screen.blit(label, (self.rect.left, self.rect.top - 18))


# =============================================================================
#  LOCAL UMEYAMA (used at runtime by CoordinateTransformer)
# =============================================================================

def _umeyama_alignment_2d(source, target, with_scale=True):
    """Runtime local-fit Umeyama, fast and minimal-allocation."""
    n, dim = source.shape

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)

    source_centered = source - mu_source
    target_centered = target - mu_target

    var_source = float(np.sum(source_centered ** 2)) / n
    if var_source < 1e-12:
        return 1.0, np.eye(dim), mu_target - mu_source

    cov = (target_centered.T @ source_centered) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_source) if with_scale else 1.0
    t = mu_target - s * R @ mu_source
    return s, R, t


# =============================================================================
#  COORDINATE TRANSFORMS (KNN local similarity)
# =============================================================================

class CoordinateTransformer:
    """
    CARLA <-> UTM <-> Nerfstudio coordinate chain (per split), with KNN local
    similarity for UTM -> NS.

    Supports both JSON formats:
      - mode == "local_similarity_knn": uses training_points + k for runtime fit.
      - mode == "2D" (or missing): falls back to the old global Umeyama
        (so the same script can still load an old _transform.json).
    """

    def __init__(self, xodr_path, utm_to_nerfstudio_path):
        # ---- xodr / projection setup ----
        with open(xodr_path, "r") as f:
            xodr_data = f.read()

        xodr_params = get_xodr_projection_params(xodr_data)
        self.xodr_offset = xodr_params["offset"]
        proj_string = xodr_params["geo_reference"].strip()
        if proj_string == "+proj=tmerc":
            proj_string = ("+proj=tmerc +lat_0=0 +lon_0=0 +k=1 "
                           "+x_0=0 +y_0=0 +datum=WGS84")

        self.transformer_proj_to_wgs84 = Transformer.from_crs(
            proj_string, "EPSG:4326", always_xy=True
        )
        self.transformer_wgs84_to_utm = Transformer.from_crs(
            "EPSG:4326", "EPSG:25832", always_xy=True
        )
        self.transformer_wgs84_to_proj = Transformer.from_crs(
            "EPSG:4326", proj_string, always_xy=True
        )
        self.transformer_utm_to_wgs84 = Transformer.from_crs(
            "EPSG:25832", "EPSG:4326", always_xy=True
        )

        self.utm_x_angle_in_carla = self._compute_utm_x_angle_in_carla()
        print(f"[CoordinateTransformer] UTM +X angle in CARLA: "
              f"{np.degrees(self.utm_x_angle_in_carla):.4f} deg")

        # ---- Load transform JSON ----
        with open(utm_to_nerfstudio_path, "r") as f:
            tf = json.load(f)

        self.transform_mode = tf.get("mode", "2D")

        # Global Umeyama params (always loaded for the yaw module &
        # position_rotation_angle, and as a fallback when KNN block is absent).
        if "scale" in tf and "rotation" in tf and "translation" in tf:
            self.ns_scale = float(tf["scale"])
            self.ns_rotation = np.array(tf["rotation"])
            self.ns_translation = np.array(tf["translation"])
            self.position_rotation_angle = float(np.arctan2(
                self.ns_rotation[1, 0], self.ns_rotation[0, 0]
            ))
        else:
            self.ns_scale = 1.0
            self.ns_rotation = np.eye(3)
            self.ns_translation = np.zeros(3)
            self.position_rotation_angle = 0.0

        # KNN block
        self.alignment_mode = "global"
        self._knn_utm = None
        self._knn_ns_xy = None
        self._knn_ns_z = None
        self._knn_k = None
        self._knn_index = None

        if self.transform_mode == "local_similarity_knn":
            tp = tf.get("training_points")
            if tp is None:
                raise RuntimeError(
                    f"JSON has mode=local_similarity_knn but no "
                    f"'training_points' block: {utm_to_nerfstudio_path}"
                )
            self._knn_utm = np.column_stack([
                np.array(tp["utm_easting"], dtype=np.float64),
                np.array(tp["utm_northing"], dtype=np.float64),
            ])
            self._knn_ns_xy = np.column_stack([
                np.array(tp["ns_x"], dtype=np.float64),
                np.array(tp["ns_y"], dtype=np.float64),
            ])
            self._knn_ns_z = np.array(tp["ns_z"], dtype=np.float64)
            self._knn_k = int(tf.get("k", 20))
            self._knn_k = max(3, min(self._knn_k, len(self._knn_utm) - 1))
            self._knn_index = NearestNeighbors(
                n_neighbors=self._knn_k
            ).fit(self._knn_utm)
            self.alignment_mode = "local_knn"
            print(f"[CoordinateTransformer] Using KNN local similarity "
                  f"(k={self._knn_k}, n_train={len(self._knn_utm)})")
        else:
            print(f"[CoordinateTransformer] Using global Umeyama "
                  f"(scale={self.ns_scale:.10f}, "
                  f"angle="
                  f"{np.degrees(self.position_rotation_angle):.2f} deg)")

        # ---- Yaw alignment (kept global for both modes) ----
        yaw_align = tf.get("yaw_alignment")
        if yaw_align:
            self.yaw_sign = int(yaw_align["yaw_sign"])
            self.yaw_offset = float(yaw_align["yaw_offset_rad"])
            self.use_orientation_yaw = True
            sign_str = "+" if self.yaw_sign > 0 else "-"
            print(f"[CoordinateTransformer] yaw map: "
                  f"ns_yaw = {sign_str}utm_yaw + "
                  f"{np.degrees(self.yaw_offset):.2f} deg "
                  f"(residual std: "
                  f"{yaw_align.get('residual_std_deg', '?')} deg)")
        else:
            self.yaw_sign = -1
            self.yaw_offset = self.position_rotation_angle - math.pi / 2
            self.use_orientation_yaw = False
            print(f"[CoordinateTransformer] WARN: no yaw_alignment in JSON. "
                  f"Using fallback formula. offset="
                  f"{np.degrees(self.yaw_offset):.2f} deg")

        # ---- Camera mount angles ----
        mount = tf.get("camera_mount_angles")
        if mount:
            self.avg_pitch = float(mount["avg_pitch_rad"])
            self.avg_roll = float(mount["avg_roll_rad"])
            print(f"[CoordinateTransformer] camera mount: "
                  f"pitch={np.degrees(self.avg_pitch):.2f} deg, "
                  f"roll={np.degrees(self.avg_roll):.2f} deg")
        else:
            self.avg_pitch = 0.0
            self.avg_roll = 0.0

    # ---- Internal helpers ----

    def _utm_to_carla_point(self, utm_easting, utm_northing):
        lon, lat = self.transformer_utm_to_wgs84.transform(utm_easting, utm_northing)
        proj_x, proj_y = self.transformer_wgs84_to_proj.transform(lon, lat)
        local_x = proj_x + self.xodr_offset[0]
        local_y = proj_y + self.xodr_offset[1]
        return local_x, -local_y

    def _compute_utm_x_angle_in_carla(self):
        proj_x = -self.xodr_offset[0]
        proj_y = -self.xodr_offset[1]
        lon, lat = self.transformer_proj_to_wgs84.transform(proj_x, proj_y)
        ref_utm_e, ref_utm_n = self.transformer_wgs84_to_utm.transform(lon, lat)

        cx0, cy0 = self._utm_to_carla_point(ref_utm_e, ref_utm_n)
        cx1, cy1 = self._utm_to_carla_point(ref_utm_e + 1.0, ref_utm_n)

        dx = cx1 - cx0
        dy = cy1 - cy0
        return math.atan2(dy, dx)

    # ---- Public API ----

    def carla_to_utm(self, carla_x, carla_y):
        local_x = carla_x
        local_y = -carla_y
        proj_x = local_x - self.xodr_offset[0]
        proj_y = local_y - self.xodr_offset[1]
        lon, lat = self.transformer_proj_to_wgs84.transform(proj_x, proj_y)
        utm_easting, utm_northing = self.transformer_wgs84_to_utm.transform(
            lon, lat
        )
        return utm_easting, utm_northing

    def utm_to_nerfstudio(self, easting, northing, altitude=0.0):
        """
        - local_knn mode: fits a local Umeyama on the k nearest training
          points and applies it. Z is inverse-distance interpolated from
          the same neighbors' ns_z.
        - global mode: scale * R @ [E, N, alt] + t.
        """
        if self.alignment_mode == "local_knn":
            query = np.array([easting, northing], dtype=np.float64)
            _, idx = self._knn_index.kneighbors(
                query.reshape(1, -1), n_neighbors=self._knn_k
            )
            local_src = self._knn_utm[idx[0]]
            local_tgt = self._knn_ns_xy[idx[0]]

            s_loc, R_loc, t_loc = _umeyama_alignment_2d(
                local_src, local_tgt, with_scale=True
            )
            xy_pred = s_loc * (R_loc @ query) + t_loc

            # Z via inverse-distance on the same neighbors' ns_z
            dists = np.linalg.norm(local_src - query, axis=1)
            weights = 1.0 / (dists + 1e-9)
            weights /= weights.sum()
            z_pred = float((weights * self._knn_ns_z[idx[0]]).sum())

            return np.array([xy_pred[0], xy_pred[1], z_pred])

        # Global Umeyama fallback
        utm = np.array([easting, northing, altitude])
        return self.ns_scale * self.ns_rotation @ utm + self.ns_translation

    def carla_to_nerfstudio(self, carla_x, carla_y, carla_z=0.0):
        utm_e, utm_n = self.carla_to_utm(carla_x, carla_y)
        return self.utm_to_nerfstudio(utm_e, utm_n, 0.0)

    def transform_yaw_carla_to_nerfstudio(self, carla_yaw_rad):
        if self.use_orientation_yaw:
            utm_yaw = self.utm_x_angle_in_carla - carla_yaw_rad + math.pi / 2
            return self.yaw_sign * utm_yaw + self.yaw_offset
        return -carla_yaw_rad + self.position_rotation_angle - math.pi / 2


# =============================================================================
#  SPLIT MODEL
# =============================================================================

class SplitModel:
    def __init__(self, name, pipeline, coord_transformer, training_cameras,
                 frame_ids, training_filenames, data_root=None):
        self.name = name
        self.pipeline = pipeline
        self.coord_transformer = coord_transformer
        self.training_cameras = training_cameras
        self.frame_ids = set(frame_ids)
        self.min_frame = min(frame_ids) if frame_ids else 0
        self.max_frame = max(frame_ids) if frame_ids else 0
        self.center_frame = (self.min_frame + self.max_frame) / 2.0
        self.training_filenames = training_filenames
        self.data_root = data_root or "."

        self.cam_idx_to_frame_id = {}
        self.frame_id_to_cam_idx = {}
        for i, fn in enumerate(training_filenames):
            fid = extract_frame_number(fn)
            if fid is not None:
                self.cam_idx_to_frame_id[i] = fid
                self.frame_id_to_cam_idx[fid] = i

        if (hasattr(coord_transformer, "avg_pitch")
                and coord_transformer.avg_pitch != 0.0):
            self.avg_pitch = coord_transformer.avg_pitch
            self.avg_roll = coord_transformer.avg_roll
            print(f"   Camera mount angles (from JSON): "
                  f"pitch={np.degrees(self.avg_pitch):.2f} deg, "
                  f"roll={np.degrees(self.avg_roll):.2f} deg")
        else:
            yaws, pitches, rolls = [], [], []
            for i in range(training_cameras.shape[0]):
                y, p, r = extract_ypr_from_c2w(training_cameras[i])
                yaws.append(y)
                pitches.append(p)
                rolls.append(r)
            self.avg_pitch = float(np.median(pitches))
            self.avg_roll = float(np.median(rolls))
            print(f"   Camera mount angles (computed): "
                  f"pitch={np.degrees(self.avg_pitch):.2f} deg, "
                  f"roll={np.degrees(self.avg_roll):.2f} deg")

        cam_positions = training_cameras[:, :3, 3]
        xy = cam_positions[:, :2]
        z = cam_positions[:, 2]

        self.z_linear = LinearNDInterpolator(xy, z)
        self.z_nearest = NearestNDInterpolator(xy, z)
        self.z_fallback = float(np.median(z))

        print(f"   Z interpolator: {len(z)} points, "
              f"Z range [{z.min():.4f}, {z.max():.4f}], "
              f"median={np.median(z):.4f}, std={np.std(z):.4f}")

    def lookup_z(self, ns_x, ns_y):
        z = self.z_linear(ns_x, ns_y)
        if np.isnan(z):
            z = self.z_nearest(ns_x, ns_y)
        if np.isnan(z):
            z = self.z_fallback
        return float(z)

    def get_training_cam_c2w(self, cam_idx):
        c2w = np.eye(4)
        c2w[:3, :] = self.training_cameras[cam_idx]
        return c2w

    def get_training_image(self, cam_idx):
        if cam_idx < 0 or cam_idx >= len(self.training_filenames):
            return None
        filepath = Path(self.training_filenames[cam_idx])

        candidates = [
            filepath,
            Path(self.data_root) / filepath,
            Path(self.data_root) / filepath.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception as e:
                    print(f"   Warning: could not load training image "
                          f"{candidate}: {e}")
        print(f"   WARN: Training image not found for cam #{cam_idx}")
        return None

    def get_first_training_frame_id(self):
        return self.cam_idx_to_frame_id.get(0, self.min_frame)


# =============================================================================
#  c2w / YPR HELPERS
# =============================================================================

def build_nerfstudio_c2w(ns_pos, yaw_rad, pitch_rad=0.0, roll_rad=0.0):
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)

    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R_roll = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    R_world = R_yaw @ R_pitch @ R_roll

    cam_forward = R_world @ np.array([0, 1, 0])
    cam_right = R_world @ np.array([1, 0, 0])
    cam_up = R_world @ np.array([0, 0, 1])

    c2w = np.eye(4)
    c2w[:3, 0] = cam_right
    c2w[:3, 1] = cam_up
    c2w[:3, 2] = -cam_forward
    c2w[:3, 3] = ns_pos
    return c2w


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


def extract_frame_number(filename):
    name = os.path.basename(str(filename))
    numbers = re.findall(r"\d+", os.path.splitext(name)[0])
    if numbers:
        return int(numbers[-1])
    return None


def render_gs(pipeline, c2w, width, height, fov):
    fov_rad = math.radians(fov)
    focal = 0.5 * width / math.tan(0.5 * fov_rad)

    camera = Cameras(
        camera_to_worlds=torch.from_numpy(c2w[:3, :]).float().unsqueeze(0),
        fx=torch.tensor([focal]),
        fy=torch.tensor([focal]),
        cx=torch.tensor([width / 2.0]),
        cy=torch.tensor([height / 2.0]),
        width=torch.tensor([width]),
        height=torch.tensor([height]),
        camera_type=CameraType.PERSPECTIVE,
    ).to(pipeline.device)

    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)
    rgb = outputs["rgb"].cpu().numpy()
    return Image.fromarray((rgb * 255).astype(np.uint8))


# =============================================================================
#  SPLIT DETECTION (cam2sim layout) - looks for the KNN transform JSON
# =============================================================================

def auto_detect_splits():
    """
    Look for splatfacto splits in:
        data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_<N>/splatfacto/<TS>/config.yml

    For each split also resolves:
        - UTM_TRANSFORM_FILENAME (next to config.yml)
        - frame_positions_split_<N>_*.txt (in GS_DATA_ROOT)
    """
    splits = []

    if not os.path.isdir(GS_OUTPUTS_DIR):
        print(f"[WARN] Outputs folder not found: {GS_OUTPUTS_DIR}")
        return splits

    split_dirs = sorted([
        d for d in os.listdir(GS_OUTPUTS_DIR)
        if os.path.isdir(os.path.join(GS_OUTPUTS_DIR, d))
        and d.startswith("splatfacto_split_")
    ])

    for split_dir in split_dirs:
        match = re.match(r"splatfacto_split_(\d+)", split_dir)
        if not match:
            continue
        split_num = int(match.group(1))

        splatfacto_dir = os.path.join(GS_OUTPUTS_DIR, split_dir, "splatfacto")
        if not os.path.isdir(splatfacto_dir):
            print(f"[WARN] Missing 'splatfacto' subfolder in {split_dir}")
            continue

        runs = sorted([
            d for d in os.listdir(splatfacto_dir)
            if os.path.isdir(os.path.join(splatfacto_dir, d))
        ])
        if not runs:
            print(f"[WARN] No runs found in {splatfacto_dir}")
            continue

        run_name = runs[-1]
        run_dir = os.path.join(splatfacto_dir, run_name)
        config_path = os.path.join(run_dir, "config.yml")
        utm_transform_path = os.path.join(run_dir, UTM_TRANSFORM_FILENAME)

        if not os.path.exists(config_path):
            print(f"[WARN] No config.yml in {run_dir}")
            continue
        if not os.path.exists(utm_transform_path):
            print(f"[WARN] No {UTM_TRANSFORM_FILENAME} in {run_dir}")
            print(f"       Run 4C_utm_yaw_to_nerfstudio_knn.py for split "
                  f"{split_num} first.")
            continue

        # Find frame_positions_split_<N>_*.txt
        frame_positions = None
        for fname in os.listdir(GS_DATA_ROOT):
            if (fname.startswith(f"frame_positions_split_{split_num}_")
                    and fname.endswith(".txt")):
                frame_positions = os.path.join(GS_DATA_ROOT, fname)
                break

        if frame_positions is None:
            print(f"[WARN] No frame_positions_split_{split_num}_*.txt found "
                  f"in {GS_DATA_ROOT}")

        splits.append({
            "name": f"split_{split_num}",
            "split_num": split_num,
            "gs_config": config_path,
            "utm_transform": utm_transform_path,
            "frame_positions": frame_positions,
            "data_root": GS_DATA_ROOT,
            "run_name": run_name,
        })
        print(f"[INFO] Found split_{split_num} (run={run_name})")

    splits.sort(key=lambda s: s["split_num"])
    return splits


def find_best_split(frame_id, splits, last_split_idx=0):
    current_split = splits[last_split_idx]

    if current_split.min_frame <= frame_id <= current_split.max_frame:
        return last_split_idx

    covering = [(i, s) for i, s in enumerate(splits)
                if s.min_frame <= frame_id <= s.max_frame]

    if len(covering) == 1:
        return covering[0][0]
    if len(covering) > 1:
        return max(covering, key=lambda x: x[1].max_frame)[0]

    best_idx = last_split_idx
    best_dist = float("inf")
    for i, s in enumerate(splits):
        if frame_id < s.min_frame:
            dist = s.min_frame - frame_id
        elif frame_id > s.max_frame:
            dist = frame_id - s.max_frame
        else:
            dist = 0
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def read_frame_ids_from_positions(filepath):
    frame_ids = []
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                try:
                    frame_ids.append(int(parts[0].strip()))
                except (ValueError, IndexError):
                    continue
    return frame_ids


def load_split_models(split_configs, xodr_path, fov):
    split_models = []

    for cfg in split_configs:
        name = cfg["name"]
        gs_config = cfg["gs_config"]
        utm_transform = cfg["utm_transform"]
        data_root = cfg["data_root"]

        print(f"\n{'='*60}")
        print(f"  Loading split: {name}")
        print(f"{'='*60}")

        coord_transformer = CoordinateTransformer(xodr_path, utm_transform)

        config_path = Path(gs_config).resolve()
        data_root_abs = Path(data_root).resolve()

        print(f"Loading GS Model from: {config_path}")
        original_cwd = os.getcwd()
        try:
            os.chdir(data_root_abs)
            _, pipeline, _, step = eval_setup(config_path, test_mode="inference")
            pipeline.model.eval()

            print(f"   Loaded checkpoint at step {step}")
            if hasattr(pipeline.model, "num_points"):
                print(f"   Number of gaussians: {pipeline.model.num_points}")
            elif (hasattr(pipeline.model, "means")
                    and pipeline.model.means is not None):
                print(f"   Number of gaussians: "
                      f"{pipeline.model.means.shape[0]}")

            dp = pipeline.datamanager.train_dataparser_outputs
            training_cameras = dp.cameras.camera_to_worlds.cpu().numpy()
            training_filenames = dp.image_filenames

            training_frame_ids = []
            for fn in training_filenames:
                fid = extract_frame_number(fn)
                if fid is not None:
                    training_frame_ids.append(fid)

            positions_frame_ids = read_frame_ids_from_positions(
                cfg.get("frame_positions")
            )
            all_frame_ids = sorted(
                set(training_frame_ids) | set(positions_frame_ids)
            )

            print(f"OK {name}: {training_cameras.shape[0]} cameras, "
                  f"frames [{min(all_frame_ids)}-{max(all_frame_ids)}] "
                  f"({len(all_frame_ids)} frame IDs)")

            split_models.append(SplitModel(
                name=name,
                pipeline=pipeline,
                coord_transformer=coord_transformer,
                training_cameras=training_cameras,
                frame_ids=all_frame_ids,
                training_filenames=training_filenames,
                data_root=str(data_root_abs),
            ))
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            import traceback
            traceback.print_exc()
            os.chdir(original_cwd)
            continue
        os.chdir(original_cwd)

    return split_models


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replay with Nerfstudio-trained GS (KNN local similarity, "
                    "multi-split, cam2sim layout)"
    )
    parser.add_argument("--only_carla", action="store_true",
                        help="Run without GS model")
    parser.add_argument("--only_split", type=int, default=None,
                        help="Load and use ONLY this split number "
                             "(useful for low-VRAM GPUs). "
                             "All trajectory frames will use this split.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to render")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip Phase 1 free camera calibration")
    parser.add_argument("--no_save", action="store_true",
                        help="Disable frame saving")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for saved frames")
    args = parser.parse_args()

    print("=" * 80)
    print("REPLAY (KNN local similarity): CARLA + Gaussian Splatting (multi-split)")
    print("=" * 80)
    print(f"[INFO] Project root:        {PROJECT_ROOT}")
    print(f"[INFO] Bag name:            {BAG_NAME}")
    print(f"[INFO] XODR file:           {XODR_FILE}")
    print(f"[INFO] Trajectory:          {TRAJECTORY_FILE}")
    print(f"[INFO] Camera config:       {CAMERA_CONFIG_FILE}")
    print(f"[INFO] GS data root:        {GS_DATA_ROOT}")
    print(f"[INFO] UTM transform file:  {UTM_TRANSFORM_FILENAME}")
    print(f"[INFO] Output dir:          {args.output_dir}")
    print(f"[INFO] CARLA:               {CARLA_IP}:{CARLA_PORT}")
    print("=" * 80)

    # ---- Camera config ----
    with open(CAMERA_CONFIG_FILE, "r") as f:
        cam_data = json.load(f)
    cam_config = cam_data["camera"]
    fov = float(cam_config["fov"])
    fps = int(cam_config["fps"])
    cam_pos_x = float(cam_config["position"]["x"])
    cam_pos_y = float(cam_config["position"]["y"])
    cam_pos_z = float(cam_config["position"]["z"])
    cam_pitch = float(cam_config.get("pitch", 0.0))
    print(f"[INFO] Camera: fov={fov}, fps={fps}, "
          f"pos=({cam_pos_x},{cam_pos_y},{cam_pos_z}), pitch={cam_pitch}")

    # ---- Discover and load split models ----
    only_carla = args.only_carla
    split_models = []

    if not only_carla:
        print("\n[INFO] Auto-detecting split models...")
        split_configs = auto_detect_splits()

        if args.only_split is not None:
            split_configs = [c for c in split_configs
                             if c["split_num"] == args.only_split]
            if not split_configs:
                print(f"[ERROR] --only_split {args.only_split} requested but "
                      f"no matching split found")
                sys.exit(1)
            print(f"[INFO] Filtered to ONLY split_{args.only_split} "
                  f"(--only_split mode)")

        if split_configs:
            split_models = load_split_models(split_configs, XODR_FILE, fov=fov)

        if not split_models:
            print("[WARN] No GS models loaded - falling back to only_carla mode")
            only_carla = True

    if split_models:
        print(f"\nLoaded {len(split_models)} split model(s):")
        for sm in split_models:
            zmin = sm.training_cameras[:, 2, 3].min()
            zmax = sm.training_cameras[:, 2, 3].max()
            print(f"   {sm.name}: frames [{sm.min_frame}-{sm.max_frame}], "
                  f"Z range [{zmin:.4f}, {zmax:.4f}], "
                  f"alignment_mode={sm.coord_transformer.alignment_mode}")

    # ---- Connect to CARLA ----
    print(f"\n[INFO] Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)

    # ---- Load trajectory ----
    with open(TRAJECTORY_FILE, "r") as f:
        trajectory_points = json.load(f)
    print(f"[INFO] Loaded {len(trajectory_points)} trajectory points.")

    trajectory_by_frame = {tp["frame_id"]: tp for tp in trajectory_points}

    # ---- Sync mode ----
    update_synchronous_mode(world, tm, True, fps)
    world.tick()

    # ---- Find or spawn hero ----
    first_pt = trajectory_points[0]["transform"]
    start_transform = carla.Transform(
        carla.Location(
            x=first_pt["location"]["x"],
            y=first_pt["location"]["y"],
            z=first_pt["location"]["z"] + 0.5,
        ),
        carla.Rotation(pitch=0, yaw=first_pt["rotation"]["yaw"], roll=0),
    )

    bp_lib = world.get_blueprint_library()

    hero_vehicle = None
    all_vehicles = world.get_actors().filter("vehicle.*")
    for v in all_vehicles:
        if v.attributes.get("role_name", "") == "hero":
            hero_vehicle = v
            print(f"[INFO] Found existing hero vehicle "
                  f"(id={v.id}, type={v.type_id})")
            break

    if hero_vehicle is None:
        for v in all_vehicles:
            if v.type_id == HERO_VEHICLE_TYPE:
                hero_vehicle = v
                print(f"[INFO] Found existing vehicle of type "
                      f"{HERO_VEHICLE_TYPE} (id={v.id})")
                break

    if hero_vehicle:
        hero_vehicle.set_transform(start_transform)
    else:
        print("[WARN] No hero vehicle found in CARLA world. "
              "Did you run prepare_carla_world.py first?")
        hero_bp = bp_lib.find(HERO_VEHICLE_TYPE)
        hero_bp.set_attribute("role_name", "hero")
        hero_vehicle = world.spawn_actor(hero_bp, start_transform)
        print(f"[INFO] Spawned new hero vehicle "
              f"(id={hero_vehicle.id}, type={HERO_VEHICLE_TYPE})")

    hero_vehicle.set_simulate_physics(False)
    hero_vehicle.set_autopilot(False)
    cleanup_old_sensors(hero_vehicle)

    # ---- Sensor ----
    sensor_tf = carla.Transform(
        carla.Location(x=cam_pos_x, y=cam_pos_y, z=cam_pos_z),
        carla.Rotation(pitch=cam_pitch),
    )
    rgb_sensor, rgb_queue = spawn_sensor(
        bp_lib, "sensor.camera.rgb", IM_WIDTH, IM_HEIGHT, str(fov),
        world, sensor_tf, hero_vehicle,
    )
    world.tick()

    # ---- Pygame ----
    pygame.init()
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    # ============================================================
    #  PHASE 1: CALIBRATION (4-panel + sliders)
    # ============================================================
    calibration_offsets = {}

    if split_models and not args.skip_calibration:
        cal_win_w = IM_WIDTH * 2
        cal_win_h = IM_HEIGHT * 2
        screen = pygame.display.set_mode((cal_win_w, cal_win_h))

        for sm in split_models:
            print(f"\n{'='*60}")
            print(f"  CALIBRATING: {sm.name}")
            print(f"  Frames [{sm.min_frame}-{sm.max_frame}]")
            print(f"  WASD=move, Mouse=look, QE=up/down, Scroll=speed")
            print(f"  [/]=prev/next training cam, T=test, P=print, R=reset, ENTER=next split")
            print(f"{'='*60}")

            pygame.display.set_caption(f"Calibrating: {sm.name} | 4-Panel View")

            current_train_cam_idx = 0

            def teleport_to_training_cam(cam_idx):
                fid = sm.cam_idx_to_frame_id.get(cam_idx)
                if fid is None:
                    return fid
                tp = trajectory_by_frame.get(fid)
                if tp is None:
                    return fid
                cal_pt = tp["transform"]
                hero_vehicle.set_transform(carla.Transform(
                    carla.Location(
                        x=cal_pt["location"]["x"],
                        y=cal_pt["location"]["y"],
                        z=cal_pt["location"]["z"],
                    ),
                    carla.Rotation(pitch=0, yaw=cal_pt["rotation"]["yaw"], roll=0),
                ))
                world.tick()
                try:
                    rgb_queue.get(block=True, timeout=1.0)
                except Empty:
                    pass
                return fid

            current_frame_id = teleport_to_training_cam(current_train_cam_idx)

            def get_reference_images(cam_idx):
                orig_img = sm.get_training_image(cam_idx)
                if orig_img is not None:
                    orig_img = orig_img.resize((IM_WIDTH, IM_HEIGHT), Image.LANCZOS)
                else:
                    orig_img = Image.new("RGB", (IM_WIDTH, IM_HEIGHT), (60, 20, 20))
                train_c2w = sm.get_training_cam_c2w(cam_idx)
                gs_train_img = render_gs(sm.pipeline, train_c2w,
                                          IM_WIDTH, IM_HEIGHT, fov)
                return orig_img, gs_train_img

            ref_orig, ref_gs_train = get_reference_images(current_train_cam_idx)

            first_c2w = sm.training_cameras[current_train_cam_idx]
            cam_pos = first_c2w[:, 3].copy().astype(np.float64)
            cam_yaw, cam_pitch_v, cam_roll_v = extract_ypr_from_c2w(first_c2w)
            move_speed = 0.05
            look_sensitivity = 0.003
            mouse_dragging = False
            last_mouse_pos = (0, 0)

            slider_x = IM_WIDTH + 20
            sliders = [
                Slider("Yaw Offset", slider_x, 100, 200, 15, -180.0, 180.0, 0.0),
                Slider("Pitch Offset", slider_x, 150, 200, 15, -45.0, 45.0, 0.0),
                Slider("Roll Offset", slider_x, 200, 200, 15, -45.0, 45.0, 0.0),
                Slider("X Offset", slider_x, 270, 200, 15, -0.02, 0.02, 0.0),
                Slider("Y Offset", slider_x, 320, 200, 15, -0.02, 0.02, 0.0),
                Slider("Z Offset", slider_x, 370, 200, 15, -0.02, 0.02, 0.0),
            ]

            calibrating = True
            while calibrating:
                adjusted_pos = cam_pos + np.array(
                    [sliders[3].val, sliders[4].val, sliders[5].val]
                )
                adjusted_yaw = cam_yaw + math.radians(sliders[0].val)
                adjusted_pitch = cam_pitch_v + math.radians(sliders[1].val)
                adjusted_roll = cam_roll_v + math.radians(sliders[2].val)

                c2w_free = build_nerfstudio_c2w(
                    adjusted_pos, adjusted_yaw, adjusted_pitch, adjusted_roll
                )
                gs_pil = render_gs(sm.pipeline, c2w_free, IM_WIDTH, IM_HEIGHT, fov)

                world.tick()
                try:
                    rgb_data = rgb_queue.get(block=True, timeout=0.5)
                    rgb_np = np.frombuffer(
                        rgb_data.raw_data, dtype=np.uint8
                    ).reshape(
                        (rgb_data.height, rgb_data.width, 4)
                    )[:, :, :3][:, :, ::-1]
                    carla_pil = Image.fromarray(rgb_np)
                except Empty:
                    carla_pil = Image.new("RGB", (IM_WIDTH, IM_HEIGHT), (30, 30, 30))

                screen.fill((30, 30, 30))
                screen.blit(pygame.image.fromstring(
                    carla_pil.tobytes(), carla_pil.size, carla_pil.mode), (0, 0))
                screen.blit(pygame.image.fromstring(
                    gs_pil.tobytes(), gs_pil.size, gs_pil.mode), (IM_WIDTH, 0))
                screen.blit(pygame.image.fromstring(
                    ref_orig.tobytes(), ref_orig.size, ref_orig.mode), (0, IM_HEIGHT))
                screen.blit(pygame.image.fromstring(
                    ref_gs_train.tobytes(), ref_gs_train.size, ref_gs_train.mode),
                    (IM_WIDTH, IM_HEIGHT))

                screen.blit(font.render("CARLA Ground Truth", True, (0, 255, 0)),
                            (10, 10))
                screen.blit(font.render(f"GS Free Cam: {sm.name}",
                            True, (0, 255, 255)), (IM_WIDTH + 10, 10))

                train_fid = sm.cam_idx_to_frame_id.get(current_train_cam_idx, "?")
                screen.blit(font.render(
                    f"Original Training Image (cam #{current_train_cam_idx}, frame {train_fid})",
                    True, (255, 200, 0)), (10, IM_HEIGHT + 10))
                screen.blit(font.render(
                    f"GS from Training Pose (cam #{current_train_cam_idx}, frame {train_fid})",
                    True, (255, 150, 255)), (IM_WIDTH + 10, IM_HEIGHT + 10))

                pos_info = (f"Pos: ({adjusted_pos[0]:.3f}, "
                            f"{adjusted_pos[1]:.3f}, {adjusted_pos[2]:.3f})")
                screen.blit(font.render(pos_info, True, (255, 255, 0)),
                            (IM_WIDTH + 10, 30))
                rot_info = (f"Yaw:{math.degrees(adjusted_yaw):.1f}deg "
                            f"P:{math.degrees(adjusted_pitch):.1f}deg "
                            f"R:{math.degrees(adjusted_roll):.1f}deg")
                screen.blit(font.render(rot_info, True, (255, 255, 0)),
                            (IM_WIDTH + 10, 50))
                screen.blit(font.render(f"Speed: {move_speed:.3f}",
                            True, (255, 255, 0)), (IM_WIDTH + 10, 70))

                for s in sliders:
                    s.draw(screen, font)

                help_y = cal_win_h - 40
                screen.blit(font.render(
                    "WASD=move Mouse=look QE=up/down Scroll=speed [/]=prev/next cam",
                    True, (0, 255, 0)), (10, help_y))
                screen.blit(font.render(
                    f"T=test P=print R=reset ENTER=next ({sm.name})",
                    True, (0, 255, 0)), (10, help_y + 20))

                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)
                    for s in sliders:
                        s.handle_event(event)
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            calibrating = False
                        elif event.key == pygame.K_RIGHTBRACKET:
                            if current_train_cam_idx < len(sm.training_cameras) - 1:
                                current_train_cam_idx += 1
                                current_frame_id = teleport_to_training_cam(
                                    current_train_cam_idx)
                                ref_orig, ref_gs_train = get_reference_images(
                                    current_train_cam_idx)
                                tc2w = sm.training_cameras[current_train_cam_idx]
                                cam_pos = tc2w[:, 3].copy().astype(np.float64)
                                cam_yaw, cam_pitch_v, cam_roll_v = \
                                    extract_ypr_from_c2w(tc2w)
                        elif event.key == pygame.K_LEFTBRACKET:
                            if current_train_cam_idx > 0:
                                current_train_cam_idx -= 1
                                current_frame_id = teleport_to_training_cam(
                                    current_train_cam_idx)
                                ref_orig, ref_gs_train = get_reference_images(
                                    current_train_cam_idx)
                                tc2w = sm.training_cameras[current_train_cam_idx]
                                cam_pos = tc2w[:, 3].copy().astype(np.float64)
                                cam_yaw, cam_pitch_v, cam_roll_v = \
                                    extract_ypr_from_c2w(tc2w)
                        elif event.key == pygame.K_r:
                            for s in sliders:
                                s.val = 0.0
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            if not any(s.rect.collidepoint(event.pos) for s in sliders):
                                mouse_dragging = True
                                last_mouse_pos = event.pos
                        elif event.button == 4:
                            move_speed *= 1.2
                        elif event.button == 5:
                            move_speed /= 1.2
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            mouse_dragging = False
                    elif event.type == pygame.MOUSEMOTION:
                        if mouse_dragging:
                            dx = event.pos[0] - last_mouse_pos[0]
                            dy = event.pos[1] - last_mouse_pos[1]
                            cam_yaw -= dx * look_sensitivity
                            cam_pitch_v = max(-math.pi/2 + 0.01,
                                              min(math.pi/2 - 0.01,
                                                  cam_pitch_v - dy * look_sensitivity))
                            last_mouse_pos = event.pos

                keys = pygame.key.get_pressed()
                forward = np.array([-np.sin(cam_yaw), np.cos(cam_yaw), 0])
                right = np.array([np.cos(cam_yaw), np.sin(cam_yaw), 0])
                if keys[pygame.K_w]: cam_pos += forward * move_speed
                if keys[pygame.K_s]: cam_pos -= forward * move_speed
                if keys[pygame.K_d]: cam_pos += right * move_speed
                if keys[pygame.K_a]: cam_pos -= right * move_speed
                if keys[pygame.K_e]: cam_pos[2] += move_speed
                if keys[pygame.K_q]: cam_pos[2] -= move_speed
                clock.tick(30)

            yaw_offset = math.radians(sliders[0].val)
            pitch_offset = math.radians(sliders[1].val)
            roll_offset = math.radians(sliders[2].val)

            cam_tf = rgb_sensor.get_transform()
            cx, cy_c = cam_tf.location.x, cam_tf.location.y
            ns_from_tf = sm.coord_transformer.carla_to_nerfstudio(cx, cy_c)
            slider_off = np.array([sliders[3].val, sliders[4].val, sliders[5].val])
            pos_offset = (cam_pos - ns_from_tf) + slider_off

            ns_z_interp = sm.lookup_z(ns_from_tf[0], ns_from_tf[1])
            z_calib = cam_pos[2] + sliders[5].val
            z_offset_from_interp = z_calib - ns_z_interp

            calibration_offsets[sm.name] = {
                "pos_offset": pos_offset,
                "yaw_offset": yaw_offset,
                "pitch_offset": pitch_offset,
                "roll_offset": roll_offset,
                "z_offset": z_offset_from_interp,
            }
            print(f"OK {sm.name} calibrated: "
                  f"pos_offset=({pos_offset[0]:.4f}, {pos_offset[1]:.4f}, "
                  f"{pos_offset[2]:.4f}), "
                  f"yaw_offset={math.degrees(yaw_offset):.2f} deg, "
                  f"z_offset_from_interp={z_offset_from_interp:.4f}")

    if args.skip_calibration and split_models:
        for sm in split_models:
            cam_pos_train = sm.training_cameras[0, :, 3].copy()
            fid = sm.cam_idx_to_frame_id.get(0)
            tp = trajectory_by_frame.get(fid)
            if tp:
                cx = tp["transform"]["location"]["x"]
                cy = tp["transform"]["location"]["y"]
                ns_from_tf = sm.coord_transformer.carla_to_nerfstudio(cx, cy)
                pos_offset = cam_pos_train - ns_from_tf
                ns_z_interp = sm.lookup_z(ns_from_tf[0], ns_from_tf[1])
                calibration_offsets[sm.name] = {
                    "pos_offset": pos_offset,
                    "yaw_offset": 0.0,
                    "pitch_offset": 0.0,
                    "roll_offset": 0.0,
                    "z_offset": cam_pos_train[2] - ns_z_interp,
                }
                print(f"[skip_calibration] {sm.name}: auto pos_offset="
                      f"({pos_offset[0]:.4f}, {pos_offset[1]:.4f}, "
                      f"{pos_offset[2]:.4f})")

    # ============================================================
    #  PHASE 2: REPLAY
    # ============================================================
    win_w = IM_WIDTH * 2
    win_h = IM_HEIGHT
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("GS Replay KNN (Multi-Split) | Driving")

    print(f"\n[INFO] Replaying with {len(split_models)} split(s)...")

    save_flag = not args.no_save
    if save_flag:
        run_name = f"{BAG_NAME}_replay_knn"
        save_dir_carla = os.path.join(args.output_dir, run_name, "carla")
        save_dir_gs = os.path.join(args.output_dir, run_name, "gs")
        save_dir_combined = os.path.join(args.output_dir, run_name, "combined")
        os.makedirs(save_dir_carla, exist_ok=True)
        os.makedirs(save_dir_gs, exist_ok=True)
        os.makedirs(save_dir_combined, exist_ok=True)
        print(f"[INFO] Saving frames to: "
              f"{os.path.join(args.output_dir, run_name)}")

    current_split_idx = 0

    start_pt = trajectory_points[0]["transform"]
    hero_vehicle.set_transform(carla.Transform(
        carla.Location(x=start_pt["location"]["x"],
                       y=start_pt["location"]["y"],
                       z=start_pt["location"]["z"]),
        carla.Rotation(pitch=0, yaw=start_pt["rotation"]["yaw"], roll=0),
    ))
    world.tick()
    try:
        rgb_queue.get(block=True, timeout=1.0)
    except Empty:
        pass
    print("[INFO] Teleported to trajectory frame 0")

    try:
        end_idx = len(trajectory_points)
        if args.max_frames is not None:
            end_idx = min(args.max_frames, end_idx)
        print(f"[INFO] Running {end_idx} frames")

        for idx in range(end_idx):
            point = trajectory_points[idx]
            frame_id = point["frame_id"]
            pt = point["transform"]
            carla_yaw_deg = pt["rotation"]["yaw"]

            # Tiny back-offset like in the old script
            offset_distance = 0.13
            yaw_rad = math.radians(carla_yaw_deg)
            ox = -offset_distance * math.cos(yaw_rad)
            oy = -offset_distance * math.sin(yaw_rad)

            hero_vehicle.set_transform(carla.Transform(
                carla.Location(x=pt["location"]["x"] + ox,
                               y=pt["location"]["y"] + oy,
                               z=pt["location"]["z"]),
                carla.Rotation(pitch=0.0, yaw=carla_yaw_deg, roll=0.0),
            ))
            world.tick()

            try:
                rgb_data = rgb_queue.get(block=True, timeout=2.0)
                rgb_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape(
                    (rgb_data.height, rgb_data.width, 4))[:, :, :3][:, :, ::-1]
                carla_pil = Image.fromarray(rgb_np)
            except Empty:
                continue

            gs_pil = None
            active_split_name = "none"
            ns_pos_raw = None
            offsets = None

            if split_models:
                new_split_idx = find_best_split(
                    frame_id, split_models, current_split_idx
                )
                if new_split_idx != current_split_idx:
                    print(f"[Frame {idx}] Switching: "
                          f"{split_models[current_split_idx].name} -> "
                          f"{split_models[new_split_idx].name}")
                    current_split_idx = new_split_idx

                sm = split_models[current_split_idx]
                active_split_name = sm.name
                offsets = calibration_offsets.get(sm.name, {
                    "pos_offset": np.zeros(3),
                    "yaw_offset": 0.0,
                    "pitch_offset": 0.0,
                    "roll_offset": 0.0,
                    "z_offset": 0.0,
                })

                cam_tf = rgb_sensor.get_transform()
                carla_x = cam_tf.location.x
                carla_y = cam_tf.location.y
                carla_yaw_rad = math.radians(cam_tf.rotation.yaw)

                ns_pos_raw = sm.coord_transformer.carla_to_nerfstudio(
                    carla_x, carla_y
                )
                ns_pos_raw[2] = sm.lookup_z(ns_pos_raw[0], ns_pos_raw[1])

                ns_pos = ns_pos_raw + offsets["pos_offset"]
                ns_pos[2] = ns_pos_raw[2] + offsets.get("z_offset", 0.0)

                ns_yaw_raw = sm.coord_transformer.transform_yaw_carla_to_nerfstudio(
                    carla_yaw_rad
                )
                ns_yaw = ns_yaw_raw + offsets["yaw_offset"]
                ns_pitch = sm.avg_pitch + offsets["pitch_offset"]
                ns_roll = sm.avg_roll + offsets["roll_offset"]

                if idx % 100 == 0:
                    print(f"[Frame {idx}] {sm.name} | CARLA: "
                          f"({carla_x:.2f}, {carla_y:.2f}) "
                          f"yaw={math.degrees(carla_yaw_rad):.1f}deg -> NS: "
                          f"({ns_pos[0]:.4f}, {ns_pos[1]:.4f}, {ns_pos[2]:.4f}) "
                          f"yaw={math.degrees(ns_yaw):.1f}deg")

                c2w = build_nerfstudio_c2w(ns_pos, ns_yaw, ns_pitch, ns_roll)
                gs_pil = render_gs(sm.pipeline, c2w, IM_WIDTH, IM_HEIGHT, fov)

            combined = Image.new("RGB", (win_w, IM_HEIGHT))
            combined.paste(carla_pil, (0, 0))
            if gs_pil:
                combined.paste(gs_pil, (IM_WIDTH, 0))

            # if save_flag:
            #     carla_pil.save(os.path.join(
            #         save_dir_carla, f"frame_{frame_id:06d}.png"))
            #     if gs_pil:
            #         gs_pil.save(os.path.join(
            #             save_dir_gs, f"frame_{frame_id:06d}.png"))
            #     combined.save(os.path.join(
            #         save_dir_combined, f"frame_{frame_id:06d}.jpg"), quality=95)

            screen.blit(pygame.image.fromstring(
                combined.tobytes(), combined.size, combined.mode), (0, 0))
            screen.blit(font.render(
                f"Frame {idx}/{len(trajectory_points)} | ID: {frame_id}",
                True, (0, 255, 0)), (10, 10))
            screen.blit(font.render(f"Split: {active_split_name}",
                        True, (255, 200, 0)), (IM_WIDTH + 10, 10))

            cam_tf = rgb_sensor.get_transform()
            cam_loc = cam_tf.location
            cam_rot = cam_tf.rotation
            screen.blit(font.render(
                f"CARLA CAM XYZ: {cam_loc.x:+.2f}, {cam_loc.y:+.2f}, {cam_loc.z:+.2f}",
                True, (0, 255, 0)), (10, 30))
            screen.blit(font.render(
                f"CARLA CAM RPY: {cam_rot.roll:+.1f}  {cam_rot.pitch:+.1f}  "
                f"{cam_rot.yaw:+.1f}",
                True, (0, 255, 0)), (10, 50))
            if split_models and ns_pos_raw is not None:
                screen.blit(font.render(
                    f"NS Z(interp): {ns_pos_raw[2]:.4f} + "
                    f"offset {offsets.get('z_offset', 0.0):.4f} = {ns_pos[2]:.4f}",
                    True, (255, 150, 0)), (IM_WIDTH + 10, 30))

            veh_tf = hero_vehicle.get_transform()
            screen.blit(font.render(
                f"CARLA VEH XYZ: {veh_tf.location.x:+.2f}, {veh_tf.location.y:+.2f}, "
                f"{veh_tf.location.z:+.2f}",
                True, (0, 200, 255)), (10, 70))
            screen.blit(font.render(
                f"CARLA VEH YAW: {veh_tf.rotation.yaw:+.1f} deg",
                True, (0, 200, 255)), (10, 90))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt

            clock.tick(60)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        if rgb_sensor:
            rgb_sensor.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()