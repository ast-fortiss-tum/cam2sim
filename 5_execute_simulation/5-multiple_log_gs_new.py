#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
replay_with_gs_nerfstudio.py

Replay script for Gaussian Splatting models trained via nerfstudio on COLMAP data.
Supports MULTIPLE SPLITS: loads N GS models and switches between them based on
which split covers the current trajectory frame.

Transform chain (per split):
    CARLA local coords → UTM (inverse of map projection) → Nerfstudio (similarity transform)

Z HANDLING:
    Since XODR maps have no real-world altitude, Z is interpolated from training
    camera positions in nerfstudio space. CARLA XY → transform to nerfstudio XY →
    interpolate Z from surrounding training cameras (Delaunay triangulation).

PHASE 1: FREE CAMERA - Explore the GS model to verify alignment
         Now includes 4-panel view:
           Top-left:     CARLA ground truth
           Top-right:    GS free camera (your current view)
           Bottom-left:  Original training image (from dataset)
           Bottom-right: GS rendered from training camera pose
PHASE 2: REPLAY - Drive through trajectory with CARLA + GS side by side

COORDINATE SYSTEMS:
    CARLA:      Left-handed, Z-up, X-forward, Y-left
    UTM:        Right-handed, X=Easting, Y=Northing
    Nerfstudio: Right-handed, Y-forward (in this dataset), X-lateral, Z-up
"""

import carla
import numpy as np
import pygame
import os
import sys
import json
import math
import re
import torch
from pathlib import Path
from PIL import Image
from queue import Queue, Empty
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType

from utils.carla_simulator import (
    update_synchronous_mode,
    cleanup_old_sensors,
    spawn_sensor,
    get_xodr_projection_params,
)
from utils.save_data import get_map_data, get_dataset_data
from config import HERO_VEHICLE_TYPE, MAPS_FOLDER_NAME

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
CARLA_IP = '127.0.0.1'
CARLA_PORT = 2000
IM_WIDTH = 800
IM_HEIGHT = 503


# ==============================================================================
#  SLIDER CLASS
# ==============================================================================
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


# ==============================================================================
#  COORDINATE TRANSFORMS
# ==============================================================================

class CoordinateTransformer:
    """
    Handles the full transform chain: CARLA → UTM → Nerfstudio
    """

    def __init__(self, map_name, utm_to_nerfstudio_path):
        xodr_path = os.path.join(MAPS_FOLDER_NAME, map_name, "map.xodr")
        with open(xodr_path, 'r') as f:
            xodr_data = f.read()

        xodr_params = get_xodr_projection_params(xodr_data)
        self.xodr_offset = xodr_params["offset"]
        proj_string = xodr_params["geo_reference"].strip()
        if proj_string == "+proj=tmerc":
            proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

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
        print(f"[CoordinateTransformer] UTM +X angle in CARLA: {np.degrees(self.utm_x_angle_in_carla):.4f}°")

        with open(utm_to_nerfstudio_path, 'r') as f:
            tf = json.load(f)

        self.ns_scale = tf['scale']
        self.ns_rotation = np.array(tf['rotation'])
        self.ns_translation = np.array(tf['translation'])
        self.transform_mode = tf.get('mode', '2D')
        self.position_rotation_angle = np.arctan2(self.ns_rotation[1, 0], self.ns_rotation[0, 0])

        yaw_align = tf.get('yaw_alignment')
        if yaw_align:
            self.yaw_sign = yaw_align['yaw_sign']
            self.yaw_offset = yaw_align['yaw_offset_rad']
            self.use_orientation_yaw = True
            print(f"[CoordinateTransformer] ✅ Using orientation-based yaw: "
                  f"ns_yaw = {'+' if self.yaw_sign > 0 else '-'}utm_yaw + {np.degrees(self.yaw_offset):.2f}° "
                  f"(residual std: {yaw_align.get('residual_std_deg', '?')}°)")
        else:
            self.yaw_sign = -1
            self.yaw_offset = self.position_rotation_angle - math.pi / 2
            self.use_orientation_yaw = False
            print(f"[CoordinateTransformer] ⚠️ No yaw_alignment in JSON — using old formula: "
                  f"rotation_angle - π/2 = {np.degrees(self.yaw_offset):.2f}°")

        mount = tf.get('camera_mount_angles')
        if mount:
            self.avg_pitch = mount['avg_pitch_rad']
            self.avg_roll = mount['avg_roll_rad']
            print(f"[CoordinateTransformer] Camera mount: pitch={np.degrees(self.avg_pitch):.2f}°, "
                  f"roll={np.degrees(self.avg_roll):.2f}°")
        else:
            self.avg_pitch = 0.0
            self.avg_roll = 0.0

        print(f"[CoordinateTransformer] scale={self.ns_scale:.10f}, "
              f"position_rotation_angle={np.degrees(self.position_rotation_angle):.2f}°")

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

    def carla_to_utm(self, carla_x, carla_y):
        local_x = carla_x
        local_y = -carla_y
        proj_x = local_x - self.xodr_offset[0]
        proj_y = local_y - self.xodr_offset[1]
        lon, lat = self.transformer_proj_to_wgs84.transform(proj_x, proj_y)
        utm_easting, utm_northing = self.transformer_wgs84_to_utm.transform(lon, lat)
        return utm_easting, utm_northing

    def utm_to_nerfstudio(self, easting, northing, altitude=0.0):
        utm = np.array([easting, northing, altitude])
        return self.ns_scale * self.ns_rotation @ utm + self.ns_translation

    def carla_to_nerfstudio(self, carla_x, carla_y, carla_z=0.0):
        utm_e, utm_n = self.carla_to_utm(carla_x, carla_y)
        return self.utm_to_nerfstudio(utm_e, utm_n, 0.0)

    def transform_yaw_carla_to_nerfstudio(self, carla_yaw_rad):
        if self.use_orientation_yaw:
            utm_yaw = self.utm_x_angle_in_carla - carla_yaw_rad + math.pi / 2
            return self.yaw_sign * utm_yaw + self.yaw_offset
        else:
            return -carla_yaw_rad + self.position_rotation_angle - math.pi / 2


# ==============================================================================
#  SPLIT MODEL
# ==============================================================================

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

        if hasattr(coord_transformer, 'avg_pitch') and coord_transformer.avg_pitch != 0.0:
            self.avg_pitch = coord_transformer.avg_pitch
            self.avg_roll = coord_transformer.avg_roll
            print(f"   Camera mount angles (from JSON): pitch={np.degrees(self.avg_pitch):.2f}°, "
                  f"roll={np.degrees(self.avg_roll):.2f}°")
        else:
            yaws, pitches, rolls = [], [], []
            for i in range(training_cameras.shape[0]):
                y, p, r = extract_ypr_from_c2w(training_cameras[i])
                yaws.append(y)
                pitches.append(p)
                rolls.append(r)
            self.avg_pitch = float(np.median(pitches))
            self.avg_roll = float(np.median(rolls))
            print(f"   Camera mount angles (computed): pitch={np.degrees(self.avg_pitch):.2f}°, "
                  f"roll={np.degrees(self.avg_roll):.2f}° "
                  f"(std: p={np.degrees(np.std(pitches)):.2f}°, r={np.degrees(np.std(rolls)):.2f}°)")

        cam_positions = training_cameras[:, :3, 3]
        xy = cam_positions[:, :2]
        z = cam_positions[:, 2]

        self.z_linear = LinearNDInterpolator(xy, z)
        self.z_nearest = NearestNDInterpolator(xy, z)
        self.z_fallback = float(np.median(z))

        print(f"   Z interpolator: {len(z)} points, "
              f"Z range [{z.min():.4f}, {z.max():.4f}], "
              f"median={np.median(z):.4f}, "
              f"std={np.std(z):.4f}")

    def lookup_z(self, ns_x, ns_y):
        z = self.z_linear(ns_x, ns_y)
        if np.isnan(z):
            z = self.z_nearest(ns_x, ns_y)
        if np.isnan(z):
            z = self.z_fallback
        return float(z)

    def covers_frame(self, frame_id):
        return frame_id in self.frame_ids

    def distance_to_center(self, frame_id):
        return abs(frame_id - self.center_frame)

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
                    return Image.open(candidate).convert('RGB')
                except Exception as e:
                    print(f"   Warning: could not load training image {candidate}: {e}")

        print(f"   ⚠️ Training image not found for cam #{cam_idx}. Tried:")
        for c in candidates:
            print(f"      {c}")
        return None

    def get_first_training_frame_id(self):
        return self.cam_idx_to_frame_id.get(0, self.min_frame)


# ==============================================================================
#  BUILD c2w MATRIX FOR NERFSTUDIO
# ==============================================================================

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


def extract_yaw_from_c2w(c2w):
    forward = -c2w[:3, 2]
    return np.arctan2(forward[0], forward[1])


def extract_ypr_from_c2w(c2w):
    right   = c2w[:3, 0]
    up      = c2w[:3, 1]
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
#  RENDERING
# ==============================================================================

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


# ==============================================================================
#  HELPERS
# ==============================================================================

def extract_frame_number(filename):
    name = os.path.basename(str(filename))
    numbers = re.findall(r'\d+', os.path.splitext(name)[0])
    if numbers:
        return int(numbers[-1])
    return None


def find_best_split(frame_id, splits, last_split_idx=0):
    current_split = splits[last_split_idx]

    if current_split.min_frame <= frame_id <= current_split.max_frame:
        return last_split_idx

    covering = [(i, s) for i, s in enumerate(splits) if s.min_frame <= frame_id <= s.max_frame]

    if len(covering) == 1:
        return covering[0][0]

    if len(covering) > 1:
        return max(covering, key=lambda x: x[1].max_frame)[0]

    best_idx = last_split_idx
    best_dist = float('inf')
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


def auto_detect_splits(dataset_dir):
    splits = []

    splatfacto_dir = os.path.join(dataset_dir, "outputs", "splatfacto")
    if os.path.exists(splatfacto_dir):
        runs = sorted([d for d in os.listdir(splatfacto_dir)
                      if os.path.isdir(os.path.join(splatfacto_dir, d))])

        if len(runs) > 1:
            for run_name in runs:
                config_path = os.path.join(splatfacto_dir, run_name, "config.yml")
                if not os.path.exists(config_path):
                    continue

                sparse_idx = _parse_sparse_index(config_path)
                if sparse_idx is None:
                    continue

                split_name = f"split_{sparse_idx}"

                utm_transform = None
                for candidate in [
                    os.path.join(splatfacto_dir, run_name, "utm_to_nerfstudio_transform.json"),
                    os.path.join(dataset_dir, f"utm_to_nerfstudio_transform_split_{sparse_idx}.json"),
                    f"utm_to_nerfstudio_transform_split_{sparse_idx}.json",
                ]:
                    if os.path.exists(candidate):
                        utm_transform = candidate
                        break

                if utm_transform is None:
                    print(f"⚠️  Found run {run_name} (sparse/{sparse_idx}) but no UTM transform")
                    continue

                frame_positions = None
                for candidate in [
                    os.path.join(dataset_dir, f"frame_positions_split_{sparse_idx + 1}.txt"),
                    os.path.join(dataset_dir, f"frame_positions_{split_name}.txt"),
                    os.path.join(dataset_dir, "frame_positions.txt"),
                ]:
                    if os.path.exists(candidate):
                        frame_positions = candidate
                        break

                splits.append({
                    "name": split_name,
                    "gs_config": config_path,
                    "utm_transform": utm_transform,
                    "frame_positions": frame_positions,
                    "data_root": dataset_dir,
                })
                print(f"[INFO] Found {split_name} (run={run_name}, sparse/{sparse_idx})")

            if splits:
                splits.sort(key=lambda s: int(s["name"].split("_")[1]))
                return splits

    for split_num in range(0, 20):
        split_name = f"split_{split_num}"
        sfdir = os.path.join(dataset_dir, "outputs", f"splatfacto_{split_name}")
        if not os.path.exists(sfdir):
            continue

        runs = sorted([d for d in os.listdir(sfdir) if os.path.isdir(os.path.join(sfdir, d))])
        if not runs:
            continue
        gs_config = os.path.join(sfdir, runs[-1], "config.yml")
        if not os.path.exists(gs_config):
            continue

        utm_transform = None
        for candidate in [
            os.path.join(dataset_dir, f"utm_to_nerfstudio_transform_{split_name}.json"),
            f"utm_to_nerfstudio_transform_{split_name}.json",
        ]:
            if os.path.exists(candidate):
                utm_transform = candidate
                break
        if utm_transform is None:
            continue

        frame_positions = None
        for candidate in [
            os.path.join(dataset_dir, f"frame_positions_{split_name}.txt"),
            os.path.join(dataset_dir, "frame_positions.txt"),
        ]:
            if os.path.exists(candidate):
                frame_positions = candidate
                break

        splits.append({
            "name": split_name,
            "gs_config": gs_config,
            "utm_transform": utm_transform,
            "frame_positions": frame_positions,
            "data_root": dataset_dir,
        })
        print(f"[INFO] Found {split_name}: config={gs_config}")

    if splits:
        return splits

    if os.path.exists(splatfacto_dir):
        runs = sorted([d for d in os.listdir(splatfacto_dir)
                      if os.path.isdir(os.path.join(splatfacto_dir, d))])
        if runs:
            gs_config = os.path.join(splatfacto_dir, runs[-1], "config.yml")
            if os.path.exists(gs_config):
                utm_transform = None
                for candidate in [
                    os.path.join(dataset_dir, "utm_to_nerfstudio_transform.json"),
                    "utm_to_nerfstudio_transform.json",
                ]:
                    if os.path.exists(candidate):
                        utm_transform = candidate
                        break
                if utm_transform:
                    splits.append({
                        "name": "full",
                        "gs_config": gs_config,
                        "utm_transform": utm_transform,
                        "frame_positions": os.path.join(dataset_dir, "frame_positions.txt"),
                        "data_root": dataset_dir,
                    })
                    print(f"[INFO] Found single model: {gs_config}")

    return splits


def _parse_sparse_index(config_path):
    try:
        with open(config_path, 'r') as f:
            text = f.read()
        match = re.search(r"- sparse\s*\n\s*- '?(\d+)'?", text)
        if match:
            return int(match.group(1))
        match = re.search(r"colmap/sparse/(\d+)", text)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None


def read_frame_ids_from_positions(filepath):
    frame_ids = []
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                try:
                    frame_ids.append(int(parts[0].strip()))
                except (ValueError, IndexError):
                    continue
    return frame_ids


def load_split_models(split_configs, map_name, fov=90.0):
    split_models = []

    for cfg in split_configs:
        name = cfg["name"]
        gs_config = cfg["gs_config"]
        utm_transform = cfg["utm_transform"]
        data_root = cfg["data_root"]

        print(f"\n{'='*60}")
        print(f"  Loading split: {name}")
        print(f"{'='*60}")

        coord_transformer = CoordinateTransformer(map_name, utm_transform)

        config_path = Path(gs_config).resolve()
        data_root_abs = Path(data_root).resolve()

        print(f"🔮 Loading GS Model from: {config_path}")
        original_cwd = os.getcwd()
        try:
            os.chdir(data_root_abs)

            _, pipeline, _, step = eval_setup(config_path, test_mode="inference")
            pipeline.model.eval()

            print(f"   Loaded checkpoint at step {step}")

            if hasattr(pipeline.model, 'num_points'):
                print(f"   Number of gaussians: {pipeline.model.num_points}")
            elif hasattr(pipeline.model, 'means') and pipeline.model.means is not None:
                print(f"   Number of gaussians: {pipeline.model.means.shape[0]}")

            dp = pipeline.datamanager.train_dataparser_outputs
            training_cameras = dp.cameras.camera_to_worlds.cpu().numpy()
            training_filenames = dp.image_filenames

            training_frame_ids = []
            for fn in training_filenames:
                fid = extract_frame_number(fn)
                if fid is not None:
                    training_frame_ids.append(fid)

            positions_frame_ids = read_frame_ids_from_positions(cfg.get("frame_positions"))
            all_frame_ids = sorted(set(training_frame_ids) | set(positions_frame_ids))

            print(f"✅ {name}: {training_cameras.shape[0]} cameras, "
                  f"frames [{min(all_frame_ids)}-{max(all_frame_ids)}] "
                  f"({len(all_frame_ids)} frame IDs)")

            print(f"   First training image: {training_filenames[0]}")
            first_frame_id = extract_frame_number(training_filenames[0])
            print(f"   First training frame ID: {first_frame_id}")

            first_c2w = np.eye(4)
            first_c2w[:3, :] = training_cameras[0]
            test_img = render_gs(pipeline, first_c2w, 160, 100, fov)
            test_arr = np.array(test_img)
            print(f"   Verification render: mean={test_arr.mean():.1f}, "
                  f"std={test_arr.std():.1f}, "
                  f"black={'YES ⚠️' if test_arr.mean() < 10 else 'no'}")

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
            print(f"❌ Error loading {name}: {e}")
            import traceback
            traceback.print_exc()
            os.chdir(original_cwd)
            continue

        os.chdir(original_cwd)

    if len(split_models) > 1:
        print(f"\n{'='*60}")
        print(f"  VERIFYING MODEL INDEPENDENCE")
        print(f"{'='*60}")
        for i, sm in enumerate(split_models):
            first_c2w = np.eye(4)
            first_c2w[:3, :] = sm.training_cameras[0]
            test_img = render_gs(sm.pipeline, first_c2w, 160, 100, fov)
            test_arr = np.array(test_img)

            if hasattr(sm.pipeline.model, 'means') and sm.pipeline.model.means is not None:
                means_hash = sm.pipeline.model.means[:5].sum().item()
            else:
                means_hash = 0

            print(f"   {sm.name}: render mean={test_arr.mean():.1f}, "
                  f"std={test_arr.std():.1f}, "
                  f"gaussians_hash={means_hash:.4f}, "
                  f"model_id={id(sm.pipeline.model)}")

        if id(split_models[0].pipeline.model) == id(split_models[1].pipeline.model):
            print(f"\n   ❌ WARNING: Both splits share the SAME model object!")
            print(f"      This means eval_setup overwrote the first model.")
            print(f"      The models are NOT independent.")
        else:
            print(f"\n   ✅ Models are different objects (independent)")

    return split_models


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replay with Nerfstudio-trained GS (multi-split)")
    parser.add_argument("--map", required=True, help="Map name")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--gs_config", type=str, default=None,
                        help="(Backwards compat) Single config path. Use --gs_configs for multi-split.")
    parser.add_argument("--gs_configs", type=str, nargs='+', default=None,
                        help="One or more nerfstudio config.yml paths (multi-split)")
    parser.add_argument("--gs_data_root", type=str, default=None,
                        help="GS data root directory (shared across all splits)")
    parser.add_argument("--utm_transform", type=str, default=None,
                        help="(Backwards compat) Single transform path. Use --utm_transforms for multi-split.")
    parser.add_argument("--utm_transforms", type=str, nargs='+', default=None,
                        help="UTM transform JSONs, one per config. If omitted, auto-resolved from config dir.")
    parser.add_argument("--only_carla", action="store_true", help="Run without GS model")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to render (default: all)")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip Phase 1 free camera calibration (use zero offsets)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for saved frames")
    args = parser.parse_args()

    dataset_data = get_dataset_data(args.dataset)
    fov = float(dataset_data["camera"]["fov"])
    only_carla = args.only_carla
    dataset_dir = os.path.join("datasets", args.dataset)

    # ---- Discover and load split models ----
    split_models = []
    if not only_carla:
        # Merge old single-arg into new multi-arg for backwards compatibility
        gs_configs = args.gs_configs or ([args.gs_config] if args.gs_config else None)
        utm_transforms = args.utm_transforms or ([args.utm_transform] if args.utm_transform else None)

        if gs_configs:
            # --- Manual multi-config mode ---
            # Auto-resolve utm transforms from config dirs if not provided
            if utm_transforms is None:
                utm_transforms = []
                for cfg_path in gs_configs:
                    cfg_dir = os.path.dirname(cfg_path)
                    tf_path = os.path.join(cfg_dir, "utm_to_nerfstudio_transform.json")
                    if not os.path.exists(tf_path):
                        print(f"ERROR: No utm_transform found at {tf_path}")
                        print(f"       Provide --utm_transforms explicitly.")
                        sys.exit(1)
                    utm_transforms.append(tf_path)

            if len(utm_transforms) == 1 and len(gs_configs) > 1:
                utm_transforms = utm_transforms * len(gs_configs)

            if len(utm_transforms) != len(gs_configs):
                print(f"ERROR: Got {len(gs_configs)} configs but {len(utm_transforms)} transforms")
                sys.exit(1)

            data_root = args.gs_data_root or dataset_dir

            print(f"[INFO] Using {len(gs_configs)} manually specified model(s)")
            split_configs = []
            for i, (cfg_path, tf_path) in enumerate(zip(gs_configs, utm_transforms)):
                # Try to parse split number from config path
                split_num = None
                match = re.search(r'split(\d+)', cfg_path)
                if match:
                    split_num = int(match.group(1))

                # Find frame_positions for this split
                frame_positions = None
                if split_num is not None:
                    for candidate in [
                        os.path.join(data_root, f"frame_positions_split_{split_num}_1_of_2.txt"),
                        os.path.join(dataset_dir, f"frame_positions_split_{split_num}_1_of_2.txt"),
                    ]:
                        if os.path.exists(candidate):
                            frame_positions = candidate
                            break
                if frame_positions is None:
                    frame_positions = os.path.join(dataset_dir, "frame_positions.txt")

                name = f"split_{split_num}" if split_num is not None else f"model_{i}"

                split_configs.append({
                    "name": name,
                    "gs_config": cfg_path,
                    "utm_transform": tf_path,
                    "frame_positions": frame_positions,
                    "data_root": data_root,
                })
                print(f"   {name}: {cfg_path}")
        else:
            # Auto-detect splits
            print("[INFO] Auto-detecting split models...")
            split_configs = auto_detect_splits(dataset_dir)

        if split_configs:
            split_models = load_split_models(split_configs, args.map, fov=fov)

        if not split_models:
            print("⚠️  No GS models loaded — falling back to only_carla mode")
            only_carla = True

    if split_models:
        print(f"\n✅ Loaded {len(split_models)} split model(s):")
        for sm in split_models:
            print(f"   {sm.name}: frames [{sm.min_frame}-{sm.max_frame}], "
                  f"Z range [{sm.training_cameras[:, 2, 3].min():.4f}, {sm.training_cameras[:, 2, 3].max():.4f}]")

    # ---- Connect to CARLA ----
    print(f"\n🔌 Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)

    map_data = get_map_data(args.map, (IM_WIDTH, IM_HEIGHT))

    # ---- Load trajectory ----
    traj_path = os.path.join("maps", args.map, "trajectory_positions_rear_odom_yaw.json")
    with open(traj_path, 'r') as f:
        trajectory_points = json.load(f)
    print(f"📂 Loaded {len(trajectory_points)} trajectory points.")

    trajectory_by_frame = {}
    for tp in trajectory_points:
        trajectory_by_frame[tp["frame_id"]] = tp

    # ---- Sync mode ----
    update_synchronous_mode(world, tm, True, dataset_data["camera"]["fps"])
    world.tick()

    # ---- Spawn hero ----
    first_pt = trajectory_points[0]["transform"]
    start_transform = carla.Transform(
        carla.Location(x=first_pt["location"]["x"], y=first_pt["location"]["y"],
                       z=first_pt["location"]["z"] + 0.5),
        carla.Rotation(pitch=0, yaw=first_pt["rotation"]["yaw"], roll=0)
    )

    bp_lib = world.get_blueprint_library()

    hero_vehicle = None
    all_vehicles = world.get_actors().filter('vehicle.*')
    for v in all_vehicles:
        attrs = v.attributes
        if attrs.get('role_name', '') == 'hero':
            hero_vehicle = v
            print(f"♻️  Found existing hero vehicle (id={v.id}, type={v.type_id})")
            break

    if hero_vehicle is None:
        for v in all_vehicles:
            if v.type_id == HERO_VEHICLE_TYPE:
                hero_vehicle = v
                print(f"♻️  Found existing vehicle of type {HERO_VEHICLE_TYPE} (id={v.id})")
                break

    if hero_vehicle:
        hero_vehicle.set_transform(start_transform)
    else:
        hero_bp = bp_lib.find(HERO_VEHICLE_TYPE)
        hero_bp.set_attribute('role_name', 'hero')
        hero_vehicle = world.spawn_actor(hero_bp, start_transform)
        print(f"🚗 Spawned new hero vehicle (id={hero_vehicle.id}, type={HERO_VEHICLE_TYPE})")

    hero_vehicle.set_simulate_physics(False)
    hero_vehicle.set_autopilot(False)
    cleanup_old_sensors(hero_vehicle)

    # ---- Sensor ----
    cam_config = dataset_data["camera"]
    sensor_tf = carla.Transform(
        carla.Location(x=cam_config["position"]["x"], y=cam_config["position"]["y"],
                       z=cam_config["position"]["z"]),
        carla.Rotation(pitch=cam_config.get("pitch", 0.0))
    )
    rgb_sensor, rgb_queue = spawn_sensor(
        bp_lib, 'sensor.camera.rgb', IM_WIDTH, IM_HEIGHT, str(fov),
        world, sensor_tf, hero_vehicle
    )
    world.tick()

    # ---- Pygame ----
    pygame.init()
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    # ==================================================================
    #  PHASE 1: FREE CAMERA CALIBRATION (per split)
    # ==================================================================
    calibration_offsets = {}

    if split_models and not args.skip_calibration:
        cal_win_w = IM_WIDTH * 2
        cal_win_h = IM_HEIGHT * 2
        screen = pygame.display.set_mode((cal_win_w, cal_win_h))

        for sm in split_models:
            first_train_frame_id = sm.get_first_training_frame_id()

            print(f"\n{'='*60}")
            print(f"  CALIBRATING: {sm.name}")
            print(f"  Frames [{sm.min_frame}-{sm.max_frame}]")
            print(f"  First training camera → frame_id={first_train_frame_id}")
            print(f"  WASD=move, Mouse=look, QE=up/down, Scroll=speed")
            print(f"  [/]=prev/next training cam, T=test, P=print, R=reset, ENTER=next split")
            print(f"{'='*60}")

            pygame.display.set_caption(f"Calibrating: {sm.name} | 4-Panel View")

            current_train_cam_idx = 0

            def teleport_to_training_cam(cam_idx):
                fid = sm.cam_idx_to_frame_id.get(cam_idx)
                if fid is None:
                    print(f"   ⚠️ Training camera {cam_idx} has no frame ID mapping")
                    return fid
                tp = trajectory_by_frame.get(fid)
                if tp is None:
                    print(f"   ⚠️ Frame {fid} not found in trajectory — cannot teleport CARLA")
                    return fid
                cal_pt = tp["transform"]
                hero_vehicle.set_transform(carla.Transform(
                    carla.Location(x=cal_pt["location"]["x"], y=cal_pt["location"]["y"],
                                   z=cal_pt["location"]["z"]),
                    carla.Rotation(pitch=0, yaw=cal_pt["rotation"]["yaw"], roll=0)
                ))
                world.tick()
                try:
                    rgb_queue.get(block=True, timeout=1.0)
                except Empty:
                    pass
                print(f"   Teleported CARLA to frame {fid} (training cam #{cam_idx})")
                return fid

            current_frame_id = teleport_to_training_cam(current_train_cam_idx)

            def get_reference_images(cam_idx):
                orig_img = sm.get_training_image(cam_idx)
                if orig_img is not None:
                    orig_img = orig_img.resize((IM_WIDTH, IM_HEIGHT), Image.LANCZOS)
                else:
                    orig_img = Image.new('RGB', (IM_WIDTH, IM_HEIGHT), (60, 20, 20))

                train_c2w = sm.get_training_cam_c2w(cam_idx)
                gs_train_img = render_gs(sm.pipeline, train_c2w, IM_WIDTH, IM_HEIGHT, fov)

                return orig_img, gs_train_img

            ref_orig, ref_gs_train = get_reference_images(current_train_cam_idx)

            first_c2w = sm.training_cameras[current_train_cam_idx]
            cam_pos = first_c2w[:, 3].copy().astype(np.float64)
            cam_yaw, cam_pitch, cam_roll = extract_ypr_from_c2w(first_c2w)
            print(f"   Extracted YPR: yaw={math.degrees(cam_yaw):.2f}°, "
                  f"pitch={math.degrees(cam_pitch):.2f}°, roll={math.degrees(cam_roll):.2f}°")
            move_speed = 0.05
            look_sensitivity = 0.003
            mouse_dragging = False
            last_mouse_pos = (0, 0)

            slider_width = 200
            slider_height = 15
            slider_x = IM_WIDTH + 20
            sliders = [
                Slider("Yaw Offset", slider_x, 100, slider_width, slider_height, -180.0, 180.0, 0.0),
                Slider("Pitch Offset", slider_x, 150, slider_width, slider_height, -45.0, 45.0, 0.0),
                Slider("Roll Offset", slider_x, 200, slider_width, slider_height, -45.0, 45.0, 0.0),
                Slider("X Offset", slider_x, 270, slider_width, slider_height, -0.02, 0.02, 0.0),
                Slider("Y Offset", slider_x, 320, slider_width, slider_height, -0.02, 0.02, 0.0),
                Slider("Z Offset", slider_x, 370, slider_width, slider_height, -0.02, 0.02, 0.0),
            ]

            calibrating = True
            while calibrating:
                yaw_offset_deg = sliders[0].val
                pitch_offset_deg = sliders[1].val
                roll_offset_deg = sliders[2].val
                x_offset = sliders[3].val
                y_offset = sliders[4].val
                z_offset = sliders[5].val

                adjusted_pos = cam_pos + np.array([x_offset, y_offset, z_offset])
                adjusted_yaw = cam_yaw + math.radians(yaw_offset_deg)
                adjusted_pitch = cam_pitch + math.radians(pitch_offset_deg)
                adjusted_roll = cam_roll + math.radians(roll_offset_deg)

                c2w_free = build_nerfstudio_c2w(adjusted_pos, adjusted_yaw, adjusted_pitch, adjusted_roll)
                gs_pil = render_gs(sm.pipeline, c2w_free, IM_WIDTH, IM_HEIGHT, fov)

                world.tick()
                try:
                    rgb_data = rgb_queue.get(block=True, timeout=0.5)
                    rgb_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape(
                        (rgb_data.height, rgb_data.width, 4))[:, :, :3][:, :, ::-1]
                    carla_pil = Image.fromarray(rgb_np)
                except Empty:
                    carla_pil = Image.new('RGB', (IM_WIDTH, IM_HEIGHT), (30, 30, 30))

                screen.fill((30, 30, 30))

                screen.blit(pygame.image.fromstring(carla_pil.tobytes(), carla_pil.size, carla_pil.mode), (0, 0))
                screen.blit(pygame.image.fromstring(gs_pil.tobytes(), gs_pil.size, gs_pil.mode), (IM_WIDTH, 0))
                screen.blit(pygame.image.fromstring(ref_orig.tobytes(), ref_orig.size, ref_orig.mode), (0, IM_HEIGHT))
                screen.blit(pygame.image.fromstring(ref_gs_train.tobytes(), ref_gs_train.size, ref_gs_train.mode), (IM_WIDTH, IM_HEIGHT))

                screen.blit(font.render("CARLA Ground Truth", True, (0, 255, 0)), (10, 10))
                screen.blit(font.render(f"GS Free Cam: {sm.name}", True, (0, 255, 255)), (IM_WIDTH + 10, 10))

                train_fid = sm.cam_idx_to_frame_id.get(current_train_cam_idx, "?")
                screen.blit(font.render(f"Original Training Image (cam #{current_train_cam_idx}, frame {train_fid})",
                                        True, (255, 200, 0)), (10, IM_HEIGHT + 10))
                screen.blit(font.render(f"GS from Training Pose (cam #{current_train_cam_idx}, frame {train_fid})",
                                        True, (255, 150, 255)), (IM_WIDTH + 10, IM_HEIGHT + 10))

                pos_info = f"Pos: ({adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}, {adjusted_pos[2]:.3f})"
                screen.blit(font.render(pos_info, True, (255, 255, 0)), (IM_WIDTH + 10, 30))
                rot_info = f"Yaw:{math.degrees(adjusted_yaw):.1f}° P:{math.degrees(adjusted_pitch):.1f}° R:{math.degrees(adjusted_roll):.1f}°"
                screen.blit(font.render(rot_info, True, (255, 255, 0)), (IM_WIDTH + 10, 50))
                screen.blit(font.render(f"Speed: {move_speed:.3f}", True, (255, 255, 0)), (IM_WIDTH + 10, 70))

                for s in sliders:
                    s.draw(screen, font)

                help_y = cal_win_h - 40
                screen.blit(font.render("WASD=move Mouse=look QE=up/down Scroll=speed [/]=prev/next cam",
                                        True, (0, 255, 0)), (10, help_y))
                screen.blit(font.render(f"T=test P=print R=reset ENTER=next ({sm.name})",
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
                                current_frame_id = teleport_to_training_cam(current_train_cam_idx)
                                ref_orig, ref_gs_train = get_reference_images(current_train_cam_idx)
                                tc2w = sm.training_cameras[current_train_cam_idx]
                                cam_pos = tc2w[:, 3].copy().astype(np.float64)
                                cam_yaw, cam_pitch, cam_roll = extract_ypr_from_c2w(tc2w)
                                print(f"   → Training cam #{current_train_cam_idx} (frame {current_frame_id}) "
                                      f"YPR: {math.degrees(cam_yaw):.2f}°, {math.degrees(cam_pitch):.2f}°, {math.degrees(cam_roll):.2f}°")
                        elif event.key == pygame.K_LEFTBRACKET:
                            if current_train_cam_idx > 0:
                                current_train_cam_idx -= 1
                                current_frame_id = teleport_to_training_cam(current_train_cam_idx)
                                ref_orig, ref_gs_train = get_reference_images(current_train_cam_idx)
                                tc2w = sm.training_cameras[current_train_cam_idx]
                                cam_pos = tc2w[:, 3].copy().astype(np.float64)
                                cam_yaw, cam_pitch, cam_roll = extract_ypr_from_c2w(tc2w)
                                print(f"   → Training cam #{current_train_cam_idx} (frame {current_frame_id}) "
                                      f"YPR: {math.degrees(cam_yaw):.2f}°, {math.degrees(cam_pitch):.2f}°, {math.degrees(cam_roll):.2f}°")
                        elif event.key == pygame.K_p:
                            print(f"\n[CAMERA STATE - {sm.name}]")
                            print(f"  Position: [{adjusted_pos[0]:.6f}, {adjusted_pos[1]:.6f}, {adjusted_pos[2]:.6f}]")
                            print(f"  Yaw: {math.degrees(adjusted_yaw):.2f}°  Pitch: {math.degrees(adjusted_pitch):.2f}°  Roll: {math.degrees(adjusted_roll):.2f}°")
                            print(f"  Training cam: #{current_train_cam_idx} (frame {train_fid})")
                            raw_c2w = sm.training_cameras[current_train_cam_idx]
                            print(f"\n  Raw training c2w (3x4):")
                            print(f"    col0 (right?):   [{raw_c2w[0,0]:.6f}, {raw_c2w[1,0]:.6f}, {raw_c2w[2,0]:.6f}]")
                            print(f"    col1 (up?):      [{raw_c2w[0,1]:.6f}, {raw_c2w[1,1]:.6f}, {raw_c2w[2,1]:.6f}]")
                            print(f"    col2 (-forward?):[{raw_c2w[0,2]:.6f}, {raw_c2w[1,2]:.6f}, {raw_c2w[2,2]:.6f}]")
                            print(f"    col3 (position): [{raw_c2w[0,3]:.6f}, {raw_c2w[1,3]:.6f}, {raw_c2w[2,3]:.6f}]")
                            raw_yaw, raw_pitch, raw_roll = extract_ypr_from_c2w(raw_c2w)
                            print(f"  Extracted YPR: yaw={math.degrees(raw_yaw):.4f}° pitch={math.degrees(raw_pitch):.4f}° roll={math.degrees(raw_roll):.4f}°")
                            rebuilt = build_nerfstudio_c2w(cam_pos, raw_yaw, raw_pitch, raw_roll)
                            print(f"  Rebuilt c2w (3x4) from extracted YPR:")
                            print(f"    col0 (right):    [{rebuilt[0,0]:.6f}, {rebuilt[1,0]:.6f}, {rebuilt[2,0]:.6f}]")
                            print(f"    col1 (up):       [{rebuilt[0,1]:.6f}, {rebuilt[1,1]:.6f}, {rebuilt[2,1]:.6f}]")
                            print(f"    col2 (-forward): [{rebuilt[0,2]:.6f}, {rebuilt[1,2]:.6f}, {rebuilt[2,2]:.6f}]")
                            print(f"    col3 (position): [{rebuilt[0,3]:.6f}, {rebuilt[1,3]:.6f}, {rebuilt[2,3]:.6f}]")
                            diff = np.abs(raw_c2w[:3,:3] - rebuilt[:3,:3]).max()
                            print(f"  Max rotation matrix difference: {diff:.8f}")
                        elif event.key == pygame.K_t:
                            cam_tf = rgb_sensor.get_transform()
                            cx, cy_c = cam_tf.location.x, cam_tf.location.y
                            cyw = math.radians(cam_tf.rotation.yaw)
                            ns_p = sm.coord_transformer.carla_to_nerfstudio(cx, cy_c)
                            ns_z = sm.lookup_z(ns_p[0], ns_p[1])
                            ns_y = sm.coord_transformer.transform_yaw_carla_to_nerfstudio(cyw)
                            print(f"\n[TRANSFORM TEST - {sm.name}]")
                            print(f"  CARLA: ({cx:.4f}, {cy_c:.4f}) yaw={math.degrees(cyw):.1f}°")
                            print(f"  NS pred: ({ns_p[0]:.4f}, {ns_p[1]:.4f}, Z_interp={ns_z:.4f}) yaw={math.degrees(ns_y):.1f}°")
                            print(f"  Free cam: ({adjusted_pos[0]:.4f}, {adjusted_pos[1]:.4f}, {adjusted_pos[2]:.4f}) yaw={math.degrees(adjusted_yaw):.1f}°")
                            print(f"  Training cam: #{current_train_cam_idx} (frame {train_fid})")
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
                            cam_pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01,
                                            cam_pitch - dy * look_sensitivity))
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
            print(f"✅ {sm.name} calibrated: pos_offset=({pos_offset[0]:.4f}, {pos_offset[1]:.4f}, {pos_offset[2]:.4f}), "
                  f"yaw_offset={math.degrees(yaw_offset):.2f}°, z_offset_from_interp={z_offset_from_interp:.4f}")

    if args.skip_calibration and split_models:
        for sm in split_models:
            first_cam_idx = 0
            cam_pos_train = sm.training_cameras[first_cam_idx, :, 3].copy()

            fid = sm.cam_idx_to_frame_id.get(first_cam_idx)
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
                      f"({pos_offset[0]:.4f}, {pos_offset[1]:.4f}, {pos_offset[2]:.4f})")

    # ==================================================================
    #  PHASE 2: REPLAY WITH AUTO-SWITCHING
    # ==================================================================
    win_w = IM_WIDTH * 2
    win_h = IM_HEIGHT
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("GS Replay (Multi-Split) | Driving")

    start_idx = 0
    print(f"\n🚀 Replaying from frame 0 with {len(split_models)} split(s)...")

    SAVE_BASE_DIR = args.output_dir or "/media/davidejannussi/ssd space/VALIDATION_RESULTS_GS_NERF"
    save_flag = 1
    if save_flag == 1:
        if args.gs_config:
            run_name = f"{Path(args.gs_config).parts[-4]}__{Path(args.gs_config).parts[-2]}"
        else:
            run_name = "auto"
        save_dir_carla = os.path.join(SAVE_BASE_DIR, run_name, "carla")
        save_dir_gs = os.path.join(SAVE_BASE_DIR, run_name, "gs")
        save_dir_combined = os.path.join(SAVE_BASE_DIR, run_name, "combined")
        os.makedirs(save_dir_carla, exist_ok=True)
        os.makedirs(save_dir_gs, exist_ok=True)
        os.makedirs(save_dir_combined, exist_ok=True)
        print(f"📁 Saving frames to: {os.path.join(SAVE_BASE_DIR, run_name)}")
    current_split_idx = 0

    start_pt = trajectory_points[0]["transform"]
    hero_vehicle.set_transform(carla.Transform(
        carla.Location(x=start_pt["location"]["x"], y=start_pt["location"]["y"],
                       z=start_pt["location"]["z"]),
        carla.Rotation(pitch=0, yaw=start_pt["rotation"]["yaw"], roll=0)
    ))
    world.tick()
    try:
        rgb_queue.get(block=True, timeout=1.0)
    except Empty:
        pass
    print("📍 Teleported to trajectory frame 0")

    try:
        end_idx = len(trajectory_points)
        if args.max_frames is not None:
            end_idx = min(start_idx + args.max_frames, end_idx)
        print(f"   Running {end_idx - start_idx} frames (max_frames={args.max_frames})")
        for idx in range(start_idx, end_idx):
            point = trajectory_points[idx]
            frame_id = point["frame_id"]
            pt = point["transform"]
            carla_yaw_deg = pt["rotation"]["yaw"]

            offset_distance = 0.13
            yaw_rad = math.radians(carla_yaw_deg)
            ox = -offset_distance * math.cos(yaw_rad)
            oy = -offset_distance * math.sin(yaw_rad)

            hero_vehicle.set_transform(carla.Transform(
                carla.Location(x=pt["location"]["x"] + ox, y=pt["location"]["y"] + oy,
                               z=pt["location"]["z"]),
                carla.Rotation(pitch=0.0, yaw=carla_yaw_deg, roll=0.0)
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

            if split_models:
                new_split_idx = find_best_split(frame_id, split_models, current_split_idx)
                if new_split_idx != current_split_idx:
                    print(f"[Frame {idx}] 🔄 Switching: {split_models[current_split_idx].name} → {split_models[new_split_idx].name}")
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
                carla_z = cam_tf.location.z
                carla_yaw_deg_cam = cam_tf.rotation.yaw
                carla_yaw_rad = math.radians(cam_tf.rotation.yaw)

                ns_pos_raw = sm.coord_transformer.carla_to_nerfstudio(carla_x, carla_y)
                ns_pos_raw[2] = sm.lookup_z(ns_pos_raw[0], ns_pos_raw[1])

                ns_pos = ns_pos_raw + offsets["pos_offset"]
                ns_pos[2] = ns_pos_raw[2] + offsets.get("z_offset", 0.0)

                ns_yaw_raw = sm.coord_transformer.transform_yaw_carla_to_nerfstudio(carla_yaw_rad)
                ns_yaw = ns_yaw_raw + offsets["yaw_offset"]

                ns_pitch = sm.avg_pitch + offsets["pitch_offset"]
                ns_roll = sm.avg_roll + offsets["roll_offset"]

                if idx % 100 == 0:
                    print(f"[Frame {idx}] {sm.name} | CARLA: ({carla_x:.2f}, {carla_y:.2f}) "
                          f"yaw={math.degrees(carla_yaw_rad):.1f}° → NS: ({ns_pos[0]:.4f}, {ns_pos[1]:.4f}, {ns_pos[2]:.4f}) "
                          f"yaw={math.degrees(ns_yaw):.1f}° pitch={math.degrees(ns_pitch):.1f}° roll={math.degrees(ns_roll):.1f}° "
                          f"[Z interp={ns_pos_raw[2]:.4f}]")

                c2w = build_nerfstudio_c2w(ns_pos, ns_yaw, ns_pitch, ns_roll)
                gs_pil = render_gs(sm.pipeline, c2w, IM_WIDTH, IM_HEIGHT, fov)

            combined = Image.new('RGB', (win_w, IM_HEIGHT))
            combined.paste(carla_pil, (0, 0))
            if gs_pil:
                combined.paste(gs_pil, (IM_WIDTH, 0))

            if save_flag == 1:
                carla_pil.save(os.path.join(save_dir_carla, f"frame_{frame_id:06d}.png"))
                if gs_pil:
                    gs_pil.save(os.path.join(save_dir_gs, f"frame_{frame_id:06d}.png"))
                combined.save(os.path.join(save_dir_combined, f"frame_{frame_id:06d}.jpg"), quality=95)

            screen.blit(pygame.image.fromstring(combined.tobytes(), combined.size, combined.mode), (0, 0))
            screen.blit(font.render(f"Frame {idx}/{len(trajectory_points)} | ID: {frame_id}", True, (0, 255, 0)), (10, 10))
            screen.blit(font.render(f"Split: {active_split_name}", True, (255, 200, 0)), (IM_WIDTH + 10, 10))
            if 'cam_tf' in locals():
                cam_loc = cam_tf.location
                cam_rot = cam_tf.rotation
                screen.blit(
                    font.render(
                        f"CARLA CAM XYZ: {cam_loc.x:+.2f}, {cam_loc.y:+.2f}, {cam_loc.z:+.2f}",
                        True, (0, 255, 0)
                    ),
                    (10, 30)
                )
                screen.blit(
                    font.render(
                        f"CARLA CAM RPY: {cam_rot.roll:+.1f}  {cam_rot.pitch:+.1f}  {cam_rot.yaw:+.1f}",
                        True, (0, 255, 0)
                    ),
                    (10, 50)
                )
                if split_models:
                    screen.blit(
                        font.render(
                            f"NS Z(interp): {ns_pos_raw[2]:.4f} + offset {offsets.get('z_offset', 0.0):.4f} = {ns_pos[2]:.4f}",
                            True, (255, 150, 0)
                        ),
                        (IM_WIDTH + 10, 30)
                    )

            veh_tf = hero_vehicle.get_transform()
            screen.blit(
                font.render(
                    f"CARLA VEH XYZ: {veh_tf.location.x:+.2f}, {veh_tf.location.y:+.2f}, {veh_tf.location.z:+.2f}",
                    True, (0, 200, 255)
                ),
                (10, 70)
            )
            screen.blit(
                font.render(
                    f"CARLA VEH YAW: {veh_tf.rotation.yaw:+.1f}°",
                    True, (0, 200, 255)
                ),
                (10, 90)
            )
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt

            clock.tick(60)

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        if 'rgb_sensor' in locals() and rgb_sensor:
            rgb_sensor.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print("👋 Done.")


if __name__ == '__main__':
    main()