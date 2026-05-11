#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4-log_gs_new.py

DAVE-2 autonomous driving with Gaussian Splatting / Nerfstudio rendering,
adapted to the cam2sim data layout.

Reads from (project root):
    data/processed_dataset/<BAG>/maps/map.xodr
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json
    data/data_for_carla/<BAG>/camera.json
    data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_N/splatfacto/<TS>/config.yml
    data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_N/splatfacto/<TS>/utm_to_nerfstudio_transform.json
    data/data_for_gaussian_splatting/<BAG>/frame_positions_split_N_1_of_K.txt

Workflow:
  1. Start CARLA server.
  2. Run prepare_carla_world.py once (loads xodr, spawns hero + parked cars).
  3. Start DAVE-2 server (separate `dave_2` conda env, port hardcoded in
     utils/dave2_connection.py).
  4. Run THIS script. It re-uses the existing hero vehicle in CARLA.

Phases:
  PHASE 1: 4-panel calibration GUI (CARLA, GS free cam, original training image,
           GS rendered from training pose). Per split.
  PHASE 2: DAVE-2 closed-loop drive. GS render -> DAVE-2 -> steer -> ackermann.
           Hero is teleported to first training camera with the proper
           road-waypoint z + back-offset, then stabilized and given a
           100-tick warmup launch (mirrors the only_carla script that
           works) before the drive loop starts.
           Termination: fall, stuck, out-of-coverage, or max_frames.

Coordinate chain (per split):
    CARLA local coords -> UTM (inverse XODR projection) -> Nerfstudio (similarity transform)
"""

import os
import sys
import json
import math
import re
import time
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
from utils.dave2_connection import connect_to_dave2_server, send_image_over_connection


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
    PROJECT_ROOT, "data", "data_for_carla", BAG_NAME, "drive_results"
)

IM_WIDTH = 800
IM_HEIGHT = 503

# Drive control
DRIVE_SPEED_KMH = 10.0          # constant forward speed for ackermann
PREDICT_EVERY = 1               # call DAVE-2 every N frames

# Hero startup (mirrors the only_carla script that works)
LAUNCH_SPEED_KMH = 12.0           # speed for warmup launch (kick-starts ackermann)
WARMUP_TICKS = 100                # ticks to apply launch control
REAR_TO_CENTER_OFFSET_METERS = 0.13   # rear axle -> car center back-offset
HERO_SPAWN_Z_OFFSET = 0.10        # +z above road waypoint (avoid clipping)

# Termination thresholds
STUCK_THRESHOLD = 0.02          # meters / frame -> below = not moving
STUCK_FRAME_LIMIT = 50          # consecutive stuck frames before terminate
MIN_Z_THRESHOLD = -0.5          # fall detection (meters)
COVERAGE_THRESHOLD = 0.15       # nerfstudio units, distance to nearest train cam
COVERAGE_FRAME_LIMIT = 30       # consecutive out-of-coverage frames

# Split switching
SWITCH_DELAY = 200               # frames before actually switching split


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
#  COORDINATE TRANSFORMS
# =============================================================================

class CoordinateTransformer:
    """CARLA <-> UTM <-> Nerfstudio coordinate chain (per split)."""

    def __init__(self, xodr_path, utm_to_nerfstudio_path):
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

        with open(utm_to_nerfstudio_path, "r") as f:
            tf = json.load(f)

        self.ns_scale = tf["scale"]
        self.ns_rotation = np.array(tf["rotation"])
        self.ns_translation = np.array(tf["translation"])
        self.transform_mode = tf.get("mode", "2D")
        self.position_rotation_angle = np.arctan2(
            self.ns_rotation[1, 0], self.ns_rotation[0, 0]
        )

        yaw_align = tf.get("yaw_alignment")
        if yaw_align:
            self.yaw_sign = yaw_align["yaw_sign"]
            self.yaw_offset = yaw_align["yaw_offset_rad"]
            self.use_orientation_yaw = True
            sign_str = "+" if self.yaw_sign > 0 else "-"
            print(f"[CoordinateTransformer] Using orientation-based yaw: "
                  f"ns_yaw = {sign_str}utm_yaw + "
                  f"{np.degrees(self.yaw_offset):.2f} deg "
                  f"(residual std: {yaw_align.get('residual_std_deg', '?')} deg)")
        else:
            self.yaw_sign = -1
            self.yaw_offset = self.position_rotation_angle - math.pi / 2
            self.use_orientation_yaw = False
            print(f"[CoordinateTransformer] WARN: No yaw_alignment in JSON. "
                  f"Using fallback formula. offset="
                  f"{np.degrees(self.yaw_offset):.2f} deg")

        mount = tf.get("camera_mount_angles")
        if mount:
            self.avg_pitch = mount["avg_pitch_rad"]
            self.avg_roll = mount["avg_roll_rad"]
            print(f"[CoordinateTransformer] Camera mount: "
                  f"pitch={np.degrees(self.avg_pitch):.2f} deg, "
                  f"roll={np.degrees(self.avg_roll):.2f} deg")
        else:
            self.avg_pitch = 0.0
            self.avg_roll = 0.0

        print(f"[CoordinateTransformer] scale={self.ns_scale:.10f}, "
              f"position_rotation_angle="
              f"{np.degrees(self.position_rotation_angle):.2f} deg")

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


# =============================================================================
#  SPLIT MODEL
# =============================================================================

class SplitModel:
    def __init__(self, name, pipeline, coord_transformer, training_cameras,
                 frame_ids, training_filenames, data_root=None,
                 filename_to_frame_id=None):
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

        filename_to_frame_id = filename_to_frame_id or {}

        for i, fn in enumerate(training_filenames):
            base = os.path.basename(str(fn))

            # IMPORTANT:
            # Prefer the real frame ID from frame_positions_split_N_*.txt.
            # This handles overlap correctly.
            fid = filename_to_frame_id.get(base)

            # Fallback only if the positions file did not contain this image.
            if fid is None:
                fid = extract_frame_number(base)

            if fid is not None:
                self.cam_idx_to_frame_id[i] = fid
                self.frame_id_to_cam_idx[fid] = i

        print(f"   Camera/frame mapping: {len(self.cam_idx_to_frame_id)} / "
            f"{len(training_filenames)} cameras mapped")

        if len(training_filenames) > 0:
            first_base = os.path.basename(str(training_filenames[0]))
            first_fid = self.cam_idx_to_frame_id.get(0)
            print(f"   First training image: {first_base} -> frame_id={first_fid}")

            last_idx = len(training_filenames) - 1
            last_base = os.path.basename(str(training_filenames[last_idx]))
            last_fid = self.cam_idx_to_frame_id.get(last_idx)
            print(f"   Last training image:  {last_base} -> frame_id={last_fid}")

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

    def nearest_cam_distance(self, ns_x, ns_y):
        cam_xy = self.training_cameras[:, :2, 3]
        dists = np.linalg.norm(cam_xy - np.array([ns_x, ns_y]), axis=1)
        return float(dists.min())

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
#  HERO STARTUP HELPERS (mirrors the only_carla script that works)
# =============================================================================

def stop_vehicle_motion(vehicle):
    """Zero out velocity/angular_velocity so the car doesn't drift after teleport."""
    try:
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except Exception:
        pass
    try:
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except Exception:
        pass


def make_drive_start_transform(world, traj_pt):
    """
    Build a clean spawn transform from a trajectory point.

    KEY FIX: the trajectory's z is the LiDAR/base_link altitude, which is
    typically a few cm below the CARLA road mesh. Spawning there makes
    physics expel the car upward.

    We instead query the road waypoint and use waypoint.z + 0.10 m, plus
    apply the rear->center back-offset (0.13 m) used by the only_carla
    script that works.
    """
    loc = traj_pt["transform"]["location"]
    rot = traj_pt["transform"]["rotation"]

    start_x = float(loc["x"])
    start_y = float(loc["y"])
    start_z = float(loc["z"])
    start_yaw = float(rot.get("yaw", 0.0))

    yaw_rad = math.radians(start_yaw)
    offset_x = -REAR_TO_CENTER_OFFSET_METERS * math.cos(yaw_rad)
    offset_y = -REAR_TO_CENTER_OFFSET_METERS * math.sin(yaw_rad)

    target_loc = carla.Location(
        x=start_x + offset_x,
        y=start_y + offset_y,
        z=start_z,
    )

    waypoint = world.get_map().get_waypoint(
        target_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is not None:
        spawn_z = waypoint.transform.location.z + HERO_SPAWN_Z_OFFSET
        print("[INFO] Hero spawn z from road waypoint:")
        print(f"       trajectory z: {start_z:.3f}")
        print(f"       waypoint z:   {waypoint.transform.location.z:.3f}")
        print(f"       spawn z:      {spawn_z:.3f}  (+{HERO_SPAWN_Z_OFFSET})")
    else:
        spawn_z = start_z + HERO_SPAWN_Z_OFFSET
        print(f"[WARN] No waypoint found at ({start_x:.2f}, {start_y:.2f}). "
              f"Using trajectory.z + offset = {spawn_z:.3f}")

    return carla.Transform(
        carla.Location(
            x=start_x + offset_x,
            y=start_y + offset_y,
            z=spawn_z,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=start_yaw,
            roll=0.0,
        ),
    )


def stabilize_and_warmup(world, hero_vehicle, rgb_queue,
                         drive_start_transform, ticks=WARMUP_TICKS):
    """
    Mirrors prepare_hero_for_driving() + warmup launch from the only_carla
    script that works.

    1. physics OFF, motion stopped, transform forced for 10 ticks
    2. physics ON, gravity ON, settle for 10 ticks
    3. apply ackermann (12 km/h, steer=0) for 100 ticks to overcome
       startup inertia (without this, ackermann won't move the car after
       a teleport, and the stuck-detector trips at frame ~50)
    """
    print("[INFO] Stabilizing hero before driving...")

    # Stage 1: physics off + force position
    hero_vehicle.set_simulate_physics(False)
    stop_vehicle_motion(hero_vehicle)
    hero_vehicle.set_transform(drive_start_transform)
    for _ in range(10):
        world.tick()
        hero_vehicle.set_transform(drive_start_transform)
        stop_vehicle_motion(hero_vehicle)

    # Stage 2: physics on, gravity on, settle
    try:
        hero_vehicle.set_enable_gravity(True)
    except Exception:
        pass
    hero_vehicle.set_simulate_physics(True)
    hero_vehicle.set_autopilot(False)
    stop_vehicle_motion(hero_vehicle)
    for _ in range(10):
        world.tick()

    # Drain old sensor frames
    while not rgb_queue.empty():
        try:
            rgb_queue.get_nowait()
        except Empty:
            break

    # Stage 3: launch warmup
    print(f"[INFO] Warmup launch: {ticks} ticks at {LAUNCH_SPEED_KMH} km/h")
    launch_ctl = carla.VehicleAckermannControl(
        speed=float(LAUNCH_SPEED_KMH / 3.6),
        steer=0.0,
    )
    for _ in range(ticks):
        hero_vehicle.apply_ackermann_control(launch_ctl)
        world.tick()
        # keep camera queue from filling up
        while not rgb_queue.empty():
            try:
                rgb_queue.get_nowait()
            except Empty:
                break

    actual = hero_vehicle.get_transform()
    print(f"[INFO] Stabilization complete. "
          f"Actual: x={actual.location.x:.2f} y={actual.location.y:.2f} "
          f"z={actual.location.z:.3f} yaw={actual.rotation.yaw:.1f}")


# =============================================================================
#  SPLIT DETECTION (cam2sim layout, splatfacto only)
# =============================================================================

def auto_detect_splits():
    """
    Look for splatfacto splits in:
        data/data_for_gaussian_splatting/<BAG>/outputs/splatfacto_split_<N>/splatfacto/<TS>/config.yml

    For each split also resolves:
        - utm_to_nerfstudio_transform.json
        - frame_positions_split_<N>_*.txt
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

    print(f"[INFO] Candidate GS split dirs: {split_dirs}")

    for split_dir in split_dirs:
        # Safer than r"splatfacto_split_(\d+)" because it avoids partial weird matches.
        match = re.match(r"^splatfacto_split_(\d+)$", split_dir)
        if not match:
            print(f"[WARN] Ignoring unexpected split folder name: {split_dir}")
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
        utm_transform_path = os.path.join(run_dir, "utm_to_nerfstudio_transform.json")

        if not os.path.exists(config_path):
            print(f"[WARN] No config.yml in {run_dir}")
            continue

        if not os.path.exists(utm_transform_path):
            print(f"[WARN] No utm_to_nerfstudio_transform.json in {run_dir}")
            print(f"       Run 4C_utm_yaw_to_nerfstudio.py for split {split_num} first.")
            continue

        # Deterministically find frame_positions_split_<N>_*.txt
        frame_position_candidates = []

        for fname in sorted(os.listdir(GS_DATA_ROOT)):
            if (
                fname.startswith(f"frame_positions_split_{split_num}_")
                and fname.endswith(".txt")
            ):
                frame_position_candidates.append(fname)

        frame_positions = None

        if frame_position_candidates:
            chosen_fname = frame_position_candidates[0]
            frame_positions = os.path.join(GS_DATA_ROOT, chosen_fname)

            print(f"[INFO] Using frame positions for split_{split_num}: "
                  f"{chosen_fname}")

            if len(frame_position_candidates) > 1:
                print(f"[WARN] Multiple frame position files found for split_{split_num}:")
                for c in frame_position_candidates:
                    print(f"       {c}")
                print(f"       Chose: {chosen_fname}")
        else:
            print(f"[WARN] No frame_positions_split_{split_num}_*.txt found "
                  f"in {GS_DATA_ROOT}")

        split_cfg = {
            "name": f"split_{split_num}",
            "split_num": split_num,
            "gs_config": config_path,
            "utm_transform": utm_transform_path,
            "frame_positions": frame_positions,
            "data_root": GS_DATA_ROOT,
            "run_name": run_name,
        }

        splits.append(split_cfg)

        print(f"[INFO] Found split_{split_num} "
              f"(run={run_name}, config={config_path})")

    splits.sort(key=lambda s: s["split_num"])

    print(f"[INFO] Total detected GS splits: {len(splits)}")

    return splits


def find_nearest_split_by_position(carla_x, carla_y, splits):
    """
    For DAVE-2 driving we don't have frame_ids - find the best split
    by checking which split's training cameras centroid is closest in NS XY.
    """
    best_idx = 0
    best_dist = float("inf")

    for i, sm in enumerate(splits):
        ns_pos = sm.coord_transformer.carla_to_nerfstudio(carla_x, carla_y)
        cam_xy = sm.training_cameras[:, :2, 3]
        centroid = cam_xy.mean(axis=0)
        dist = np.linalg.norm(ns_pos[:2] - centroid)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def read_frame_positions_mapping(filepath):
    """
    Read frame_positions_split_N_*.txt.

    Returns:
        frame_ids: list[int]
        filename_to_frame_id: dict[str, int]

    The positions file is the source of truth for mapping:
        Nerfstudio image filename -> original CARLA/trajectory frame_id
    """
    frame_ids = []
    filename_to_frame_id = {}

    if filepath and os.path.exists(filepath):
        print(f"[INFO] Reading frame positions: {filepath}")

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]

                try:
                    frame_id = int(parts[0])
                    image_file = os.path.basename(parts[-1])
                except (ValueError, IndexError):
                    continue

                frame_ids.append(frame_id)
                filename_to_frame_id[image_file] = frame_id

    return frame_ids, filename_to_frame_id

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

            print(f"   Pipeline device: {pipeline.device}")

            param_devices = sorted({str(p.device) for p in pipeline.model.parameters()})
            print(f"   Model parameter devices: {param_devices}")

            if torch.cuda.is_available():
                print(f"   GPU allocated after load: "
                    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"   GPU reserved after load:  "
                    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB")

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

            positions_frame_ids, filename_to_frame_id = read_frame_positions_mapping(
                cfg.get("frame_positions")
            )

            # IMPORTANT:
            # If the positions file exists, trust it over filename parsing.
            # Filename parsing can be wrong when splits overlap or filenames are renumbered.
            if positions_frame_ids:
                all_frame_ids = sorted(set(positions_frame_ids))
            else:
                print(f"[WARN] {name}: no frame IDs from positions file; "
                    f"falling back to filename parsing")
                all_frame_ids = sorted(set(training_frame_ids))

            if not all_frame_ids:
                raise RuntimeError(f"{name}: could not determine frame IDs")

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
                filename_to_frame_id=filename_to_frame_id,
            ))
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            import traceback
            traceback.print_exc()
            os.chdir(original_cwd)
            continue
        os.chdir(original_cwd)

    return split_models


def save_drive_data(frame_id, output_dir, carla_pil, gs_pil):
    filename = f"{frame_id:06d}"
    carla_pil.save(os.path.join(output_dir, "rgb_gt", f"{filename}.png"))
    if gs_pil:
        gs_pil.save(os.path.join(output_dir, "generated_gs", f"{filename}.png"))


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DAVE-2 + GS multi-split closed-loop driving (cam2sim layout)"
    )
    parser.add_argument("--only_carla", action="store_true",
                        help="Run without GS (DAVE-2 sees CARLA images directly)")
    parser.add_argument("--only_split", type=int, default=None,
                        help="Load and use ONLY this split number")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames before stopping")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip Phase 1 free camera calibration")
    parser.add_argument("--no_save", action="store_true",
                        help="Disable frame saving")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-named under "
                             "data/data_for_carla/<bag>/drive_results/)")
    args = parser.parse_args()

    print("=" * 80)
    print("DAVE-2 DRIVE: CARLA + Gaussian Splatting (multi-split)")
    print("=" * 80)
    print(f"[INFO] Project root:    {PROJECT_ROOT}")
    print(f"[INFO] Bag name:        {BAG_NAME}")
    print(f"[INFO] XODR file:       {XODR_FILE}")
    print(f"[INFO] Trajectory:      {TRAJECTORY_FILE}")
    print(f"[INFO] Camera config:   {CAMERA_CONFIG_FILE}")
    print(f"[INFO] GS data root:    {GS_DATA_ROOT}")
    print(f"[INFO] CARLA:           {CARLA_IP}:{CARLA_PORT}")
    print("=" * 80)

    print("[GPU CHECK]")
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda version: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        print(f"cuda device 0: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] PyTorch does not see CUDA. GS will likely run on CPU.")

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
            print(f"[INFO] Filtered to ONLY split_{args.only_split}")

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
                  f"Z range [{zmin:.4f}, {zmax:.4f}]")

    # ---- Connect to CARLA ----
    print(f"\n[INFO] Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(40.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)

    # ---- Load trajectory (used to teleport vehicle for calibration & start) ----
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

    hero_vehicle.set_simulate_physics(False)  # off until phase 2
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

    # ---- Connect to DAVE-2 server ----
    print("\n[INFO] Connecting to DAVE-2 server...")
    dave2_conn = connect_to_dave2_server()
    print("[INFO] DAVE-2 connected.")

    # ============================================================
    #  PHASE 1: CALIBRATION (4-panel + sliders, per split)
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
            print(f"  [/]=prev/next training cam, R=reset, ENTER=next split")
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
                    f"Original (cam #{current_train_cam_idx}, frame {train_fid})",
                    True, (255, 200, 0)), (10, IM_HEIGHT + 10))
                screen.blit(font.render(
                    f"GS Training Pose (cam #{current_train_cam_idx})",
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
                    "WASD=move Mouse=look QE=up/down [/]=prev/next cam ENTER=drive",
                    True, (0, 255, 0)), (10, help_y))

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
            computed_pos_offset = (cam_pos - ns_from_tf) + slider_off

            ns_z_interp = sm.lookup_z(ns_from_tf[0], ns_from_tf[1])
            z_calib = cam_pos[2] + sliders[5].val
            z_offset_from_interp = z_calib - ns_z_interp

            # IMPORTANT:
            # Do not store XY translation offset for driving.
            # It is anchored at the calibration/training camera and can pull the live
            # render back toward the overlap-start pose when switching splits.
            pos_offset = np.zeros(3, dtype=np.float64)

            calibration_offsets[sm.name] = {
                "pos_offset": pos_offset,
                "yaw_offset": yaw_offset,
                "pitch_offset": pitch_offset,
                "roll_offset": roll_offset,
                "z_offset": z_offset_from_interp,
            }
            print(f"OK {sm.name} calibrated: "
                f"computed_pos_offset=({computed_pos_offset[0]:.4f}, "
                f"{computed_pos_offset[1]:.4f}, {computed_pos_offset[2]:.4f}) "
                f"[XY disabled for driving], "
                f"yaw_offset={math.degrees(yaw_offset):.2f} deg, "
                f"z_offset_from_interp={z_offset_from_interp:.4f}")

    if args.skip_calibration and split_models:
        for sm in split_models:
            cam_pos_train = sm.training_cameras[0, :, 3].copy()
            fid = sm.cam_idx_to_frame_id.get(0)
            tp = trajectory_by_frame.get(fid)
            if tp:
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

                cam_tf = rgb_sensor.get_transform()
                cx, cy_c = cam_tf.location.x, cam_tf.location.y
                ns_from_tf = sm.coord_transformer.carla_to_nerfstudio(cx, cy_c)

                computed_pos_offset = cam_pos_train - ns_from_tf
                ns_z_interp = sm.lookup_z(ns_from_tf[0], ns_from_tf[1])
                z_offset = cam_pos_train[2] - ns_z_interp

                # IMPORTANT:
                # Do not use XY offset from training_cameras[0].
                # For split_2, camera 0 is the overlap-start camera, so using its XY offset
                # can make the live render jump backward at split switch.
                pos_offset = np.zeros(3, dtype=np.float64)

                calibration_offsets[sm.name] = {
                    "pos_offset": pos_offset,
                    "yaw_offset": 0.0,
                    "pitch_offset": 0.0,
                    "roll_offset": 0.0,
                    "z_offset": z_offset,
                }
                print(f"[skip_calibration] {sm.name}: "
                    f"computed_pos_offset=({computed_pos_offset[0]:.4f}, "
                    f"{computed_pos_offset[1]:.4f}, {computed_pos_offset[2]:.4f}) "
                    f"[XY disabled], z_offset={z_offset:.4f}")

    # ============================================================
    #  PHASE 2: DAVE-2 AUTONOMOUS DRIVING
    # ============================================================
    win_w = IM_WIDTH * 2
    win_h = IM_HEIGHT
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("DAVE-2 + GS | Multi-Split Driving")
    print("\n[INFO] Starting Autonomous Drive...")

    # ---- Output dir setup ----
    save_flag = not args.no_save
    if save_flag:
        if args.output_dir:
            run_folder = args.output_dir
        else:
            ts = int(time.time())
            run_folder = os.path.join(
                DEFAULT_OUTPUT_DIR,
                f"{BAG_NAME}_drive_{ts}"
            )
        os.makedirs(os.path.join(run_folder, "rgb_gt"), exist_ok=True)
        os.makedirs(os.path.join(run_folder, "generated_gs"), exist_ok=True)
        print(f"[INFO] Output: {run_folder}")
    else:
        run_folder = None

    # ---- Build drive-start transform (waypoint.z + back-offset) ----
    if split_models:
        sm_first = split_models[0]
        first_fid = sm_first.get_first_training_frame_id()
        tp = trajectory_by_frame.get(first_fid)
        if tp:
            drive_start_transform = make_drive_start_transform(world, tp)
            print(f"[INFO] Starting at first training camera (frame {first_fid})")
        else:
            drive_start_transform = start_transform
            print(f"[WARN] Frame {first_fid} not in trajectory - using default start")
    else:
        drive_start_transform = start_transform

    # ---- Stabilize physics + warmup launch (kicks ackermann) ----
    stabilize_and_warmup(world, hero_vehicle, rgb_queue, drive_start_transform)

    # ---- Drive loop state ----
    current_split_idx = 0
    switch_pending_idx = -1
    switch_pending_frame = 0
    frame = 0
    trajectory_log = []
    raw_steer = 0.0
    norm_steer = 0.0
    out_of_coverage_count = 0
    stuck_counter = 0
    prev_loc = None

    try:
        while True:
            frame += 1

            if args.max_frames is not None and frame > args.max_frames:
                print(f"[F{frame}] Reached max_frames={args.max_frames} - stopping.")
                break

            world.tick()

            # --- TERMINATION CHECKS ---
            cur_loc = hero_vehicle.get_location()

            if cur_loc.z < MIN_Z_THRESHOLD:
                print(f"[F{frame}] FAIL: Car fell off map (z={cur_loc.z:.2f}). Terminating.")
                break

            if prev_loc is not None:
                if cur_loc.distance(prev_loc) < STUCK_THRESHOLD:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if stuck_counter > STUCK_FRAME_LIMIT:
                    print(f"[F{frame}] FAIL: Car stuck for {STUCK_FRAME_LIMIT} frames. Terminating.")
                    break
            prev_loc = cur_loc

            vehicle_transform = hero_vehicle.get_transform()
            spectator = world.get_spectator()
            spectator.set_transform(carla.Transform(
                vehicle_transform.location + carla.Location(z=10),
                carla.Rotation(pitch=-90)
            ))

            try:
                rgb_data = rgb_queue.get(block=True, timeout=1.0)
                rgb_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape(
                    (rgb_data.height, rgb_data.width, 4))[:, :, :3][:, :, ::-1]
                carla_pil = Image.fromarray(rgb_np)
            except Empty:
                continue

            steering_image = None
            gs_pil = None

            if only_carla:
                steering_image = carla_pil
            else:
                cam_tf = rgb_sensor.get_transform()
                carla_x = cam_tf.location.x
                carla_y = cam_tf.location.y
                carla_yaw_rad = math.radians(cam_tf.rotation.yaw)

                # --- SPLIT SWITCHING (with delay) ---
                if len(split_models) > 1:
                    new_split_idx = find_nearest_split_by_position(
                        carla_x, carla_y, split_models)
                    if new_split_idx != current_split_idx:
                        if switch_pending_idx != new_split_idx:
                            switch_pending_idx = new_split_idx
                            switch_pending_frame = frame
                        elif frame - switch_pending_frame >= SWITCH_DELAY:
                            print(f"[F{frame}] Switching: "
                                  f"{split_models[current_split_idx].name} -> "
                                  f"{split_models[new_split_idx].name} "
                                  f"(delayed {SWITCH_DELAY} frames)")
                            current_split_idx = new_split_idx
                            switch_pending_idx = -1

                            # Warmup new pipeline
                            sm_new = split_models[current_split_idx]
                            warmup_c2w = sm_new.get_training_cam_c2w(0)
                            for _ in range(5):
                                render_gs(sm_new.pipeline, warmup_c2w,
                                          IM_WIDTH, IM_HEIGHT, fov)
                            print(f"   Warmup: 5 full-res frames rendered for {sm_new.name}")
                    else:
                        switch_pending_idx = -1

                sm = split_models[current_split_idx]
                offsets = calibration_offsets.get(sm.name, {
                    "pos_offset": np.zeros(3),
                    "yaw_offset": 0.0,
                    "pitch_offset": 0.0,
                    "roll_offset": 0.0,
                    "z_offset": 0.0,
                })

                # CARLA -> Nerfstudio
                ns_pos_raw = sm.coord_transformer.carla_to_nerfstudio(
                    carla_x, carla_y)
                ns_pos_raw[2] = sm.lookup_z(ns_pos_raw[0], ns_pos_raw[1])

                # --- Coverage check ---
                coverage_dist = sm.nearest_cam_distance(ns_pos_raw[0], ns_pos_raw[1])
                if coverage_dist > COVERAGE_THRESHOLD:
                    out_of_coverage_count += 1
                    if out_of_coverage_count >= COVERAGE_FRAME_LIMIT:
                        print(f"\n[F{frame}] Out of training coverage for "
                              f"{out_of_coverage_count} frames "
                              f"(dist={coverage_dist:.4f} > {COVERAGE_THRESHOLD}) - stopping.")
                        break
                    if out_of_coverage_count == 1:
                        print(f"[F{frame}] WARN: Approaching coverage edge "
                              f"(dist={coverage_dist:.4f})")
                else:
                    out_of_coverage_count = 0

                ns_pos = ns_pos_raw + offsets["pos_offset"]
                ns_pos[2] = ns_pos_raw[2] + offsets.get("z_offset", 0.0)

                ns_yaw_raw = sm.coord_transformer.transform_yaw_carla_to_nerfstudio(
                    carla_yaw_rad)
                ns_yaw = ns_yaw_raw + offsets["yaw_offset"]
                ns_pitch = sm.avg_pitch + offsets["pitch_offset"]
                ns_roll = sm.avg_roll + offsets["roll_offset"]

                if frame % 100 == 0:
                    print(f"[F{frame}] {sm.name} | CARLA: ({carla_x:.1f}, {carla_y:.1f}) "
                          f"yaw={math.degrees(carla_yaw_rad):.1f}deg -> "
                          f"NS: ({ns_pos[0]:.4f}, {ns_pos[1]:.4f}, {ns_pos[2]:.4f}) "
                          f"yaw={math.degrees(ns_yaw):.1f}deg")

                c2w = build_nerfstudio_c2w(ns_pos, ns_yaw, ns_pitch, ns_roll)
                gs_pil = render_gs(sm.pipeline, c2w, IM_WIDTH, IM_HEIGHT, fov)

                steering_image = gs_pil

            # --- DAVE-2 INFERENCE (every PREDICT_EVERY frames) ---
            if (frame - 1) % PREDICT_EVERY == 0:
                raw_steer, _ = send_image_over_connection(dave2_conn, steering_image)
                norm_steer = raw_steer / (3 * np.pi)

            if frame % 50 == 0:
                print(f"[F{frame}] Steer: raw={raw_steer:.4f} norm={norm_steer:.4f}")

            # --- DISPLAY: CARLA on left, GS (or CARLA again) on right ---
            display_img = Image.new("RGB", (win_w, win_h))
            display_img.paste(carla_pil, (0, 0))
            if gs_pil is not None:
                display_img.paste(gs_pil, (IM_WIDTH, 0))
            else:
                display_img.paste(carla_pil, (IM_WIDTH, 0))
            screen_surf = pygame.image.fromstring(
                display_img.tobytes(), display_img.size, display_img.mode)
            screen.blit(screen_surf, (0, 0))

            screen.blit(font.render(f"Frame {frame}", True, (0, 255, 0)), (10, 10))
            if split_models and not only_carla:
                screen.blit(font.render(
                    f"Split: {split_models[current_split_idx].name}",
                    True, (255, 200, 0)), (IM_WIDTH + 10, 10))
            screen.blit(font.render(
                f"Steer: {norm_steer:+.3f}",
                True, (0, 200, 255)), (IM_WIDTH + 10, 30))

            pygame.display.flip()

            # --- ACKERMANN CONTROL ---
            ackermann_control = carla.VehicleAckermannControl(
                speed=float(DRIVE_SPEED_KMH / 3.6),
                steer=float(-norm_steer)
            )
            hero_vehicle.apply_ackermann_control(ackermann_control)

            # --- SAVE ---
            if save_flag:
                # save_drive_data(frame, run_folder, carla_pil, gs_pil)

                veh_tf = hero_vehicle.get_transform()
                trajectory_log.append({
                    "frame": frame,
                    "x": round(veh_tf.location.x, 4),
                    "y": round(veh_tf.location.y, 4),
                    "z": round(veh_tf.location.z, 4),
                    "yaw": round(veh_tf.rotation.yaw, 4),
                    "steer_raw": round(float(raw_steer), 6),
                    "steer_norm": round(float(norm_steer), 6),
                    "split": (split_models[current_split_idx].name
                              if split_models and not only_carla else "none"),
                })

            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt

            clock.tick(60)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        if save_flag and trajectory_log:
            traj_out = os.path.join(run_folder, "trajectory.json")
            with open(traj_out, "w") as f:
                json.dump(trajectory_log, f, indent=2)
            print(f"[INFO] Trajectory saved: {traj_out} ({len(trajectory_log)} frames)")

        if rgb_sensor:
            rgb_sensor.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()