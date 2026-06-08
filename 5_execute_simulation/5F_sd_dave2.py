#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5F_sd_dave2.py

DAVE-2 closed-loop driving with Stable Diffusion rendering, cam2sim layout.

For each frame:
  - CARLA renders the ground-truth sensors
  - SD generates an RGB image from CARLA's semantic + instance + previous
  - DAVE-2 takes the SD frame as input and produces steering
  - hero is controlled via ackermann
  - model switching across the 3 SD splits is done by position along trajectory

Reads from (project root):
    data/data_for_carla/camera.json                       (shared)
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json
    data/data_for_carla/<BAG>/instance_color_map.json  (from 3F_OPT)

Reads SD models from (external SSD):
    <EXTERNAL_DRIVE>/cam2sim_sd/<BAG>/SD_Training_Outputs_Split/part_<N>/

Writes to (project root):
    data/results/sd_run<N>/
        trajectory.json
        rgb/           CARLA RGB frames (used as fallback in only_carla mode)
        semantic/      cleaned semantic maps
        instance/      cleaned + bag-remapped instance maps
        generated/     SD generated frames (only when SD is active)
        combined/      side-by-side preview
        prompts.txt    per-frame prompt + control schedule + model part

Auto-increments run folder: scans data/results/ for existing sd_run<N>
and picks the next free integer.
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
import cv2
import torch
from PIL import Image


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


# =============================================================================
#  LOCAL UTILS
# =============================================================================

from utils.config import (
    CARLA_IP,
    CARLA_PORT,
    HERO_VEHICLE_TYPE,
)

from utils.carla_simulator import (
    update_synchronous_mode,
    cleanup_old_sensors,
    spawn_sensor,
    process_instance_map_fixed,
    remap_segmentation_colors,
    carla_image_to_pil,
    load_instance_color_map,
)

from utils.stable_diffusion import (
    load_pipeline_models,
    generate_image_realtime,
)

from utils.dave2_connection import (
    connect_to_dave2_server,
    send_image_over_connection,
)


# =============================================================================
#  CONFIG (non bag-dependent)
# =============================================================================

# Bag name (with .bag extension): must match an existing bag from step 1.
DEFAULT_BAG_NAME = "reference_bag.bag"

# Note: bag-dependent paths (TRAJECTORY_FILE, MODELS_BASE_DIR, ...) are
# built in main() after argparse parses --bag-name.

NUM_PARTS = 3

# Best control schedule from thesis
CONTROL_START = [0.0, 0.0, 0.35]   # [seg, inst, temp]
CONTROL_END = [1.0, 0.6, 0.55]
GUIDANCE_SCALE = 3.0

# Output path (project root, same as 5D)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "results")
RUN_PREFIX = "sd_run"

# External SSD root (shared, not bag-dependent)
EXTERNAL_DRIVE = "/media/davidejannussi/ssd space"
CAM2SIM_SD_ROOT = os.path.join(EXTERNAL_DRIVE, "cam2sim_sd")
os.environ["HF_HOME"] = os.path.join(CAM2SIM_SD_ROOT, "huggingface_cache")

# Sensors
IM_WIDTH = 800
IM_HEIGHT = 503
TARGET_SIZE = 512   # SD pipeline input/output resolution

# Driving
DRIVE_SPEED_KMH = 10.0
WARMUP_SPEED_KMH = 12.0
WARMUP_TICKS = 20

# Termination
STUCK_THRESHOLD = 0.02
STUCK_FRAME_LIMIT = 50
MIN_Z_THRESHOLD = -0.5

# Model switching hysteresis (frames waited in new region before switching)
SWITCH_HYSTERESIS_FRAMES = 5

# Cleaning
MIN_PIXEL_AREA = 250
EGO_HOOD_HEIGHT = 42
ROAD_COLOR_BGR = (128, 64, 128)
BACKGROUND_COLOR_BGR = (0, 0, 0)
TARGET_COLORS_BGR = [
    (142, 0, 0),     # Car
    (70, 0, 0),      # Truck
    (100, 60, 0),    # Bus
    (230, 0, 0),     # Motorcycle
    (32, 11, 119),   # Bicycle
    (128, 64, 128),  # Road
    (50, 234, 157),  # RoadLines
    (232, 35, 244),  # Sidewalk
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
#  CLEANING
# =============================================================================

def clean_semantic_and_instance(sem_pil, inst_pil, min_area=MIN_PIXEL_AREA):
    sem_img = cv2.cvtColor(np.array(sem_pil), cv2.COLOR_RGB2BGR)
    inst_img = cv2.cvtColor(np.array(inst_pil), cv2.COLOR_RGB2BGR)

    cleaned_sem = sem_img.copy()
    cleaned_inst = inst_img.copy()

    for target_color in TARGET_COLORS_BGR:
        lower_bound = np.array(target_color, dtype=np.uint8)
        upper_bound = np.array(target_color, dtype=np.uint8)
        mask = cv2.inRange(sem_img, lower_bound, upper_bound)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                object_mask = labels == i
                cleaned_sem[object_mask] = BACKGROUND_COLOR_BGR

    background_mask = np.all(cleaned_sem == BACKGROUND_COLOR_BGR, axis=-1)
    cleaned_inst[background_mask] = [0, 0, 0]

    if EGO_HOOD_HEIGHT > 0:
        cleaned_sem[-EGO_HOOD_HEIGHT:, :] = ROAD_COLOR_BGR
        cleaned_inst[-(EGO_HOOD_HEIGHT + 10):, :] = [0, 0, 0]

    cleaned_sem_pil = Image.fromarray(
        cv2.cvtColor(cleaned_sem, cv2.COLOR_BGR2RGB)
    )
    cleaned_inst_pil = Image.fromarray(
        cv2.cvtColor(cleaned_inst, cv2.COLOR_BGR2RGB)
    )
    return cleaned_sem_pil, cleaned_inst_pil


# =============================================================================
#  TRAJECTORY-BASED MODEL SELECTION
# =============================================================================

def split_trajectory_into_parts(trajectory_points, num_parts=3):
    """Split trajectory into equal chunks (same as training)."""
    total = len(trajectory_points)
    chunk_size = total // num_parts
    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = total if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(trajectory_points[start:end])
    return chunks


def find_closest_trajectory_point(current_pos, trajectory_chunk):
    """current_pos is a carla.Location."""
    min_dist = float("inf")
    closest_idx = 0
    for idx, point in enumerate(trajectory_chunk):
        traj_x = point["transform"]["location"]["x"]
        traj_y = point["transform"]["location"]["y"]
        dist = np.sqrt(
            (current_pos.x - traj_x) ** 2 + (current_pos.y - traj_y) ** 2
        )
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    return closest_idx, min_dist


def select_model_part(current_pos, trajectory_chunks):
    best_part = 0
    best_distance = float("inf")
    for part_idx, chunk in enumerate(trajectory_chunks):
        _, dist = find_closest_trajectory_point(current_pos, chunk)
        if dist < best_distance:
            best_distance = dist
            best_part = part_idx
    return best_part, best_distance


# =============================================================================
#  OUTPUT FOLDER
# =============================================================================

def next_run_folder(base_dir, prefix=RUN_PREFIX, forced_id=None):
    os.makedirs(base_dir, exist_ok=True)
    if forced_id is not None:
        return os.path.join(base_dir, f"{prefix}{int(forced_id)}")

    existing = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for entry in os.listdir(base_dir):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue
        m = pattern.match(entry)
        if m:
            existing.append(int(m.group(1)))

    next_n = max(existing) + 1 if existing else 1
    return os.path.join(base_dir, f"{prefix}{next_n}")


def make_output_dirs(run_folder):
    for sub in ["rgb", "semantic", "instance", "generated", "combined"]:
        os.makedirs(os.path.join(run_folder, sub), exist_ok=True)


def save_frame_outputs(frame, run_folder, final_rgb, final_seg, final_inst,
                       generated_image, prompt, model_part):
    filename = f"{frame:06d}"
    final_rgb.save(os.path.join(run_folder, "rgb", f"{filename}.png"))
    final_seg.save(os.path.join(run_folder, "semantic", f"{filename}.png"))
    final_inst.save(os.path.join(run_folder, "instance", f"{filename}.png"))
    if generated_image is not None:
        generated_image.save(
            os.path.join(run_folder, "generated", f"{filename}.png")
        )
    with open(os.path.join(run_folder, "prompts.txt"), "a") as f:
        f.write(
            f"{filename} | Part:{model_part} | "
            f"S:{CONTROL_START} | E:{CONTROL_END} | {prompt}\n"
        )


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DAVE-2 closed-loop driving with Stable Diffusion (cam2sim)"
    )
    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help="Bag filename including .bag extension "
             "(default: env BAG_NAME or 'reference_bag.bag').",
    )
    parser.add_argument("--only_carla", action="store_true",
                        help="Run without SD - feed CARLA RGB directly to DAVE-2.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to drive.")
    parser.add_argument("--no_save", action="store_true",
                        help="Disable frame saving.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output dir. Default: data/results/sd_run<N>")
    parser.add_argument("--run_id", type=int, default=None,
                        help="Force a specific run number.")
    args = parser.parse_args()

    bag_name = args.bag_name               # e.g. "reference_bag.bag"
    bag_stem = Path(bag_name).stem         # e.g. "reference_bag"

    # ---------- Build bag-dependent paths ----------
    trajectory_file = os.path.join(
        PROJECT_ROOT, "data", "data_for_carla", bag_stem,
        "trajectory_positions_rear_odom_yaw.json",
    )
    camera_config_file = os.path.join(
        PROJECT_ROOT, "data", "data_for_carla", "camera.json"
    )
    instance_color_map_path = os.path.join(
        PROJECT_ROOT, "data", "data_for_carla", bag_stem,
        "instance_color_map.json",
    )

    # SD models (external SSD, per-bag)
    models_base_dir = os.path.join(
        CAM2SIM_SD_ROOT, bag_stem, "SD_Training_Outputs_Split"
    )

    print("=" * 80)
    print("DAVE-2 CLOSED-LOOP WITH STABLE DIFFUSION (cam2sim, SD branch)")
    print("=" * 80)
    print(f"[INFO] Project root:    {PROJECT_ROOT}")
    print(f"[INFO] Bag:             {bag_name}")
    print(f"[INFO] Bag stem:        {bag_stem}")
    print(f"[INFO] Trajectory:      {trajectory_file}")
    print(f"[INFO] Camera config:   {camera_config_file}  (shared)")
    print(f"[INFO] Instance map:    {instance_color_map_path}")
    print(f"[INFO] SD models:       {models_base_dir}")
    print(f"[INFO] CARLA:           {CARLA_IP}:{CARLA_PORT}")
    print(f"[INFO] Device:          {DEVICE}")
    print(f"[INFO] only_carla mode: {args.only_carla}")
    print(f"[INFO] Control start:   {CONTROL_START}")
    print(f"[INFO] Control end:     {CONTROL_END}")
    print("=" * 80)

    # ---------- Camera config ----------
    with open(camera_config_file, "r") as f:
        cam_data = json.load(f)
    cam_config = cam_data.get("camera", cam_data)
    fov = float(cam_config["fov"])
    fps = int(cam_config.get("fps", 20))
    cam_pos_x = float(cam_config["position"]["x"])
    cam_pos_y = float(cam_config["position"]["y"])
    cam_pos_z = float(cam_config["position"]["z"])
    cam_pitch = float(cam_config.get("pitch", 0.0))
    print(f"[INFO] Camera: fov={fov}, fps={fps}, "
          f"pos=({cam_pos_x},{cam_pos_y},{cam_pos_z}), pitch={cam_pitch}")

    # ---------- Instance color map ----------
    color_map = {}
    if os.path.exists(instance_color_map_path):
        color_map = load_instance_color_map(instance_color_map_path)
        print(f"[INFO] Loaded {len(color_map)} CARLA-to-bag color mappings.")
    else:
        print(f"[WARN] No instance_color_map.json. Instance maps will use "
              f"CARLA synthetic colors only.")

    # ---------- Trajectory + chunks ----------
    with open(trajectory_file, "r") as f:
        trajectory_points = json.load(f)
    print(f"[INFO] Trajectory points: {len(trajectory_points)}")

    trajectory_chunks = split_trajectory_into_parts(
        trajectory_points, NUM_PARTS
    )
    for i, chunk in enumerate(trajectory_chunks):
        print(f"   Part {i}: {len(chunk)} frames")

    # ---------- Connect CARLA ----------
    print(f"\n[INFO] Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(40.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)

    # ---------- Sync mode ----------
    update_synchronous_mode(world, tm, True, fps)
    world.tick()

    # ---------- Hero spawn / teleport ----------
    first_pt = trajectory_points[0]["transform"]
    start_yaw = float(first_pt["rotation"]["yaw"])

    # Try to project to road waypoint for z
    target_loc = carla.Location(
        x=float(first_pt["location"]["x"]),
        y=float(first_pt["location"]["y"]),
        z=float(first_pt["location"]["z"]),
    )
    carla_map = world.get_map()
    waypoint = carla_map.get_waypoint(
        target_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    # Small backward offset (matches 5D pattern)
    offset_distance = 0.13
    yaw_rad = math.radians(start_yaw)
    offset_x = -offset_distance * math.cos(yaw_rad)
    offset_y = -offset_distance * math.sin(yaw_rad)

    start_transform = carla.Transform(
        carla.Location(
            x=target_loc.x + offset_x,
            y=target_loc.y + offset_y,
            z=waypoint.transform.location.z if waypoint else target_loc.z,
        ),
        carla.Rotation(pitch=0, yaw=start_yaw, roll=0),
    )

    bp_lib = world.get_blueprint_library()

    # Find existing hero or spawn new
    hero_vehicle = None
    all_vehicles = world.get_actors().filter("vehicle.*")
    for v in all_vehicles:
        if v.attributes.get("role_name", "") == "hero":
            hero_vehicle = v
            print(f"[INFO] Found existing hero vehicle id={v.id}")
            break

    if hero_vehicle is None:
        for v in all_vehicles:
            if v.type_id == HERO_VEHICLE_TYPE:
                hero_vehicle = v
                print(f"[INFO] Found existing vehicle of type "
                      f"{HERO_VEHICLE_TYPE} id={v.id}")
                break

    if hero_vehicle is not None:
        print(f"[INFO] Teleporting existing hero to start...")
        hero_vehicle.set_simulate_physics(False)
        hero_vehicle.set_transform(start_transform)
        hero_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        hero_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        world.tick()
        hero_vehicle.set_simulate_physics(True)
    else:
        print(f"[INFO] No hero found, spawning new {HERO_VEHICLE_TYPE}")
        vehicle_bp = bp_lib.find(HERO_VEHICLE_TYPE)
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")
        hero_vehicle = world.spawn_actor(vehicle_bp, start_transform)

    hero_vehicle.set_simulate_physics(True)
    hero_vehicle.set_autopilot(False)
    cleanup_old_sensors(hero_vehicle)

    # ---------- Sensors ----------
    sensor_tf = carla.Transform(
        carla.Location(x=cam_pos_x, y=cam_pos_y, z=cam_pos_z),
        carla.Rotation(pitch=cam_pitch),
    )

    fov_str = str(fov)
    print("[INFO] Spawning sensors on hero...")

    rgb_sensor, rgb_queue = spawn_sensor(
        bp_lib, "sensor.camera.rgb",
        IM_WIDTH, IM_HEIGHT, fov_str, world, sensor_tf, hero_vehicle,
    )
    sem_sensor, sem_queue = spawn_sensor(
        bp_lib, "sensor.camera.semantic_segmentation",
        IM_WIDTH, IM_HEIGHT, fov_str, world, sensor_tf, hero_vehicle,
    )
    inst_sensor, inst_queue = spawn_sensor(
        bp_lib, "sensor.camera.instance_segmentation",
        IM_WIDTH, IM_HEIGHT, fov_str, world, sensor_tf, hero_vehicle,
    )

    world.tick()

    # ---------- Pygame ----------
    win_w = TARGET_SIZE * 2
    win_h = TARGET_SIZE * 2
    pygame.init()
    pygame.display.set_caption("DAVE-2 + Stable Diffusion (cam2sim)")
    pygame_screen = pygame.display.set_mode((win_w, win_h))
    pygame_clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)

    # ---------- DAVE-2 ----------
    print("[INFO] Connecting to DAVE-2 server...")
    dave2_conn = connect_to_dave2_server()
    print("[INFO] DAVE-2 connected.")

    # ---------- Output run folder ----------
    save_flag = not args.no_save
    run_folder = None
    if save_flag:
        if args.output_dir:
            run_folder = args.output_dir
        else:
            run_folder = next_run_folder(
                DEFAULT_OUTPUT_DIR,
                prefix=RUN_PREFIX,
                forced_id=args.run_id,
            )
        make_output_dirs(run_folder)
        print(f"[INFO] Output folder: {run_folder}")

    # ---------- SD model holders ----------
    pipe = None
    model_data_gen = None
    current_model_part = None
    pending_switch_to = None
    frames_in_new_part = 0
    prev_image = None

    # ---------- Helper to drain sensor queues ----------
    def drain_queues():
        for q in [rgb_queue, sem_queue, inst_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

    # ---------- Warmup launch ----------
    print(f"[INFO] Warmup: {WARMUP_TICKS} ticks at {WARMUP_SPEED_KMH} km/h...")
    launch_control = carla.VehicleAckermannControl(
        speed=float(WARMUP_SPEED_KMH / 3.6),
        steer=0.0,
    )
    drain_queues()
    for _ in range(WARMUP_TICKS):
        hero_vehicle.apply_ackermann_control(launch_control)
        world.tick()
        drain_queues()
    time.sleep(0.1)
    drain_queues()

    # ---------- Main loop ----------
    print("[INFO] Starting closed-loop drive.")

    trajectory_log = []
    frame = 0
    stuck_counter = 0
    prev_loc = hero_vehicle.get_location()
    spectator = world.get_spectator()

    try:
        while True:
            if args.max_frames is not None and frame >= args.max_frames:
                print(f"[F{frame}] Reached max_frames={args.max_frames}. Stop.")
                break

            frame += 1
            world.tick()

            cur_loc = hero_vehicle.get_location()

            # ---- Fall check ----
            if cur_loc.z < MIN_Z_THRESHOLD:
                print(f"[F{frame}] FAIL: Car fell (z={cur_loc.z:.2f}). Terminate.")
                break

            # ---- Stuck check ----
            if cur_loc.distance(prev_loc) < STUCK_THRESHOLD:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter > STUCK_FRAME_LIMIT:
                print(f"[F{frame}] FAIL: Stuck for {STUCK_FRAME_LIMIT} frames. Terminate.")
                break
            prev_loc = cur_loc

            # ---- Chase spectator ----
            veh_tf = hero_vehicle.get_transform()
            spec_loc = veh_tf.location - (veh_tf.get_forward_vector() * 10)
            spec_loc.z += 5
            spec_rot = veh_tf.rotation
            spec_rot.pitch = -15
            spectator.set_transform(carla.Transform(spec_loc, spec_rot))

            # ---- Get sensor data ----
            try:
                rgb_data = rgb_queue.get(block=True, timeout=1.0)
                sem_data = sem_queue.get(block=True, timeout=1.0)
                inst_data = inst_queue.get(block=True, timeout=1.0)
            except Empty:
                print(f"[F{frame}] WARN: Sensor timeout, skipping.")
                continue

            # ---- Process sensors ----
            rgb_np = np.frombuffer(
                rgb_data.raw_data, dtype=np.uint8
            ).reshape((rgb_data.height, rgb_data.width, 4))[:, :, :3][:, :, ::-1]
            rgb_pil = Image.fromarray(rgb_np)

            sem_data.convert(carla.ColorConverter.CityScapesPalette)
            sem_pil = remap_segmentation_colors(carla_image_to_pil(sem_data))

            inst_pil = process_instance_map_fixed(inst_data, color_map)

            sem_cleaned, inst_cleaned = clean_semantic_and_instance(
                sem_pil, inst_pil
            )

            final_rgb = rgb_pil.resize(
                (TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.LANCZOS
            )
            final_seg = sem_cleaned.resize(
                (TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.NEAREST
            )
            final_inst = inst_cleaned.resize(
                (TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.NEAREST
            )

            # ---- Decide steering image source ----
            steering_image = None
            generated_image = None
            dynamic_prompt = ""
            traj_distance = 0.0
            status_indicator = ""

            if args.only_carla:
                steering_image = final_rgb
                generated_image = None
                current_model_part = 0
            else:
                # ---- Model selection ----
                required_part, traj_distance = select_model_part(
                    cur_loc, trajectory_chunks
                )

                # ---- Hysteresis-based switching ----
                if required_part != current_model_part:
                    if pending_switch_to == required_part:
                        frames_in_new_part += 1
                        if frames_in_new_part >= SWITCH_HYSTERESIS_FRAMES:
                            print(f"\n[F{frame}] MODEL SWITCH: "
                                  f"{current_model_part} -> {required_part} | "
                                  f"pos=({cur_loc.x:.1f},{cur_loc.y:.1f}) | "
                                  f"dist={traj_distance:.2f}m")
                            previous_last_image = prev_image
                            if pipe is not None:
                                del pipe
                                del model_data_gen
                                torch.cuda.empty_cache()
                                print(f"   Freed GPU memory.")

                            model_path = os.path.join(
                                models_base_dir, f"part_{required_part}"
                            )
                            print(f"   Loading from: {model_path}")
                            pipe, model_data_gen = load_pipeline_models(
                                model_path, DEVICE
                            )
                            current_model_part = required_part
                            prev_image = previous_last_image
                            frames_in_new_part = 0
                            pending_switch_to = None
                            print(f"   Model part {required_part} loaded.\n")
                    else:
                        pending_switch_to = required_part
                        frames_in_new_part = 1
                else:
                    frames_in_new_part = 0
                    pending_switch_to = None

                if pending_switch_to is not None:
                    status_indicator = (
                        f"[switching to {pending_switch_to} in "
                        f"{SWITCH_HYSTERESIS_FRAMES - frames_in_new_part}]"
                    )

                dynamic_prompt = (
                    f"pos x: {cur_loc.x:.2f}, y: {cur_loc.y:.2f}"
                )

                if pipe is not None:
                    generated_image = generate_image_realtime(
                        pipe,
                        seg_image=final_seg,
                        inst_image=final_inst,
                        model_data=model_data_gen,
                        prev_image=prev_image,
                        prompt=dynamic_prompt,
                        guidance=GUIDANCE_SCALE,
                        control_start=CONTROL_START,
                        control_end=CONTROL_END,
                    )
                    prev_image = generated_image
                    steering_image = generated_image
                else:
                    # First frame: load initial model based on spawn position
                    initial_part, initial_dist = select_model_part(
                        cur_loc, trajectory_chunks
                    )
                    print(f"[F{frame}] Loading initial model part "
                          f"{initial_part} (dist={initial_dist:.2f}m)")
                    model_path = os.path.join(
                        models_base_dir, f"part_{initial_part}"
                    )
                    pipe, model_data_gen = load_pipeline_models(
                        model_path, DEVICE
                    )
                    current_model_part = initial_part
                    print(f"[F{frame}] Initial model loaded.")
                    # Use CARLA RGB this frame as fallback
                    steering_image = final_rgb
                    generated_image = None

            # ---- DAVE-2 inference ----
            raw_steer, throttle = send_image_over_connection(
                dave2_conn, steering_image
            )
            max_rotation = 3 * np.pi
            normalized_steer = raw_steer / max_rotation

            print(f"[F{frame}] Part:{current_model_part} {status_indicator} | "
                  f"dist={traj_distance:.1f}m | "
                  f"steer={normalized_steer:.4f}")

            # ---- Control ----
            ackermann_control = carla.VehicleAckermannControl(
                speed=float(DRIVE_SPEED_KMH / 3.6),
                steer=float(normalized_steer * -1.0),
            )
            hero_vehicle.apply_ackermann_control(ackermann_control)

            # ---- Log trajectory ----
            trajectory_log.append({
                "frame": frame,
                "x": round(cur_loc.x, 4),
                "y": round(cur_loc.y, 4),
                "z": round(cur_loc.z, 4),
                "yaw": round(veh_tf.rotation.yaw, 4),
                "steer_raw": round(float(raw_steer), 6),
                "steer_norm": round(float(normalized_steer), 6),
                "model_part": current_model_part,
            })

            # ---- Save outputs ----
            if save_flag:
                save_frame_outputs(
                    frame, run_folder,
                    final_rgb, final_seg, final_inst,
                    generated_image, dynamic_prompt, current_model_part,
                )

            # ---- 2x2 display ----
            display_image = Image.new("RGB", (win_w, win_h))
            display_image.paste(final_rgb, (0, 0))
            if generated_image is not None:
                display_image.paste(generated_image, (TARGET_SIZE, 0))
            else:
                display_image.paste(final_rgb, (TARGET_SIZE, 0))
            display_image.paste(final_seg, (0, TARGET_SIZE))
            display_image.paste(final_inst, (TARGET_SIZE, TARGET_SIZE))

            surf = pygame.image.fromstring(
                display_image.tobytes(),
                display_image.size,
                display_image.mode,
            )
            pygame_screen.blit(surf, (0, 0))

            pygame_screen.blit(font.render(
                f"Frame {frame} | Part {current_model_part} {status_indicator}",
                True, (0, 255, 0)
            ), (10, 10))
            pygame_screen.blit(font.render(
                f"Pos ({cur_loc.x:.1f}, {cur_loc.y:.1f}) | "
                f"Steer {normalized_steer:+.4f}",
                True, (0, 255, 0)
            ), (10, 30))

            pygame.display.flip()
            pygame_clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[INFO] User interrupt.")

    finally:
        # ---- Save trajectory ----
        if save_flag and trajectory_log and run_folder is not None:
            traj_out = os.path.join(run_folder, "trajectory.json")
            with open(traj_out, "w") as f:
                json.dump(trajectory_log, f, indent=2)
            print(f"[INFO] Trajectory saved: {traj_out} ({len(trajectory_log)} frames)")

        # ---- Cleanup ----
        print("[INFO] Cleaning up...")
        for s in [rgb_sensor, sem_sensor, inst_sensor]:
            try:
                if s is not None:
                    s.stop()
                    s.destroy()
            except Exception:
                pass

        if pipe is not None:
            del pipe
            del model_data_gen
            torch.cuda.empty_cache()

        try:
            update_synchronous_mode(world, tm, False, fps)
        except Exception:
            pass

        pygame.quit()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()