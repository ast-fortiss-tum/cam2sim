#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5A_sd_trajectory_only_carla.py

Same as 5A_trajectory_only_carla.py but for the Stable Diffusion branch:
the instance segmentation maps are remapped to use the real-world bag colors
(via instance_color_map.json produced by 3F_OPT_generate_scenario_with_mapping.py).

CARLA instance colors are replaced with the corresponding bag RGB colors for
every parked car that was mapped. Unmapped instances keep CARLA synthetic
colors. This produces the exact format expected by the SD training and
generation pipeline.

Reads from (project root):
    data/data_for_carla/camera.json                        (shared)
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json
    data/data_for_carla/<BAG>/instance_color_map.json (from 3F_OPT)

Writes to (project root):
    data/processed_dataset/<BAG>/carla_replay_dataset_sd/
        data/all_frame_data.json
        rgb/        (CARLA rgb frames)
        semantic/   (CARLA cleaned semantic maps)
        instance/   (CARLA instance maps with bag colors)
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from queue import Empty

import carla
import numpy as np
import pygame
import cv2
from PIL import Image


# =======================
# PATH SETUP
# =======================

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


# =======================
# LOCAL UTILS IMPORTS
# =======================

from utils.config import (
    CARLA_IP,
    CARLA_PORT,
    HERO_VEHICLE_TYPE,
)

from utils.carla_simulator import (
    update_synchronous_mode,
    get_inverse_transform,
    remap_segmentation_colors,
    carla_image_to_pil,
    cleanup_old_sensors,
    spawn_sensor,
    process_instance_map_fixed,
    load_instance_color_map,
)


# =======================
# CONFIG (non bag-dependent)
# =======================

# Bag name (with .bag extension): must match an existing bag from step 1.
DEFAULT_BAG_NAME = "reference_bag.bag"

# Note: bag-dependent paths (CARLA_DATA_FOLDER, TRAJECTORY_PATH, ...) are
# built in main() after argparse parses --bag-name.

IM_WIDTH = 800
IM_HEIGHT = 503

DEFAULT_TARGET_SIZE = 512
DEFAULT_FPS = 20
DEFAULT_FOV = 90.0

MAX_FRAMES = None
# MAX_FRAMES = 300

REAR_TO_CENTER_OFFSET_METERS = 0.13

USE_SYNCHRONOUS_MODE = True

FOLLOW_HERO_WITH_SPECTATOR = True
SPECTATOR_BACK_DISTANCE = -10.0
SPECTATOR_HEIGHT = 8.0
SPECTATOR_PITCH = -25.0

ENABLE_PYGAME_DISPLAY = True


# =======================
# CLEANING CONFIGURATION
# =======================

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


# =======================
# HELPERS
# =======================

def load_json_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_first_existing_number(*values, default=None):
    for value in values:
        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return default


def get_first_existing_int(*values, default=None):
    value = get_first_existing_number(*values, default=default)

    if value is None:
        return None

    return int(value)


def load_camera_data(path):
    """
    Robust camera.json loader. Supports nested {"size":..., "camera":...}
    or flat {"fov":..., "position":...} formats.
    """
    raw = load_json_file(path)

    camera = raw.get("camera", raw)

    if not isinstance(camera, dict):
        raise RuntimeError(f"Invalid camera.json format: {path}")

    position = camera.get("position", raw.get("position"))

    if position is None:
        raise KeyError(
            "camera.json must contain camera.position or position with x/y/z."
        )

    for key in ["x", "y", "z"]:
        if key not in position:
            raise KeyError(
                f"camera.json position must contain '{key}'. File: {path}"
            )

    size = raw.get("size", camera.get("size", {}))
    original_size = raw.get("original_size", camera.get("original_size", {}))

    target_size = get_first_existing_int(
        raw.get("target_size"),
        raw.get("output_size"),
        size.get("x") if isinstance(size, dict) else None,
        camera.get("target_size"),
        camera.get("output_size"),
        default=DEFAULT_TARGET_SIZE,
    )

    fps = get_first_existing_int(
        camera.get("fps"),
        raw.get("fps"),
        default=DEFAULT_FPS,
    )

    fov = get_first_existing_number(
        camera.get("fov"),
        raw.get("fov"),
        default=DEFAULT_FOV,
    )

    pitch = get_first_existing_number(
        camera.get("pitch"),
        raw.get("pitch"),
        default=0.0,
    )

    camera_data = {
        "size": {
            "x": target_size,
            "y": target_size,
        },
        "camera": {
            "fov": fov,
            "fps": fps,
            "position": {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            },
            "pitch": float(pitch),
        },
    }

    if isinstance(original_size, dict):
        camera_data["camera"]["original_size"] = original_size

    print("[INFO] Loaded camera config:")
    print(f"       fov:         {fov}")
    print(f"       fps:         {fps}")
    print(f"       target size: {target_size}")
    print(
        f"       position:    "
        f"x={position['x']}, y={position['y']}, z={position['z']}"
    )
    print(f"       pitch:       {pitch}")

    if "size" not in raw:
        print(
            f"[WARN] camera.json has no size field. "
            f"Using DEFAULT_TARGET_SIZE={DEFAULT_TARGET_SIZE}."
        )

    return camera_data


def create_output_folders(output_folder):
    os.makedirs(os.path.join(output_folder, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "instance"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "data"), exist_ok=True)


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
            mask,
            connectivity=8,
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


def carla_rgb_to_pil(rgb_obj):
    rgb_arr = np.frombuffer(rgb_obj.raw_data, dtype=np.uint8)
    rgb_arr = np.reshape(rgb_arr, (rgb_obj.height, rgb_obj.width, 4))
    rgb_arr = rgb_arr[:, :, :3][:, :, ::-1]
    return Image.fromarray(rgb_arr)


def carla_rgb_to_array(rgb_obj):
    rgb_arr = np.frombuffer(rgb_obj.raw_data, dtype=np.uint8)
    rgb_arr = np.reshape(rgb_arr, (rgb_obj.height, rgb_obj.width, 4))
    rgb_arr = rgb_arr[:, :, :3][:, :, ::-1]
    return rgb_arr


def save_frame_data(
    frame_id,
    rgb_obj,
    sem_obj,
    inst_obj,
    transform_data,
    target_size,
    output_folder,
    color_map,
    all_frame_data,
):
    filename = f"{frame_id:06d}"
    target_res = (target_size, target_size)

    # ---------- RGB ----------
    rgb_pil = carla_rgb_to_pil(rgb_obj)

    final_rgb = rgb_pil.resize(
        target_res,
        resample=Image.Resampling.LANCZOS,
    )

    final_rgb.save(
        os.path.join(output_folder, "rgb", f"{filename}.png")
    )

    # ---------- SEMANTIC ----------
    sem_obj.convert(carla.ColorConverter.CityScapesPalette)
    sem_pil_raw = remap_segmentation_colors(carla_image_to_pil(sem_obj))

    # ---------- INSTANCE WITH BAG MAPPING ----------
    # Replaces raw CARLA BGRA with synthetic per-instance colors,
    # then remaps any color that matches color_map to its bag RGB.
    inst_pil_raw = process_instance_map_fixed(inst_obj, color_map)

    # ---------- CLEANING ----------
    sem_cleaned, inst_cleaned = clean_semantic_and_instance(
        sem_pil_raw,
        inst_pil_raw,
    )

    # ---------- RESIZE ----------
    final_sem = sem_cleaned.resize(
        target_res,
        resample=Image.Resampling.NEAREST,
    )

    final_inst = inst_cleaned.resize(
        target_res,
        resample=Image.Resampling.NEAREST,
    )

    final_sem.save(
        os.path.join(output_folder, "semantic", f"{filename}.png")
    )

    final_inst.save(
        os.path.join(output_folder, "instance", f"{filename}.png")
    )

    # ---------- METADATA ----------
    all_frame_data.append(
        {
            "frame": int(frame_id),
            "location": transform_data["location"],
            "rotation": transform_data["rotation"],
            "caption": (
                f"pos x: {transform_data['location']['x']:.2f}, "
                f"y: {transform_data['location']['y']:.2f}"
            ),
        }
    )


def find_existing_hero(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for vehicle in vehicles:
        try:
            if vehicle.attributes.get("role_name") == "hero":
                return vehicle
        except Exception:
            pass

    actors = list(world.get_actors())

    for vehicle in vehicles:
        for actor in actors:
            if actor.parent and actor.parent.id == vehicle.id:
                return vehicle

    if vehicles:
        return vehicles[0]

    return None


def spawn_hero_if_needed(world, blueprint_library, start_transform):
    hero_vehicle = find_existing_hero(world)

    if hero_vehicle is not None:
        print(f"[INFO] Found existing hero/vehicle actor: {hero_vehicle.id}")
        hero_vehicle.set_simulate_physics(False)
        hero_vehicle.set_autopilot(False)
        hero_vehicle.set_transform(start_transform)
        return hero_vehicle

    print("[INFO] No existing hero found. Spawning new hero.")

    vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "hero")

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", "255,0,0")

    hero_vehicle = world.spawn_actor(vehicle_bp, start_transform)
    hero_vehicle.set_simulate_physics(False)
    hero_vehicle.set_autopilot(False)

    print(f"[INFO] Spawned hero actor: {hero_vehicle.id}")

    return hero_vehicle


def make_trajectory_transform(point):
    loc = point["transform"]["location"]
    rot = point["transform"]["rotation"]

    x = float(loc["x"])
    y = float(loc["y"])
    z = float(loc["z"])
    yaw = float(rot.get("yaw", 0.0))

    if REAR_TO_CENTER_OFFSET_METERS != 0.0:
        yaw_rad = math.radians(yaw)
        x -= REAR_TO_CENTER_OFFSET_METERS * math.cos(yaw_rad)
        y -= REAR_TO_CENTER_OFFSET_METERS * math.sin(yaw_rad)

    return carla.Transform(
        carla.Location(
            x=x,
            y=y,
            z=z,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=yaw,
            roll=0.0,
        ),
    )


def move_spectator_to_hero(world, hero_vehicle):
    spectator = world.get_spectator()
    hero_transform = hero_vehicle.get_transform()

    spectator_location = hero_transform.transform(
        carla.Location(
            x=SPECTATOR_BACK_DISTANCE,
            y=0.0,
            z=SPECTATOR_HEIGHT,
        )
    )

    spectator_rotation = carla.Rotation(
        pitch=SPECTATOR_PITCH,
        yaw=hero_transform.rotation.yaw,
        roll=0.0,
    )

    spectator.set_transform(
        carla.Transform(
            spectator_location,
            spectator_rotation,
        )
    )


def drain_queues(*queues):
    for q in queues:
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break


# =======================
# MAIN
# =======================

def main():
    parser = argparse.ArgumentParser(
        description="CARLA-only trajectory replay (SD branch, with instance color mapping)"
    )
    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help="Bag filename including .bag extension "
             "(default: env BAG_NAME or 'reference_bag.bag').",
    )
    args = parser.parse_args()

    bag_name = args.bag_name               # e.g. "reference_bag.bag"
    bag_stem = Path(bag_name).stem         # e.g. "reference_bag"

    # Build bag-dependent paths now that we know the bag.
    carla_data_folder = os.path.join(
        PROJECT_ROOT, "data", "data_for_carla", bag_stem
    )
    camera_json_path = os.path.join(
        PROJECT_ROOT, "data", "data_for_carla", "camera.json"
    )
    trajectory_path = os.path.join(
        carla_data_folder, "trajectory_positions_rear_odom_yaw.json"
    )
    instance_color_map_path = os.path.join(
        carla_data_folder, "instance_color_map.json"
    )
    output_folder = os.path.join(
        PROJECT_ROOT, "data", "processed_dataset", bag_stem,
        "carla_replay_dataset_sd",
    )

    print("=" * 80)
    print("REPLAY HERO TRAJECTORY (SD branch, with instance color mapping)")
    print("=" * 80)
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] Script folder:      {SCRIPT_DIR}")
    print(f"[INFO] Local utils:        {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag:                {bag_name}")
    print(f"[INFO] Bag stem:           {bag_stem}")
    print(f"[INFO] CARLA data folder:  {carla_data_folder}")
    print(f"[INFO] Camera json:        {camera_json_path}  (shared)")
    print(f"[INFO] Trajectory path:    {trajectory_path}")
    print(f"[INFO] Instance color map: {instance_color_map_path}")
    print(f"[INFO] Output folder:      {output_folder}")
    print(f"[INFO] CARLA:              {CARLA_IP}:{CARLA_PORT}")
    print("=" * 80)

    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(f"camera.json not found: {camera_json_path}")

    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    create_output_folders(output_folder)

    # ---------- Load instance color map ----------
    color_map = {}   # CARLA instance color tuple -> [R, G, B] from bag

    if os.path.exists(instance_color_map_path):
        color_map = load_instance_color_map(instance_color_map_path)
        print(
            f"[INFO] Loaded instance color map: "
            f"{len(color_map)} CARLA-to-bag mappings."
        )
    else:
        print(
            f"[WARN] No instance_color_map.json found at "
            f"{instance_color_map_path}."
        )
        print(
            "       Instance maps will use CARLA synthetic colors only "
            "(no bag remapping)."
        )
        print(
            "       Run 3F_OPT_generate_scenario_with_mapping.py first to "
            "produce the mapping."
        )

    # ---------- Load camera + trajectory ----------
    camera_data = load_camera_data(camera_json_path)

    fov = str(camera_data["camera"]["fov"])
    target_size = int(camera_data["size"]["x"])
    fps = int(camera_data["camera"]["fps"])

    trajectory_points = load_json_file(trajectory_path)

    if MAX_FRAMES is not None and MAX_FRAMES > 0:
        trajectory_points = trajectory_points[:MAX_FRAMES]

    if not trajectory_points:
        raise RuntimeError(f"Trajectory is empty: {trajectory_path}")

    print(f"[INFO] Loaded trajectory frames: {len(trajectory_points)}")
    print(f"[INFO] Target output size: {target_size} x {target_size}")
    print(f"[INFO] Replay FPS: {fps}")
    print(f"[INFO] Camera FOV: {fov}")

    # ---------- Connect to CARLA ----------
    print("[INFO] Connecting to existing CARLA world...")

    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(20.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)

    blueprint_library = world.get_blueprint_library()

    update_synchronous_mode(
        world,
        traffic_manager,
        USE_SYNCHRONOUS_MODE,
        fps,
    )

    for _ in range(5):
        world.tick()

    first_transform = make_trajectory_transform(trajectory_points[0])

    # ---------- Spawn / find hero ----------
    print("[INFO] Preparing hero vehicle...")

    hero_vehicle = spawn_hero_if_needed(
        world,
        blueprint_library,
        first_transform,
    )

    hero_vehicle.set_simulate_physics(False)
    hero_vehicle.set_autopilot(False)

    cleanup_old_sensors(hero_vehicle)

    # ---------- Sensors ----------
    cam_config = camera_data["camera"]

    sensor_tf = carla.Transform(
        carla.Location(
            x=float(cam_config["position"]["x"]),
            y=float(cam_config["position"]["y"]),
            z=float(cam_config["position"]["z"]),
        ),
        carla.Rotation(
            pitch=float(cam_config.get("pitch", 0.0)),
        ),
    )

    print("[INFO] Spawning sensors on hero...")

    rgb_sensor, rgb_queue = spawn_sensor(
        blueprint_library,
        "sensor.camera.rgb",
        IM_WIDTH,
        IM_HEIGHT,
        fov,
        world,
        sensor_tf,
        hero_vehicle,
    )

    sem_sensor, sem_queue = spawn_sensor(
        blueprint_library,
        "sensor.camera.semantic_segmentation",
        IM_WIDTH,
        IM_HEIGHT,
        fov,
        world,
        sensor_tf,
        hero_vehicle,
    )

    inst_sensor, inst_queue = spawn_sensor(
        blueprint_library,
        "sensor.camera.instance_segmentation",
        IM_WIDTH,
        IM_HEIGHT,
        fov,
        world,
        sensor_tf,
        hero_vehicle,
    )

    for _ in range(10):
        hero_vehicle.set_transform(first_transform)
        world.tick()
        drain_queues(rgb_queue, sem_queue, inst_queue)

    # ---------- Pygame display ----------
    display = None
    clock = None

    if ENABLE_PYGAME_DISPLAY:
        pygame.init()
        display = pygame.display.set_mode(
            (IM_WIDTH, IM_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("CARLA trajectory replay (SD branch)")
        clock = pygame.time.Clock()

    print("[INFO] Starting trajectory replay.")

    # Per-run frame metadata (was a module-level global before).
    all_frame_data = []

    try:
        for idx, point in enumerate(trajectory_points):
            target_transform = make_trajectory_transform(point)

            hero_vehicle.set_simulate_physics(False)
            hero_vehicle.set_transform(target_transform)

            if FOLLOW_HERO_WITH_SPECTATOR:
                move_spectator_to_hero(world, hero_vehicle)

            drain_queues(rgb_queue, sem_queue, inst_queue)

            world.tick()

            try:
                rgb_data = rgb_queue.get(block=True, timeout=2.0)
                sem_data = sem_queue.get(block=True, timeout=2.0)
                inst_data = inst_queue.get(block=True, timeout=2.0)
            except Empty:
                print(f"[WARN] Timeout waiting for sensors at replay index {idx}.")
                continue

            actual_transform = hero_vehicle.get_transform()
            current_transform_mapped = get_inverse_transform(actual_transform)

            frame_id = int(point.get("frame_id", idx))

            save_frame_data(
                frame_id=frame_id,
                rgb_obj=rgb_data,
                sem_obj=sem_data,
                inst_obj=inst_data,
                transform_data=current_transform_mapped,
                target_size=target_size,
                output_folder=output_folder,
                color_map=color_map,
                all_frame_data=all_frame_data,
            )

            if ENABLE_PYGAME_DISPLAY:
                rgb_array = carla_rgb_to_array(rgb_data)
                surface = pygame.surfarray.make_surface(
                    rgb_array.swapaxes(0, 1)
                )
                display.blit(surface, (0, 0))
                pygame.display.flip()
                pygame.display.set_caption(
                    f"Replay frame {idx + 1}/{len(trajectory_points)} (SD)"
                )

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise KeyboardInterrupt

                clock.tick(60)

            if (idx + 1) % 100 == 0:
                print(f"[INFO] Saved {idx + 1}/{len(trajectory_points)} frames.")

    except KeyboardInterrupt:
        print("\n[INFO] Replay interrupted by user.")

    finally:
        metadata_path = os.path.join(
            output_folder,
            "data",
            "all_frame_data.json",
        )

        print("[INFO] Saving metadata...")

        with open(metadata_path, "w") as f:
            json.dump(all_frame_data, f, indent=4)

        print(f"[INFO] Metadata saved: {metadata_path}")
        print(f"[INFO] Saved frames: {len(all_frame_data)}")

        print("[INFO] Cleaning up sensors...")

        for sensor in [rgb_sensor, sem_sensor, inst_sensor]:
            try:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass

        try:
            update_synchronous_mode(
                world,
                traffic_manager,
                False,
                fps,
            )
        except Exception:
            pass

        if ENABLE_PYGAME_DISPLAY:
            pygame.quit()

        print("[INFO] Done.")


if __name__ == "__main__":
    main()