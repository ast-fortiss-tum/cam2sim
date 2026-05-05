#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay hero trajectory in an already-loaded CARLA world and save sensor data.

This script does NOT load the map.

It assumes another script already:
  - loaded the OpenDRIVE map into CARLA
  - spawned parked cars
  - optionally spawned the hero car

Reads camera config from:
  data/data_for_carla/<BAG_NAME>/camera.json

Reads trajectory from:
  data/data_for_carla/<BAG_NAME>/trajectory_positions_rear_odom_yaw.json

Saves output to:
  data/generated_data_from_extracted_data/<BAG_NAME>/carla_replay_dataset

The utils folder is expected next to this script.
"""

import os
import sys
import json
import math
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
)


# =======================
# HARDCODED CONFIG
# =======================

BAG_NAME = "reference_bag"

CARLA_DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_carla",
    BAG_NAME,
)

CAMERA_JSON_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "camera.json",
)

TRAJECTORY_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "trajectory_positions_rear_odom_yaw.json",
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "generated_data_from_extracted_data",
    BAG_NAME,
    "carla_replay_dataset",
)

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
# GLOBAL STORAGE
# =======================

ALL_FRAME_DATA = []


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
    Robust camera.json loader.

    Supports either:

    1. Nested format:
       {
         "size": {"x": 512, "y": 512},
         "camera": {
           "fov": 90,
           "fps": 20,
           "position": {"x": ..., "y": ..., "z": ...},
           "pitch": ...
         }
       }

    2. Direct camera format:
       {
         "fov": 90,
         "fps": 20,
         "position": {"x": ..., "y": ..., "z": ...},
         "pitch": ...
       }

    If size is missing, DEFAULT_TARGET_SIZE is used.
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


def create_output_folders():
    os.makedirs(os.path.join(OUTPUT_FOLDER, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "instance"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)


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
):
    filename = f"{frame_id:06d}"
    target_res = (target_size, target_size)

    rgb_pil = carla_rgb_to_pil(rgb_obj)

    final_rgb = rgb_pil.resize(
        target_res,
        resample=Image.Resampling.LANCZOS,
    )

    final_rgb.save(
        os.path.join(OUTPUT_FOLDER, "rgb", f"{filename}.png")
    )

    sem_obj.convert(carla.ColorConverter.CityScapesPalette)
    sem_pil_raw = remap_segmentation_colors(carla_image_to_pil(sem_obj))

    # No instance_map.txt is used anymore.
    # This saves CARLA's raw instance segmentation colors.
    inst_pil_raw = carla_image_to_pil(inst_obj)

    sem_cleaned, inst_cleaned = clean_semantic_and_instance(
        sem_pil_raw,
        inst_pil_raw,
    )

    final_sem = sem_cleaned.resize(
        target_res,
        resample=Image.Resampling.NEAREST,
    )

    final_inst = inst_cleaned.resize(
        target_res,
        resample=Image.Resampling.NEAREST,
    )

    final_sem.save(
        os.path.join(OUTPUT_FOLDER, "semantic", f"{filename}.png")
    )

    final_inst.save(
        os.path.join(OUTPUT_FOLDER, "instance", f"{filename}.png")
    )

    ALL_FRAME_DATA.append(
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
    print("=" * 80)
    print("REPLAY HERO TRAJECTORY AND SAVE SENSOR DATA")
    print("=" * 80)
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] Script folder:      {SCRIPT_DIR}")
    print(f"[INFO] Local utils:        {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag name:           {BAG_NAME}")
    print(f"[INFO] CARLA data folder:  {CARLA_DATA_FOLDER}")
    print(f"[INFO] Camera json:        {CAMERA_JSON_PATH}")
    print(f"[INFO] Trajectory path:    {TRAJECTORY_PATH}")
    print(f"[INFO] Output folder:      {OUTPUT_FOLDER}")
    print(f"[INFO] CARLA:              {CARLA_IP}:{CARLA_PORT}")
    print("=" * 80)

    if not os.path.exists(CAMERA_JSON_PATH):
        raise FileNotFoundError(f"camera.json not found: {CAMERA_JSON_PATH}")

    if not os.path.exists(TRAJECTORY_PATH):
        raise FileNotFoundError(f"Trajectory file not found: {TRAJECTORY_PATH}")

    create_output_folders()

    camera_data = load_camera_data(CAMERA_JSON_PATH)

    fov = str(camera_data["camera"]["fov"])
    target_size = int(camera_data["size"]["x"])
    fps = int(camera_data["camera"]["fps"])

    trajectory_points = load_json_file(TRAJECTORY_PATH)

    if MAX_FRAMES is not None and MAX_FRAMES > 0:
        trajectory_points = trajectory_points[:MAX_FRAMES]

    if not trajectory_points:
        raise RuntimeError(f"Trajectory is empty: {TRAJECTORY_PATH}")

    print(f"[INFO] Loaded trajectory frames: {len(trajectory_points)}")
    print(f"[INFO] Target output size: {target_size} x {target_size}")
    print(f"[INFO] Replay FPS: {fps}")
    print(f"[INFO] Camera FOV: {fov}")

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

    print("[INFO] Preparing hero vehicle...")

    hero_vehicle = spawn_hero_if_needed(
        world,
        blueprint_library,
        first_transform,
    )

    hero_vehicle.set_simulate_physics(False)
    hero_vehicle.set_autopilot(False)

    cleanup_old_sensors(hero_vehicle)

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

    display = None
    clock = None

    if ENABLE_PYGAME_DISPLAY:
        pygame.init()
        display = pygame.display.set_mode(
            (IM_WIDTH, IM_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("CARLA trajectory replay")
        clock = pygame.time.Clock()

    print("[INFO] Starting trajectory replay.")

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
            )

            if ENABLE_PYGAME_DISPLAY:
                rgb_array = carla_rgb_to_array(rgb_data)
                surface = pygame.surfarray.make_surface(
                    rgb_array.swapaxes(0, 1)
                )
                display.blit(surface, (0, 0))
                pygame.display.flip()
                pygame.display.set_caption(
                    f"Replay frame {idx + 1}/{len(trajectory_points)}"
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
            OUTPUT_FOLDER,
            "data",
            "all_frame_data.json",
        )

        print("[INFO] Saving metadata...")

        with open(metadata_path, "w") as f:
            json.dump(ALL_FRAME_DATA, f, indent=4)

        print(f"[INFO] Metadata saved: {metadata_path}")
        print(f"[INFO] Saved frames: {len(ALL_FRAME_DATA)}")

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