#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run DAVE-2 in an already-prepared CARLA world.

This version:
  - does NOT load the map
  - does NOT use Stable Diffusion
  - does NOT use YOLO
  - sends raw CARLA RGB directly to DAVE-2
  - expects the prepare-world script to have already loaded the map and spawned cars

Expected workflow:
  1. Start CARLA server.
  2. Run prepare-world script once.
  3. Start DAVE-2 socket server.
  4. Run this script.

Reads camera config from:
  data/data_for_carla/<BAG_NAME>/camera.json

Reads trajectory start from:
  data/data_for_carla/<BAG_NAME>/trajectory_positions_rear_odom_yaw.json

Saves output to:
  data/processed_dataset/<BAG_NAME>/dave2_runs/only_carla_run<RUN_NUMBER>
"""

import os
import sys
import json
import math
import time
from queue import Empty

import carla
import numpy as np
import pygame
import cv2
from PIL import Image, ImageOps


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
    ROTATION_DEGREES,
)

from utils.carla_simulator import (
    update_synchronous_mode,
    remove_sensor,
    remap_segmentation_colors,
    carla_image_to_pil,
    cleanup_old_sensors,
    spawn_sensor,
)

from utils.dave2_connection import (
    connect_to_dave2_server,
    send_image_over_connection,
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

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_dataset",
    BAG_NAME,
    "dave2_runs",
)

RUN_NUMBER = 1

OUTPUT_FOLDER = os.path.join(
    OUTPUT_ROOT,
    f"only_carla_run{RUN_NUMBER}",
)

IM_WIDTH = 800
IM_HEIGHT = 503

DEFAULT_TARGET_SIZE = 512
DEFAULT_FPS = 30
DEFAULT_FOV = 54.7

MAX_FRAMES = None
# MAX_FRAMES = 1000

NO_SAVE = False

USE_SYNCHRONOUS_MODE = True

# Keep old working logic.
REAR_TO_CENTER_OFFSET_METERS = 0.13
HERO_SPAWN_Z_OFFSET = 0.10

WARMUP_TICKS = 100

DRIVE_SPEED_KMH = 10.0
LAUNCH_SPEED_KMH = 12.0

STUCK_THRESHOLD = 0.02
STUCK_FRAME_LIMIT = 100
MIN_Z_THRESHOLD = -0.5

MAX_ROTATION_RAD = 3.0 * math.pi
INVERT_STEERING = True

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


def get_number(value, default):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_int(value, default):
    return int(get_number(value, default))


def load_camera_data(path):
    raw = load_json_file(path)

    camera = raw.get("camera", raw)

    if not isinstance(camera, dict):
        raise RuntimeError(f"Invalid camera.json format: {path}")

    position = camera.get("position", raw.get("position"))

    if position is None:
        raise KeyError(
            "camera.json must contain camera.position or position with x/y/z."
        )

    target_size = DEFAULT_TARGET_SIZE

    if isinstance(raw.get("size"), dict) and "x" in raw["size"]:
        target_size = int(raw["size"]["x"])
    elif "target_size" in raw:
        target_size = int(raw["target_size"])
    elif "output_size" in raw:
        target_size = int(raw["output_size"])

    fps = get_int(camera.get("fps", raw.get("fps")), DEFAULT_FPS)
    fov = get_number(camera.get("fov", raw.get("fov")), DEFAULT_FOV)
    pitch = get_number(camera.get("pitch", raw.get("pitch")), 0.0)

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

    print("[INFO] Loaded camera config:")
    print(f"       fov:         {fov}")
    print(f"       fps:         {fps}")
    print(f"       target size: {target_size}")
    print(
        f"       position:    "
        f"x={position['x']}, y={position['y']}, z={position['z']}"
    )
    print(f"       pitch:       {pitch}")

    return camera_data


def create_output_folders():
    for subdir in [
        "rgb",
        "semantic",
        "instance",
        "depth",
        "data",
    ]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, subdir), exist_ok=True)


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


def carla_rgb_to_pil(rgb_data):
    rgb_image_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
    rgb_image_np = rgb_image_np.reshape(
        (rgb_data.height, rgb_data.width, 4)
    )
    rgb_image_np = rgb_image_np[:, :, :3][:, :, ::-1]
    return Image.fromarray(rgb_image_np)


def carla_rgb_to_array(rgb_data):
    rgb_image_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
    rgb_image_np = rgb_image_np.reshape(
        (rgb_data.height, rgb_data.width, 4)
    )
    rgb_image_np = rgb_image_np[:, :, :3][:, :, ::-1]
    return rgb_image_np


def process_depth_to_pil(depth_data):
    depth_data.convert(carla.ColorConverter.LogarithmicDepth)

    depth_image_np = np.array(carla_image_to_pil(depth_data))

    depth_image_np = cv2.filter2D(
        depth_image_np,
        -1,
        np.ones((3, 3), np.float32) / 10,
    )

    return ImageOps.invert(Image.fromarray(depth_image_np))


def save_data(
    frame_id,
    final_rgb,
    final_seg,
    final_inst,
    final_depth,
):
    filename = f"{frame_id:06d}"

    final_rgb.save(os.path.join(OUTPUT_FOLDER, "rgb", f"{filename}.png"))
    final_seg.save(os.path.join(OUTPUT_FOLDER, "semantic", f"{filename}.png"))
    final_inst.save(os.path.join(OUTPUT_FOLDER, "instance", f"{filename}.png"))
    final_depth.save(os.path.join(OUTPUT_FOLDER, "depth", f"{filename}.png"))


def drain_queues(*queues):
    for q in queues:
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break


def find_existing_hero(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for vehicle in vehicles:
        try:
            if vehicle.attributes.get("role_name") == "hero":
                return vehicle
        except Exception:
            pass

    if vehicles:
        print("[WARN] No role_name=hero actor found.")
        print(f"[WARN] Using first existing vehicle as hero: {vehicles[0].id}")
        return vehicles[0]

    return None


def stop_vehicle_motion(vehicle):
    try:
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except Exception:
        pass

    try:
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except Exception:
        pass


def enable_vehicle_physics(vehicle):
    try:
        vehicle.set_enable_gravity(True)
    except Exception:
        pass

    vehicle.set_simulate_physics(True)
    vehicle.set_autopilot(False)


def make_start_transform(world, first_point):
    loc = first_point["transform"]["location"]
    rot = first_point["transform"]["rotation"]

    start_x = float(loc["x"])
    start_y = float(loc["y"])
    start_z = float(loc["z"])

    # Keep old working logic.
    start_yaw = (float(rot.get("yaw", 0.0)) + ROTATION_DEGREES) % 360.0

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
        print("[INFO] Using waypoint z for hero spawn.")
        print(f"       trajectory z: {start_z:.3f}")
        print(f"       waypoint z:   {waypoint.transform.location.z:.3f}")
        print(f"       spawn z:      {spawn_z:.3f}")
    else:
        spawn_z = start_z + HERO_SPAWN_Z_OFFSET
        print("[WARN] No waypoint found. Using trajectory z + offset.")

    start_transform = carla.Transform(
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

    print("[INFO] Hero start transform:")
    print(
        f"       x={start_transform.location.x:.3f}, "
        f"y={start_transform.location.y:.3f}, "
        f"z={start_transform.location.z:.3f}, "
        f"yaw={start_transform.rotation.yaw:.3f}"
    )

    return start_transform


def prepare_hero_for_driving(world, blueprint_library, start_transform):
    hero_vehicle = find_existing_hero(world)

    if hero_vehicle is not None:
        print(f"[INFO] Found existing hero/vehicle actor: {hero_vehicle.id}")
        print("[INFO] Teleporting existing hero to DAVE-2 start.")

        hero_vehicle.set_simulate_physics(False)
        stop_vehicle_motion(hero_vehicle)
        hero_vehicle.set_transform(start_transform)

        for _ in range(10):
            world.tick()
            hero_vehicle.set_transform(start_transform)
            stop_vehicle_motion(hero_vehicle)

    else:
        print("[INFO] No existing hero found. Spawning new hero.")

        vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)

        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")

        if vehicle_bp.has_attribute("color"):
            vehicle_bp.set_attribute("color", "255,0,0")

        hero_vehicle = world.spawn_actor(vehicle_bp, start_transform)

        hero_vehicle.set_simulate_physics(False)
        hero_vehicle.set_autopilot(False)

        for _ in range(10):
            world.tick()
            hero_vehicle.set_transform(start_transform)
            stop_vehicle_motion(hero_vehicle)

    enable_vehicle_physics(hero_vehicle)
    stop_vehicle_motion(hero_vehicle)

    for _ in range(10):
        world.tick()

    actual = hero_vehicle.get_transform()

    print("[INFO] Hero ready for DAVE-2 driving.")
    print(f"       actor id: {hero_vehicle.id}")
    print(
        f"       actual x={actual.location.x:.3f}, "
        f"y={actual.location.y:.3f}, "
        f"z={actual.location.z:.3f}, "
        f"yaw={actual.rotation.yaw:.3f}"
    )

    return hero_vehicle


def move_spectator_to_hero(world, hero_vehicle):
    spectator = world.get_spectator()
    hero_transform = hero_vehicle.get_transform()

    spectator_loc = (
        hero_transform.location
        - hero_transform.get_forward_vector() * 10.0
    )
    spectator_loc.z += 5.0

    spectator_rot = hero_transform.rotation
    spectator_rot.pitch = -15.0

    spectator.set_transform(
        carla.Transform(
            spectator_loc,
            spectator_rot,
        )
    )


def pil_to_pygame_surface(pil_image):
    image = np.array(pil_image)
    return pygame.surfarray.make_surface(image.swapaxes(0, 1))


# =======================
# MAIN
# =======================

def main():
    print("=" * 80)
    print("DAVE-2 ONLY-CARLA DRIVING")
    print("=" * 80)
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] Script folder:      {SCRIPT_DIR}")
    print(f"[INFO] Local utils:        {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag name:           {BAG_NAME}")
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

    fov = float(camera_data["camera"]["fov"])
    fov_str = str(fov)
    fps = int(camera_data["camera"]["fps"])
    target_size = int(camera_data["size"]["x"])
    target_res = (target_size, target_size)

    full_trajectory = load_json_file(TRAJECTORY_PATH)

    if MAX_FRAMES is not None and MAX_FRAMES > 0:
        full_trajectory = full_trajectory[:MAX_FRAMES]

    if not full_trajectory:
        raise RuntimeError(f"Trajectory is empty: {TRAJECTORY_PATH}")

    print(f"[INFO] Loaded trajectory frames: {len(full_trajectory)}")
    print(f"[INFO] Target size: {target_size}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] FOV: {fov}")

    print("[INFO] Connecting to CARLA world...")

    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(40.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    blueprint_library = world.get_blueprint_library()

    rgb_sensor = None
    sem_sensor = None
    inst_sensor = None
    depth_sensor = None

    pygame_screen = None
    pygame_clock = None

    trajectory_log = []

    try:
        update_synchronous_mode(
            world,
            traffic_manager,
            USE_SYNCHRONOUS_MODE,
            fps,
        )

        for _ in range(10):
            world.tick()

        start_transform = make_start_transform(world, full_trajectory[0])

        hero_vehicle = prepare_hero_for_driving(
            world,
            blueprint_library,
            start_transform,
        )

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
            fov_str,
            world,
            sensor_tf,
            hero_vehicle,
        )

        sem_sensor, sem_queue = spawn_sensor(
            blueprint_library,
            "sensor.camera.semantic_segmentation",
            IM_WIDTH,
            IM_HEIGHT,
            fov_str,
            world,
            sensor_tf,
            hero_vehicle,
        )

        inst_sensor, inst_queue = spawn_sensor(
            blueprint_library,
            "sensor.camera.instance_segmentation",
            IM_WIDTH,
            IM_HEIGHT,
            fov_str,
            world,
            sensor_tf,
            hero_vehicle,
        )

        depth_sensor, depth_queue = spawn_sensor(
            blueprint_library,
            "sensor.camera.depth",
            IM_WIDTH,
            IM_HEIGHT,
            fov_str,
            world,
            sensor_tf,
            hero_vehicle,
        )

        for _ in range(10):
            world.tick()

        if ENABLE_PYGAME_DISPLAY:
            pygame.init()
            pygame.display.set_caption("DAVE-2 only CARLA")
            pygame_screen = pygame.display.set_mode(
                (target_size * 2, target_size * 2)
            )
            pygame_clock = pygame.time.Clock()

        print("[INFO] Connecting to DAVE-2 server...")
        dave2_conn = connect_to_dave2_server()

        print(f"[INFO] Warmup with launch control: {WARMUP_TICKS} ticks")

        launch_control = carla.VehicleAckermannControl(
            speed=float(LAUNCH_SPEED_KMH / 3.6),
            steer=0.0,
        )

        drain_queues(rgb_queue, sem_queue, inst_queue, depth_queue)

        for _ in range(WARMUP_TICKS):
            hero_vehicle.apply_ackermann_control(launch_control)
            world.tick()
            drain_queues(rgb_queue, sem_queue, inst_queue, depth_queue)

        time.sleep(0.1)
        drain_queues(rgb_queue, sem_queue, inst_queue, depth_queue)

        frame = 0
        stuck_counter = 0
        prev_loc = hero_vehicle.get_location()

        print("[INFO] Starting DAVE-2 driving loop.")

        while True:
            frame += 1

            if MAX_FRAMES is not None and frame > MAX_FRAMES:
                print(f"[INFO] Reached MAX_FRAMES={MAX_FRAMES}. Stopping.")
                break

            world.tick()

            cur_loc = hero_vehicle.get_location()

            if cur_loc.z < MIN_Z_THRESHOLD:
                print(
                    f"[ERROR] Car fell below threshold "
                    f"(z={cur_loc.z:.2f}). Stopping."
                )
                break

            if cur_loc.distance(prev_loc) < STUCK_THRESHOLD:
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter > STUCK_FRAME_LIMIT:
                print("[ERROR] Car appears stuck. Stopping.")
                break

            prev_loc = cur_loc

            move_spectator_to_hero(world, hero_vehicle)

            try:
                rgb_data = rgb_queue.get(block=True, timeout=1.0)
                sem_data = sem_queue.get(block=True, timeout=1.0)
                inst_data = inst_queue.get(block=True, timeout=1.0)
                depth_data = depth_queue.get(block=True, timeout=1.0)
            except Empty:
                print(f"[WARN] Timeout waiting for sensor data at frame {frame}.")
                continue

            rgb_image_pil = carla_rgb_to_pil(rgb_data)

            sem_data.convert(carla.ColorConverter.CityScapesPalette)
            seg_image = remap_segmentation_colors(carla_image_to_pil(sem_data))

            # No instance map in this version.
            instance_pil = carla_image_to_pil(inst_data)

            depth_image_pil = process_depth_to_pil(depth_data)

            seg_image, instance_pil = clean_semantic_and_instance(
                seg_image,
                instance_pil,
            )

            final_rgb = rgb_image_pil.resize(
                target_res,
                resample=Image.Resampling.LANCZOS,
            )

            final_seg = seg_image.resize(
                target_res,
                resample=Image.Resampling.NEAREST,
            )

            final_inst = instance_pil.resize(
                target_res,
                resample=Image.Resampling.NEAREST,
            )

            final_depth = depth_image_pil.resize(
                target_res,
                resample=Image.Resampling.BILINEAR,
            )

            raw_steering_rad, raw_throttle = send_image_over_connection(
                dave2_conn,
                final_rgb,
            )

            normalized_steer = raw_steering_rad / MAX_ROTATION_RAD

            if INVERT_STEERING:
                applied_steer = normalized_steer * -1.0
            else:
                applied_steer = normalized_steer

            print(
                f"Frame: {frame} | "
                f"Steer raw: {raw_steering_rad:.4f} | "
                f"Steer norm: {normalized_steer:.4f} | "
                f"Applied: {applied_steer:.4f} | "
                f"x={cur_loc.x:.2f}, y={cur_loc.y:.2f}, z={cur_loc.z:.3f}"
            )

            hero_tf = hero_vehicle.get_transform()

            trajectory_log.append(
                {
                    "frame": frame,
                    "x": round(cur_loc.x, 4),
                    "y": round(cur_loc.y, 4),
                    "z": round(cur_loc.z, 4),
                    "yaw": round(hero_tf.rotation.yaw, 4),
                    "steering_raw_rad": round(float(raw_steering_rad), 6),
                    "steering_normalized": round(float(normalized_steer), 6),
                    "steering_applied": round(float(applied_steer), 6),
                    "throttle_raw": round(float(raw_throttle), 6),
                }
            )

            if ENABLE_PYGAME_DISPLAY:
                display_image = Image.new(
                    "RGB",
                    (target_size * 2, target_size * 2),
                )

                display_image.paste(final_rgb, (0, 0))
                display_image.paste(final_rgb, (target_size, 0))
                display_image.paste(final_seg, (0, target_size))
                display_image.paste(final_inst, (target_size, target_size))

                surface = pil_to_pygame_surface(display_image)
                pygame_screen.blit(surface, (0, 0))
                pygame.display.flip()
                pygame_clock.tick(fps)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise KeyboardInterrupt

            ackermann_control = carla.VehicleAckermannControl()
            ackermann_control.speed = float(DRIVE_SPEED_KMH / 3.6)
            ackermann_control.steer = float(applied_steer)
            ackermann_control.steer_speed = 0.0
            ackermann_control.acceleration = 0.0
            ackermann_control.jerk = 0.0

            hero_vehicle.apply_ackermann_control(ackermann_control)

            if not NO_SAVE:
                save_data(
                    frame_id=frame,
                    final_rgb=final_rgb,
                    final_seg=final_seg,
                    final_inst=final_inst,
                    final_depth=final_depth,
                )

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted.")

    finally:
        print("[INFO] Cleaning up sensors...")

        for sensor in [
            rgb_sensor,
            sem_sensor,
            inst_sensor,
            depth_sensor,
        ]:
            try:
                if sensor is not None:
                    remove_sensor(sensor)
            except Exception:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass

        if trajectory_log:
            traj_file = os.path.join(OUTPUT_FOLDER, "data", "trajectory.json")

            with open(traj_file, "w") as f:
                json.dump(trajectory_log, f, indent=2)

            print(f"[INFO] Trajectory saved: {traj_file}")
            print(f"[INFO] Logged frames: {len(trajectory_log)}")

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