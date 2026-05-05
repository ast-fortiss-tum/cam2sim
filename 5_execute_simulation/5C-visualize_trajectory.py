#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize hero trajectory as static ghost cars in CARLA.

Reads map from:

    data/generated_data_from_extracted_data/<BAG_NAME>/maps/map.xodr

Reads trajectory from:

    data/data_for_carla/<BAG_NAME>/trajectory_positions_odom_yaw.json

This script:
  - uploads / loads the OpenDRIVE map into CARLA
  - reads the converted CARLA trajectory
  - spawns static ghost cars along the trajectory
  - freezes them
  - moves spectator near the first ghost car
"""

import os
import sys
import json

import carla


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
# HARDCODED CONFIG
# =======================

# Change this to select another bag.
# No command-line parameters are used.
BAG_NAME = "reference_bag"

MAP_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "generated_data_from_extracted_data",
    BAG_NAME,
    "maps",
)

XODR_FILE = os.path.join(
    MAP_FOLDER,
    "map.xodr",
)

CARLA_DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_carla",
    BAG_NAME,
)

TRAJECTORY_FILE = os.path.join(
    CARLA_DATA_FOLDER,
    "trajectory_positions_odom_yaw.json",
)

VEHICLE_DATA_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "vehicle_data.json",
)

# Spawn every Nth frame to avoid overcrowding.
SPAWN_STEP = 5

# Z offset to avoid clipping into the road.
SPAWN_Z_OFFSET = 0.1

# Since trajectory_positions_odom_yaw.json is already in CARLA coordinates,
# keep this at 0 unless you intentionally need visual fine tuning.
MANUAL_OFFSET_X = 0.0
MANUAL_OFFSET_Y = 0.0
MANUAL_Z_OFFSET = 0.0
MANUAL_YAW_OFFSET = 0.0


# =======================
# LOCAL UTILS IMPORTS
# =======================

from utils.config import (
    CARLA_IP,
    CARLA_PORT,
    ROTATION_DEGREES,
    HERO_VEHICLE_TYPE,
)

from utils.carla_simulator import generate_world_map


# =======================
# HELPERS
# =======================

def load_text_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        return f.read()


def load_json_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_hero_blueprint(blueprint_library):
    """
    Load configured hero vehicle blueprint.
    Falls back to Tesla Model 3 if the configured blueprint is unavailable.
    """
    try:
        hero_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
    except Exception:
        print(f"[WARN] Could not find HERO_VEHICLE_TYPE: {HERO_VEHICLE_TYPE}")
        print("[WARN] Falling back to vehicle.tesla.model3")
        hero_bp = blueprint_library.find("vehicle.tesla.model3")

    if hero_bp.has_attribute("color"):
        hero_bp.set_attribute("color", "255,0,0")

    return hero_bp


# =======================
# MAIN
# =======================

def main():
    print("=" * 80)
    print("VISUALIZE HERO TRAJECTORY AS STATIC CARS")
    print("=" * 80)
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] Script folder:      {SCRIPT_DIR}")
    print(f"[INFO] Local utils:        {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag name:           {BAG_NAME}")
    print(f"[INFO] Map folder:         {MAP_FOLDER}")
    print(f"[INFO] XODR file:          {XODR_FILE}")
    print(f"[INFO] CARLA data folder:  {CARLA_DATA_FOLDER}")
    print(f"[INFO] Trajectory file:    {TRAJECTORY_FILE}")
    print(f"[INFO] Vehicle data path:  {VEHICLE_DATA_PATH}")
    print(f"[INFO] CARLA:              {CARLA_IP}:{CARLA_PORT}")
    print(f"[INFO] Hero vehicle type:  {HERO_VEHICLE_TYPE}")
    print(f"[INFO] Spawn step:         {SPAWN_STEP}")
    print("=" * 80)

    # 1. Load map and trajectory.
    xodr_data = load_text_file(XODR_FILE)
    trajectory_points = load_json_file(TRAJECTORY_FILE)

    if not trajectory_points:
        raise RuntimeError(f"Trajectory file is empty: {TRAJECTORY_FILE}")

    print(f"[INFO] Loaded {len(trajectory_points)} trajectory frames.")

    # Optional: load vehicle_data only for debug / consistency.
    if os.path.exists(VEHICLE_DATA_PATH):
        vehicle_data = load_json_file(VEHICLE_DATA_PATH)
        hero_car = vehicle_data.get("hero_car")
        if hero_car is not None:
            print("[INFO] vehicle_data.json hero_car:")
            print(f"       position={hero_car.get('position')}")
            print(f"       heading={hero_car.get('heading')}")
    else:
        print("[WARN] vehicle_data.json not found. Continuing with trajectory only.")

    # 2. Connect to CARLA.
    print(f"\n[INFO] Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(30.0)

    # 3. Upload / generate OpenDRIVE world.
    print("[INFO] Loading OpenDRIVE map into CARLA...")
    world = generate_world_map(client, xodr_data)

    blueprint_library = world.get_blueprint_library()
    hero_bp = get_hero_blueprint(blueprint_library)

    spawned_actors = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print("\n[INFO] Spawning trajectory ghost cars...")

        failed_count = 0

        for i in range(0, len(trajectory_points), SPAWN_STEP):
            point = trajectory_points[i]

            tf_data = point.get("transform", {})
            loc_data = tf_data.get("location", {})
            rot_data = tf_data.get("rotation", {})

            if not loc_data or not rot_data:
                failed_count += 1
                continue

            raw_x = float(loc_data.get("x", 0.0))
            raw_y = float(loc_data.get("y", 0.0))
            raw_z = float(loc_data.get("z", 0.0))
            raw_yaw = float(rot_data.get("yaw", 0.0))

            final_x = raw_x + MANUAL_OFFSET_X
            final_y = raw_y + MANUAL_OFFSET_Y
            final_z = raw_z + SPAWN_Z_OFFSET + MANUAL_Z_OFFSET

            # trajectory_positions_odom_yaw.json is already in CARLA yaw.
            # ROTATION_DEGREES is kept for compatibility with your config.
            final_yaw = (
                raw_yaw
                + ROTATION_DEGREES
                + MANUAL_YAW_OFFSET
            ) % 360.0

            carla_loc = carla.Location(
                x=final_x,
                y=final_y,
                z=final_z,
            )

            carla_rot = carla.Rotation(
                pitch=0.0,
                yaw=final_yaw,
                roll=0.0,
            )

            transform = carla.Transform(carla_loc, carla_rot)

            ghost_car = world.try_spawn_actor(hero_bp, transform)

            if ghost_car is None:
                frame_id = point.get("frame_id", i)
                print(f"[WARN] Collision or invalid spawn at frame {frame_id}, skipping.")
                failed_count += 1
                continue

            ghost_car.set_simulate_physics(False)
            spawned_actors.append(ghost_car)

        print(f"[INFO] Successfully spawned {len(spawned_actors)} ghost cars.")
        print(f"[INFO] Failed spawns: {failed_count}")

        # 4. Move spectator to first ghost car.
        if spawned_actors:
            spectator = world.get_spectator()
            first_car_tf = spawned_actors[0].get_transform()

            spectator_location = first_car_tf.location + carla.Location(
                z=30.0,
                x=-10.0,
            )

            spectator_rotation = carla.Rotation(
                pitch=-60.0,
                yaw=first_car_tf.rotation.yaw,
                roll=0.0,
            )

            spectator.set_transform(
                carla.Transform(
                    spectator_location,
                    spectator_rotation,
                )
            )

            print("[INFO] Spectator moved near first ghost car.")

        print("-" * 80)
        print(f"[INFO] Visualization active. Press Ctrl+C to clear {len(spawned_actors)} cars.")
        print("-" * 80)

        while True:
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Cleaning up...")

    finally:
        print(f"[INFO] Destroying {len(spawned_actors)} ghost actors...")

        if spawned_actors:
            client.apply_batch(
                [
                    carla.command.DestroyActor(actor)
                    for actor in spawned_actors
                ]
            )

        print("[INFO] Done.")


if __name__ == "__main__":
    main()