#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize parked vehicle spawn positions in CARLA.

Reads map from:

    data/generated_data_from_extracted_data/<BAG_NAME>/maps/map.xodr

Reads final CARLA vehicle data from:

    data/data_for_carla/<BAG_NAME>/vehicle_data.json

This script:
  - uploads / loads the OpenDRIVE map into CARLA
  - reads spawn_positions from vehicle_data.json
  - spawns parked cars at those positions
  - keeps them frozen
  - moves spectator above the first parked car
"""

import os
import sys
import json
import random

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

OSM_FILE = os.path.join(
    MAP_FOLDER,
    "map.osm",
)

CARLA_DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_carla",
    BAG_NAME,
)

VEHICLE_DATA_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "vehicle_data.json",
)

# Since spawn_positions are already converted to CARLA coordinates,
# keep this False unless you intentionally want manual fine-tuning.
APPLY_VEHICLE_DATA_OFFSET = False

# Extra manual tuning if needed.
MANUAL_OFFSET_X = 0.0
MANUAL_OFFSET_Y = 0.0
MANUAL_HEADING_OFFSET = 0.0

# Spawn height offset to avoid clipping into the road.
SPAWN_Z_OFFSET = 0.5

# Use deterministic random vehicle choices.
RANDOM_SEED = 42


# =======================
# LOCAL UTILS IMPORTS
# =======================

from utils.config import CARLA_IP, CARLA_PORT

from utils.carla_simulator import (
    generate_world_map,
    get_filtered_vehicle_blueprints,
    get_osm_center,
    get_xodr_geo_reference,
)


# =======================
# HELPERS
# =======================

def load_text_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        return f.read()


def load_vehicle_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"vehicle_data.json not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "spawn_positions" not in data:
        raise RuntimeError(
            f"vehicle_data.json has no 'spawn_positions' key: {path}"
        )

    return data


def get_offsets(vehicle_data):
    offset = vehicle_data.get(
        "offset",
        {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
        },
    )

    if offset is None:
        offset = {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
        }

    if APPLY_VEHICLE_DATA_OFFSET:
        offset_x = float(offset.get("x", 0.0)) + MANUAL_OFFSET_X
        offset_y = float(offset.get("y", 0.0)) + MANUAL_OFFSET_Y
        heading_offset = float(offset.get("heading", 0.0)) + MANUAL_HEADING_OFFSET
    else:
        offset_x = MANUAL_OFFSET_X
        offset_y = MANUAL_OFFSET_Y
        heading_offset = MANUAL_HEADING_OFFSET

    return offset_x, offset_y, heading_offset


def print_map_debug(xodr_data):
    if os.path.exists(OSM_FILE):
        try:
            osm_center_lat, osm_center_lon = get_osm_center(OSM_FILE)
            print("[INFO] OSM center:")
            print(f"       lat={osm_center_lat:.6f}, lon={osm_center_lon:.6f}")
        except Exception as e:
            print("[WARN] Could not read OSM center:")
            print(f"       {e}")

    xodr_proj = get_xodr_geo_reference(xodr_data)

    if xodr_proj:
        print("[INFO] XODR projection:")
        print(f"       {xodr_proj}")


def choose_vehicle_blueprint(vehicle_bp_list):
    if not vehicle_bp_list:
        raise RuntimeError("No valid vehicle blueprints found.")

    return random.choice(vehicle_bp_list)


# =======================
# MAIN
# =======================

def main():
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("VISUALIZE PARKED VEHICLE SPAWN POSITIONS")
    print("=" * 80)
    print(f"[INFO] Project root:         {PROJECT_ROOT}")
    print(f"[INFO] Script folder:        {SCRIPT_DIR}")
    print(f"[INFO] Local utils:          {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag name:             {BAG_NAME}")
    print(f"[INFO] Map folder:           {MAP_FOLDER}")
    print(f"[INFO] XODR file:            {XODR_FILE}")
    print(f"[INFO] Vehicle data path:    {VEHICLE_DATA_PATH}")
    print(f"[INFO] CARLA:                {CARLA_IP}:{CARLA_PORT}")
    print(f"[INFO] Apply JSON offset:    {APPLY_VEHICLE_DATA_OFFSET}")
    print("=" * 80)

    # 1. Load input files.
    xodr_data = load_text_file(XODR_FILE)
    vehicle_data = load_vehicle_data(VEHICLE_DATA_PATH)

    spawn_list = vehicle_data.get("spawn_positions", [])

    print(f"[INFO] Loaded {len(spawn_list)} parked vehicle spawn positions.")

    if not spawn_list:
        print("[WARN] No spawn_positions found. Nothing to spawn.")
        return

    print_map_debug(xodr_data)

    # 2. Connect to CARLA.
    print(f"\n[INFO] Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(30.0)

    # 3. Upload / generate the OpenDRIVE map.
    print("[INFO] Loading OpenDRIVE map into CARLA...")
    world = generate_world_map(client, xodr_data)

    # 4. Prepare vehicle blueprints.
    vehicle_bp_list = get_filtered_vehicle_blueprints(world)
    print(f"[INFO] Available vehicle blueprints: {len(vehicle_bp_list)}")

    # 5. Offset setup.
    offset_x, offset_y, heading_offset = get_offsets(vehicle_data)

    print("[INFO] Final offsets:")
    print(f"       x={offset_x:.3f}")
    print(f"       y={offset_y:.3f}")
    print(f"       heading={heading_offset:.3f}")

    spawned_actors = []

    try:
        # Use asynchronous mode for simple visualization.
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print("\n[INFO] Spawning parked vehicles...")

        failed_count = 0

        for item in spawn_list:
            raw_loc = item.get("start")
            heading = float(item.get("heading", 0.0))

            if raw_loc is None or len(raw_loc) < 3:
                failed_count += 1
                continue

            x = float(raw_loc[0]) + offset_x
            y = float(raw_loc[1]) + offset_y
            z = float(raw_loc[2]) + SPAWN_Z_OFFSET

            yaw = (heading + heading_offset) % 360.0

            carla_loc = carla.Location(
                x=x,
                y=y,
                z=z,
            )

            carla_rot = carla.Rotation(
                pitch=0.0,
                yaw=yaw,
                roll=0.0,
            )

            transform = carla.Transform(carla_loc, carla_rot)

            bp = choose_vehicle_blueprint(vehicle_bp_list)

            vehicle = world.try_spawn_actor(bp, transform)

            if vehicle is None:
                failed_count += 1
                continue

            vehicle.set_simulate_physics(False)
            spawned_actors.append(vehicle)

        print(f"[INFO] Successfully spawned {len(spawned_actors)} vehicles.")
        print(f"[INFO] Failed spawns: {failed_count}")

        # 6. Move spectator above first parked car.
        if spawned_actors:
            spectator = world.get_spectator()
            first_transform = spawned_actors[0].get_transform()

            spectator_location = first_transform.location + carla.Location(z=50.0)

            spectator_rotation = carla.Rotation(
                pitch=-90.0,
                yaw=first_transform.rotation.yaw,
                roll=0.0,
            )

            spectator.set_transform(
                carla.Transform(
                    spectator_location,
                    spectator_rotation,
                )
            )

            print("[INFO] Spectator moved above first parked car.")

        print("\n[INFO] Press Ctrl+C to exit and destroy spawned vehicles.")

        while True:
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print(f"[INFO] Destroying {len(spawned_actors)} spawned vehicles...")

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