#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spawn the hero car and parked cars in CARLA.

This version is simplified for the new pipeline.

Reads map from:

    data/generated_data_from_extracted_data/<BAG_NAME>/maps/map.xodr

Reads CARLA-ready vehicle data from:

    data/data_for_carla/<BAG_NAME>/vehicle_data.json

Reads hero trajectory start from:

    data/data_for_carla/<BAG_NAME>/trajectory_positions_rear.json

No color mapping.
No instance segmentation.
No pygame visualization.
No instance_map.txt output.
"""

import os
import sys
import json
import random
import gc

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

VEHICLE_DATA_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "vehicle_data.json",
)

TRAJECTORY_REAR_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "trajectory_positions_rear.json",
)

TRAJECTORY_ODOM_REAR_PATH = os.path.join(
    CARLA_DATA_FOLDER,
    "trajectory_positions_rear_odom_yaw.json",
)

# Prefer odom-yaw rear trajectory if available.
USE_ODOM_YAW_TRAJECTORY = True

# Parked car spawn settings.
PARKED_CAR_Z = 0.05
PARKED_CAR_SPAWN_TEST_Z = 0.25

# Hero spawn settings.
HERO_Z_OFFSET = 0.05

# If True, parked cars with mode == perpendicular may randomly face opposite direction.
RANDOM_FLIP_PERPENDICULAR = True

# Limit for debugging. Set to None to spawn all.
MAX_PARKED_CARS = None
# MAX_PARKED_CARS = 100

# Deterministic vehicle choices.
RANDOM_SEED = 42

# Synchronous mode.
USE_SYNCHRONOUS_MODE = True
SIM_FPS = 20


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
    generate_world_map,
    get_filtered_vehicle_blueprints,
)


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


def choose_hero_start():
    """
    Load hero initial transform.

    Priority:
      1. trajectory_positions_rear_odom_yaw.json, if USE_ODOM_YAW_TRAJECTORY=True
      2. trajectory_positions_rear.json
      3. vehicle_data.json hero_car
    """
    if USE_ODOM_YAW_TRAJECTORY and os.path.exists(TRAJECTORY_ODOM_REAR_PATH):
        trajectory_points = load_json_file(TRAJECTORY_ODOM_REAR_PATH)
        source = TRAJECTORY_ODOM_REAR_PATH

    elif os.path.exists(TRAJECTORY_REAR_PATH):
        trajectory_points = load_json_file(TRAJECTORY_REAR_PATH)
        source = TRAJECTORY_REAR_PATH

    else:
        vehicle_data = load_json_file(VEHICLE_DATA_PATH)
        hero_car = vehicle_data.get("hero_car")

        if hero_car is None:
            raise RuntimeError(
                "No trajectory file found and vehicle_data.json has no hero_car."
            )

        position = hero_car.get("position", [0.0, 0.0, 0.0])
        heading = hero_car.get("heading", 0.0)

        print("[INFO] Using hero_car from vehicle_data.json.")
        print(f"       position={position}")
        print(f"       heading={heading}")

        return {
            "location": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]) + HERO_Z_OFFSET,
            },
            "rotation": {
                "pitch": 0.0,
                "yaw": float(heading),
                "roll": 0.0,
            },
            "source": VEHICLE_DATA_PATH,
        }

    if not trajectory_points:
        raise RuntimeError(f"Trajectory file is empty: {source}")

    first_transform = trajectory_points[0]["transform"]

    loc = first_transform["location"]
    rot = first_transform["rotation"]

    print("[INFO] Using hero start from trajectory:")
    print(f"       {source}")
    print(f"       x={loc['x']:.3f}, y={loc['y']:.3f}, z={loc['z']:.3f}")
    print(f"       yaw={rot['yaw']:.3f}")

    return {
        "location": {
            "x": float(loc["x"]),
            "y": float(loc["y"]),
            "z": float(loc["z"]) + HERO_Z_OFFSET,
        },
        "rotation": {
            "pitch": float(rot.get("pitch", 0.0)),
            "yaw": float(rot.get("yaw", 0.0)),
            "roll": float(rot.get("roll", 0.0)),
        },
        "source": source,
    }


def make_carla_transform(transform_data):
    loc = transform_data["location"]
    rot = transform_data["rotation"]

    return carla.Transform(
        carla.Location(
            x=float(loc["x"]),
            y=float(loc["y"]),
            z=float(loc["z"]),
        ),
        carla.Rotation(
            pitch=float(rot.get("pitch", 0.0)),
            yaw=float(rot.get("yaw", 0.0)),
            roll=float(rot.get("roll", 0.0)),
        ),
    )


def choose_vehicle_blueprint(vehicle_library):
    if not vehicle_library:
        raise RuntimeError("No valid vehicle blueprints found.")

    return random.choice(vehicle_library)


def spawn_static_actor(world, blueprint, transform):
    actor = world.try_spawn_actor(blueprint, transform)

    if actor is None:
        return None

    actor.set_simulate_physics(False)
    return actor


def spawn_hero(world, blueprint_library):
    hero_start = choose_hero_start()

    hero_transform = make_carla_transform(hero_start)

    hero_bp = blueprint_library.find(HERO_VEHICLE_TYPE)

    if hero_bp.has_attribute("role_name"):
        hero_bp.set_attribute("role_name", "hero")

    vehicle = world.spawn_actor(hero_bp, hero_transform)
    vehicle.set_simulate_physics(True)
    vehicle.set_autopilot(False)

    print("[INFO] Spawned hero vehicle.")
    print(f"       type={HERO_VEHICLE_TYPE}")
    print(f"       source={hero_start['source']}")

    return vehicle


def spawn_parked_cars(world, vehicle_library, spawn_positions):
    spawned_actors = []
    failed_count = 0

    total = len(spawn_positions)

    if MAX_PARKED_CARS is not None:
        total = min(total, MAX_PARKED_CARS)

    print(f"[INFO] Spawning parked cars: {total}")

    for idx, entry in enumerate(spawn_positions[:total]):
        if idx % 25 == 0:
            gc.collect()

        start = entry.get("start")
        heading = float(entry.get("heading", 0.0))
        mode = str(entry.get("mode", "")).strip().lower()

        if start is None or len(start) < 2:
            failed_count += 1
            continue

        x = float(start[0])
        y = float(start[1])
        z = float(start[2]) if len(start) > 2 else 0.0

        vehicle_heading = heading

        if (
            RANDOM_FLIP_PERPENDICULAR
            and mode == "perpendicular"
            and random.random() < 0.5
        ):
            vehicle_heading = (vehicle_heading + 180.0) % 360.0

        # Test spawn slightly above the road first.
        test_transform = carla.Transform(
            carla.Location(
                x=x,
                y=y,
                z=z + PARKED_CAR_SPAWN_TEST_Z,
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=vehicle_heading,
                roll=0.0,
            ),
        )

        final_transform = carla.Transform(
            carla.Location(
                x=x,
                y=y,
                z=z + PARKED_CAR_Z,
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=vehicle_heading,
                roll=0.0,
            ),
        )

        blueprint = choose_vehicle_blueprint(vehicle_library)

        actor = world.try_spawn_actor(blueprint, test_transform)

        if actor is None:
            failed_count += 1
            continue

        actor.set_transform(final_transform)
        actor.set_simulate_physics(False)
        spawned_actors.append(actor)

        if (idx + 1) % 50 == 0:
            print(f"       spawned {idx + 1}/{total}")

    print(f"[INFO] Parked cars spawned: {len(spawned_actors)}")
    print(f"[INFO] Parked car failed spawns: {failed_count}")

    return spawned_actors


def move_spectator_to_hero(world, hero_vehicle):
    spectator = world.get_spectator()
    hero_transform = hero_vehicle.get_transform()

    spectator_loc = hero_transform.transform(
        carla.Location(
            x=-10.0,
            z=8.0,
        )
    )

    spectator.set_transform(
        carla.Transform(
            spectator_loc,
            carla.Rotation(
                pitch=-25.0,
                yaw=hero_transform.rotation.yaw,
                roll=0.0,
            ),
        )
    )

    print("[INFO] Spectator moved near hero vehicle.")


# =======================
# MAIN
# =======================

def main():
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("SPAWN HERO AND PARKED CARS")
    print("=" * 80)
    print(f"[INFO] Project root:          {PROJECT_ROOT}")
    print(f"[INFO] Script folder:         {SCRIPT_DIR}")
    print(f"[INFO] Local utils:           {LOCAL_UTILS_DIR}")
    print(f"[INFO] Bag name:              {BAG_NAME}")
    print(f"[INFO] Map folder:            {MAP_FOLDER}")
    print(f"[INFO] XODR file:             {XODR_FILE}")
    print(f"[INFO] CARLA data folder:     {CARLA_DATA_FOLDER}")
    print(f"[INFO] Vehicle data path:     {VEHICLE_DATA_PATH}")
    print(f"[INFO] Rear trajectory:       {TRAJECTORY_REAR_PATH}")
    print(f"[INFO] Odom rear trajectory:  {TRAJECTORY_ODOM_REAR_PATH}")
    print(f"[INFO] CARLA:                 {CARLA_IP}:{CARLA_PORT}")
    print("=" * 80)

    xodr_data = load_text_file(XODR_FILE)
    vehicle_data = load_json_file(VEHICLE_DATA_PATH)

    spawn_positions = vehicle_data.get("spawn_positions", [])

    if not spawn_positions:
        print("[WARN] vehicle_data.json has no parked spawn_positions.")

    print(f"[INFO] Loaded parked spawn positions: {len(spawn_positions)}")

    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(40.0)

    traffic_manager = client.get_trafficmanager(8000)

    print("[INFO] Generating CARLA world from XODR...")
    world = generate_world_map(client, xodr_data)

    blueprint_library = world.get_blueprint_library()
    vehicle_library = get_filtered_vehicle_blueprints(world)

    spawned_parked_actors = []
    hero_vehicle = None

    try:
        update_synchronous_mode(
            world,
            traffic_manager,
            USE_SYNCHRONOUS_MODE,
            SIM_FPS,
        )

        world.tick()

        hero_vehicle = spawn_hero(world, blueprint_library)

        world.tick()

        spawned_parked_actors = spawn_parked_cars(
            world=world,
            vehicle_library=vehicle_library,
            spawn_positions=spawn_positions,
        )

        world.tick()

        move_spectator_to_hero(world, hero_vehicle)

        print("-" * 80)
        print("[INFO] Simulation ready.")
        print("[INFO] Hero vehicle and parked cars are in position.")
        print("[INFO] Press Ctrl+C to destroy actors and exit.")
        print("-" * 80)

        while True:
            world.tick()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        actors_to_destroy = []

        if hero_vehicle is not None:
            actors_to_destroy.append(hero_vehicle)

        actors_to_destroy.extend(spawned_parked_actors)

        print(f"[INFO] Destroying {len(actors_to_destroy)} actors...")

        if actors_to_destroy:
            client.apply_batch(
                [
                    carla.command.DestroyActor(actor)
                    for actor in actors_to_destroy
                ]
            )

        try:
            update_synchronous_mode(
                world,
                traffic_manager,
                False,
                SIM_FPS,
            )
        except Exception:
            pass

        print("[INFO] Done.")


if __name__ == "__main__":
    main()