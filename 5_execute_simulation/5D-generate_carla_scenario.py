#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare CARLA world with hero car and parked cars.

This script prepares the CARLA world for the replay script.

It does:
  - load the OpenDRIVE map
  - spawn parked cars
  - spawn the hero car at the first odom-yaw rear trajectory pose
  - disable physics for all spawned actors
  - leave the actors alive in CARLA
  - disable synchronous mode
  - exit

Important:
  This script should NOT keep running while the replay script runs.
  The replay script should be the only script calling world.tick() during replay.

Reads map from:

    data/processed_dataset/<BAG_NAME>/maps/map.xodr

Reads CARLA-ready vehicle data from:

    data/data_for_carla/<BAG_NAME>/vehicle_data.json

Reads hero trajectory start from:

    data/data_for_carla/<BAG_NAME>/trajectory_positions_rear_odom_yaw.json

Expected workflow:

    1. Start CARLA server.
    2. Run this script once.
    3. Run the replay script.
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

# Force imports from:
#   5_execute_simulation/utils/
# instead of:
#   project_root/utils/
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
    "processed_dataset",
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

# Use odom-yaw rear trajectory for the hero start.
USE_ODOM_YAW_TRAJECTORY = True

# Keep exact z from trajectory.
HERO_Z_OFFSET = 0.0

# Static preparation mode.
# The hero is teleported to the target pose and physics is disabled.
HERO_SIMULATE_PHYSICS = False

# This script prepares the world and exits.
# It does not stay in an infinite world.tick() loop.
LEAVE_WORLD_READY_AND_EXIT = True

# Important:
# False means actors stay alive in CARLA after this script exits.
DESTROY_ACTORS_ON_EXIT = False

# Destroy old vehicles before spawning a new prepared scene.
DESTROY_EXISTING_VEHICLES = True

# Parked car spawn settings.
PARKED_CAR_Z = 0.0
PARKED_CAR_SPAWN_TEST_Z = 0.25

# Perpendicular cars may face either direction.
RANDOM_FLIP_PERPENDICULAR = True

# Set to None to spawn all parked cars.
MAX_PARKED_CARS = None
# MAX_PARKED_CARS = 100

RANDOM_SEED = 42

# Use synchronous mode only while preparing the world.
# The script disables it again before exit.
USE_SYNCHRONOUS_MODE_DURING_PREP = True
SIM_FPS = 20

# Spectator camera.
MOVE_SPECTATOR_TO_HERO = True
SPECTATOR_BACK_DISTANCE = -10.0
SPECTATOR_HEIGHT = 8.0
SPECTATOR_PITCH = -25.0


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
      1. trajectory_positions_rear_odom_yaw.json
      2. trajectory_positions_rear.json
      3. vehicle_data.json hero_car
    """

    if USE_ODOM_YAW_TRAJECTORY and os.path.exists(TRAJECTORY_ODOM_REAR_PATH):
        trajectory_path = TRAJECTORY_ODOM_REAR_PATH

    elif os.path.exists(TRAJECTORY_REAR_PATH):
        trajectory_path = TRAJECTORY_REAR_PATH

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

    trajectory_points = load_json_file(trajectory_path)

    if not trajectory_points:
        raise RuntimeError(f"Trajectory file is empty: {trajectory_path}")

    first_transform = trajectory_points[0]["transform"]

    loc = first_transform["location"]
    rot = first_transform["rotation"]

    print("[INFO] Using hero start from trajectory:")
    print(f"       {trajectory_path}")
    print(f"       x={loc['x']:.3f}, y={loc['y']:.3f}, z={loc['z']:.3f}")
    print(f"       yaw={rot['yaw']:.3f}")

    return {
        "location": {
            "x": float(loc["x"]),
            "y": float(loc["y"]),
            "z": float(loc["z"]) + HERO_Z_OFFSET,
        },
        "rotation": {
            "pitch": 0.0,
            "yaw": float(rot.get("yaw", 0.0)),
            "roll": 0.0,
        },
        "source": trajectory_path,
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


def stop_actor_motion(actor):
    """
    Stop linear/angular motion and disable physics/gravity where available.
    """
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    try:
        actor.set_enable_gravity(False)
    except Exception:
        pass

    try:
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except Exception:
        pass

    try:
        actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except Exception:
        pass


def clear_existing_vehicles(world, client):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    if not vehicles:
        print("[INFO] No existing vehicles to destroy.")
        return

    print(f"[INFO] Destroying existing vehicles from previous runs: {len(vehicles)}")

    client.apply_batch(
        [
            carla.command.DestroyActor(actor)
            for actor in vehicles
        ]
    )

    for _ in range(5):
        world.tick()


def choose_vehicle_blueprint(vehicle_library):
    if not vehicle_library:
        raise RuntimeError("No valid vehicle blueprints found.")

    return random.choice(vehicle_library)


def get_safe_spawn_transform(world, target_transform):
    """
    Pick a temporary safe spawn transform.

    The hero is spawned here first, then teleported to the exact target pose.
    This avoids CARLA rejecting the actor spawn at the exact target location.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    if spawn_points:
        safe_transform = spawn_points[0]
        safe_transform.location.z += 1.0
        return safe_transform

    return carla.Transform(
        carla.Location(
            x=target_transform.location.x,
            y=target_transform.location.y,
            z=target_transform.location.z + 3.0,
        ),
        target_transform.rotation,
    )


def spawn_hero(world, blueprint_library):
    """
    Spawn hero safely, then teleport it to the exact target transform.

    Physics stays disabled so the hero cannot fall or fly away.
    """
    hero_start = choose_hero_start()
    target_transform = make_carla_transform(hero_start)

    hero_bp = blueprint_library.find(HERO_VEHICLE_TYPE)

    if hero_bp.has_attribute("role_name"):
        hero_bp.set_attribute("role_name", "hero")

    if hero_bp.has_attribute("color"):
        hero_bp.set_attribute("color", "255,0,0")

    print("[INFO] Warming up world before hero spawn...")
    for _ in range(20):
        world.tick()

    safe_spawn_transform = get_safe_spawn_transform(world, target_transform)

    print("[INFO] Spawning hero at temporary safe location:")
    print(
        f"       safe x={safe_spawn_transform.location.x:.3f}, "
        f"y={safe_spawn_transform.location.y:.3f}, "
        f"z={safe_spawn_transform.location.z:.3f}, "
        f"yaw={safe_spawn_transform.rotation.yaw:.3f}"
    )

    vehicle = world.try_spawn_actor(hero_bp, safe_spawn_transform)

    if vehicle is None:
        print("[WARN] Safe spawn failed. Trying target transform with z + 2.0.")

        fallback_transform = carla.Transform(
            carla.Location(
                x=target_transform.location.x,
                y=target_transform.location.y,
                z=target_transform.location.z + 2.0,
            ),
            target_transform.rotation,
        )

        vehicle = world.try_spawn_actor(hero_bp, fallback_transform)

    if vehicle is None:
        raise RuntimeError("Failed to spawn hero vehicle.")

    vehicle.set_autopilot(False)
    stop_actor_motion(vehicle)

    print("[INFO] Teleporting hero to trajectory position.")

    for _ in range(20):
        stop_actor_motion(vehicle)
        vehicle.set_transform(target_transform)
        world.tick()

    actual_transform = vehicle.get_transform()

    print("[INFO] Spawned and pinned hero vehicle.")
    print(f"       actor id: {vehicle.id}")
    print(f"       type:     {HERO_VEHICLE_TYPE}")
    print(f"       source:   {hero_start['source']}")
    print(
        f"       target    x={target_transform.location.x:.3f}, "
        f"y={target_transform.location.y:.3f}, "
        f"z={target_transform.location.z:.3f}, "
        f"yaw={target_transform.rotation.yaw:.3f}"
    )
    print(
        f"       actual    x={actual_transform.location.x:.3f}, "
        f"y={actual_transform.location.y:.3f}, "
        f"z={actual_transform.location.z:.3f}, "
        f"yaw={actual_transform.rotation.yaw:.3f}"
    )

    return vehicle, target_transform


def pin_actor_to_transform(actor, transform):
    """
    Put actor exactly at the desired transform and freeze it.
    """
    stop_actor_motion(actor)
    actor.set_transform(transform)
    stop_actor_motion(actor)


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

        stop_actor_motion(actor)
        actor.set_transform(final_transform)
        stop_actor_motion(actor)

        spawned_actors.append(actor)

        if (idx + 1) % 50 == 0:
            print(f"       spawned {idx + 1}/{total}")

    print(f"[INFO] Parked cars spawned: {len(spawned_actors)}")
    print(f"[INFO] Parked car failed spawns: {failed_count}")

    return spawned_actors


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

    print("[INFO] Spectator moved behind hero.")


def set_world_async(world, traffic_manager):
    """
    Disable synchronous mode so the next script can control simulation safely.
    """
    try:
        update_synchronous_mode(
            world,
            traffic_manager,
            False,
            SIM_FPS,
        )
        print("[INFO] Synchronous mode disabled.")
    except Exception as e:
        print(f"[WARN] Failed to disable synchronous mode: {e}")


# =======================
# MAIN
# =======================

def main():
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("PREPARE CARLA WORLD WITH HERO AND PARKED CARS")
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
    print(f"[INFO] Use odom yaw traj:     {USE_ODOM_YAW_TRAJECTORY}")
    print(f"[INFO] Hero z offset:         {HERO_Z_OFFSET}")
    print(f"[INFO] Destroy on exit:       {DESTROY_ACTORS_ON_EXIT}")
    print(f"[INFO] Leave ready and exit:  {LEAVE_WORLD_READY_AND_EXIT}")
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
    hero_target_transform = None

    try:
        update_synchronous_mode(
            world,
            traffic_manager,
            USE_SYNCHRONOUS_MODE_DURING_PREP,
            SIM_FPS,
        )

        for _ in range(10):
            world.tick()

        if DESTROY_EXISTING_VEHICLES:
            clear_existing_vehicles(world, client)

        hero_vehicle, hero_target_transform = spawn_hero(
            world,
            blueprint_library,
        )

        world.tick()

        if MOVE_SPECTATOR_TO_HERO:
            move_spectator_to_hero(world, hero_vehicle)

        spawned_parked_actors = spawn_parked_cars(
            world=world,
            vehicle_library=vehicle_library,
            spawn_positions=spawn_positions,
        )

        world.tick()

        # Final pin after parked cars are spawned.
        if hero_vehicle is not None and hero_target_transform is not None:
            pin_actor_to_transform(hero_vehicle, hero_target_transform)

        if MOVE_SPECTATOR_TO_HERO:
            move_spectator_to_hero(world, hero_vehicle)

        # Freeze all actors one last time.
        if hero_vehicle is not None:
            stop_actor_motion(hero_vehicle)

        for actor in spawned_parked_actors:
            stop_actor_motion(actor)

        print("-" * 80)
        print("[INFO] World prepared.")
        print("[INFO] Hero vehicle is static.")
        print("[INFO] Parked cars are static.")
        print("[INFO] Actors will remain alive in CARLA.")
        print("[INFO] This script will now exit.")
        print("[INFO] Run the replay script next.")
        print("-" * 80)

        if LEAVE_WORLD_READY_AND_EXIT:
            set_world_async(world, traffic_manager)
            return

        # Optional interactive mode, disabled by default.
        while True:
            world.tick()

            if hero_vehicle is not None and hero_target_transform is not None:
                pin_actor_to_transform(hero_vehicle, hero_target_transform)

            if MOVE_SPECTATOR_TO_HERO and hero_vehicle is not None:
                move_spectator_to_hero(world, hero_vehicle)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        if DESTROY_ACTORS_ON_EXIT:
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
        else:
            kept_count = len(spawned_parked_actors)
            if hero_vehicle is not None:
                kept_count += 1

            print(f"[INFO] Leaving actors alive in CARLA: {kept_count}")

        set_world_async(world, traffic_manager)

        print("[INFO] Done.")


if __name__ == "__main__":
    main()