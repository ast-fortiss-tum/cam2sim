#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3F_OPT_generate_scenario_with_mapping.py

Stable Diffusion branch counterpart of 3F_generate_carla_scenario.py.

For each parked car:
  1. Spawn it temporarily 15m in front of the hero (so the instance sensor
     attached to the hero captures it).
  2. Read the CARLA instance segmentation sensor to find the new CARLA
     instance color assigned to that actor.
  3. Map that CARLA instance color -> the real-world RGB color from the
     bag (entry["color"] from vehicle_data.json).
  4. Teleport the car to its final parking position and freeze physics.

At the end writes:
    data/data_for_carla/<BAG>/instance_color_map.json

which maps:
    "(carla_R, carla_G, carla_B)" -> "real_R,real_G,real_B"

This file is later read at runtime by the SD inference script to recolor
CARLA instance maps with bag colors before feeding them to the SD model.

Reads from (project root):
    data/processed_dataset/<BAG>/maps/map.xodr
    data/data_for_carla/<BAG>/vehicle_data.json  (with color field, from 3B_OPT)
    data/data_for_carla/<BAG>/trajectory_positions_rear_odom_yaw.json (optional)

Writes to (project root):
    data/data_for_carla/<BAG>/instance_color_map.json
"""

import os
import sys
import json
import random
import gc
from queue import Empty

import carla


# =======================
# PATH SETUP
# =======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to this script, not found: {LOCAL_UTILS_DIR}"
    )

if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


# =======================
# HARDCODED CONFIG
# =======================

BAG_NAME = "reference_bag"

MAP_FOLDER = os.path.join(
    PROJECT_ROOT, "data", "processed_dataset", BAG_NAME, "maps",
)
XODR_FILE = os.path.join(MAP_FOLDER, "map.xodr")

CARLA_DATA_FOLDER = os.path.join(
    PROJECT_ROOT, "data", "data_for_carla", BAG_NAME,
)
VEHICLE_DATA_PATH = os.path.join(CARLA_DATA_FOLDER, "vehicle_data.json")
TRAJECTORY_ODOM_REAR_PATH = os.path.join(
    CARLA_DATA_FOLDER, "trajectory_positions_rear_odom_yaw.json",
)
TRAJECTORY_REAR_PATH = os.path.join(
    CARLA_DATA_FOLDER, "trajectory_positions_rear.json",
)
INSTANCE_MAP_OUTPUT = os.path.join(
    CARLA_DATA_FOLDER, "instance_color_map.json",
)

# Sensor settings for the mapping pass
IM_WIDTH = 800
IM_HEIGHT = 503
SENSOR_FOV = "54.7"
SIM_FPS = 20

# Sensor mount on hero (front, slightly up)
SENSOR_X = 0.762
SENSOR_Y = -0.015
SENSOR_Z = 1.21
SENSOR_PITCH = 0.6

# Spawn limit (None = all)
MAX_PARKED_CARS = None

# Retries / timeouts
SPAWN_MAX_RETRIES = 100
COLOR_DETECTION_MAX_ATTEMPTS = 50
SENSOR_QUEUE_TIMEOUT_S = 5.0
CALIBRATION_TICKS = 20

# Random
RANDOM_SEED = 42


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
    spawn_parked_cars_front_of_hero,
    spawn_sensor,
    flush_all_queues,
    get_unique_colors_from_sensor,
    cleanup_old_sensors,
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


def choose_hero_start(vehicle_data):
    """
    Hero start priority:
      1. trajectory_positions_rear_odom_yaw.json (first transform)
      2. trajectory_positions_rear.json (first transform)
      3. vehicle_data.json hero_car
    """
    for trajectory_path in [TRAJECTORY_ODOM_REAR_PATH, TRAJECTORY_REAR_PATH]:
        if os.path.exists(trajectory_path):
            trajectory_points = load_json_file(trajectory_path)
            if trajectory_points:
                first = trajectory_points[0]["transform"]
                loc = first["location"]
                rot = first["rotation"]
                print(f"[INFO] Hero start from: {trajectory_path}")
                return carla.Transform(
                    carla.Location(x=float(loc["x"]),
                                   y=float(loc["y"]),
                                   z=float(loc["z"])),
                    carla.Rotation(pitch=0.0,
                                   yaw=float(rot.get("yaw", 0.0)),
                                   roll=0.0),
                )

    hero_car = vehicle_data.get("hero_car")
    if hero_car is None:
        raise RuntimeError(
            "No trajectory file and no hero_car in vehicle_data.json."
        )
    position = hero_car.get("position", [0.0, 0.0, 0.0])
    heading = hero_car.get("heading", 0.0)
    print("[INFO] Hero start from vehicle_data.json hero_car")
    return carla.Transform(
        carla.Location(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        ),
        carla.Rotation(pitch=0.0, yaw=float(heading), roll=0.0),
    )


def color_list_to_str(color_list):
    """[R, G, B] -> 'R,G,B' (format expected by spawn_parked_cars_front_of_hero)."""
    if color_list is None:
        return None
    return ",".join(str(int(c)) for c in color_list)


def stop_motion(actor):
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass
    try:
        actor.set_target_velocity(carla.Vector3D(0, 0, 0))
    except Exception:
        pass
    try:
        actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    except Exception:
        pass


# =======================
# MAIN
# =======================

def main():
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("GENERATE CARLA SCENARIO WITH INSTANCE-COLOR MAPPING (SD branch)")
    print("=" * 80)
    print(f"[INFO] Project root:        {PROJECT_ROOT}")
    print(f"[INFO] Bag name:            {BAG_NAME}")
    print(f"[INFO] XODR file:           {XODR_FILE}")
    print(f"[INFO] Vehicle data:        {VEHICLE_DATA_PATH}")
    print(f"[INFO] Output map:          {INSTANCE_MAP_OUTPUT}")
    print(f"[INFO] CARLA:               {CARLA_IP}:{CARLA_PORT}")
    print("=" * 80)

    if not os.path.exists(XODR_FILE):
        raise FileNotFoundError(f"XODR not found: {XODR_FILE}")
    if not os.path.exists(VEHICLE_DATA_PATH):
        raise FileNotFoundError(
            f"vehicle_data.json not found: {VEHICLE_DATA_PATH}\n"
            f"Run 3B_OPT_transform_parked_vehicles_with_colors.py first."
        )

    xodr_data = load_text_file(XODR_FILE)
    vehicle_data = load_json_file(VEHICLE_DATA_PATH)
    spawn_positions = vehicle_data.get("spawn_positions", [])

    if not spawn_positions:
        raise RuntimeError("vehicle_data.json has no spawn_positions.")

    n_with_color = sum(
        1 for e in spawn_positions if e.get("color") is not None
    )
    print(f"[INFO] Parked spawn positions: {len(spawn_positions)}")
    print(f"[INFO] Of which with color:    {n_with_color}")

    if n_with_color == 0:
        raise RuntimeError(
            "No spawn entry has a 'color'. Did you run 3B_OPT instead of 3B?"
        )

    # ---------- Connect to CARLA ----------
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(40.0)
    traffic_manager = client.get_trafficmanager(8000)

    print("[INFO] Generating world from XODR...")
    world = generate_world_map(client, xodr_data)

    blueprint_library = world.get_blueprint_library()
    vehicle_library = get_filtered_vehicle_blueprints(world)

    # ---------- Synchronous mode ----------
    update_synchronous_mode(world, traffic_manager, True, SIM_FPS)
    for _ in range(10):
        world.tick()

    # ---------- Destroy any existing vehicles ----------
    existing = list(world.get_actors().filter("vehicle.*"))
    if existing:
        print(f"[INFO] Destroying {len(existing)} existing vehicles.")
        client.apply_batch([
            carla.command.DestroyActor(actor) for actor in existing
        ])
        for _ in range(5):
            world.tick()

    # ---------- Spawn hero ----------
    hero_transform = choose_hero_start(vehicle_data)
    hero_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
    if hero_bp.has_attribute("role_name"):
        hero_bp.set_attribute("role_name", "hero")
    if hero_bp.has_attribute("color"):
        hero_bp.set_attribute("color", "255,0,0")

    hero_vehicle = world.try_spawn_actor(hero_bp, hero_transform)
    if hero_vehicle is None:
        # safe fallback
        fallback = carla.Transform(
            carla.Location(
                x=hero_transform.location.x,
                y=hero_transform.location.y,
                z=hero_transform.location.z + 2.0,
            ),
            hero_transform.rotation,
        )
        hero_vehicle = world.try_spawn_actor(hero_bp, fallback)
    if hero_vehicle is None:
        raise RuntimeError("Failed to spawn hero vehicle.")

    hero_vehicle.set_autopilot(False)
    stop_motion(hero_vehicle)

    # pin to the exact desired transform
    for _ in range(10):
        stop_motion(hero_vehicle)
        hero_vehicle.set_transform(hero_transform)
        world.tick()

    print(f"[INFO] Hero spawned. id={hero_vehicle.id}")

    # ---------- Attach instance segmentation sensor ----------
    cleanup_old_sensors(hero_vehicle)

    sensor_tf = carla.Transform(
        carla.Location(x=SENSOR_X, y=SENSOR_Y, z=SENSOR_Z),
        carla.Rotation(pitch=SENSOR_PITCH),
    )
    inst_sensor, inst_queue = spawn_sensor(
        blueprint_library,
        "sensor.camera.instance_segmentation",
        IM_WIDTH, IM_HEIGHT, SENSOR_FOV,
        world, sensor_tf, hero_vehicle,
    )

    # We also create RGB and SEG queues only because flush_all_queues wants
    # three arguments; they remain empty and are flushed harmlessly.
    rgb_sensor, rgb_queue = spawn_sensor(
        blueprint_library, "sensor.camera.rgb",
        IM_WIDTH, IM_HEIGHT, SENSOR_FOV,
        world, sensor_tf, hero_vehicle,
    )
    seg_sensor, seg_queue = spawn_sensor(
        blueprint_library, "sensor.camera.semantic_segmentation",
        IM_WIDTH, IM_HEIGHT, SENSOR_FOV,
        world, sensor_tf, hero_vehicle,
    )

    world.tick()

    # ---------- Calibration: log "ignored" colors (hero, any leftover) ----------
    print("[INFO] Calibration: capturing baseline known colors...")
    known_colors = set()
    for _ in range(CALIBRATION_TICKS):
        world.tick()
        flush_all_queues(seg_queue, rgb_queue, inst_queue)
        try:
            inst_data = inst_queue.get(timeout=1.0)
            initial = get_unique_colors_from_sensor(inst_data)
            known_colors.update(initial)
        except Empty:
            pass
    print(f"[INFO] Baseline known colors: {len(known_colors)}")

    # ---------- Mapping loop ----------
    detected_color_map = {}    # (carla_R, carla_G, carla_B) -> "real_R,real_G,real_B"
    spawned_actors_keep = []   # actors we keep alive (teleported to final pos)
    n_mapped = 0
    n_no_color = 0

    total = (
        len(spawn_positions) if MAX_PARKED_CARS is None
        else min(len(spawn_positions), MAX_PARKED_CARS)
    )

    print(f"[INFO] Processing {total} parked cars...")

    for count, entry in enumerate(spawn_positions[:total]):
        if count % 10 == 0:
            gc.collect()

        # ---- Skip entries without color ----
        color_list = entry.get("color")
        if color_list is None:
            n_no_color += 1
            continue

        # Normalize color into "R,G,B" string (expected by spawn helper)
        # because the helper does `entry["color"].split(',')`.
        entry_for_spawn = dict(entry)
        entry_for_spawn["color"] = color_list_to_str(color_list)

        # ---- PHASE 1: spawn in front of hero ----
        actor = None
        for _ in range(SPAWN_MAX_RETRIES):
            spawned = spawn_parked_cars_front_of_hero(
                world, vehicle_library, entry_for_spawn, hero_vehicle,
            )
            if spawned:
                actor = spawned[0]
                break
            world.tick()
            flush_all_queues(seg_queue, rgb_queue, inst_queue)

        if actor is None:
            print(f"[WARN] Could not spawn entry {count} after retries.")
            continue

        # ---- PHASE 2: detect new instance color ----
        detected_key = None
        for attempt in range(COLOR_DETECTION_MAX_ATTEMPTS):
            world.tick()
            # flush stale frames, keep only the latest
            while not seg_queue.empty():
                seg_queue.get()
            while not rgb_queue.empty():
                rgb_queue.get()

            try:
                inst_data = inst_queue.get(timeout=SENSOR_QUEUE_TIMEOUT_S)
            except Empty:
                continue

            current_colors = get_unique_colors_from_sensor(inst_data)
            new_colors = current_colors - known_colors
            if new_colors:
                detected_key = list(new_colors)[0]
                detected_color_map[detected_key] = color_list_to_str(color_list)
                known_colors.add(detected_key)
                n_mapped += 1
                break

        if detected_key is None:
            print(f"[WARN] Timeout: no color detected for entry {count}.")

        # ---- PHASE 3: teleport to final parking position ----
        start = entry["start"]
        heading = float(entry["heading"])
        mode = str(entry.get("mode", "")).strip().lower()

        veh_heading = heading
        if mode == "perpendicular" and random.random() < 0.5:
            veh_heading = (veh_heading + 180.0) % 360.0

        final_transform = carla.Transform(
            carla.Location(x=float(start[0]),
                           y=float(start[1]),
                           z=float(start[2]) if len(start) > 2 else 0.0),
            carla.Rotation(yaw=veh_heading),
        )

        try:
            actor.set_transform(final_transform)
            actor.set_simulate_physics(False)
            spawned_actors_keep.append(actor)
        except RuntimeError as e:
            print(f"[WARN] Teleport failed entry {count}: {e}")
            try:
                actor.destroy()
            except Exception:
                pass
            continue

        world.tick()
        flush_all_queues(seg_queue, rgb_queue, inst_queue)

        if (count + 1) % 25 == 0:
            print(f"   processed {count + 1}/{total}, "
                  f"mapped={n_mapped}, kept={len(spawned_actors_keep)}")

    print("=" * 80)
    print(f"[INFO] Total processed:        {total}")
    print(f"[INFO] Entries without color:  {n_no_color}")
    print(f"[INFO] Colors mapped:          {n_mapped}")
    print(f"[INFO] Parked actors kept:     {len(spawned_actors_keep)}")
    print("=" * 80)

    # ---------- Save mapping ----------
    # Key is a CARLA instance color tuple, JSON keys must be strings.
    json_ready = {str(tuple(k)): v for k, v in detected_color_map.items()}
    with open(INSTANCE_MAP_OUTPUT, "w") as f:
        json.dump(json_ready, f, indent=2)
    print(f"[INFO] Saved instance color map: {INSTANCE_MAP_OUTPUT}")

    # ---------- Cleanup sensors ----------
    print("[INFO] Stopping sensors...")
    for s in [inst_sensor, rgb_sensor, seg_sensor]:
        try:
            s.stop()
            s.destroy()
        except Exception:
            pass

    # ---------- Freeze everything ----------
    for actor in spawned_actors_keep:
        stop_motion(actor)
    stop_motion(hero_vehicle)

    # ---------- Disable sync mode so next script can take over ----------
    update_synchronous_mode(world, traffic_manager, False, SIM_FPS)

    print("[INFO] World left alive in CARLA. Done.")


if __name__ == "__main__":
    main()