#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys


# =======================
# PATH SETUP
# =======================

# Folder where this script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Force imports from the utils folder next to this script.
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)

LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )


def find_project_root(start_dir):
    """
    Find the project root by walking upward until a data/ folder is found.

    This avoids creating data folders relative to the terminal launch directory.
    """
    current = os.path.abspath(start_dir)

    while True:
        data_dir = os.path.join(current, "data")

        if os.path.isdir(data_dir):
            return current

        parent = os.path.dirname(current)

        if parent == current:
            # Fallback: assume the parent of the script folder is project root.
            return os.path.abspath(os.path.join(start_dir, ".."))

        current = parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)


def project_path(*parts):
    """
    Build an absolute path relative to the project root.
    """
    return os.path.abspath(os.path.join(PROJECT_ROOT, *parts))


# =======================
# IMPORTS
# =======================

from utils.map_data import get_street_data, fetch_osm_data

from utils.save_data import (
    create_map_folders,
    save_vehicle_data,
    save_map_data,
    save_osm_data,
    get_existing_osm_data,
)

from utils.other import ensure_carla_functionality
from utils.plotting import create_plot, show_plot, get_output


# =======================
# CONFIGURATION
# =======================

# Address used to fetch the OSM map.
ADDRESS = "Guerickestraße, Alte Heide, Munich"

# Bag / map name.
MAP_NAME = "reference_bag"

# Output folder.
# Everything for this bag is saved under:
# data/processed_dataset/<MAP_NAME>/maps/
MAP_OUTPUT_ROOT = project_path(
    "data",
    "processed_dataset",
    MAP_NAME,
    "maps",
)

# Radius around the address used for OSM data extraction.
DIST = 200

# Map creation mode.
# Use:
# - "manual" to select hero car and parking areas manually
# - "auto" to select only hero car; parked cars are injected later from clusters
MODE = "auto"

# If True, CARLA functionality is not checked.
# If False, the script checks that CARLA functionality is available.
NO_CARLA = False

# If True, load previously saved OSM data from disk.
# If False, fetch fresh OSM data from ADDRESS.
SKIP_FETCH = False


# =======================
# HELPERS
# =======================

def load_existing_vehicle_data(folder_name):
    """
    Load existing vehicle data from the selected map folder.

    This is used to preserve previous offset, hero_car, and spawn_positions.
    """
    vehicle_data_path = os.path.join(folder_name, "vehicle_data.json")

    if not os.path.exists(vehicle_data_path):
        return None

    with open(vehicle_data_path, "r") as f:
        return json.load(f)


def extract_hero_car_from_gui_output(output_json):
    """
    Convert GUI output into the current hero_car schema.

    Supports old GUI format:
      {
        "start": {
          "x": ...,
          "y": ...,
          "z": ...,
          "yaw": ...
        }
      }

    Also supports already-normalized format:
      {
        "hero_car": {
          "position": [x, y, z],
          "heading": yaw
        }
      }
    """
    if output_json is None:
        return None

    # Already correct format
    if isinstance(output_json.get("hero_car"), dict):
        hero_car = output_json["hero_car"]

        position = hero_car.get("position", [0.0, 0.0, 0.0])
        heading = hero_car.get("heading", 0.0)

        return {
            "position": [
                float(position[0]),
                float(position[1]),
                float(position[2]) if len(position) > 2 else 0.0,
            ],
            "heading": float(heading),
        }

    # Old GUI format
    if isinstance(output_json.get("start"), dict):
        start = output_json["start"]

        return {
            "position": [
                float(start.get("x", 0.0)),
                float(start.get("y", 0.0)),
                float(start.get("z", 0.0)),
            ],
            "heading": float(start.get("yaw", 0.0)),
        }

    # Some plotting code may return position as a list/tuple.
    if isinstance(output_json.get("start"), (list, tuple)):
        start = output_json["start"]

        return {
            "position": [
                float(start[0]) if len(start) > 0 else 0.0,
                float(start[1]) if len(start) > 1 else 0.0,
                float(start[2]) if len(start) > 2 else 0.0,
            ],
            "heading": float(output_json.get("heading", 0.0)),
        }

    raise RuntimeError(
        "Could not extract hero car from GUI output. "
        "Expected key 'start' or 'hero_car'."
    )


def normalize_vehicle_data_schema(output_json, existing_vehicle_data, dist):
    """
    Build final vehicle_data.json using the current expected schema.

    Final format:
    {
      "offset": {
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0
      },
      "dist": 200,
      "hero_car": {
        "position": [x, y, z],
        "heading": yaw
      },
      "spawn_positions": [...]
    }

    Preserves existing:
      - offset
      - spawn_positions

    Uses clicked GUI output for:
      - hero_car
    """
    if output_json is None:
        return None

    hero_car = extract_hero_car_from_gui_output(output_json)

    # Default offset
    offset = {
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0,
    }

    # Default spawn positions
    spawn_positions = []

    # Preserve existing data if available
    if isinstance(existing_vehicle_data, dict):
        if existing_vehicle_data.get("offset") is not None:
            offset = existing_vehicle_data["offset"]

        if isinstance(existing_vehicle_data.get("spawn_positions"), list):
            spawn_positions = existing_vehicle_data["spawn_positions"]

    # If current GUI output has offset, use it only when no existing offset exists.
    if output_json.get("offset") is not None and not isinstance(existing_vehicle_data, dict):
        offset = output_json["offset"]

    vehicle_data = {
        "offset": offset,
        "dist": dist,
        "hero_car": hero_car,
        "spawn_positions": spawn_positions,
    }

    return vehicle_data


# =======================
# MAIN SCRIPT
# =======================

def main():
    # 1. Validate config
    if not MAP_NAME or not isinstance(MAP_NAME, str):
        raise ValueError("MAP_NAME must be a non-empty string.")

    if MODE not in {"manual", "auto"}:
        raise ValueError("MODE must be either 'manual' or 'auto'.")

    # 2. Set up output paths
    folder_name = MAP_OUTPUT_ROOT

    print("=" * 70)
    print("MAP DATA GENERATION")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Script folder: {SCRIPT_DIR}")
    print(f"Local utils: {LOCAL_UTILS_DIR}")
    print(f"Address: {ADDRESS}")
    print(f"Map name: {MAP_NAME}")
    print(f"Distance: {DIST}")
    print(f"Mode: {MODE}")
    print(f"No CARLA: {NO_CARLA}")
    print(f"Skip OSM fetch: {SKIP_FETCH}")
    print(f"Output folder: {folder_name}")
    print("=" * 70)

    # 3. Load existing vehicle data to preserve offset and spawn_positions
    existing_vehicle_data = load_existing_vehicle_data(folder_name)

    # 4. Check CARLA functionality if requested
    if not NO_CARLA:
        ensure_carla_functionality()

    # 5. Fetch or load OSM data
    if not SKIP_FETCH:
        print(f"\nFetching OSM data for: {ADDRESS}")
        osm_data = get_street_data(ADDRESS, dist=DIST)
    else:
        print(f"\nLoading existing OSM data from: {folder_name}")
        osm_data = get_existing_osm_data(folder_name)

    # 6. Create map folders
    create_map_folders(folder_name)

    # 7. Save OSM data if freshly fetched
    if not SKIP_FETCH:
        save_osm_data(folder_name, osm_data)

    # 8. Process map geometry
    print("\nProcessing map geometry: nodes, edges, and buildings")
    graph, edges, buildings = fetch_osm_data(folder_name)

    # 9. Open GUI
    print(f"\nMode: {MODE.upper()}. Opening GUI.")

    if MODE == "manual":
        print("Instructions: Select the Hero Car and parking areas.")
    else:
        print("Instructions: Select ONLY the Hero Car initial position.")
        print("Parking areas will be injected later from clusters.")
        print("Click the start position and then close the window.")

    create_plot(buildings, edges, ADDRESS)
    show_plot()

    # 10. Retrieve GUI output
    raw_output_json = get_output(DIST)

    # 11. Convert GUI output to correct vehicle_data.json schema
    if raw_output_json is not None:
        output_json = normalize_vehicle_data_schema(
            output_json=raw_output_json,
            existing_vehicle_data=existing_vehicle_data,
            dist=DIST,
        )

        print("\nHero car selected:")
        print(f"  Position: {output_json['hero_car']['position']}")
        print(f"  Heading:  {output_json['hero_car']['heading']}")
        print(f"  Preserved spawn_positions: {len(output_json['spawn_positions'])}")
    else:
        output_json = None

    # 12. Save map and vehicle data
    if output_json is not None:
        save_map_data(folder_name, osm_data, NO_CARLA)
        save_vehicle_data(folder_name, output_json)

        print(f"\n[{MODE.upper()}] Map and vehicle data saved to:")
        print(folder_name)

        if MODE != "manual":
            print(
                "vehicle_data.json created with clicked Hero Car. "
                "Run the centroid script to add parked vehicle spawn_positions."
            )
    else:
        print("\nPlot closed without selection. Nothing saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()