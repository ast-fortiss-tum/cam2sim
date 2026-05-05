#!/usr/bin/env python3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd
from pyproj import Transformer


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

# Bag / dataset name.
DATASET_NAME = "reference_bag"

# Input trajectory file.
TRAJECTORY_FILE = project_path(
    "data",
    "raw_dataset",
    DATASET_NAME,
    "trajectory.csv",
)

# UTM coordinate reference system.
# Munich is in UTM zone 32N, WGS84.
# EPSG:32632 = WGS84 / UTM zone 32N.
UTM_EPSG = 32632

# Output folder.
# Everything for this bag is saved under:
# data/processed_dataset/<DATASET_NAME>/maps/
MAP_OUTPUT_ROOT = project_path(
    "data",
    "processed_dataset",
    DATASET_NAME,
    "maps",
)

# Radius around the detected road-level location used for OSM extraction.
DIST = 200

# Map creation mode.
# Use:
# - "manual" to select hero car and parking areas manually
# - "auto" to skip GUI and write placeholder hero_car
MODE = "auto"

# In auto mode, use placeholder hero car data.
# The real CARLA hero position is later written by:
# 3A_transform_coordinates_yaw_to_carla.py
USE_FIRST_TRAJECTORY_POSE_AS_START = True

# If True, CARLA functionality is not checked.
# If False, the script checks that CARLA functionality is available.
NO_CARLA = False

# If True, load previously saved OSM data from disk.
# If False, fetch fresh OSM data from the automatically detected road-level address.
SKIP_FETCH = False

# Reverse-geocoding settings.
# Nominatim requires a user agent.
NOMINATIM_USER_AGENT = "cam2sim-map-generation"

# If this is not None, the script uses this exact address instead of the
# automatically built road-level address.
ADDRESS_OVERRIDE = None
# Example:
# ADDRESS_OVERRIDE = "Guerickestraße, Alte Heide, Munich"


# =======================
# HELPERS
# =======================

def load_first_trajectory_pose(trajectory_file):
    """
    Load the first pose from trajectory file.

    Supports both:

    New CSV format:
      timestamp,x,y,z,yaw
      1771259998.585840702,692933.421271,5339067.195751,550.456116,1.065336

    Old headerless format:
      timestamp, x, y, z, yaw
    """
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    required_columns = ["timestamp", "x", "y", "z", "yaw"]

    # First try normal CSV with header.
    df = pd.read_csv(
        trajectory_file,
        comment="#",
        skipinitialspace=True,
    )

    df.columns = [str(col).strip().lower() for col in df.columns]

    # If the CSV did not contain the expected header, fall back to old format.
    if not all(col in df.columns for col in required_columns):
        df = pd.read_csv(
            trajectory_file,
            comment="#",
            header=None,
            names=required_columns,
            skipinitialspace=True,
        )

    # Remove accidental header rows if a header was parsed as data.
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)

    if df.empty:
        raise RuntimeError(
            f"Trajectory file has no valid trajectory rows: {trajectory_file}"
        )

    first_pose = df.iloc[0]

    return {
        "timestamp": float(first_pose["timestamp"]),
        "x": float(first_pose["x"]),
        "y": float(first_pose["y"]),
        "z": float(first_pose["z"]),
        "yaw": float(first_pose["yaw"]),
    }

def utm_to_latlon(easting, northing, utm_epsg):
    """
    Convert UTM coordinates to WGS84 latitude/longitude.
    """
    transformer = Transformer.from_crs(
        f"EPSG:{utm_epsg}",
        "EPSG:4326",
        always_xy=True,
    )

    lon, lat = transformer.transform(easting, northing)

    return lat, lon


def reverse_geocode_latlon(lat, lon):
    """
    Reverse-geocode latitude/longitude using Nominatim.

    Returns the full Nominatim JSON object so we can build a cleaner
    road-level query address instead of using the full house-number address.
    """
    query = urllib.parse.urlencode({
        "lat": f"{lat:.8f}",
        "lon": f"{lon:.8f}",
        "format": "jsonv2",
        "addressdetails": 1,
    })

    url = f"https://nominatim.openstreetmap.org/reverse?{query}"

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": NOMINATIM_USER_AGENT,
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    if not data.get("display_name"):
        raise RuntimeError(
            f"Reverse geocoding failed for lat={lat}, lon={lon}. "
            "No display_name returned."
        )

    return data


def build_road_level_address(reverse_data):
    """
    Build an OSM query address similar to the manually written address.

    This intentionally removes house number, postcode, state, and country,
    because querying the full reverse-geocoded address can shift the OSM
    download center to a specific building.

    Example target:
    Guerickestraße, Alte Heide, München
    """
    address = reverse_data.get("address", {})

    road = address.get("road")

    area = (
        address.get("neighbourhood")
        or address.get("suburb")
        or address.get("quarter")
        or address.get("city_district")
    )

    city = (
        address.get("city")
        or address.get("town")
        or address.get("municipality")
        or address.get("village")
    )

    parts = [road, area, city]
    parts = [part for part in parts if part]

    if not parts:
        return reverse_data["display_name"]

    return ", ".join(parts)


def load_existing_vehicle_data(folder_name):
    """
    Load existing vehicle data from the selected map folder.

    This is used only to preserve previously saved offset values when re-running.
    """
    vehicle_data_path = os.path.join(folder_name, "vehicle_data.json")

    if not os.path.exists(vehicle_data_path):
        return None

    with open(vehicle_data_path, "r") as f:
        return json.load(f)


def build_vehicle_output_from_first_pose(first_pose, dist):
    """
    Build placeholder vehicle data.

    The real hero car CARLA position is generated later by:
      3A_transform_coordinates_yaw_to_carla.py

    Therefore we do not store odom / UTM coordinates here.

    Correct vehicle_data.json format:
    {
      "offset": {
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0
      },
      "dist": 200,
      "hero_car": {
        "position": [0.0, 0.0, 0.0],
        "heading": 0.0
      },
      "spawn_positions": []
    }
    """
    return {
        "offset": {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
        },
        "dist": dist,
        "hero_car": {
            "position": [
                0.0,
                0.0,
                0.0,
            ],
            "heading": 0.0,
        },
        "spawn_positions": [],
    }


def normalize_vehicle_data_schema(vehicle_data, dist):
    """
    Convert any old vehicle_data.json schema into the current expected schema.

    Removes old keys:
      - start
      - parking

    Preserves:
      - offset
      - spawn_positions
      - hero_car if already present
    """
    if vehicle_data is None:
        vehicle_data = {}

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

    hero_car = vehicle_data.get(
        "hero_car",
        {
            "position": [
                0.0,
                0.0,
                0.0,
            ],
            "heading": 0.0,
        },
    )

    spawn_positions = vehicle_data.get("spawn_positions", [])

    normalized = {
        "offset": offset,
        "dist": dist,
        "hero_car": hero_car,
        "spawn_positions": spawn_positions,
    }

    return normalized


# =======================
# MAIN SCRIPT
# =======================

def main():
    # 1. Validate config
    if not DATASET_NAME or not isinstance(DATASET_NAME, str):
        raise ValueError("DATASET_NAME must be a non-empty string.")

    if MODE not in {"manual", "auto"}:
        raise ValueError("MODE must be either 'manual' or 'auto'.")

    # 2. Set up output paths
    folder_name = MAP_OUTPUT_ROOT

    # 3. Read trajectory and convert UTM to lat/lon
    first_pose = load_first_trajectory_pose(TRAJECTORY_FILE)

    lat, lon = utm_to_latlon(
        first_pose["x"],
        first_pose["y"],
        UTM_EPSG,
    )

    print("=" * 70)
    print("MAP DATA GENERATION")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Script folder: {SCRIPT_DIR}")
    print(f"Local utils: {LOCAL_UTILS_DIR}")
    print(f"Map name: {DATASET_NAME}")
    print(f"Trajectory file: {TRAJECTORY_FILE}")
    print(f"UTM EPSG: {UTM_EPSG}")
    print(f"UTM easting: {first_pose['x']}")
    print(f"UTM northing: {first_pose['y']}")
    print(f"Latitude: {lat}")
    print(f"Longitude: {lon}")
    print(f"Distance: {DIST}")
    print(f"Mode: {MODE}")
    print(f"No CARLA: {NO_CARLA}")
    print(f"Skip OSM fetch: {SKIP_FETCH}")
    print(f"Output folder: {folder_name}")
    print("=" * 70)

    # 4. Determine OSM query address
    if SKIP_FETCH:
        osm_query_address = None
        print("\nSkipping reverse geocoding because SKIP_FETCH=True.")

    elif ADDRESS_OVERRIDE is not None:
        osm_query_address = ADDRESS_OVERRIDE
        print("\nUsing configured address override:")
        print(osm_query_address)

    else:
        print("\nReverse-geocoding trajectory position...")

        reverse_data = reverse_geocode_latlon(lat, lon)

        # Be polite to Nominatim before the next possible network request.
        time.sleep(1.0)

        full_detected_address = reverse_data["display_name"]
        osm_query_address = build_road_level_address(reverse_data)

        print(f"Detected full address: {full_detected_address}")
        print(f"Road-level OSM query address: {osm_query_address}")

    # 5. Load existing vehicle data to preserve offset and spawn positions
    existing_vehicle_data = load_existing_vehicle_data(folder_name)

    # 6. Check CARLA functionality if requested
    if not NO_CARLA:
        ensure_carla_functionality()

    # 7. Fetch or load OSM data
    if not SKIP_FETCH:
        print("\nFetching OSM data for road-level address:")
        print(osm_query_address)

        osm_data = get_street_data(osm_query_address, dist=DIST)

    else:
        print(f"\nLoading existing OSM data from: {folder_name}")
        osm_data = get_existing_osm_data(folder_name)

    # 8. Create map folders
    create_map_folders(folder_name)

    # 9. Save OSM data if freshly fetched
    if not SKIP_FETCH:
        save_osm_data(folder_name, osm_data)

    # 10. Process map geometry
    print("\nProcessing map geometry: nodes, edges, and buildings")
    graph, edges, buildings = fetch_osm_data(folder_name)

    # 11. Get hero/start position
    if MODE == "auto" and USE_FIRST_TRAJECTORY_POSE_AS_START:
        print("\nMode: AUTO.")
        print("Writing placeholder hero_car position.")
        print("The real CARLA hero position will be written later by:")
        print("3A_transform_coordinates_yaw_to_carla.py")

        output_json = build_vehicle_output_from_first_pose(first_pose, DIST)

        # Preserve existing offset / spawn_positions if vehicle_data.json already exists.
        if existing_vehicle_data is not None:
            existing_normalized = normalize_vehicle_data_schema(
                existing_vehicle_data,
                DIST,
            )

            output_json["offset"] = existing_normalized["offset"]
            output_json["spawn_positions"] = existing_normalized["spawn_positions"]

            # Preserve existing hero_car too if it was already computed.
            if existing_normalized.get("hero_car") is not None:
                output_json["hero_car"] = existing_normalized["hero_car"]

            print("Preserved existing offset, hero_car, and spawn_positions where available.")

    else:
        print(f"\nMode: {MODE.upper()}. Opening GUI.")

        if MODE == "manual":
            print("Instructions: Select the hero car position and parking areas.")
        else:
            print("Instructions: Select only the hero car position.")
            print("Parking areas will be injected later from cluster data.")
            print("Click the start position and then close the window.")

        plot_title = osm_query_address if osm_query_address is not None else DATASET_NAME

        create_plot(buildings, edges, plot_title)
        show_plot()

        # Retrieve GUI output
        output_json = get_output(DIST)

        # Normalize old GUI output into current vehicle_data.json format.
        output_json = normalize_vehicle_data_schema(output_json, DIST)

        # Preserve existing spawn positions if present.
        if existing_vehicle_data is not None:
            existing_normalized = normalize_vehicle_data_schema(
                existing_vehicle_data,
                DIST,
            )

            output_json["spawn_positions"] = existing_normalized["spawn_positions"]

            if existing_normalized.get("offset") is not None:
                output_json["offset"] = existing_normalized["offset"]

            print("Preserved existing offset and spawn_positions where available.")

    # 12. Save map and vehicle data
    if output_json is not None:
        save_map_data(folder_name, osm_data, NO_CARLA)
        save_vehicle_data(folder_name, output_json)

        print(f"\n[{MODE.upper()}] Map and vehicle data saved to:")
        print(folder_name)

        if MODE != "manual":
            print(
                "vehicle_data.json created with placeholder hero_car data. "
                "Run 3A_transform_coordinates_yaw_to_carla.py to write the real hero_car, "
                "and run the centroid script to add spawn_positions."
            )
    else:
        print("\nPlot closed without selection. Nothing saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()