#!/usr/bin/env python3

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
# This makes imports work even if the script is launched from another directory.
# The utils/ folder is expected to be in the same folder as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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

# Required map name.
# This name defines the output folder:
# data/maps/<DATASET_NAME>/
DATASET_NAME = "reference_bag"

# Input trajectory file.
# This path is relative to the directory where you run the script.
TRAJECTORY_FILE = os.path.join(
    "data",
    "extracted_ros_data",
    DATASET_NAME,
    "trajectory.txt",
)

# UTM coordinate reference system.
# Munich is in UTM zone 32N, WGS84.
# EPSG:32632 = WGS84 / UTM zone 32N.
UTM_EPSG = 32632

# Output root.
# This is relative to the directory where you run the script.
MAP_OUTPUT_ROOT = os.path.join("data", "maps")

# Radius around the detected road-level location used for OSM extraction.
DIST = 200

# Map creation mode.
# Use:
# - "manual" to select hero car and parking areas manually
# - "auto" to use first trajectory pose as hero car/start position
MODE = "auto"

# In auto mode, use the first pose from trajectory.txt instead of clicking.
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
# Keep this as None for full automatic mode.
ADDRESS_OVERRIDE = None
# Example:
# ADDRESS_OVERRIDE = "Guerickestraße, Alte Heide, Munich"


# =======================
# HELPERS
# =======================

def load_first_trajectory_pose(trajectory_file):
    """
    Load the first pose from trajectory.txt.

    Expected format:
    # timestamp x y z yaw
    timestamp, x, y, z, yaw
    """
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    df = pd.read_csv(
        trajectory_file,
        comment="#",
        header=None,
        names=["timestamp", "x", "y", "z", "yaw"],
        skipinitialspace=True,
    )

    if df.empty:
        raise RuntimeError(f"Trajectory file is empty: {trajectory_file}")

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

    This is used only to preserve previously saved offset values when re-running
    the script with the same distance.
    """
    vehicle_data_path = os.path.join(folder_name, "vehicle_data.json")

    if not os.path.exists(vehicle_data_path):
        return None

    with open(vehicle_data_path, "r") as f:
        return json.load(f)


def build_vehicle_output_from_first_pose(first_pose, dist):
    """
    Build vehicle data from the first trajectory pose instead of a GUI click.

    This assumes the downstream vehicle_data.json format stores the hero/start
    position under the key "start".

    If your old GUI-generated vehicle_data.json uses a different key, adjust
    only this function.
    """
    return {
        "dist": dist,
        "start": {
            "x": float(first_pose["x"]),
            "y": float(first_pose["y"]),
            "z": float(first_pose["z"]),
            "yaw": float(first_pose["yaw"]),
        },
        "parking": [],
        "offset": None,
    }


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
    maps_root = os.path.abspath(MAP_OUTPUT_ROOT)
    folder_name = os.path.join(maps_root, DATASET_NAME)

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
    print(f"Map name: {DATASET_NAME}")
    print(f"Trajectory file: {os.path.abspath(TRAJECTORY_FILE)}")
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

    # 5. Load existing vehicle data to preserve offsets when re-running
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
        print("Using first trajectory pose as hero car/start position.")
        print("No GUI click needed.")
        print(f"Start x: {first_pose['x']}")
        print(f"Start y: {first_pose['y']}")
        print(f"Start z: {first_pose['z']}")
        print(f"Start yaw: {first_pose['yaw']}")

        output_json = build_vehicle_output_from_first_pose(first_pose, DIST)

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

    # 12. Preserve previous offset values if possible
    if existing_vehicle_data is not None:
        if (
            existing_vehicle_data.get("dist") == DIST
            and existing_vehicle_data.get("offset") is not None
            and output_json is not None
        ):
            output_json["offset"] = existing_vehicle_data["offset"]
            print("Copied offset values from existing vehicle data.")

    # 13. Save map and vehicle data
    if output_json is not None:
        save_map_data(folder_name, osm_data, NO_CARLA)
        save_vehicle_data(folder_name, output_json)

        print(f"\n[{MODE.upper()}] Map and vehicle data saved to:")
        print(folder_name)

        if MODE != "manual":
            print(
                "JSON created with hero car data from first trajectory pose. "
                "Run create_vehicle_data_from_centroids.py to add clusters."
            )
    else:
        print("\nPlot closed without selection. Nothing saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()