#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2C_create_map.py

Generate OSM map data and vehicle metadata for a dataset.

Default behavior is automatic and non-interactive:
    - read the first pose from data/raw_dataset/<BAG>/trajectory.csv
    - convert UTM to latitude/longitude
    - reverse-geocode the position
    - fetch OSM data around the detected road-level address
    - write map files and a placeholder vehicle_data.json

If --manual is enabled, the script opens the GUI after map generation and lets
the user manually select the hero_car position. The selected hero_car is saved
to vehicle_data.json.

Reads from:
    data/raw_dataset/<BAG>/trajectory.csv                 # default automatic mode
    data/processed_dataset/<BAG>/maps/map.osm             # when --skip-fetch is used
    data/processed_dataset/<BAG>/maps/vehicle_data.json   # optional existing metadata

Writes to:
    data/processed_dataset/<BAG>/maps/
        map.osm
        map.xodr
        buildings.obj
        vehicle_data.json

Parameters:
    --bag-name <BAG>.bag
        Bag filename including .bag extension.
        The input dataset is read from:
            data/raw_dataset/<BAG>/
        Default: env BAG_NAME or reference_bag.bag.

    --manual
        Open the GUI to manually select the hero_car position.
        Disabled by default.

    --address <ADDRESS>
        Use a manually provided address instead of reverse-geocoding the first
        trajectory pose.

    --skip-fetch
        Reuse existing OSM data from:
            data/processed_dataset/<BAG>/maps/
        instead of fetching fresh OSM data.

    --no-carla
        Skip the CARLA functionality check.

    --dist <METERS>
        Radius around the selected address / position used for OSM extraction.
        Default: 200.

    --utm-epsg <EPSG>
        EPSG code for the UTM coordinates in trajectory.csv.
        Default: 32632.

Usage:
    python 2_process_datasets/2C_create_map.py --bag-name snowy.bag

    python 2_process_datasets/2C_create_map.py \
        --bag-name snowy.bag \
        --manual

    python 2_process_datasets/2C_create_map.py \
        --bag-name snowy.bag \
        --address "Guerickestraße, Alte Heide, Munich" \
        --manual

    python 2_process_datasets/2C_create_map.py \
        --bag-name snowy.bag \
        --skip-fetch \
        --manual
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from pyproj import Transformer


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_UTILS_DIR = SCRIPT_DIR / "utils"

if not LOCAL_UTILS_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )

# Force imports from 2_process_datasets/utils.
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

sys.path.insert(0, str(SCRIPT_DIR))


# =============================================================================
# IMPORTS FROM LOCAL UTILS
# =============================================================================

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


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_DIST = 200
DEFAULT_UTM_EPSG = 32632
NOMINATIM_USER_AGENT = "cam2sim-map-generation"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate OSM map data and vehicle metadata for a dataset."
    )

    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help=(
            "Bag filename including .bag extension "
            "(default: env BAG_NAME or 'reference_bag.bag')."
        ),
    )

    parser.add_argument(
        "--manual",
        action="store_true",
        help="Open GUI to manually select the hero_car position. Disabled by default.",
    )

    parser.add_argument(
        "--address",
        default=None,
        help="Manual address to use instead of reverse-geocoding the first trajectory pose.",
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse existing OSM data instead of fetching fresh OSM data.",
    )

    parser.add_argument(
        "--no-carla",
        action="store_true",
        help="Skip CARLA functionality check.",
    )

    parser.add_argument(
        "--dist",
        type=int,
        default=DEFAULT_DIST,
        help=f"OSM extraction radius in meters. Default: {DEFAULT_DIST}",
    )

    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=DEFAULT_UTM_EPSG,
        help=f"EPSG code for UTM coordinates. Default: {DEFAULT_UTM_EPSG}",
    )

    return parser.parse_args()


# =============================================================================
# TRAJECTORY / GEOCODING HELPERS
# =============================================================================

def load_first_trajectory_pose(trajectory_file):
    """
    Load the first valid pose from trajectory.csv.

    Supports both:

    Header format:
        timestamp,x,y,z,yaw

    Headerless format:
        timestamp, x, y, z, yaw
    """
    if not trajectory_file.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    required_columns = ["timestamp", "x", "y", "z", "yaw"]

    df = pd.read_csv(
        trajectory_file,
        comment="#",
        skipinitialspace=True,
    )

    df.columns = [str(col).strip().lower() for col in df.columns]

    if not all(col in df.columns for col in required_columns):
        df = pd.read_csv(
            trajectory_file,
            comment="#",
            header=None,
            names=required_columns,
            skipinitialspace=True,
        )

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

    Returns the full Nominatim JSON object so a cleaner road-level OSM query
    address can be built.
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
    Build a road-level address for OSM queries.

    This intentionally removes house number, postcode, state, and country,
    because querying the full reverse-geocoded address can shift the OSM
    download center to a specific building.
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


# =============================================================================
# VEHICLE DATA HELPERS
# =============================================================================

def load_existing_vehicle_data(maps_dir):
    """
    Load existing vehicle_data.json, if present.
    """
    vehicle_data_path = maps_dir / "vehicle_data.json"

    if not vehicle_data_path.is_file():
        return None

    with open(vehicle_data_path, "r") as file:
        return json.load(file)


def build_placeholder_vehicle_data(dist):
    """
    Build placeholder vehicle data.

    The real CARLA hero position is generated later by:
        3A_transform_coordinates_yaw_to_carla.py

    Output schema:
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


def normalize_existing_vehicle_data(vehicle_data, dist):
    """
    Normalize old or partial vehicle_data.json into the current schema.

    Preserves:
      - offset
      - hero_car
      - spawn_positions

    Removes old keys:
      - start
      - parking
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

    return {
        "offset": offset,
        "dist": dist,
        "hero_car": hero_car,
        "spawn_positions": spawn_positions,
    }


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
        "Could not extract hero_car from GUI output. "
        "Expected key 'start' or 'hero_car'."
    )


def build_vehicle_data_from_gui_output(raw_output_json, existing_vehicle_data, dist):
    """
    Build final vehicle_data.json using GUI-selected hero_car.

    Preserves existing:
      - offset
      - spawn_positions

    Uses clicked GUI output for:
      - hero_car
    """
    if raw_output_json is None:
        return None

    hero_car = extract_hero_car_from_gui_output(raw_output_json)

    existing_normalized = normalize_existing_vehicle_data(
        existing_vehicle_data,
        dist,
    )

    return {
        "offset": existing_normalized["offset"],
        "dist": dist,
        "hero_car": hero_car,
        "spawn_positions": existing_normalized["spawn_positions"],
    }


def build_vehicle_data_auto(existing_vehicle_data, dist):
    """
    Build automatic placeholder vehicle_data.json while preserving existing data.
    """
    output_json = build_placeholder_vehicle_data(dist)

    if existing_vehicle_data is not None:
        existing_normalized = normalize_existing_vehicle_data(
            existing_vehicle_data,
            dist,
        )

        output_json["offset"] = existing_normalized["offset"]
        output_json["hero_car"] = existing_normalized["hero_car"]
        output_json["spawn_positions"] = existing_normalized["spawn_positions"]

        print("Preserved existing offset, hero_car, and spawn_positions where available.")

    return output_json


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem

    trajectory_file = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem / "trajectory.csv"
    maps_dir = PROJECT_ROOT / "data" / "processed_dataset" / bag_stem / "maps"

    print("=" * 70)
    print("MAP DATA GENERATION")
    print("=" * 70)
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Script folder:     {SCRIPT_DIR}")
    print(f"Local utils:       {LOCAL_UTILS_DIR}")
    print(f"Bag:               {bag_name}")
    print(f"Bag stem:          {bag_stem}")
    print(f"Trajectory file:   {trajectory_file}")
    print(f"Output folder:     {maps_dir}")
    print(f"Manual mode:       {args.manual}")
    print(f"Address override:  {args.address}")
    print(f"Distance:          {args.dist}")
    print(f"UTM EPSG:          {args.utm_epsg}")
    print(f"No CARLA:          {args.no_carla}")
    print(f"Skip OSM fetch:    {args.skip_fetch}")
    print("=" * 70)

    if not bag_stem:
        raise ValueError("bag_stem must be a non-empty string.")

    if args.address is None and not args.skip_fetch:
        if not trajectory_file.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    existing_vehicle_data = load_existing_vehicle_data(maps_dir)

    if not args.no_carla:
        ensure_carla_functionality()

    # -------------------------------------------------------------------------
    # 1. Determine OSM query address
    # -------------------------------------------------------------------------
    osm_query_address = None

    if args.skip_fetch:
        print("\n[INFO] Skipping reverse geocoding because --skip-fetch is enabled.")

    elif args.address is not None:
        osm_query_address = args.address

        print("\n[INFO] Using manual address:")
        print(osm_query_address)

    else:
        print("\n[INFO] Loading first trajectory pose...")

        first_pose = load_first_trajectory_pose(trajectory_file)

        lat, lon = utm_to_latlon(
            first_pose["x"],
            first_pose["y"],
            args.utm_epsg,
        )

        print(f"[INFO] UTM easting:  {first_pose['x']}")
        print(f"[INFO] UTM northing: {first_pose['y']}")
        print(f"[INFO] Latitude:     {lat}")
        print(f"[INFO] Longitude:    {lon}")

        print("\n[INFO] Reverse-geocoding trajectory position...")

        reverse_data = reverse_geocode_latlon(lat, lon)

        # Be polite to Nominatim before the next possible network request.
        time.sleep(1.0)

        full_detected_address = reverse_data["display_name"]
        osm_query_address = build_road_level_address(reverse_data)

        print(f"[INFO] Detected full address:       {full_detected_address}")
        print(f"[INFO] Road-level OSM query address: {osm_query_address}")

    # -------------------------------------------------------------------------
    # 2. Fetch or load OSM data
    # -------------------------------------------------------------------------
    if not args.skip_fetch:
        print("\n[INFO] Fetching OSM data for:")
        print(osm_query_address)

        osm_data = get_street_data(
            osm_query_address,
            dist=args.dist,
        )
    else:
        print(f"\n[INFO] Loading existing OSM data from: {maps_dir}")

        osm_data = get_existing_osm_data(str(maps_dir))

    # -------------------------------------------------------------------------
    # 3. Create map folders and save raw OSM if needed
    # -------------------------------------------------------------------------
    create_map_folders(str(maps_dir))

    if not args.skip_fetch:
        save_osm_data(str(maps_dir), osm_data)

    # -------------------------------------------------------------------------
    # 4. Process map geometry
    # -------------------------------------------------------------------------
    print("\n[INFO] Processing map geometry: nodes, edges, and buildings")

    graph, edges, buildings = fetch_osm_data(str(maps_dir))

    # -------------------------------------------------------------------------
    # 5. Build vehicle_data.json
    # -------------------------------------------------------------------------
    if args.manual:
        print("\n[INFO] Manual mode enabled. Opening GUI.")
        print("[INFO] Select the hero_car position, then close the window.")

        plot_title = osm_query_address if osm_query_address is not None else bag_stem

        create_plot(buildings, edges, plot_title)
        show_plot()

        raw_output_json = get_output(args.dist)

        if raw_output_json is not None:
            output_json = build_vehicle_data_from_gui_output(
                raw_output_json=raw_output_json,
                existing_vehicle_data=existing_vehicle_data,
                dist=args.dist,
            )

            print("\n[INFO] Hero car selected:")
            print(f"  Position: {output_json['hero_car']['position']}")
            print(f"  Heading:  {output_json['hero_car']['heading']}")
            print(f"  Preserved spawn_positions: {len(output_json['spawn_positions'])}")
        else:
            output_json = None

    else:
        print("\n[INFO] Automatic mode.")
        print("[INFO] Writing placeholder vehicle_data.json.")
        print("[INFO] The real CARLA hero position will be written later by:")
        print("       3A_transform_coordinates_yaw_to_carla.py")

        output_json = build_vehicle_data_auto(
            existing_vehicle_data=existing_vehicle_data,
            dist=args.dist,
        )

    # -------------------------------------------------------------------------
    # 6. Save map and vehicle data
    # -------------------------------------------------------------------------
    if output_json is not None:
        save_map_data(
            str(maps_dir),
            osm_data,
            args.no_carla,
        )

        save_vehicle_data(
            str(maps_dir),
            output_json,
        )

        print("\n[OK] Map and vehicle data saved to:")
        print(maps_dir)

        if not args.manual:
            print(
                "[INFO] vehicle_data.json contains placeholder hero_car data. "
                "Run 3A_transform_coordinates_yaw_to_carla.py to write the real hero_car, "
                "and run the centroid script to add spawn_positions."
            )

    else:
        print("\n[WARN] GUI closed without selection. Nothing saved.")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()