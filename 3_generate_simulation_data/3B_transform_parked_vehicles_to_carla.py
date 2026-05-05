#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create / overwrite spawn_positions in final CARLA vehicle_data.json.

Reads map data from:

    data/generated_data_from_extracted_data/<BAG_NAME>/maps

Reads centroid input from:

    data/generated_data_from_extracted_data/<BAG_NAME>/lidar_detections/unified_clusters.txt

Writes final vehicle data to:

    data/data_for_carla/<BAG_NAME>/vehicle_data.json

The centroid parser is flexible and ignores RGB/color columns.

It requires at least:

    cluster_id, x, y

It then scans the remaining columns for:

    orientation: parallel / perpendicular
    side:        left / right
"""

import os
import sys
import json
from collections import Counter

import numpy as np
from shapely.geometry import Point, LineString, MultiLineString


# =======================
# PATH SETUP
# =======================

# Folder where this script is located, for example:
# /home/cam2sim/Documents/cam2sim/3_generate_simulation_data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root, for data paths only:
# /home/cam2sim/Documents/cam2sim
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Local utils folder:
# /home/cam2sim/Documents/cam2sim/3_generate_simulation_data/utils
LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to script, but not found: {LOCAL_UTILS_DIR}"
    )

# Force Python to import utils from SCRIPT_DIR first.
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)


# =======================
# HARDCODED BAG NAME
# =======================

# Change this to select another bag.
# No command-line parameters are used.
BAG_NAME = "reference_bag"


# =======================
# HARDCODED INPUTS / OUTPUTS
# =======================

# Read map geometry and map.xodr from here.
MAP_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "generated_data_from_extracted_data",
    BAG_NAME,
    "maps",
)

# Read centroid file from here.
CENTROIDS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "generated_data_from_extracted_data",
    BAG_NAME,
    "lidar_detections",
    "unified_clusters.txt",
)

# Save final CARLA-ready data here.
OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_carla",
    BAG_NAME,
)

# Final CARLA vehicle_data.json.
FINAL_VEHICLE_DATA_PATH = os.path.join(
    OUTPUT_FOLDER,
    "vehicle_data.json",
)

# Optional source vehicle_data.json from map-generation step.
SOURCE_VEHICLE_DATA_PATH = os.path.join(
    MAP_FOLDER,
    "vehicle_data.json",
)


# =======================
# LOCAL SCRIPT-FOLDER UTILS IMPORTS
# =======================

from utils.config import (
    SPAWN_OFFSET_METERS,
    SPAWN_OFFSET_METERS_LEFT,
    SPAWN_OFFSET_METERS_RIGHT,
    CARLA_OFFSET_X,
    CARLA_OFFSET_Y,
)

from utils.map_data import (
    fetch_osm_data,
    generate_spawn_gdf,
    get_heading,
)

from utils.coordinates import odom_xy_to_wgs84_vec

from utils.carla_simulator import (
    get_xodr_projection_params,
    calculate_grid_convergence,
)


# =======================
# VEHICLE DATA HELPERS
# =======================

def load_base_vehicle_data():
    """
    Load vehicle_data.json for preservation.

    Priority:
      1. Existing final CARLA vehicle_data.json in data/data_for_carla/<BAG_NAME>
      2. Source vehicle_data.json from generated map folder
      3. Empty dict

    This preserves hero_car if trajectory script already wrote it.
    """
    if os.path.exists(FINAL_VEHICLE_DATA_PATH):
        with open(FINAL_VEHICLE_DATA_PATH, "r") as f:
            return json.load(f)

    if os.path.exists(SOURCE_VEHICLE_DATA_PATH):
        with open(SOURCE_VEHICLE_DATA_PATH, "r") as f:
            return json.load(f)

    return {}


def normalize_vehicle_data(vehicle_data):
    """
    Normalize vehicle_data.json into final expected format.

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

    hero_car = vehicle_data.get("hero_car")

    if hero_car is None and isinstance(vehicle_data.get("start"), dict):
        start = vehicle_data["start"]
        hero_car = {
            "position": [
                float(start.get("x", 0.0)),
                float(start.get("y", 0.0)),
                float(start.get("z", 0.0)),
            ],
            "heading": float(start.get("yaw", 0.0)),
        }

    if hero_car is None:
        hero_car = {
            "position": [
                0.0,
                0.0,
                0.0,
            ],
            "heading": 0.0,
        }

    spawn_positions = vehicle_data.get("spawn_positions", [])

    normalized = {
        "offset": offset,
        "dist": int(vehicle_data.get("dist", 200)),
        "hero_car": hero_car,
        "spawn_positions": spawn_positions,
    }

    return normalized


# =======================
# CENTROID LOADER
# =======================

def load_centroids_xy(path: str):
    """
    Parse centroid file.

    Required:
      column 0: cluster_id
      column 1: x
      column 2: y

    Remaining columns may contain z, count, confidence, RGB/color values,
    orientation, side, etc.

    This function ignores RGB/color columns and scans all later columns for:
      orientation: parallel / perpendicular
      side:        left / right

    Returns:
      cluster_ids, x, y, orientations, sides
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Centroid file not found: {path}")

    ids = []
    xs = []
    ys = []
    orientations = []
    sides = []

    skipped = 0

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 3:
                skipped += 1
                continue

            try:
                cid = int(float(parts[0]))
                x_val = float(parts[1])
                y_val = float(parts[2])
            except ValueError:
                skipped += 1
                continue

            orientation = "unknown"
            side = "unknown"

            # Scan remaining columns so RGB/color columns are ignored.
            for raw_value in parts[3:]:
                value = raw_value.strip().lower()

                if value in ("parallel", "perpendicular"):
                    orientation = value

                elif value in ("left", "right"):
                    side = value

            ids.append(cid)
            xs.append(x_val)
            ys.append(y_val)
            orientations.append(orientation)
            sides.append(side)

    if not xs:
        raise RuntimeError(
            f"Failed to load centroids from {path}: no valid lines parsed."
        )

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} malformed centroid rows.")

    id_arr = np.array(ids, dtype=int)
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    print("[INFO] Parsed centroid labels:")
    print(f"   Orientation counts: {Counter(orientations)}")
    print(f"   Side counts:        {Counter(sides)}")

    return id_arr, x_arr, y_arr, orientations, sides


# =======================
# SPAWN POSITION BUILDER
# =======================

def build_spawn_positions_from_centroids_xodr(
    cluster_ids,
    cent_x,
    cent_y,
    orientation,
    side_labels,
    edges,
    xodr_params,
):
    """
    Convert centroid odom / UTM coordinates to CARLA spawn positions.

    Coordinate chain:
      odom / UTM x,y
        -> WGS84 lat/lon
        -> XODR projection
        -> XODR local coordinates
        -> CARLA coordinates

    CARLA conversion:
      carla_x = local_x
      carla_y = -local_y
    """

    from pyproj import Transformer

    assert (
        cent_x.size
        == cent_y.size
        == len(cluster_ids)
        == len(orientation)
        == len(side_labels)
    ), "All input arrays must have the same length"

    print(
        f"[INFO] Building spawn_positions for "
        f"{cent_x.size} centroids using XODR local coordinates..."
    )

    if xodr_params is None:
        raise ValueError("XODR params are required.")

    # Setup XODR projection.
    proj_string = xodr_params["geo_reference"].strip()

    if proj_string == "+proj=tmerc":
        proj_string = (
            "+proj=tmerc "
            "+lat_0=0 "
            "+lon_0=0 "
            "+k=1 "
            "+x_0=0 "
            "+y_0=0 "
            "+datum=WGS84"
        )

    xodr_offset = xodr_params["offset"]
    xodr_center = xodr_params.get("center_local", None)

    print(f"[INFO] XODR projection: {proj_string}")
    print(f"[INFO] XODR offset: ({xodr_offset[0]:.2f}, {xodr_offset[1]:.2f})")

    if xodr_center is not None:
        print(
            f"[INFO] XODR center local: "
            f"({xodr_center[0]:.2f}, {xodr_center[1]:.2f})"
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        proj_string,
        always_xy=True,
    )

    # Convert odom / UTM centroids to WGS84.
    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)

    # Grid convergence correction.
    avg_lat = float(np.mean(cent_lat))
    avg_lon = float(np.mean(cent_lon))

    grid_convergence = calculate_grid_convergence(
        avg_lat,
        avg_lon,
        central_meridian=0,
    )

    print(
        f"[INFO] Grid convergence at "
        f"({avg_lat:.6f}, {avg_lon:.6f}): {grid_convergence:.2f} degrees"
    )

    # Generate candidate parking lines from road edges.
    spawn_gdf = generate_spawn_gdf(
        edges,
        offset=SPAWN_OFFSET_METERS,
        offset_left=SPAWN_OFFSET_METERS_LEFT,
        offset_right=SPAWN_OFFSET_METERS_RIGHT,
        override=True,
    )

    if spawn_gdf.empty:
        raise RuntimeError("spawn_gdf is empty: no parking lines generated.")

    print(f"[INFO] spawn_gdf has {len(spawn_gdf)} parking lines.")

    spawn_positions = []

    for i in range(cent_lat.size):
        lat_i = float(cent_lat[i])
        lon_i = float(cent_lon[i])

        pt = Point(lon_i, lat_i)

        requested_side = str(side_labels[i]).lower()

        if requested_side not in ("left", "right"):
            requested_side = "unknown"

        # Filter candidate parking lines by side if possible.
        if requested_side in ("left", "right"):
            candidates = spawn_gdf[spawn_gdf["side"] == requested_side]

            if candidates.empty:
                candidates = spawn_gdf
        else:
            candidates = spawn_gdf

        # Find nearest candidate parking line.
        dists = candidates.geometry.distance(pt)
        min_idx = int(dists.idxmin())
        row = candidates.loc[min_idx]

        parking_line = row.geometry
        side = row["side"]

        if isinstance(parking_line, MultiLineString):
            parking_line = max(parking_line, key=lambda g: g.length)

        if not isinstance(parking_line, LineString):
            continue

        # Heading from parking line.
        coords_pl = list(parking_line.coords)

        start_lat_pl = coords_pl[0][1]
        start_lon_pl = coords_pl[0][0]
        end_lat_pl = coords_pl[-1][1]
        end_lon_pl = coords_pl[-1][0]

        line_heading = get_heading(
            start_lat_pl,
            start_lon_pl,
            end_lat_pl,
            end_lon_pl,
        )

        if side == "left":
            line_heading = (line_heading + 180.0) % 360.0

        raw_mode = str(orientation[i]).lower()

        if raw_mode in ("parallel", "perpendicular"):
            mode = raw_mode
        else:
            mode = "parallel"

        vehicle_heading = line_heading

        if mode == "perpendicular":
            vehicle_heading = (vehicle_heading + 90.0) % 360.0

        # Correct heading from true north to XODR grid north.
        vehicle_heading = (vehicle_heading - grid_convergence) % 360.0

        # Convert centroid to XODR local coordinates.
        proj_x, proj_y = transformer.transform(lon_i, lat_i)

        local_x = proj_x + xodr_offset[0]
        local_y = proj_y + xodr_offset[1]

        # Convert XODR local to CARLA coordinates.
        carla_x = local_x
        carla_y = -local_y

        start_pos = (float(carla_x), float(carla_y), 0.0)

        # Tiny end offset to keep same segment-style structure.
        end_pos = (
            float(carla_x + 0.0001),
            float(carla_y + 0.0001),
            0.0,
        )

        street_id = row.get("osmid_str", row.get("osmid", None))

        spawn_positions.append(
            {
                "cluster_id": int(cluster_ids[i]),
                "side": side,
                "street_id": street_id,
                "mode": mode,
                "start": start_pos,
                "end": end_pos,
                "heading": float(vehicle_heading),
            }
        )

    print(f"[INFO] Created {len(spawn_positions)} spawn segments.")

    side_mode_counts = Counter(
        (spawn["side"], spawn["mode"])
        for spawn in spawn_positions
    )

    print("[INFO] Spawn side/mode counts:")

    for (side, mode), count in sorted(side_mode_counts.items()):
        print(f"   {side:5s} {mode:12s}: {count}")

    return spawn_positions


# =======================
# MAIN
# =======================

def main():
    print("=" * 80)
    print("CREATE PARKED VEHICLE SPAWN POSITIONS FROM CENTROIDS")
    print("=" * 80)
    print(f"[INFO] Script folder:              {SCRIPT_DIR}")
    print(f"[INFO] Local utils:                {LOCAL_UTILS_DIR}")
    print(f"[INFO] Project root:               {PROJECT_ROOT}")
    print(f"[INFO] Bag name:                   {BAG_NAME}")
    print(f"[INFO] Map folder:                 {MAP_FOLDER}")
    print(f"[INFO] Centroid file:              {CENTROIDS_FILE}")
    print(f"[INFO] Final CARLA output folder:  {OUTPUT_FOLDER}")
    print(f"[INFO] Source vehicle data path:   {SOURCE_VEHICLE_DATA_PATH}")
    print(f"[INFO] Final vehicle data path:    {FINAL_VEHICLE_DATA_PATH}")
    print("=" * 80)

    if not os.path.isdir(MAP_FOLDER):
        raise FileNotFoundError(f"Map folder not found: {MAP_FOLDER}")

    if not os.path.exists(CENTROIDS_FILE):
        raise FileNotFoundError(f"Centroid file not found: {CENTROIDS_FILE}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load centroid data.
    cluster_ids, cent_x, cent_y, orientation, side_labels = load_centroids_xy(
        CENTROIDS_FILE
    )

    print(f"[INFO] Loaded {cent_x.size} centroids.")

    # Load map geometry.
    _, edges, _ = fetch_osm_data(MAP_FOLDER)

    # Load XODR projection.
    xodr_file = os.path.join(MAP_FOLDER, "map.xodr")

    if not os.path.exists(xodr_file):
        raise FileNotFoundError(f"XODR file not found: {xodr_file}")

    print(f"[INFO] Loading XODR projection from: {xodr_file}")

    with open(xodr_file, "r") as f:
        xodr_data = f.read()

    xodr_params = get_xodr_projection_params(xodr_data)

    # Build spawn positions.
    spawn_positions = build_spawn_positions_from_centroids_xodr(
        cluster_ids=cluster_ids,
        cent_x=cent_x,
        cent_y=cent_y,
        orientation=orientation,
        side_labels=side_labels,
        edges=edges,
        xodr_params=xodr_params,
    )

    # Load and normalize existing vehicle data.
    vehicle_data = load_base_vehicle_data()
    vehicle_data = normalize_vehicle_data(vehicle_data)

    # Overwrite spawn_positions only.
    # Preserve hero_car from trajectory script if it already exists.
    vehicle_data["spawn_positions"] = spawn_positions

    # Fine-tuning offsets.
    vehicle_data["offset"] = {
        "x": CARLA_OFFSET_X,
        "y": CARLA_OFFSET_Y,
        "heading": 0.0,
    }

    # Save final CARLA vehicle_data.json.
    with open(FINAL_VEHICLE_DATA_PATH, "w") as f:
        json.dump(vehicle_data, f, indent=2)

    print("\nUpdated final CARLA vehicle_data.json")
    print(f"   File: {FINAL_VEHICLE_DATA_PATH}")
    print(f"   Number of parked cars: {len(spawn_positions)}")
    print(f"   Hero car preserved: {vehicle_data.get('hero_car') is not None}")
    print("\nDone.")


if __name__ == "__main__":
    main()