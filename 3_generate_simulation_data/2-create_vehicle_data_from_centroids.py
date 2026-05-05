#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create / overwrite spawn_positions in maps/<map_name>/vehicle_data.json
from a file of cluster centroids expressed in odom coordinates.

Supports input format (8 columns, no color):
cluster_id, x, y, z, count, conf, orientation, side
Example:
1, 692955.845, 5339059.252, 549.017, 274, 0.695, perpendicular, left

Usage:
    python 3-create_vehicle_data_from_centroids.py \
        --map guerickestrasse_alte_heide_munich_25_11_03 \
        --centroids datasets/2026-02-16-23-07-19/unified_clusters_filtered.txt
"""

import os
import sys

# =======================
# PATH SETUP (workaround per import da root del progetto)
# =======================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import math
import argparse

import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from collections import Counter

from config import (
    MAPS_FOLDER_NAME, SPAWN_OFFSET_METERS, CAR_SPACING,
    SPAWN_OFFSET_METERS_LEFT, SPAWN_OFFSET_METERS_RIGHT,
    CARLA_OFFSET_X, CARLA_OFFSET_Y
)
from utils.map_data import (
    fetch_osm_data,
    generate_spawn_gdf,
    latlon_to_carla,
    get_heading,
)
# Use centralized math
from utils.coordinates import odom_xy_to_wgs84_vec
from utils.carla_simulator import get_xodr_projection_params, calculate_grid_convergence


# =======================
#  Centroid loader
# =======================

def load_centroids_xy(path: str):
    """
    Parses file format (8 columns, no color):
    cluster_id, x, y, z, count, last_conf, orientation, side

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

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            # We need at least id, x, y (3 columns)
            if len(parts) < 3:
                continue

            # --- ID, X, Y ---
            try:
                cid = int(float(parts[0]))
                x_val = float(parts[1])
                y_val = float(parts[2])
            except ValueError:
                continue

            # --- orientation (index 6) ---
            orient = "unknown"
            if len(parts) >= 7:
                o = parts[6].lower()
                if o in ("parallel", "perpendicular"):
                    orient = o

            # --- side (index 7) ---
            side = "unknown"
            if len(parts) >= 8:
                s = parts[7].lower()
                if s in ("left", "right"):
                    side = s

            ids.append(cid)
            xs.append(x_val)
            ys.append(y_val)
            orientations.append(orient)
            sides.append(side)

    if not xs:
        raise RuntimeError(f"Failed to load centroids from {path}: no valid lines parsed.")

    id_arr = np.array(ids, dtype=int)
    x_arr  = np.array(xs, dtype=float)
    y_arr  = np.array(ys, dtype=float)

    return id_arr, x_arr, y_arr, orientations, sides


# =======================
#  Spawn positions builder (XODR local coordinates)
# =======================

def build_spawn_positions_from_centroids_xodr(cluster_ids, cent_x, cent_y,
                                              orientation, side_labels, edges,
                                              xodr_params):
    """
    Maps centroids to XODR local coordinates directly.
    This uses the EXACT same projection as the XODR file, ensuring coordinates
    align perfectly with the roads in CARLA.

    The output coordinates are in XODR local space (0 to size_x, 0 to size_y).

    Grid convergence is calculated automatically from the location to correct headings.
    """
    from pyproj import Transformer

    assert cent_x.size == cent_y.size == len(cluster_ids) == len(orientation) == len(side_labels), \
        "All input arrays must have the same length"

    print(f"[INFO] Building spawn_positions for {cent_x.size} centroids (XODR local coords)...")

    if xodr_params is None:
        raise ValueError("XODR params required for this function")

    # Setup XODR projection
    proj_string = xodr_params["geo_reference"].strip()
    if proj_string == "+proj=tmerc":
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

    xodr_offset = xodr_params["offset"]
    xodr_center = xodr_params["center_local"]

    print(f"[INFO] XODR projection: {proj_string}")
    print(f"[INFO] XODR offset: ({xodr_offset[0]:.2f}, {xodr_offset[1]:.2f})")
    print(f"[INFO] XODR center (local): ({xodr_center[0]:.2f}, {xodr_center[1]:.2f})")

    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    # Convert UTM centroids to WGS84
    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)

    # Calculate grid convergence at the center of the data
    # This corrects for the difference between true north and XODR grid north
    avg_lat = float(np.mean(cent_lat))
    avg_lon = float(np.mean(cent_lon))
    grid_convergence = calculate_grid_convergence(avg_lat, avg_lon, central_meridian=0)
    print(f"[INFO] Grid convergence at ({avg_lat:.4f}, {avg_lon:.4f}): {grid_convergence:.2f}°")

    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS,
                                   offset_left=SPAWN_OFFSET_METERS_LEFT,
                                   offset_right=SPAWN_OFFSET_METERS_RIGHT, override=True)
    if spawn_gdf.empty:
        raise RuntimeError("spawn_gdf is empty: no parking lines generated.")

    spawn_positions = []
    print(f"[INFO] spawn_gdf has {len(spawn_gdf)} parking lines.")

    for i in range(cent_lat.size):
        lat_i = float(cent_lat[i])
        lon_i = float(cent_lon[i])
        pt = Point(lon_i, lat_i)

        side_label = str(side_labels[i]).lower()
        if side_label not in ("left", "right"):
            side_label = "unknown"

        # Filter by side
        if side_label in ("left", "right"):
            cand = spawn_gdf[spawn_gdf["side"] == side_label]
            if cand.empty:
                cand = spawn_gdf
        else:
            cand = spawn_gdf

        # Find nearest line
        dists = cand.geometry.distance(pt)
        min_idx = int(dists.idxmin())
        row = cand.loc[min_idx]
        parking_line = row.geometry
        side = row["side"]

        if isinstance(parking_line, MultiLineString):
            parking_line = max(parking_line, key=lambda g: g.length)

        if not isinstance(parking_line, LineString):
            continue

        # Heading from parking line
        coords_pl = list(parking_line.coords)
        start_lat_pl, start_lon_pl = coords_pl[0][1], coords_pl[0][0]
        end_lat_pl, end_lon_pl = coords_pl[-1][1], coords_pl[-1][0]
        line_heading = get_heading(start_lat_pl, start_lon_pl, end_lat_pl, end_lon_pl)

        if side == "left":
            line_heading = (line_heading + 180.0) % 360.0

        raw_mode = str(orientation[i]).lower()
        mode = raw_mode if raw_mode in ("parallel", "perpendicular") else "parallel"

        veh_heading = line_heading
        if mode == "perpendicular":
            veh_heading = (veh_heading + 90.0) % 360.0

        # Adjust heading for grid convergence
        # The heading is calculated relative to true north, but CARLA roads
        # follow the XODR grid which is rotated by grid_convergence degrees.
        # Subtract grid convergence to align with XODR grid north.
        veh_heading = (veh_heading - grid_convergence) % 360.0

        # === CONVERT TO XODR LOCAL COORDINATES ===
        # 1. Project lat/lon to XODR projection space
        proj_x, proj_y = transformer.transform(lon_i, lat_i)

        # 2. Apply XODR offset to get local coords
        local_x = proj_x + xodr_offset[0]
        local_y = proj_y + xodr_offset[1]

        # 3. Convert to CARLA coords (Y-flip: CARLA Y = -XODR Y)
        carla_x = local_x
        carla_y = -local_y

        start_pos = (carla_x, carla_y, 0.0)
        end_pos = (carla_x + 0.0001, carla_y + 0.0001, 0.0)

        street_id = row.get("osmid_str", row.get("osmid", None))

        spawn_positions.append({
            "cluster_id": int(cluster_ids[i]),
            "side": side,
            "street_id": street_id,
            "mode": mode,
            "start": start_pos,
            "end": end_pos,
            "heading": veh_heading,
        })

    print(f"[INFO] Created {len(spawn_positions)} spawn segments.")
    side_mode = Counter((s["side"], s["mode"]) for s in spawn_positions)
    print("[INFO] Spawn side/mode counts:")
    for (side, mode), n in sorted(side_mode.items()):
        print(f"   {side:5s} {mode:12s}: {n}")

    return spawn_positions


# =======================
#  Main
# =======================

def main():
    parser = argparse.ArgumentParser(
        description="Overwrite spawn_positions in vehicle_data.json from odom centroids."
    )
    parser.add_argument(
        "--map", required=True,
        help="Map folder name (without 'maps/')"
    )
    parser.add_argument(
        "--centroids", required=True,
        help="Path to centroid file"
    )
    args = parser.parse_args()

    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)
    if not os.path.isdir(map_folder):
        raise FileNotFoundError(f"Map folder not found: {map_folder}")

    centroid_file = args.centroids

    # LOAD DATA
    cluster_ids, cent_x, cent_y, orientation, side_labels = load_centroids_xy(centroid_file)

    if cent_x.size == 0:
        raise RuntimeError(f"No centroids loaded from {centroid_file}.")

    print(f"[INFO] Loaded {cent_x.size} centroids from {centroid_file}")

    _, edges, _ = fetch_osm_data(map_folder)

    # Load XODR file and extract projection parameters
    xodr_file = os.path.join(map_folder, "map.xodr")
    xodr_params = None
    if os.path.exists(xodr_file):
        print(f"[INFO] Loading XODR projection from: {xodr_file}")
        with open(xodr_file, "r") as f:
            xodr_data = f.read()
        xodr_params = get_xodr_projection_params(xodr_data)
    else:
        print(f"[WARN] XODR file not found: {xodr_file}")
        print(f"       Will use OSM-center-based projection (may have offset)")

    # BUILD POSITIONS using XODR local coordinates directly
    spawn_positions = build_spawn_positions_from_centroids_xodr(
        cluster_ids, cent_x, cent_y, orientation, side_labels, edges,
        xodr_params
    )

    # SAVE
    vehicle_data_path = os.path.join(map_folder, "vehicle_data.json")
    if not os.path.exists(vehicle_data_path):
        raise FileNotFoundError(
            f"vehicle_data.json not found in {map_folder}.\n"
            f"First run 2-create_map.py to generate it."
        )

    with open(vehicle_data_path, "r") as f:
        vehicle_data = json.load(f)

    vehicle_data["spawn_positions"] = spawn_positions

    vehicle_data["offset"] = {
        "x": CARLA_OFFSET_X,  # Fine-tuning only (default 0)
        "y": CARLA_OFFSET_Y,  # Fine-tuning only (default 0)
        "heading": 0.0
    }

    with open(vehicle_data_path, "w") as f:
        json.dump(vehicle_data, f, indent=2)

    print(f"\n✅ Updated spawn_positions in: {vehicle_data_path}")
    print(f"   Number of parked cars (segments): {len(spawn_positions)}")


if __name__ == "__main__":
    main()