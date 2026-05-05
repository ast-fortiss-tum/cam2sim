#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates TWO JSON files:
1. trajectory_positions.json (Center of car - Original)
2. trajectory_positions_rear.json (Rear Axle/Back of car - New)

Updated to use XODR local coordinates (same pipeline as parked cars).
"""

import os
import json
import math
import argparse
import numpy as np
from pyproj import Transformer

# ==========================================
# IMPORTS
# ==========================================
from config import MAPS_FOLDER_NAME
from utils.coordinates import odom_xy_to_wgs84_vec, load_shift_values
from utils.carla_simulator import get_xodr_projection_params

# ==========================================
# CONFIGURATION
# ==========================================
# Distance from Center to Rear Axle (in meters).
# Based on your logic: extent.x * 0.9.
# A Tesla Model 3 extent.x is ~2.3m, so 2.3 * 0.9 ≈ 2.07m.
REAR_OFFSET = -1.393

# ==========================================
# DATA LOADING
# ==========================================
def load_positions(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Position file not found: {path}")

    frames, timestamps, oxs, oys = [], [], [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4: continue
            try:
                frames.append(int(parts[0]))
                timestamps.append(float(parts[1]))
                oxs.append(float(parts[2]))
                oys.append(float(parts[3]))
            except ValueError: continue

    return (np.array(frames), np.array(timestamps), np.array(oxs), np.array(oys))

# ==========================================
# COORDINATE CONVERSION (matches parked cars pipeline)
# ==========================================
def convert_to_xodr_local(lat, lon, transformer, xodr_offset):
    """
    Convert WGS84 lat/lon to XODR local coordinates.
    Same method as parked cars pipeline.
    """
    # Project to TM space
    proj_x, proj_y = transformer.transform(lon, lat)

    # Apply XODR offset to get local coords
    local_x = proj_x + xodr_offset[0]
    local_y = proj_y + xodr_offset[1]

    return local_x, local_y




# ==========================================
# HEADING CALCULATION
# ==========================================
def calculate_carla_heading(current_carla, next_carla):
    """
    Calculate heading in CARLA coordinate system.

    CARLA yaw convention:
    - 0° = positive X (East)
    - 90° = positive Y (South in CARLA)

    No grid convergence needed here because positions are already
    in CARLA/XODR coordinate space. Grid convergence is only needed
    when converting from geographic (true north) coordinates.
    """
    dx = next_carla[0] - current_carla[0]
    dy = next_carla[1] - current_carla[1]

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return None

    # atan2(dy, dx) for CARLA's coordinate system
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return angle_deg % 360.0


# ==========================================
# MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="Map folder name")
    parser.add_argument("--positions", required=True, help="Path to position.txt (trajectory output)")
    args = parser.parse_args()

    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)

    # 1. Load Trajectory Data
    frames, times, ox, oy = load_positions(args.positions)
    print(f"[INFO] Loaded {len(frames)} frames.")



    # 3. Load XODR projection parameters
    xodr_file = os.path.join(map_folder, "map.xodr")
    if not os.path.exists(xodr_file):
        raise FileNotFoundError(f"XODR file not found: {xodr_file}")

    with open(xodr_file, "r") as f:
        xodr_data = f.read()

    xodr_params = get_xodr_projection_params(xodr_data)
    xodr_offset = xodr_params["offset"]

    # Setup projection (same as XODR uses)
    proj_string = xodr_params["geo_reference"].strip()
    if proj_string == "+proj=tmerc":
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)



    # 4. Convert Odom -> Lat/Lon
    print(f"[INFO] Converting UTM to WGS84...")
    traj_lats, traj_lons = odom_xy_to_wgs84_vec(ox, oy)

    # 5. Convert to CARLA coordinates (center-relative)
    carla_positions = []
    for i in range(len(frames)):
        # WGS84 -> XODR local
        local_x, local_y = convert_to_xodr_local(
            traj_lats[i], traj_lons[i], transformer, xodr_offset
        )
        # XODR local -> CARLA 
        cx = local_x
        cy = -local_y
        carla_positions.append((cx, cy))

    # 6. Compute headings and build output
    trajectory_center = []
    trajectory_rear = []

    print("[INFO] Computing headings and rear offsets...")

    LOOKAHEAD = 5
    last_valid_heading = 0.0

    for i in range(len(frames)):
        cx, cy = carla_positions[i]

        # --- KINEMATIC HEADING LOGIC ---
        target_idx = min(i + LOOKAHEAD, len(frames) - 1)

        heading_val = None
        if target_idx == i and i > 0:
            prev_idx = max(0, i - LOOKAHEAD)
            heading_val = calculate_carla_heading(
                carla_positions[prev_idx], (cx, cy)
            )
        else:
            heading_val = calculate_carla_heading(
                (cx, cy), carla_positions[target_idx]
            )

        if heading_val is not None:
            last_valid_heading = heading_val

        # --- REAR AXLE CALCULATION ---
        yaw_rad = math.radians(last_valid_heading)
        rx = cx - math.cos(yaw_rad) * REAR_OFFSET
        ry = cy - math.sin(yaw_rad) * REAR_OFFSET

        # --- CREATE ENTRIES ---
        entry_center = {
            "frame_id": int(frames[i]),
            "timestamp": float(times[i]),
            "transform": {
                "location": { "x": cx, "y": cy, "z": 0.0 },
                "rotation": { "pitch": 0.0, "yaw": last_valid_heading, "roll": 0.0 }
            }
        }
        trajectory_center.append(entry_center)

        entry_rear = {
            "frame_id": int(frames[i]),
            "timestamp": float(times[i]),
            "transform": {
                "location": { "x": rx, "y": ry, "z": 0.0 },
                "rotation": { "pitch": 0.0, "yaw": last_valid_heading, "roll": 0.0 }
            }
        }
        trajectory_rear.append(entry_rear)

    # 7. Save Outputs
    output_center = os.path.join(map_folder, "trajectory_positions.json")
    output_rear   = os.path.join(map_folder, "trajectory_positions_rear.json")

    with open(output_center, "w") as f:
        json.dump(trajectory_center, f, indent=2)

    with open(output_rear, "w") as f:
        json.dump(trajectory_rear, f, indent=2)

    print(f"\n✅ Saved CENTER trajectory to: {output_center}")
    print(f"✅ Saved REAR trajectory to:   {output_rear}")
    print(f"   Total frames: {len(trajectory_center)}")

    if len(trajectory_center) > 0:
        first = trajectory_center[0]['transform']['location']
        last = trajectory_center[-1]['transform']['location']
        print(f"   First position: ({first['x']:.2f}, {first['y']:.2f})")
        print(f"   Last position:  ({last['x']:.2f}, {last['y']:.2f})")
        print(f"   First heading:  {trajectory_center[0]['transform']['rotation']['yaw']:.2f}°")

    # 8. Note about coordinate system
    # The trajectory uses the same offset as parked cars (should be 0,0)
    print(f"\n[INFO] Trajectory uses same coordinate system as parked cars.")
    print(f"       Apply translation_vector [size/2, -size/2] when spawning.")


if __name__ == "__main__":
    main()