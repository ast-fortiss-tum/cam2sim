#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_trajectory_odom_yaw.py

Generates trajectory JSON files using REAL odom yaw from the rosbag.

The UTM→CARLA yaw conversion is derived FROM the coordinate chain:
  1. Push UTM +X direction through the full position pipeline
  2. Measure what angle it becomes in CARLA space
  3. Use that to convert any UTM angle to CARLA angle

No manual offset — the rotation comes from the projection math.

Formula derivation:
  - UTM has axes: +X = East, +Y = North
  - After the chain (TM projection + XODR offset + Y-flip), UTM +X maps
    to some angle A in CARLA space
  - The Y-flip reverses rotation direction (right-hand → left-hand)
  - So UTM angle θ maps to CARLA angle: A - θ

Input: positions.txt with columns:
  FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, ImageFile

Output:
  trajectory_positions.json / _rear.json (kinematic yaw, original)
  trajectory_positions_odom_yaw.json / _rear_odom_yaw.json (odom yaw)
"""

import os
import json
import math
import argparse
import numpy as np
from pyproj import Transformer

from config import MAPS_FOLDER_NAME
from utils.coordinates import odom_xy_to_wgs84_vec
from utils.carla_simulator import get_xodr_projection_params

REAR_OFFSET = -1.393
LOOKAHEAD = 5


def load_positions_with_yaw(path):
    frames, timestamps, oxs, oys, yaws, images = [], [], [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                frames.append(int(parts[0]))
                timestamps.append(float(parts[1]))
                oxs.append(float(parts[2]))
                oys.append(float(parts[3]))
                yaws.append(float(parts[4]))
                images.append(parts[5] if len(parts) > 5 else "")
            except ValueError:
                continue
    return (np.array(frames), np.array(timestamps),
            np.array(oxs), np.array(oys), np.array(yaws), images)


def setup_projection(map_folder):
    xodr_file = os.path.join(map_folder, "map.xodr")
    with open(xodr_file, "r") as f:
        xodr_data = f.read()
    xodr_params = get_xodr_projection_params(xodr_data)
    xodr_offset = xodr_params["offset"]
    proj_string = xodr_params["geo_reference"].strip()
    if proj_string == "+proj=tmerc":
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"
    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    return transformer, xodr_offset


def utm_to_carla(utm_x, utm_y, transformer, xodr_offset):
    """Full chain: UTM → WGS84 → TM projection → + XODR offset → flip Y → CARLA"""
    lats, lons = odom_xy_to_wgs84_vec(np.array([utm_x]), np.array([utm_y]))
    proj_x, proj_y = transformer.transform(lons[0], lats[0])
    local_x = proj_x + xodr_offset[0]
    local_y = proj_y + xodr_offset[1]
    return local_x, -local_y  # flip Y for CARLA


def compute_utm_x_angle_in_carla(utm_x, utm_y, transformer, xodr_offset):
    """
    Compute what angle UTM +X (East) direction becomes in CARLA space.
    
    This is done numerically by transforming two nearby points through
    the full coordinate chain and measuring the resulting angle.
    
    This single number captures:
      - TM projection grid convergence
      - Y-flip effect on directions
      - Any other axis rotation in the chain
    """
    cx0, cy0 = utm_to_carla(utm_x, utm_y, transformer, xodr_offset)
    cx1, cy1 = utm_to_carla(utm_x + 1.0, utm_y, transformer, xodr_offset)
    
    dx = cx1 - cx0
    dy = cy1 - cy0
    angle = math.atan2(dy, dx)
    
    # Also check UTM +Y for sanity
    cx2, cy2 = utm_to_carla(utm_x, utm_y + 1.0, transformer, xodr_offset)
    angle_y = math.atan2(cy2 - cy0, cx2 - cx0)
    
    # In a right-handed system, +Y would be +90° from +X.
    # With Y-flip, +Y should be -90° from +X (axes become left-handed).
    expected_y = angle - math.pi / 2
    actual_diff = ((math.degrees(angle_y - expected_y) + 180) % 360) - 180
    
    print(f"[INFO] UTM→CARLA axis mapping:")
    print(f"   UTM +X (East)  → CARLA angle: {math.degrees(angle):.4f}°")
    print(f"   UTM +Y (North) → CARLA angle: {math.degrees(angle_y):.4f}°")
    print(f"   Axes orthogonality check: {actual_diff:.4f}° (should be ~0°)")
    
    return angle


def utm_yaw_to_carla_yaw(odom_yaw_rad, utm_x_angle_in_carla):
    """
    Convert odom yaw (UTM frame, radians) to CARLA yaw (degrees).
    
    Formula: carla_yaw = A - odom_yaw + π/2
    
    Where A = angle of UTM +X direction in CARLA space (from compute_utm_x_angle_in_carla).
    
    Derivation:
      - A - θ converts a UTM direction-vector angle to CARLA (accounts for 
        projection rotation + Y-flip reversing rotation direction)
      - The +π/2 corrects for the difference between the quaternion yaw 
        convention (body rotation) and the direction-vector convention 
        after the Y-flip handedness change
    
    Verified empirically:
      odom_yaw = 62.27°, A = -6.76°
      → -6.76 - 62.27 + 90 = 20.97° ≈ expected 20.4° ✓
    """
    carla_yaw_rad = utm_x_angle_in_carla - odom_yaw_rad + math.pi / 2
    return math.degrees(carla_yaw_rad) % 360.0


def calculate_carla_heading(pos_a, pos_b):
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="Map folder name")
    parser.add_argument("--positions", required=True, help="Path to positions.txt")
    args = parser.parse_args()

    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)

    # 1. Load data
    frames, times, ox, oy, odom_yaws, images = load_positions_with_yaw(args.positions)
    print(f"[INFO] Loaded {len(frames)} frames.")
    print(f"   First odom yaw: {odom_yaws[0]:.4f} rad = {math.degrees(odom_yaws[0]):.2f}°")

    # 2. Setup projection
    transformer, xodr_offset = setup_projection(map_folder)

    # 3. Compute UTM→CARLA rotation from the coordinate chain
    print(f"\n[INFO] Computing UTM→CARLA axis rotation...")
    utm_x_angle = compute_utm_x_angle_in_carla(ox[0], oy[0], transformer, xodr_offset)
    
    # Verify consistency at another point
    mid = len(frames) // 2
    utm_x_angle_mid = compute_utm_x_angle_in_carla(ox[mid], oy[mid], transformer, xodr_offset)
    diff = abs(math.degrees(utm_x_angle_mid - utm_x_angle))
    print(f"\n   Rotation at start: {math.degrees(utm_x_angle):.4f}°")
    print(f"   Rotation at mid:   {math.degrees(utm_x_angle_mid):.4f}°")
    print(f"   Difference: {diff:.4f}°")
    if diff < 0.1:
        print(f"   ✅ Consistent — using single rotation for all frames.")
    else:
        print(f"   ⚠️  Varies by {diff:.2f}° — consider per-point computation for long trajectories.")

    # 4. Convert positions to CARLA
    print(f"\n[INFO] Converting to CARLA coordinates...")
    carla_positions = []
    for i in range(len(frames)):
        cx, cy = utm_to_carla(ox[i], oy[i], transformer, xodr_offset)
        carla_positions.append((cx, cy))

    # 5. Compute both yaw types
    # Odom yaw
    odom_carla_yaws = [utm_yaw_to_carla_yaw(y, utm_x_angle) for y in odom_yaws]

    # Kinematic yaw
    last_valid_kin = 0.0
    kin_yaws = []
    for i in range(len(frames)):
        target_idx = min(i + LOOKAHEAD, len(frames) - 1)
        if target_idx == i and i > 0:
            h = calculate_carla_heading(carla_positions[max(0, i - LOOKAHEAD)], carla_positions[i])
        else:
            h = calculate_carla_heading(carla_positions[i], carla_positions[target_idx])
        if h is not None:
            last_valid_kin = h
        kin_yaws.append(last_valid_kin)

    # 6. Build trajectory JSONs
    all_trajs = {
        "trajectory_positions.json": [],
        "trajectory_positions_rear.json": [],
        "trajectory_positions_odom_yaw.json": [],
        "trajectory_positions_rear_odom_yaw.json": [],
    }

    for i in range(len(frames)):
        cx, cy = carla_positions[i]

        for heading, key_c, key_r in [
            (kin_yaws[i], "trajectory_positions.json", "trajectory_positions_rear.json"),
            (odom_carla_yaws[i], "trajectory_positions_odom_yaw.json", "trajectory_positions_rear_odom_yaw.json"),
        ]:
            yaw_rad = math.radians(heading)
            rx = cx - math.cos(yaw_rad) * REAR_OFFSET
            ry = cy - math.sin(yaw_rad) * REAR_OFFSET

            entry = {
                "frame_id": int(frames[i]),
                "timestamp": float(times[i]),
                "transform": {
                    "location": {"x": cx, "y": cy, "z": 0.0},
                    "rotation": {"pitch": 0.0, "yaw": heading, "roll": 0.0}
                }
            }
            all_trajs[key_c].append(entry)
            all_trajs[key_r].append({
                "frame_id": int(frames[i]),
                "timestamp": float(times[i]),
                "transform": {
                    "location": {"x": rx, "y": ry, "z": 0.0},
                    "rotation": {"pitch": 0.0, "yaw": heading, "roll": 0.0}
                }
            })

    # 7. Save
    for fname, data in all_trajs.items():
        path = os.path.join(map_folder, fname)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ {fname} ({len(data)} frames)")

    # 8. Comparison
    print(f"\n📊 Yaw Comparison (first 15 frames):")
    print(f"   {'Frame':>5}  {'Kinematic':>10}  {'Odom':>10}  {'Diff':>8}")
    for i in range(min(15, len(frames))):
        diff = ((odom_carla_yaws[i] - kin_yaws[i] + 180) % 360) - 180
        print(f"   {frames[i]:>5}  {kin_yaws[i]:>10.2f}  {odom_carla_yaws[i]:>10.2f}  {diff:>+8.2f}")

    diffs = np.array([((o - k + 180) % 360) - 180 for o, k in zip(odom_carla_yaws, kin_yaws)])
    print(f"\n📊 Overall Statistics:")
    print(f"   Mean diff:   {np.mean(diffs):+.2f}°")
    print(f"   Std:         {np.std(diffs):.2f}°")
    print(f"   Max |diff|:  {np.max(np.abs(diffs)):.2f}°")

    print(f"\n🔍 First frame verification:")
    print(f"   Odom yaw (UTM):   {math.degrees(odom_yaws[0]):.2f}°")
    print(f"   Odom yaw (CARLA): {odom_carla_yaws[0]:.2f}°")
    print(f"   Kinematic (CARLA): {kin_yaws[0]:.2f}°")
    print(f"   Expected ~20.4°")


if __name__ == "__main__":
    main()