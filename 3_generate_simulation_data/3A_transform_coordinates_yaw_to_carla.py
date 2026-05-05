#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3A_transform_coordinates_yaw_to_carla.py

Generates trajectory JSON files using real odom yaw from images_positions.txt.

This version:
  - Uses hardcoded input paths
  - Reads the new images_positions.txt format:
      FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z,
      Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile
  - Imports utils from the same folder where this script is located:
      3_generate_simulation_data/utils/
  - Reads the map from:
      /home/cam2sim/Documents/cam2sim/data/maps/reference_bag
  - Reads trajectory positions from:
      /home/cam2sim/Documents/cam2sim/data/extracted_ros_data/reference_bag/images_positions.txt
  - Saves trajectory JSON outputs to:
      /home/cam2sim/Documents/cam2sim/data/data_for_carla
  - Updates:
      /home/cam2sim/Documents/cam2sim/data/maps/reference_bag/vehicle_data.json

vehicle_data.json format:
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

Outputs:
  trajectory_positions.json
  trajectory_positions_rear.json
  trajectory_positions_odom_yaw.json
  trajectory_positions_rear_odom_yaw.json
"""

import os
import sys
import json
import math
import numpy as np
from pyproj import Transformer


# ==========================================
# PATH SETUP
# ==========================================

# Folder where this script is located:
# /home/cam2sim/Documents/cam2sim/3_generate_simulation_data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root:
# /home/cam2sim/Documents/cam2sim
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Local utils folder:
# /home/cam2sim/Documents/cam2sim/3_generate_simulation_data/utils
LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected local utils folder not found: {LOCAL_UTILS_DIR}"
    )

# IMPORTANT:
# Put SCRIPT_DIR first so Python imports:
#   3_generate_simulation_data/utils/...
# instead of:
#   /home/cam2sim/Documents/cam2sim/utils/...
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)


# ==========================================
# HARDCODED INPUTS / OUTPUTS
# ==========================================

MAP_NAME = "reference_bag"

POSITIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "extracted_ros_data",
    MAP_NAME,
    "images_positions.txt",
)

MAP_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "maps",
    MAP_NAME,
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "data_for_carla",
)

VEHICLE_DATA_PATH = os.path.join(
    MAP_FOLDER,
    "vehicle_data.json",
)


# ==========================================
# LOCAL PROJECT IMPORTS
# ==========================================

from utils.coordinates import odom_xy_to_wgs84_vec
from utils.carla_simulator import get_xodr_projection_params


# ==========================================
# CONFIGURATION
# ==========================================

MAP_DIST = 200

# Distance from vehicle center to rear axle.
#
# Keeping your original value/sign.
# Note: with the formula below:
#   rear_x = cx - cos(yaw) * REAR_OFFSET
# a negative REAR_OFFSET moves the point forward.
#
# If you want the rear axle behind the center, use:
#   REAR_OFFSET = 1.393
REAR_OFFSET = -1.393

LOOKAHEAD = 5

# Use fixed CARLA z.
# Odom_Z is loaded from the file but not used for CARLA spawn height.
OUTPUT_Z = 0.0


# ==========================================
# DATA LOADING
# ==========================================

def load_positions_with_yaw(path):
    """
    Load trajectory positions from images_positions.txt.

    Expected format:
      # FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z,
      # Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile

    Required columns:
      0: FrameID
      1: Timestamp_Sec
      2: Odom_X
      3: Odom_Y
      4: Odom_Z
      9: Odom_Yaw

    Optional:
      10: ImageFile
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")

    frames = []
    timestamps = []
    oxs = []
    oys = []
    ozs = []
    yaws = []
    images = []

    skipped = 0

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            # Need at least up to Odom_Yaw, index 9
            if len(parts) < 10:
                skipped += 1
                continue

            try:
                frame_id = int(parts[0])
                timestamp = float(parts[1])
                odom_x = float(parts[2])
                odom_y = float(parts[3])
                odom_z = float(parts[4])
                odom_yaw = float(parts[9])
                image_file = parts[10] if len(parts) > 10 else ""

                frames.append(frame_id)
                timestamps.append(timestamp)
                oxs.append(odom_x)
                oys.append(odom_y)
                ozs.append(odom_z)
                yaws.append(odom_yaw)
                images.append(image_file)

            except ValueError:
                skipped += 1
                continue

    if len(frames) == 0:
        raise RuntimeError(
            f"No valid trajectory rows loaded from: {path}\n"
            "Expected format:\n"
            "FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, "
            "Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile"
        )

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} malformed rows.")

    return (
        np.array(frames),
        np.array(timestamps),
        np.array(oxs),
        np.array(oys),
        np.array(ozs),
        np.array(yaws),
        images,
    )


# ==========================================
# PROJECTION SETUP
# ==========================================

def setup_projection(map_folder):
    """
    Read map.xodr and construct the projection used by the CARLA/OpenDRIVE map.
    """
    xodr_file = os.path.join(map_folder, "map.xodr")

    if not os.path.exists(xodr_file):
        raise FileNotFoundError(f"XODR file not found: {xodr_file}")

    with open(xodr_file, "r") as f:
        xodr_data = f.read()

    xodr_params = get_xodr_projection_params(xodr_data)

    xodr_offset = xodr_params["offset"]
    proj_string = xodr_params["geo_reference"].strip()

    # Fallback for incomplete XODR projection string
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

    transformer = Transformer.from_crs(
        "EPSG:4326",
        proj_string,
        always_xy=True,
    )

    return transformer, xodr_offset


# ==========================================
# ANGLE HELPERS
# ==========================================

def normalize_angle_deg(angle):
    """
    Normalize angle to [-180, 180].
    """
    return ((angle + 180.0) % 360.0) - 180.0


def calculate_carla_heading(pos_a, pos_b):
    """
    Compute kinematic heading from two CARLA positions.
    """
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return None

    return math.degrees(math.atan2(dy, dx)) % 360.0


# ==========================================
# COORDINATE CONVERSION
# ==========================================

def utm_to_carla(utm_x, utm_y, transformer, xodr_offset):
    """
    Full coordinate chain:

      Odom/UTM x,y
        -> WGS84 lat/lon
        -> XODR projection
        -> XODR local coordinates
        -> CARLA coordinates

    CARLA conversion:
      carla_x = local_x
      carla_y = -local_y
    """
    lats, lons = odom_xy_to_wgs84_vec(
        np.array([utm_x]),
        np.array([utm_y]),
    )

    proj_x, proj_y = transformer.transform(lons[0], lats[0])

    local_x = proj_x + xodr_offset[0]
    local_y = proj_y + xodr_offset[1]

    carla_x = local_x
    carla_y = -local_y

    return carla_x, carla_y


def compute_utm_x_angle_in_carla(utm_x, utm_y, transformer, xodr_offset):
    """
    Compute what angle UTM +X, East, becomes in CARLA space.

    This is done numerically by transforming:
      point A = utm_x, utm_y
      point B = utm_x + 1.0, utm_y

    Then measuring the resulting angle in CARLA coordinates.
    """
    cx0, cy0 = utm_to_carla(utm_x, utm_y, transformer, xodr_offset)
    cx1, cy1 = utm_to_carla(utm_x + 1.0, utm_y, transformer, xodr_offset)

    dx = cx1 - cx0
    dy = cy1 - cy0

    angle = math.atan2(dy, dx)

    # Also check UTM +Y for sanity
    cx2, cy2 = utm_to_carla(utm_x, utm_y + 1.0, transformer, xodr_offset)
    angle_y = math.atan2(cy2 - cy0, cx2 - cx0)

    expected_y = angle - math.pi / 2
    actual_diff = normalize_angle_deg(
        math.degrees(angle_y - expected_y)
    )

    print("[INFO] UTM -> CARLA axis mapping:")
    print(f"   UTM +X, East  -> CARLA angle: {math.degrees(angle):.4f}°")
    print(f"   UTM +Y, North -> CARLA angle: {math.degrees(angle_y):.4f}°")
    print(f"   Axes orthogonality check: {actual_diff:.4f}° should be near 0°")

    return angle


def utm_yaw_to_carla_yaw(odom_yaw_rad, utm_x_angle_in_carla):
    """
    Convert odom yaw in UTM frame, radians, to CARLA yaw in degrees.

    Formula:
      carla_yaw = A - odom_yaw + pi/2

    where:
      A = angle of UTM +X direction in CARLA space.

    The +pi/2 is kept from your empirically verified version.
    """
    carla_yaw_rad = utm_x_angle_in_carla - odom_yaw_rad + math.pi / 2
    return math.degrees(carla_yaw_rad) % 360.0


# ==========================================
# JSON ENTRY CREATION
# ==========================================

def make_transform_entry(frame_id, timestamp, x, y, z, yaw):
    """
    Create one CARLA-style transform JSON entry.
    """
    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "transform": {
            "location": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            },
            "rotation": {
                "pitch": 0.0,
                "yaw": float(yaw),
                "roll": 0.0,
            },
        },
    }


# ==========================================
# VEHICLE DATA UPDATE
# ==========================================

def update_vehicle_data_hero_car(
    vehicle_data_path,
    hero_x,
    hero_y,
    hero_z,
    hero_heading,
):
    """
    Update vehicle_data.json using the correct format:

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

    Existing spawn_positions are preserved.
    Old keys "start" and "parking" are removed.
    """
    if os.path.exists(vehicle_data_path):
        with open(vehicle_data_path, "r") as f:
            vehicle_data = json.load(f)
    else:
        vehicle_data = {}

    # Preserve existing parked vehicles if present.
    spawn_positions = vehicle_data.get("spawn_positions", [])

    # Preserve offset if present, otherwise use zeros.
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

    # Remove old format keys.
    vehicle_data.pop("start", None)
    vehicle_data.pop("parking", None)

    # Write correct structure.
    vehicle_data["offset"] = offset
    vehicle_data["dist"] = MAP_DIST
    vehicle_data["hero_car"] = {
        "position": [
            float(hero_x),
            float(hero_y),
            float(hero_z),
        ],
        "heading": float(hero_heading),
    }
    vehicle_data["spawn_positions"] = spawn_positions

    with open(vehicle_data_path, "w") as f:
        json.dump(vehicle_data, f, indent=2)

    print("\n✅ Updated vehicle_data.json hero_car")
    print(f"   File:    {vehicle_data_path}")
    print(f"   x:       {hero_x:.3f}")
    print(f"   y:       {hero_y:.3f}")
    print(f"   z:       {hero_z:.3f}")
    print(f"   heading: {hero_heading:.3f}")
    print(f"   Preserved spawn_positions: {len(spawn_positions)}")


# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    print("=" * 80)
    print("GENERATE TRAJECTORY WITH ODOM YAW")
    print("=" * 80)
    print(f"[INFO] Script folder:      {SCRIPT_DIR}")
    print(f"[INFO] Local utils:        {LOCAL_UTILS_DIR}")
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] Map name:           {MAP_NAME}")
    print(f"[INFO] Map folder:         {MAP_FOLDER}")
    print(f"[INFO] Positions file:     {POSITIONS_FILE}")
    print(f"[INFO] Output folder:      {OUTPUT_FOLDER}")
    print(f"[INFO] Vehicle data path:  {VEHICLE_DATA_PATH}")
    print(f"[INFO] Output z:           {OUTPUT_Z}")
    print(f"[INFO] Rear offset:        {REAR_OFFSET}")
    print("=" * 80)

    if not os.path.exists(MAP_FOLDER):
        raise FileNotFoundError(f"Map folder not found: {MAP_FOLDER}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load trajectory data
    frames, times, ox, oy, oz, odom_yaws, images = load_positions_with_yaw(
        POSITIONS_FILE
    )

    print(f"\n[INFO] Loaded {len(frames)} frames.")
    print(f"   First frame:     {frames[0]}")
    print(f"   First timestamp: {times[0]}")
    print(f"   First odom x:    {ox[0]:.4f}")
    print(f"   First odom y:    {oy[0]:.4f}")
    print(f"   First odom z:    {oz[0]:.4f}")
    print(
        f"   First odom yaw:  {odom_yaws[0]:.4f} rad "
        f"= {math.degrees(odom_yaws[0]):.2f}°"
    )

    if images[0]:
        print(f"   First image:     {images[0]}")

    # 2. Setup projection from XODR
    print("\n[INFO] Loading XODR projection...")
    transformer, xodr_offset = setup_projection(MAP_FOLDER)
    print(f"   XODR offset: {xodr_offset}")

    # 3. Compute UTM to CARLA axis rotation
    print("\n[INFO] Computing UTM -> CARLA axis rotation at start...")
    utm_x_angle = compute_utm_x_angle_in_carla(
        ox[0],
        oy[0],
        transformer,
        xodr_offset,
    )

    # Verify consistency at middle point
    mid = len(frames) // 2

    print("\n[INFO] Computing UTM -> CARLA axis rotation at middle...")
    utm_x_angle_mid = compute_utm_x_angle_in_carla(
        ox[mid],
        oy[mid],
        transformer,
        xodr_offset,
    )

    angle_diff = normalize_angle_deg(
        math.degrees(utm_x_angle_mid - utm_x_angle)
    )

    print("\n[INFO] Axis rotation consistency:")
    print(f"   Rotation at start: {math.degrees(utm_x_angle):.4f}°")
    print(f"   Rotation at mid:   {math.degrees(utm_x_angle_mid):.4f}°")
    print(f"   Difference:        {angle_diff:.4f}°")

    if abs(angle_diff) < 0.1:
        print("   ✅ Consistent — using single rotation for all frames.")
    else:
        print(
            f"   ⚠️ Varies by {angle_diff:.2f}°. "
            "For very long trajectories, consider per-frame yaw conversion."
        )

    # 4. Convert all positions to CARLA coordinates
    print("\n[INFO] Converting trajectory positions to CARLA coordinates...")

    carla_positions = []

    for i in range(len(frames)):
        cx, cy = utm_to_carla(
            ox[i],
            oy[i],
            transformer,
            xodr_offset,
        )
        carla_positions.append((cx, cy))

    print("   Done.")

    # 5. Compute odom yaw converted to CARLA yaw
    print("\n[INFO] Converting odom yaw to CARLA yaw...")

    odom_carla_yaws = [
        utm_yaw_to_carla_yaw(yaw, utm_x_angle)
        for yaw in odom_yaws
    ]

    # 6. Compute kinematic yaw from trajectory movement
    print("[INFO] Computing kinematic yaw from trajectory movement...")

    last_valid_kinematic_yaw = 0.0
    kinematic_yaws = []

    for i in range(len(frames)):
        target_idx = min(i + LOOKAHEAD, len(frames) - 1)

        if target_idx == i and i > 0:
            previous_idx = max(0, i - LOOKAHEAD)
            heading = calculate_carla_heading(
                carla_positions[previous_idx],
                carla_positions[i],
            )
        else:
            heading = calculate_carla_heading(
                carla_positions[i],
                carla_positions[target_idx],
            )

        if heading is not None:
            last_valid_kinematic_yaw = heading

        kinematic_yaws.append(last_valid_kinematic_yaw)

    # 7. Build output trajectories
    print("\n[INFO] Building output JSON files...")

    all_trajectories = {
        "trajectory_positions.json": [],
        "trajectory_positions_rear.json": [],
        "trajectory_positions_odom_yaw.json": [],
        "trajectory_positions_rear_odom_yaw.json": [],
    }

    for i in range(len(frames)):
        cx, cy = carla_positions[i]

        trajectory_variants = [
            (
                kinematic_yaws[i],
                "trajectory_positions.json",
                "trajectory_positions_rear.json",
            ),
            (
                odom_carla_yaws[i],
                "trajectory_positions_odom_yaw.json",
                "trajectory_positions_rear_odom_yaw.json",
            ),
        ]

        for yaw, center_key, rear_key in trajectory_variants:
            yaw_rad = math.radians(yaw)

            rear_x = cx - math.cos(yaw_rad) * REAR_OFFSET
            rear_y = cy - math.sin(yaw_rad) * REAR_OFFSET

            center_entry = make_transform_entry(
                frame_id=frames[i],
                timestamp=times[i],
                x=cx,
                y=cy,
                z=OUTPUT_Z,
                yaw=yaw,
            )

            rear_entry = make_transform_entry(
                frame_id=frames[i],
                timestamp=times[i],
                x=rear_x,
                y=rear_y,
                z=OUTPUT_Z,
                yaw=yaw,
            )

            all_trajectories[center_key].append(center_entry)
            all_trajectories[rear_key].append(rear_entry)

    # 8. Save trajectory outputs
    print("\n[INFO] Saving output trajectory files...")

    for filename, data in all_trajectories.items():
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"   ✅ {output_path} ({len(data)} frames)")

    # 9. First/last position verification
    first_center = all_trajectories["trajectory_positions_odom_yaw.json"][0]
    last_center = all_trajectories["trajectory_positions_odom_yaw.json"][-1]

    first_location = first_center["transform"]["location"]
    first_rotation = first_center["transform"]["rotation"]

    last_location = last_center["transform"]["location"]
    last_rotation = last_center["transform"]["rotation"]

    # 10. Update vehicle_data.json hero_car with first computed CARLA position
    update_vehicle_data_hero_car(
        vehicle_data_path=VEHICLE_DATA_PATH,
        hero_x=first_location["x"],
        hero_y=first_location["y"],
        hero_z=first_location["z"],
        hero_heading=first_rotation["yaw"],
    )

    # 11. Print comparison
    print("\n📊 Yaw comparison, first 15 frames:")
    print(f"   {'Frame':>5}  {'Kinematic':>10}  {'Odom':>10}  {'Diff':>8}")

    for i in range(min(15, len(frames))):
        diff = normalize_angle_deg(odom_carla_yaws[i] - kinematic_yaws[i])

        print(
            f"   {frames[i]:>5}  "
            f"{kinematic_yaws[i]:>10.2f}  "
            f"{odom_carla_yaws[i]:>10.2f}  "
            f"{diff:>+8.2f}"
        )

    diffs = np.array([
        normalize_angle_deg(odom_yaw - kin_yaw)
        for odom_yaw, kin_yaw in zip(odom_carla_yaws, kinematic_yaws)
    ])

    print("\n📊 Overall yaw statistics:")
    print(f"   Mean diff:   {np.mean(diffs):+.2f}°")
    print(f"   Std:         {np.std(diffs):.2f}°")
    print(f"   Max |diff|:  {np.max(np.abs(diffs)):.2f}°")

    print("\n🔍 First frame verification:")
    print(f"   Input odom yaw:    {math.degrees(odom_yaws[0]):.2f}°")
    print(f"   CARLA odom yaw:    {first_rotation['yaw']:.2f}°")
    print(f"   Kinematic yaw:     {kinematic_yaws[0]:.2f}")
    print(
        f"   CARLA position:    "
        f"({first_location['x']:.2f}, "
        f"{first_location['y']:.2f}, "
        f"{first_location['z']:.2f})"
    )

    print("\n🔍 Last frame verification:")
    print(f"   CARLA odom yaw:    {last_rotation['yaw']:.2f}°")
    print(
        f"   CARLA position:    "
        f"({last_location['x']:.2f}, "
        f"{last_location['y']:.2f}, "
        f"{last_location['z']:.2f})"
    )

    print("\n✅ Done.")
    print(f"   Trajectory outputs saved in: {OUTPUT_FOLDER}")
    print(f"   vehicle_data.json updated:   {VEHICLE_DATA_PATH}")


if __name__ == "__main__":
    main()