#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1C_poses_and_trajectory.py

Extract rear-axle odometry poses from a ROS bag and save both full odometry
and compact trajectory CSV files.

Reads from:
    data/raw_ros_data/<BAG>.bag

Writes to:
    data/raw_dataset/<BAG>/
        odometry.csv
            Full odometry samples:
            timestamp, tx, ty, tz, qx, qy, qz, qw, yaw

        trajectory.csv
            Compact trajectory samples:
            timestamp, x, y, z, yaw

Parameters:
    --bag-name <BAG>.bag
        ROS bag filename to process.
        Default: env BAG_NAME or reference_bag.bag.

    --odom-topic <TOPIC>
        ROS odometry topic to extract poses from.
        Default: /odom.

Usage:
    python 1_process_datasets/1C_poses_and_trajectory.py --bag-name snowy.bag

    python 1_process_datasets/1C_poses_and_trajectory.py \
        --bag-name snowy.bag \
        --odom-topic /odom
"""

import os
import math
import csv
import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_ODOM_TOPIC = "/odom"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract full odometry and compact trajectory CSV files from a ROS bag."
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
        "--odom-topic",
        default=DEFAULT_ODOM_TOPIC,
        help=f"Odometry topic to extract. Default: {DEFAULT_ODOM_TOPIC}",
    )

    return parser.parse_args()


# =============================================================================
# UTILS
# =============================================================================

def quat_to_yaw(x, y, z, w):
    """
    Convert quaternion to yaw angle in radians.
    """
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


# =============================================================================
# ODOMETRY EXTRACTION
# =============================================================================

def read_odometry_from_bag(bag_path, odom_topic):
    """
    Read odometry messages from a ROS bag.

    Returns:
        list of tuples:
            timestamp, tx, ty, tz, qx, qy, qz, qw, yaw
    """
    odom_data = []

    with AnyReader([bag_path]) as reader:
        odom_conns = [
            conn for conn in reader.connections
            if conn.topic == odom_topic
        ]

        if not odom_conns:
            raise RuntimeError(f"Odometry topic not found: {odom_topic}")

        print(f"[INFO] Reading odometry topic: {odom_topic}")

        for conn in odom_conns:
            for connection, ts, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, connection.msgtype)

                p = msg.pose.pose.position
                q = msg.pose.pose.orientation

                timestamp = ts * 1e-9
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

                odom_data.append((
                    float(timestamp),
                    float(p.x),
                    float(p.y),
                    float(p.z),
                    float(q.x),
                    float(q.y),
                    float(q.z),
                    float(q.w),
                    float(yaw),
                ))

                if len(odom_data) % 1000 == 0:
                    print(f"[INFO] Extracted {len(odom_data)} odometry messages...")

    odom_data.sort(key=lambda item: item[0])

    if not odom_data:
        raise RuntimeError(f"No odometry messages found on topic: {odom_topic}")

    print(f"[OK] Loaded {len(odom_data)} odometry samples.")

    return odom_data


# =============================================================================
# CSV WRITING
# =============================================================================

def write_odometry_csv(odom_path, odom_data):
    """
    Write full odometry CSV.
    """
    print(f"[INFO] Writing full odometry CSV: {odom_path}")

    with open(odom_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
            "yaw",
        ])

        for timestamp, tx, ty, tz, qx, qy, qz, qw, yaw in odom_data:
            writer.writerow([
                f"{timestamp:.9f}",
                f"{tx:.6f}",
                f"{ty:.6f}",
                f"{tz:.6f}",
                f"{qx:.6f}",
                f"{qy:.6f}",
                f"{qz:.6f}",
                f"{qw:.6f}",
                f"{yaw:.6f}",
            ])

    print(f"[OK] Wrote odometry CSV: {odom_path}")


def write_trajectory_csv(traj_path, odom_data):
    """
    Write compact trajectory CSV.
    """
    print(f"[INFO] Writing compact trajectory CSV: {traj_path}")

    with open(traj_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "x",
            "y",
            "z",
            "yaw",
        ])

        for timestamp, tx, ty, tz, _, _, _, _, yaw in odom_data:
            writer.writerow([
                f"{timestamp:.9f}",
                f"{tx:.6f}",
                f"{ty:.6f}",
                f"{tz:.6f}",
                f"{yaw:.6f}",
            ])

    print(f"[OK] Wrote trajectory CSV: {traj_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem

    bag_path = PROJECT_ROOT / "data" / "raw_ros_data" / bag_name
    dataset_dir = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem

    odom_path = dataset_dir / "odometry.csv"
    traj_path = dataset_dir / "trajectory.csv"

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ODOMETRY AND TRAJECTORY EXTRACTION")
    print("=" * 70)
    print(f"[INFO] Project root:      {PROJECT_ROOT}")
    print(f"[INFO] Bag:               {bag_name}")
    print(f"[INFO] Bag path:          {bag_path}")
    print(f"[INFO] Output directory:  {dataset_dir}")
    print(f"[INFO] Odometry topic:    {args.odom_topic}")
    print("=" * 70)

    odom_data = read_odometry_from_bag(
        bag_path=bag_path,
        odom_topic=args.odom_topic,
    )

    write_odometry_csv(
        odom_path=odom_path,
        odom_data=odom_data,
    )

    write_trajectory_csv(
        traj_path=traj_path,
        odom_data=odom_data,
    )

    print("=" * 70)
    print("[OK] Odometry and trajectory extraction completed.")
    print(f"[OK] Samples:        {len(odom_data)}")
    print(f"[OK] Odometry CSV:   {odom_path}")
    print(f"[OK] Trajectory CSV: {traj_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()