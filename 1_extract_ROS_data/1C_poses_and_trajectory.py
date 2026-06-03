#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1C_poses_and_trajectory.py

Extract rear-axle odometry poses from a ROS bag to CSV files.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag

Writes to (project root):
    data/raw_dataset/<BAG>/
        odometry.csv (full odometry: timestamp, tx, ty, tz, qx, qy, qz, qw, yaw)
        trajectory.csv (compact trajectory: timestamp, x, y, z, yaw)
"""


import os
import math
import csv
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
DEFAULT_BAG_NAME   = "reference_bag.bag"
DEFAULT_ODOM_TOPIC = "/odom"

parser = argparse.ArgumentParser(
    description="Extract full odometry and a simplified trajectory from a ROS bag."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
args = parser.parse_args()

bag_name   = args.bag_name                # e.g. "reference_bag.bag"
bag_stem   = Path(bag_name).stem          # e.g. "reference_bag"
odom_topic = args.odom_topic

bag_path = Path("data") / "raw_ros_data" / bag_name
if not bag_path.is_file():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

dataset_dir = Path("data") / "raw_dataset" / bag_stem
dataset_dir.mkdir(parents=True, exist_ok=True)

odom_path = dataset_dir / "odometry.csv"
traj_path = dataset_dir / "trajectory.csv"

print(f"Bag:              {bag_path}")
print(f"Output directory: {dataset_dir}")
# ----------------------------------------


def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def main():
    if not bag_path.exists():
        print(f"Bag file not found: {bag_path}")
        return

    print(f"Reading: {bag_path}")

    odom_data = []

    with AnyReader([bag_path]) as reader:
        conns = {}
        for c in reader.connections:
            conns.setdefault(c.topic, []).append(c)

        if odom_topic not in conns:
            raise RuntimeError(f"Odom topic not found: {odom_topic}")

        print("Reading odometry...")

        for c in conns[odom_topic]:
            for conn, ts, raw in reader.messages(connections=[c]):
                msg = reader.deserialize(raw, conn.msgtype)

                p = msg.pose.pose.position
                q = msg.pose.pose.orientation

                timestamp = ts * 1e-9
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

                odom_data.append((
                    timestamp,
                    p.x,
                    p.y,
                    p.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                    yaw,
                ))

                if len(odom_data) % 1000 == 0:
                    print(f"Extracted {len(odom_data)} odometry messages...")

    odom_data.sort(key=lambda x: x[0])

    # =========================
    # SAVE FULL ODOMETRY CSV
    # =========================
    print(f"Writing: {odom_path}")

    with open(odom_path, "w", newline="") as f:
        writer = csv.writer(f)

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

        for t, x, y, z, qx, qy, qz, qw, yaw in odom_data:
            writer.writerow([
                f"{t:.9f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{qx:.6f}",
                f"{qy:.6f}",
                f"{qz:.6f}",
                f"{qw:.6f}",
                f"{yaw:.6f}",
            ])

    # =========================
    # SAVE TRAJECTORY CSV
    # =========================
    print(f"Writing: {traj_path}")

    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "x",
            "y",
            "z",
            "yaw",
        ])

        for t, x, y, z, _, _, _, _, yaw in odom_data:
            writer.writerow([
                f"{t:.9f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{yaw:.6f}",
            ])

    print("Done.")
    print(f"Saved {len(odom_data)} odometry samples")
    print(f"Odometry CSV: {odom_path}")
    print(f"Trajectory CSV: {traj_path}")


if __name__ == "__main__":
    main()