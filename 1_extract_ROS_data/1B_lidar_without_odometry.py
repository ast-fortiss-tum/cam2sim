#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1B_lidar_without_odometry.py

Extract LiDAR point clouds from a ROS bag to binary files.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag

Writes to (project root):
    data/raw_dataset/<BAG>/
        point_clouds/point_cloud_<N:06d>.bin (Velodyne XYZI scans)
"""

import os
import argparse
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
DEFAULT_BAG_NAME    = "reference_bag.bag"
DEFAULT_LIDAR_TOPIC = "/velodyne_points"

parser = argparse.ArgumentParser(
    description="Extract LiDAR point clouds from a ROS bag (no odometry sync)."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
args = parser.parse_args()

bag_name    = args.bag_name                # e.g. "reference_bag.bag"
bag_stem    = Path(bag_name).stem          # e.g. "reference_bag"
lidar_topic = args.lidar_topic

bag_path = Path("data") / "raw_ros_data" / bag_name
if not bag_path.is_file():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

dataset_dir = Path("data") / "raw_dataset" / bag_stem
pc_dir      = dataset_dir / "point_clouds"
pc_dir.mkdir(parents=True, exist_ok=True)

print(f"Bag:              {bag_path}")
print(f"Output directory: {dataset_dir}")
# ---------------------------------------


def decode_pointcloud2(msg):

    field_names = [f.name for f in msg.fields]

    raw = msg.data
    buffer = np.frombuffer(raw, dtype=np.uint8)

    point_step = msg.point_step
    num_points = msg.width * msg.height

    points = []

    for i in range(num_points):
        offset = i * point_step

        x = np.frombuffer(buffer[offset:offset+4], dtype=np.float32)[0]
        y = np.frombuffer(buffer[offset+4:offset+8], dtype=np.float32)[0]
        z = np.frombuffer(buffer[offset+8:offset+12], dtype=np.float32)[0]

        intensity = 0.0
        if "intensity" in field_names:
            intensity = np.frombuffer(buffer[offset+12:offset+16], dtype=np.float32)[0]

        points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)


def main():

    if not bag_path.exists():
        print("Bag not found")
        return

    print(f"Reading: {bag_path}")

    frame_idx = 0

    with AnyReader([bag_path]) as reader:

        lidar_conns = [c for c in reader.connections if c.topic == lidar_topic]

        if not lidar_conns:
            raise RuntimeError("LiDAR not found")

        print("Extracting LiDAR...")

        for conn, ts, raw in reader.messages(connections=lidar_conns):

            msg = reader.deserialize(raw, conn.msgtype)

            try:
                points = decode_pointcloud2(msg)

                if len(points) == 0:
                    continue

                save_path = os.path.join(pc_dir, f"point_cloud_{frame_idx:06d}.bin")
                points.tofile(save_path)

                frame_idx += 1

                if frame_idx % 50 == 0:
                    print(f"[LiDAR] saved {frame_idx} scans...")

            except Exception as e:
                print(f"Skip scan {ts}: {e}")

    print("Done.")
    print(f"Saved {frame_idx} point clouds")


if __name__ == "__main__":
    main()