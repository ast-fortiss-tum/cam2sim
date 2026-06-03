#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1B_lidar_with_odometry.py

Extract synchronized LiDAR point clouds and odometry poses from a ROS bag.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag

Writes to (project root):
    data/raw_dataset/<BAG>/
        point_clouds/point_cloud_<N:06d>.bin (Velodyne XYZI scans)
        lidar_positions.txt (per-frame UTM pose interpolated from /odom)
"""

import os
import math
import argparse
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
DEFAULT_BAG_NAME    = "reference_bag.bag"
DEFAULT_LIDAR_TOPIC = "/velodyne_points"
DEFAULT_ODOM_TOPIC  = "/odom"

parser = argparse.ArgumentParser(
    description="Extract synchronized LiDAR point clouds and odometry poses from a ROS bag."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
parser.add_argument("--odom-topic",  default=DEFAULT_ODOM_TOPIC)
args = parser.parse_args()

bag_name    = args.bag_name                # e.g. "reference_bag.bag"
bag_stem    = Path(bag_name).stem          # e.g. "reference_bag"
lidar_topic = args.lidar_topic
odom_topic  = args.odom_topic

bag_path = Path("data") / "raw_ros_data" / bag_name
if not bag_path.is_file():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

dataset_dir     = Path("data") / "raw_dataset" / bag_stem
pc_dir          = dataset_dir / "point_clouds"
lidar_sync_path = dataset_dir / "lidar_positions.txt"
pc_dir.mkdir(parents=True, exist_ok=True)

print(f"Bag:              {bag_path}")
print(f"Output directory: {dataset_dir}")
# ---------------------------------------


def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


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

    odoms = []
    lidar_data = []

    with AnyReader([bag_path]) as reader:

        conns = {}
        for c in reader.connections:
            conns.setdefault(c.topic, []).append(c)

        # =========================
        # ODOM (FULL TIMESTAMP LIST)
        # =========================
        if odom_topic not in conns:
            raise RuntimeError("Odom not found")

        print("Reading odometry...")

        for c in conns[odom_topic]:
            for conn, ts, raw in reader.messages(connections=[c]):

                msg = reader.deserialize(raw, conn.msgtype)

                p = msg.pose.pose.position
                q = msg.pose.pose.orientation

                t = ts * 1e-9
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

                odoms.append((t, p.x, p.y, yaw))

        odoms.sort(key=lambda x: x[0])

        odom_t = np.array([o[0] for o in odoms])
        odom_x = np.array([o[1] for o in odoms])
        odom_y = np.array([o[2] for o in odoms])
        odom_yaw = np.array([o[3] for o in odoms])

        # =========================
        # LIDAR
        # =========================
        lidar_conns = [c for c in reader.connections if c.topic == lidar_topic]

        if not lidar_conns:
            raise RuntimeError("LiDAR not found")

        print("Extracting LiDAR...")

        frame_idx = 0

        for conn, ts, raw in reader.messages(connections=lidar_conns):

            msg = reader.deserialize(raw, conn.msgtype)

            try:
                points = decode_pointcloud2(msg)

                if len(points) == 0:
                    continue

                # save pointcloud
                save_path = os.path.join(pc_dir, f"point_cloud_{frame_idx:06d}.bin")
                points.tofile(save_path)

                t = ts * 1e-9

                lidar_data.append({
                    "id": frame_idx,
                    "ts": t
                })

                frame_idx += 1

                if frame_idx % 50 == 0:
                    print(f"[LiDAR] saved {frame_idx} scans...")

            except Exception as e:
                print(f"Skip scan {ts}: {e}")

    # =========================
    # SYNC (CAMERA-STYLE LOGIC)
    # =========================
    print("Computing LiDAR odometry sync...")

    lidar_ts = np.array([l["ts"] for l in lidar_data])
    lidar_id = np.array([l["id"] for l in lidar_data])

    sync_x = np.interp(lidar_ts, odom_t, odom_x)
    sync_y = np.interp(lidar_ts, odom_t, odom_y)
    sync_yaw = np.interp(lidar_ts, odom_t, odom_yaw)

    print(f"Writing: {lidar_sync_path}")

    with open(lidar_sync_path, "w") as f:
        f.write("# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, PointCloudFile\n")

        for i in range(len(lidar_ts)):
            f.write(
                f"{lidar_id[i]}, {lidar_ts[i]:.6f}, "
                f"{sync_x[i]:.4f}, {sync_y[i]:.4f}, {sync_yaw[i]:.4f}, "
                f"point_cloud_{lidar_id[i]:06d}.bin\n"
            )

    print(f"Done.")
    print(f"Saved {len(lidar_ts)} LiDAR scans")


if __name__ == "__main__":
    main()