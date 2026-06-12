#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1B_extract_lidar.py

Extract LiDAR point clouds from a ROS bag, optionally synchronized with
odometry poses.

Reads from:
    data/raw_ros_data/<BAG>.bag

Writes to:
    data/raw_dataset/<BAG>/
        point_clouds/point_cloud_<N:06d>.bin
        lidar_positions.txt   # only when odometry is enabled

Parameters:
    --bag-name <BAG>.bag
        ROS bag filename to process.
        The file is read from:
            data/raw_ros_data/<BAG>.bag
        Default: env BAG_NAME or reference_bag.bag.

    --lidar-topic <TOPIC>
        ROS PointCloud2 topic to extract LiDAR scans from.
        Default: /velodyne_points.

    --odom-topic <TOPIC>
        ROS odometry topic used to interpolate the vehicle pose at each
        LiDAR scan timestamp.
        Default: /odom.

    --no-odom
        Extract only LiDAR point clouds. Do not read odometry and do not write
        lidar_positions.txt.

Usage:
    python 1_process_datasets/1B_extract_lidar.py --bag-name snowy.bag

    python 1_process_datasets/1B_extract_lidar.py \
        --bag-name snowy.bag \
        --no-odom

    python 1_process_datasets/1B_extract_lidar.py \
        --bag-name snowy.bag \
        --lidar-topic /velodyne_points \
        --odom-topic /odom
"""

import os
import math
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_LIDAR_TOPIC = "/velodyne_points"
DEFAULT_ODOM_TOPIC = "/odom"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract LiDAR point clouds from a ROS bag, optionally "
            "synchronized with odometry."
        )
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
        "--lidar-topic",
        default=DEFAULT_LIDAR_TOPIC,
        help=f"LiDAR topic to extract. Default: {DEFAULT_LIDAR_TOPIC}",
    )

    parser.add_argument(
        "--odom-topic",
        default=DEFAULT_ODOM_TOPIC,
        help=f"Odometry topic to synchronize. Default: {DEFAULT_ODOM_TOPIC}",
    )

    parser.add_argument(
        "--no-odom",
        action="store_true",
        help="Extract point clouds only. Do not read odometry or write lidar_positions.txt.",
    )

    return parser.parse_args()


# =============================================================================
# POINT CLOUD DECODING
# =============================================================================

def decode_pointcloud2(msg):
    """
    Decode ROS PointCloud2 message to XYZI float32 array.

    This assumes the classic Velodyne layout:
        x         float32 at offset 0
        y         float32 at offset 4
        z         float32 at offset 8
        intensity float32 at offset 12, if present

    Output shape:
        (N, 4) with columns x, y, z, intensity
    """
    field_names = [field.name for field in msg.fields]

    raw = msg.data
    buffer = np.frombuffer(raw, dtype=np.uint8)

    point_step = msg.point_step
    num_points = msg.width * msg.height

    points = []

    for index in range(num_points):
        offset = index * point_step

        x = np.frombuffer(buffer[offset:offset + 4], dtype=np.float32)[0]
        y = np.frombuffer(buffer[offset + 4:offset + 8], dtype=np.float32)[0]
        z = np.frombuffer(buffer[offset + 8:offset + 12], dtype=np.float32)[0]

        intensity = 0.0
        if "intensity" in field_names:
            intensity = np.frombuffer(
                buffer[offset + 12:offset + 16],
                dtype=np.float32,
            )[0]

        points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)


# =============================================================================
# ODOMETRY
# =============================================================================

def quat_to_yaw(x, y, z, w):
    """
    Convert quaternion to yaw angle in radians.
    """
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def read_odometry(reader, odom_topic):
    """
    Read odometry messages from the selected topic.

    Returns:
        list of tuples:
            timestamp_sec, x, y, yaw
    """
    odom_conns = [
        conn for conn in reader.connections
        if conn.topic == odom_topic
    ]

    if not odom_conns:
        raise RuntimeError(f"Odometry topic not found: {odom_topic}")

    print(f"[INFO] Reading odometry topic: {odom_topic}")

    odoms = []

    for conn in odom_conns:
        for connection, ts, raw in reader.messages(connections=[conn]):
            msg = reader.deserialize(raw, connection.msgtype)

            p = msg.pose.pose.position
            q = msg.pose.pose.orientation

            timestamp_sec = ts * 1e-9
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

            odoms.append((
                timestamp_sec,
                float(p.x),
                float(p.y),
                float(yaw),
            ))

    odoms.sort(key=lambda item: item[0])

    if not odoms:
        raise RuntimeError(f"No odometry messages found on topic: {odom_topic}")

    print(f"[INFO] Loaded {len(odoms)} odometry messages.")

    return odoms


def write_lidar_positions(lidar_sync_path, lidar_data, odoms):
    """
    Interpolate odometry poses at LiDAR timestamps and write lidar_positions.txt.
    """
    print(f"[INFO] Writing synchronized LiDAR poses to: {lidar_sync_path}")

    odom_arr = np.array(odoms)

    odom_t = odom_arr[:, 0]
    odom_x = odom_arr[:, 1]
    odom_y = odom_arr[:, 2]
    odom_yaw = odom_arr[:, 3]

    lidar_ts = np.array([item["ts"] for item in lidar_data])
    lidar_id = np.array([item["id"] for item in lidar_data])

    sync_x = np.interp(lidar_ts, odom_t, odom_x)
    sync_y = np.interp(lidar_ts, odom_t, odom_y)
    sync_yaw = np.interp(lidar_ts, odom_t, odom_yaw)

    with open(lidar_sync_path, "w") as file:
        file.write(
            "# FrameID, Timestamp_Sec, Odom_X, Odom_Y, "
            "Odom_Yaw, PointCloudFile\n"
        )

        for index in range(len(lidar_ts)):
            file.write(
                f"{lidar_id[index]}, {lidar_ts[index]:.6f}, "
                f"{sync_x[index]:.4f}, {sync_y[index]:.4f}, "
                f"{sync_yaw[index]:.4f}, "
                f"point_cloud_{lidar_id[index]:06d}.bin\n"
            )

    print(f"[OK] Wrote {len(lidar_data)} synchronized LiDAR poses.")


# =============================================================================
# LIDAR EXTRACTION
# =============================================================================

def extract_lidar_scans(reader, lidar_topic, pc_dir):
    """
    Extract LiDAR scans from the selected topic.

    Returns:
        list of per-scan timestamp metadata.
    """
    lidar_conns = [
        conn for conn in reader.connections
        if conn.topic == lidar_topic
    ]

    if not lidar_conns:
        raise RuntimeError(f"LiDAR topic not found: {lidar_topic}")

    print(f"[INFO] Reading LiDAR topic: {lidar_topic}")
    print(f"[INFO] Output point clouds: {pc_dir}")

    lidar_data = []
    frame_idx = 0

    for conn in lidar_conns:
        for connection, ts, raw in reader.messages(connections=[conn]):
            msg = reader.deserialize(raw, connection.msgtype)
            timestamp_sec = ts * 1e-9

            filename = f"point_cloud_{frame_idx:06d}.bin"
            save_path = pc_dir / filename

            try:
                points = decode_pointcloud2(msg)

                if len(points) == 0:
                    continue

                points.tofile(str(save_path))

                lidar_data.append({
                    "id": frame_idx,
                    "ts": timestamp_sec,
                    "filename": filename,
                })

                frame_idx += 1

                if frame_idx % 50 == 0:
                    print(f"[INFO] Saved {frame_idx} LiDAR scans...", end="\r")

            except Exception as exc:
                print(f"[WARN] Skipping scan at timestamp {timestamp_sec:.6f}: {exc}")

    lidar_data.sort(key=lambda item: item["ts"])

    print(f"\n[OK] Extracted {len(lidar_data)} LiDAR scans.")

    return lidar_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem

    bag_path = PROJECT_ROOT / "data" / "raw_ros_data" / bag_name
    dataset_dir = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem
    pc_dir = dataset_dir / "point_clouds"
    lidar_sync_path = dataset_dir / "lidar_positions.txt"

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIDAR EXTRACTION")
    print("=" * 70)
    print(f"[INFO] Project root:      {PROJECT_ROOT}")
    print(f"[INFO] Bag:               {bag_name}")
    print(f"[INFO] Bag path:          {bag_path}")
    print(f"[INFO] Output directory:  {dataset_dir}")
    print(f"[INFO] LiDAR topic:       {args.lidar_topic}")
    print(f"[INFO] Odometry enabled:  {not args.no_odom}")
    if not args.no_odom:
        print(f"[INFO] Odometry topic:    {args.odom_topic}")
    print("=" * 70)

    with AnyReader([bag_path]) as reader:
        odoms = None

        if not args.no_odom:
            odoms = read_odometry(reader, args.odom_topic)

        lidar_data = extract_lidar_scans(
            reader=reader,
            lidar_topic=args.lidar_topic,
            pc_dir=pc_dir,
        )

    if not lidar_data:
        raise RuntimeError("No LiDAR scans were extracted.")

    if not args.no_odom:
        write_lidar_positions(
            lidar_sync_path=lidar_sync_path,
            lidar_data=lidar_data,
            odoms=odoms,
        )
    else:
        print("[INFO] --no-odom enabled: lidar_positions.txt was not written.")

    print("=" * 70)
    print("[OK] LiDAR extraction completed.")
    print(f"[OK] Point clouds: {pc_dir}")
    if not args.no_odom:
        print(f"[OK] Sync file: {lidar_sync_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()