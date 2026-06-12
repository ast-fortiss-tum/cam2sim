#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1A_extract_camera.py

Extract RGB frames from a ROS bag camera topic, optionally synchronized with
rear-axle odometry poses.

Reads from:
    data/raw_ros_data/<BAG>.bag

Writes to:
    data/raw_dataset/<BAG>/
        images/frame_<N:06d>.png
        images_positions.txt   # only when odometry is enabled

Parameters:
    --bag-name <BAG>.bag
        ROS bag filename to process.
        Default: env BAG_NAME or reference_bag.bag.

    --cam-topic <TOPIC>
        ROS image topic to extract RGB frames from.
        Default: /gmsl_camera/front_narrow/image_raw.

    --odom-topic <TOPIC>
        ROS odometry topic used to interpolate the vehicle pose at each
        camera frame timestamp.
        Default: /odom.

    --no-odom
        Extract only RGB frames. Do not read odometry and do not write
        images_positions.txt.

Usage:
    python 1_process_datasets/1A_extract_camera.py --bag-name snowy.bag
    python 1_process_datasets/1A_extract_camera.py --bag-name snowy.bag --no-odom
"""

import os
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_CAM_TOPIC = "/gmsl_camera/front_narrow/image_raw"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract RGB frames from a ROS bag camera topic, optionally "
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
        "--cam-topic",
        default=DEFAULT_CAM_TOPIC,
        help=f"Camera topic to extract. Default: {DEFAULT_CAM_TOPIC}",
    )

    parser.add_argument(
        "--odom-topic",
        default=DEFAULT_ODOM_TOPIC,
        help=f"Odometry topic to synchronize. Default: {DEFAULT_ODOM_TOPIC}",
    )

    parser.add_argument(
        "--no-odom",
        action="store_true",
        help="Extract images only. Do not read odometry or write images_positions.txt.",
    )

    return parser.parse_args()


# =============================================================================
# IMAGE DECODING
# =============================================================================

def decode_image_msg(msg):
    """
    Decode a ROS image message to an OpenCV BGR image.
    """
    img_data = np.frombuffer(msg.data, dtype=np.uint8)
    encoding = msg.encoding.lower()

    if encoding == "mono8":
        img = img_data.reshape(msg.height, msg.width)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif encoding == "rgb8":
        img = img_data.reshape(msg.height, msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif encoding == "bgr8":
        img = img_data.reshape(msg.height, msg.width, 3)

    elif "bayer" in encoding:
        img = img_data.reshape(msg.height, msg.width)

        if "rggb" in encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
        elif "bggr" in encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        elif "gbrg" in encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
        elif "grbg" in encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)

    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    return img


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
            timestamp_sec, x, y, z, qx, qy, qz, qw
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

            odoms.append((
                ts * 1e-9,
                float(p.x),
                float(p.y),
                float(p.z),
                float(q.x),
                float(q.y),
                float(q.z),
                float(q.w),
            ))

    odoms.sort(key=lambda item: item[0])

    if not odoms:
        raise RuntimeError(f"No odometry messages found on topic: {odom_topic}")

    print(f"[INFO] Loaded {len(odoms)} odometry messages.")

    return odoms


def write_images_positions(camera_sync_path, cam_data, odoms):
    """
    Interpolate odometry poses at camera timestamps and write images_positions.txt.
    """
    print(f"[INFO] Writing synchronized odometry to: {camera_sync_path}")

    odom_arr = np.array(odoms)

    odom_t_all = odom_arr[:, 0]
    odom_tx_all = odom_arr[:, 1]
    odom_ty_all = odom_arr[:, 2]
    odom_tz_all = odom_arr[:, 3]
    odom_qx_all = odom_arr[:, 4]
    odom_qy_all = odom_arr[:, 5]
    odom_qz_all = odom_arr[:, 6]
    odom_qw_all = odom_arr[:, 7]

    cam_timestamps = np.array([item["ts"] for item in cam_data])

    cam_tx = np.interp(cam_timestamps, odom_t_all, odom_tx_all)
    cam_ty = np.interp(cam_timestamps, odom_t_all, odom_ty_all)
    cam_tz = np.interp(cam_timestamps, odom_t_all, odom_tz_all)

    cam_qx = np.interp(cam_timestamps, odom_t_all, odom_qx_all)
    cam_qy = np.interp(cam_timestamps, odom_t_all, odom_qy_all)
    cam_qz = np.interp(cam_timestamps, odom_t_all, odom_qz_all)
    cam_qw = np.interp(cam_timestamps, odom_t_all, odom_qw_all)

    q_norm = np.sqrt(cam_qx**2 + cam_qy**2 + cam_qz**2 + cam_qw**2)
    q_norm[q_norm == 0.0] = 1.0

    cam_qx /= q_norm
    cam_qy /= q_norm
    cam_qz /= q_norm
    cam_qw /= q_norm

    cam_yaw = np.array([
        quat_to_yaw(qx, qy, qz, qw)
        for qx, qy, qz, qw in zip(cam_qx, cam_qy, cam_qz, cam_qw)
    ])

    with open(camera_sync_path, "w") as file:
        file.write(
            "# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, "
            "Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile\n"
        )

        for idx, item in enumerate(cam_data):
            file.write(
                f"{item['id']}, {item['ts']:.6f}, "
                f"{cam_tx[idx]:.4f}, {cam_ty[idx]:.4f}, {cam_tz[idx]:.4f}, "
                f"{cam_qx[idx]:.6f}, {cam_qy[idx]:.6f}, "
                f"{cam_qz[idx]:.6f}, {cam_qw[idx]:.6f}, "
                f"{cam_yaw[idx]:.4f}, "
                f"{item['filename']}\n"
            )

    print(f"[OK] Wrote {len(cam_data)} synchronized camera poses.")


# =============================================================================
# CAMERA EXTRACTION
# =============================================================================

def extract_camera_frames(reader, cam_topic, images_dir):
    """
    Extract camera frames from the selected topic.

    Returns:
        list of per-frame timestamp metadata.
    """
    cam_conns = [
        conn for conn in reader.connections
        if conn.topic == cam_topic
    ]

    if not cam_conns:
        raise RuntimeError(f"Camera topic not found: {cam_topic}")

    print(f"[INFO] Reading camera topic: {cam_topic}")
    print(f"[INFO] Output images: {images_dir}")

    cam_data = []
    frame_idx = 0

    for conn in cam_conns:
        for connection, ts, raw in reader.messages(connections=[conn]):
            msg = reader.deserialize(raw, connection.msgtype)
            timestamp_sec = ts * 1e-9

            filename = f"frame_{frame_idx:06d}.png"
            save_path = images_dir / filename

            try:
                img = decode_image_msg(msg)
                cv2.imwrite(str(save_path), img)

                cam_data.append({
                    "ts": timestamp_sec,
                    "id": frame_idx,
                    "filename": filename,
                })

                frame_idx += 1

                if frame_idx % 50 == 0:
                    print(f"[INFO] Saved {frame_idx} frames...", end="\r")

            except Exception as exc:
                print(f"[WARN] Error decoding frame {frame_idx}: {exc}")

    cam_data.sort(key=lambda item: item["ts"])

    print(f"\n[OK] Extracted {len(cam_data)} camera frames.")

    return cam_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem

    bag_path = PROJECT_ROOT / "data" / "raw_ros_data" / bag_name
    dataset_dir = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem
    images_dir = dataset_dir / "images"
    camera_sync_path = dataset_dir / "images_positions.txt"

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CAMERA EXTRACTION")
    print("=" * 70)
    print(f"[INFO] Project root:      {PROJECT_ROOT}")
    print(f"[INFO] Bag:               {bag_name}")
    print(f"[INFO] Bag path:          {bag_path}")
    print(f"[INFO] Output directory:  {dataset_dir}")
    print(f"[INFO] Camera topic:      {args.cam_topic}")
    print(f"[INFO] Odometry enabled:  {not args.no_odom}")
    if not args.no_odom:
        print(f"[INFO] Odometry topic:    {args.odom_topic}")
    print("=" * 70)

    with AnyReader([bag_path]) as reader:
        odoms = None

        if not args.no_odom:
            odoms = read_odometry(reader, args.odom_topic)

        cam_data = extract_camera_frames(
            reader=reader,
            cam_topic=args.cam_topic,
            images_dir=images_dir,
        )

    if not cam_data:
        raise RuntimeError("No camera frames were extracted.")

    if not args.no_odom:
        write_images_positions(
            camera_sync_path=camera_sync_path,
            cam_data=cam_data,
            odoms=odoms,
        )
    else:
        print("[INFO] --no-odom enabled: images_positions.txt was not written.")

    print("=" * 70)
    print("[OK] Camera extraction completed.")
    print(f"[OK] Images: {images_dir}")
    if not args.no_odom:
        print(f"[OK] Sync file: {camera_sync_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()