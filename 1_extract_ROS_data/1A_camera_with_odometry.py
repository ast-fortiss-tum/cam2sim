#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1A_camera_with_odometry.py

Extract synchronized RGB frames and rear-axle odometry poses from a ROS bag.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag

Writes to (project root):
    data/raw_dataset/<BAG>/
        images/frame_<N:06d>.png (RGB frames from the camera topic)
        images_positions.txt (per-frame UTM pose interpolated from /odom)
"""

import os
import math
import argparse
import numpy as np
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader

# --------------------- USER SETTINGS ---------------------
DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_CAM_TOPIC  = "/gmsl_camera/front_narrow/image_raw"

parser = argparse.ArgumentParser(
    description="Extract synchronized RGB frames and odometry poses from a ROS bag."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
parser.add_argument("--cam-topic",  default=DEFAULT_CAM_TOPIC)
args = parser.parse_args()

bag_name   = args.bag_name                # e.g. "reference_bag.bag"
bag_stem   = Path(bag_name).stem          # e.g. "reference_bag"
odom_topic = args.odom_topic
cam_topic  = args.cam_topic

bag_path = Path("data") / "raw_ros_data" / bag_name
if not bag_path.is_file():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

dataset_dir = Path("data") / "raw_dataset" / bag_stem
images_dir  = dataset_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

camera_sync_path = dataset_dir / "images_positions.txt"

print(f"Bag:              {bag_path}")
print(f"Output directory: {dataset_dir}")
# --------------------------------------------------------


def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w*z + x*y)
    c = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(s, c)


def decode_image_msg(msg):
    dtype = np.uint8
    img_data = np.frombuffer(msg.data, dtype=dtype)

    if "bayer" in msg.encoding or msg.encoding == "mono8":
        img = img_data.reshape(msg.height, msg.width)
        if "bayer" in msg.encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img_data.reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


print(f"Reading bag: {bag_path} ...")

# 1. READ DATA
odoms = []
cam_data = []

with AnyReader([bag_path]) as reader:

    conns = {}
    for c in reader.connections:
        conns.setdefault(c.topic, []).append(c)

    if odom_topic not in conns:
        raise RuntimeError(f"Odom topic {odom_topic} not found.")
    if cam_topic not in conns:
        print(f"[WARNING] Camera topic {cam_topic} not found.")

    # --- ODOM ---
    if odom_topic in conns:
        for c in conns[odom_topic]:
            for connection, ts, raw in reader.messages(connections=[c]):
                m = reader.deserialize(raw, connection.msgtype)
                p = m.pose.pose.position
                q = m.pose.pose.orientation
                odoms.append((
                    ts * 1e-9,
                    float(p.x), float(p.y), float(p.z),
                    float(q.x), float(q.y), float(q.z), float(q.w),
                ))

    # --- IMAGES ---
    if cam_topic in conns:
        frame_idx = 0
        print("Extracting images...")
        for c in conns[cam_topic]:
            for connection, ts, raw in reader.messages(connections=[c]):
                msg = reader.deserialize(raw, connection.msgtype)
                timestamp_sec = ts * 1e-9

                try:
                    img = decode_image_msg(msg)
                    filename = f"frame_{frame_idx:06d}.png"
                    save_path = os.path.join(images_dir, filename)
                    cv2.imwrite(save_path, img)

                    cam_data.append({
                        'ts': timestamp_sec,
                        'id': frame_idx,
                        'filename': filename
                    })

                    frame_idx += 1

                    if frame_idx % 10 == 0:
                        print(f"Saved {frame_idx} images...", end='\r')

                except Exception as e:
                    print(f"Error decoding frame {frame_idx}: {e}")

        print(f"\nFinished extracting {frame_idx} images.")

# Sort data
odoms.sort(key=lambda x: x[0])
cam_data.sort(key=lambda x: x['ts'])

if not odoms:
    raise RuntimeError("No odometry messages found.")

odom_arr = np.array(odoms)
odom_t_all  = odom_arr[:, 0]
odom_tx_all = odom_arr[:, 1]
odom_ty_all = odom_arr[:, 2]
odom_tz_all = odom_arr[:, 3]
odom_qx_all = odom_arr[:, 4]
odom_qy_all = odom_arr[:, 5]
odom_qz_all = odom_arr[:, 6]
odom_qw_all = odom_arr[:, 7]

# =========================================================================
# 2. Export Camera Sync
# =========================================================================
print(f"\nProcessing Sync for {len(cam_data)} camera frames...")

if len(cam_data) > 0:
    cam_timestamps = np.array([c['ts'] for c in cam_data])

    # Interpolate position (x, y, z)
    cam_tx = np.interp(cam_timestamps, odom_t_all, odom_tx_all)
    cam_ty = np.interp(cam_timestamps, odom_t_all, odom_ty_all)
    cam_tz = np.interp(cam_timestamps, odom_t_all, odom_tz_all)

    # Interpolate quaternion component-by-component, then renormalize.
    # Linear interpolation + renormalization is a good approximation of slerp
    # when adjacent poses are close, which is the case here (~100 Hz).
    cam_qx = np.interp(cam_timestamps, odom_t_all, odom_qx_all)
    cam_qy = np.interp(cam_timestamps, odom_t_all, odom_qy_all)
    cam_qz = np.interp(cam_timestamps, odom_t_all, odom_qz_all)
    cam_qw = np.interp(cam_timestamps, odom_t_all, odom_qw_all)

    q_norm = np.sqrt(cam_qx**2 + cam_qy**2 + cam_qz**2 + cam_qw**2)
    cam_qx /= q_norm
    cam_qy /= q_norm
    cam_qz /= q_norm
    cam_qw /= q_norm

    # Yaw derived from the interpolated quaternion (consistent with qx, qy, qz, qw).
    cam_yaw = np.array([
        quat_to_yaw(qx, qy, qz, qw)
        for qx, qy, qz, qw in zip(cam_qx, cam_qy, cam_qz, cam_qw)
    ])

    print(f"Exporting synchronized camera data to: {camera_sync_path}")

    try:
        with open(camera_sync_path, 'w') as f:
            f.write("# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, "
                    "Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile\n")

            for i, c in enumerate(cam_data):
                f.write(
                    f"{c['id']}, {c['ts']:.6f}, "
                    f"{cam_tx[i]:.4f}, {cam_ty[i]:.4f}, {cam_tz[i]:.4f}, "
                    f"{cam_qx[i]:.6f}, {cam_qy[i]:.6f}, {cam_qz[i]:.6f}, {cam_qw[i]:.6f}, "
                    f"{cam_yaw[i]:.4f}, "
                    f"{c['filename']}\n"
                )

        print(f"Successfully wrote {len(cam_data)} synchronized frames.")

    except Exception as e:
        print(f"Error writing sync file: {e}")