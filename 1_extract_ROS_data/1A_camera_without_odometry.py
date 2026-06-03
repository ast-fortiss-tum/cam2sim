#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1A_camera_without_odometry.py

Extract RGB frames from a ROS bag camera topic to PNG files.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag

Writes to (project root):
    data/raw_dataset/<BAG>/
        images/frame_<N:06d>.png (RGB frames from the camera topic)
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# --------------------- USER SETTINGS ---------------------
DEFAULT_BAG_NAME  = "reference_bag.bag"
DEFAULT_CAM_TOPIC = "/gmsl_camera/front_narrow/image_raw"

parser = argparse.ArgumentParser(
    description="Extract RGB frames from a ROS bag camera topic to PNG files."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument("--cam-topic", default=DEFAULT_CAM_TOPIC)
args = parser.parse_args()

bag_name  = args.bag_name                # e.g. "reference_bag.bag"
bag_stem  = Path(bag_name).stem          # e.g. "reference_bag"
cam_topic = args.cam_topic

bag_path = Path("data") / "raw_ros_data" / bag_name
if not bag_path.is_file():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

dataset_dir = Path("data") / "raw_dataset" / bag_stem
images_dir  = dataset_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

print(f"Bag:              {bag_path}")
print(f"Output directory: {dataset_dir}")
# --------------------------------------------------------

def decode_image_msg(msg):
    """Decode ROS image message to OpenCV (BGR)."""
    img_data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "mono8":
        img = img_data.reshape(msg.height, msg.width)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif msg.encoding == "rgb8":
        img = img_data.reshape(msg.height, msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif msg.encoding == "bgr8":
        img = img_data.reshape(msg.height, msg.width, 3)

    elif "bayer" in msg.encoding:
        img = img_data.reshape(msg.height, msg.width)
        img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)

    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    return img


print(f"Reading bag: {bag_path}")

with AnyReader([bag_path]) as reader:

    # Find camera connections
    cam_conns = [c for c in reader.connections if c.topic == cam_topic]

    if not cam_conns:
        raise RuntimeError(f"Camera topic not found: {cam_topic}")

    frame_idx = 0

    print("Extracting camera frames...")

    for conn, ts, raw in reader.messages(connections=cam_conns):
        msg = reader.deserialize(raw, conn.msgtype)

        try:
            img = decode_image_msg(msg)

            filename = f"frame_{frame_idx:06d}.png"
            save_path = os.path.join(images_dir, filename)

            cv2.imwrite(save_path, img)

            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"Saved {frame_idx} frames...", end="\r")

        except Exception as e:
            print(f"Error at frame {frame_idx}: {e}")

print(f"\nDone. Saved {frame_idx} images to:")
print(images_dir)