#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1D_steering_status.py

Extract vehicle steering values data from a ROS bag to a text file.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag

Writes to (project root):
    data/raw_dataset/<BAG>/
        steering_pct.txt (timestamp, steering value per row)
"""


import os
import argparse
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_TOPIC    = "/vehicle/steering_pct"

parser = argparse.ArgumentParser(
    description="Extract vehicle steering status from a ROS bag."
)
parser.add_argument(
    "--bag-name",
    default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
    help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
)
parser.add_argument("--topic", default=DEFAULT_TOPIC)
args = parser.parse_args()

bag_name = args.bag_name                # e.g. "reference_bag.bag"
bag_stem = Path(bag_name).stem          # e.g. "reference_bag"
topic    = args.topic

bag_path = Path("data") / "raw_ros_data" / bag_name
if not bag_path.is_file():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

dataset_dir = Path("data") / "raw_dataset" / bag_stem
dataset_dir.mkdir(parents=True, exist_ok=True)

output_path = dataset_dir / "steering_pct.txt"

print(f"Bag:              {bag_path}")
print(f"Output directory: {dataset_dir}")
# ---------------------------------------


def main():

    if not bag_path.exists():
        print("Bag not found")
        return

    print(f"Reading: {bag_path}")
    print(f"Extracting: {topic}")

    data = []

    with AnyReader([bag_path]) as reader:

        conns = [c for c in reader.connections if c.topic == topic]

        if not conns:
            raise RuntimeError(f"Topic not found: {topic}")

        for conn, ts, raw in reader.messages(connections=conns):

            msg = reader.deserialize(raw, conn.msgtype)

            timestamp = ts * 1e-9
            value = msg.data

            data.append((timestamp, value))

    # sort just in case
    data.sort(key=lambda x: x[0])

    # write output
    with open(output_path, "w") as f:
        f.write("# timestamp value\n")
        for t, v in data:
            f.write(f"{t:.9f}, {v}\n")

    print(f"Done. Saved {len(data)} messages")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()