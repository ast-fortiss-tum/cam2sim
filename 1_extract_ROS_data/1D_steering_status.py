#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1D_steering_status.py

Extract vehicle steering values from a ROS bag to a text file.

Reads from:
    data/raw_ros_data/<BAG>.bag

Writes to:
    data/raw_dataset/<BAG>/
        steering_pct.txt
            Steering samples:
            timestamp, steering_value

Parameters:
    --bag-name <BAG>.bag
        ROS bag filename to process.
        Default: env BAG_NAME or reference_bag.bag.

    --topic <TOPIC>
        ROS topic containing steering percentage values.
        Default: /vehicle/steering_pct.

Usage:
    python 1_process_datasets/1D_steering_status.py --bag-name snowy.bag

    python 1_process_datasets/1D_steering_status.py \
        --bag-name snowy.bag \
        --topic /vehicle/steering_pct
"""

import os
import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BAG_NAME = "reference_bag.bag"
DEFAULT_TOPIC = "/vehicle/steering_pct"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract vehicle steering values from a ROS bag."
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
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Steering topic to extract. Default: {DEFAULT_TOPIC}",
    )

    return parser.parse_args()


# =============================================================================
# STEERING EXTRACTION
# =============================================================================

def read_steering_from_bag(bag_path, topic):
    """
    Read steering values from a ROS bag.

    Returns:
        list of tuples:
            timestamp, steering_value
    """
    steering_data = []

    with AnyReader([bag_path]) as reader:
        conns = [
            conn for conn in reader.connections
            if conn.topic == topic
        ]

        if not conns:
            raise RuntimeError(f"Steering topic not found: {topic}")

        print(f"[INFO] Reading steering topic: {topic}")

        for conn in conns:
            for connection, ts, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, connection.msgtype)

                timestamp = ts * 1e-9
                value = msg.data

                steering_data.append((
                    float(timestamp),
                    value,
                ))

                if len(steering_data) % 1000 == 0:
                    print(f"[INFO] Extracted {len(steering_data)} steering messages...")

    steering_data.sort(key=lambda item: item[0])

    if not steering_data:
        raise RuntimeError(f"No steering messages found on topic: {topic}")

    print(f"[OK] Loaded {len(steering_data)} steering samples.")

    return steering_data


# =============================================================================
# OUTPUT WRITING
# =============================================================================

def write_steering_txt(output_path, steering_data):
    """
    Write steering samples to steering_pct.txt.
    """
    print(f"[INFO] Writing steering data: {output_path}")

    with open(output_path, "w") as file:
        file.write("# timestamp, steering_value\n")

        for timestamp, value in steering_data:
            file.write(f"{timestamp:.9f}, {value}\n")

    print(f"[OK] Wrote steering data: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem

    bag_path = PROJECT_ROOT / "data" / "raw_ros_data" / bag_name
    dataset_dir = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem
    output_path = dataset_dir / "steering_pct.txt"

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEERING EXTRACTION")
    print("=" * 70)
    print(f"[INFO] Project root:      {PROJECT_ROOT}")
    print(f"[INFO] Bag:               {bag_name}")
    print(f"[INFO] Bag path:          {bag_path}")
    print(f"[INFO] Output directory:  {dataset_dir}")
    print(f"[INFO] Steering topic:    {args.topic}")
    print("=" * 70)

    steering_data = read_steering_from_bag(
        bag_path=bag_path,
        topic=args.topic,
    )

    write_steering_txt(
        output_path=output_path,
        steering_data=steering_data,
    )

    print("=" * 70)
    print("[OK] Steering extraction completed.")
    print(f"[OK] Samples: {len(steering_data)}")
    print(f"[OK] Output:  {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()