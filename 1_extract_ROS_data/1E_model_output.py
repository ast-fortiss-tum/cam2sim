#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1E_model_output.py

Extract driving model steering predictions from a ROS bag to a text file.

Reads from:
    data/raw_ros_data/<BAG>.bag

Writes to:
    data/raw_dataset/<BAG>/
        steering_predictions.txt
            Model-output steering samples:
            timestamp, steering_target

Parameters:
    --bag-name <BAG>.bag
        ROS bag filename to process.
        Default: env BAG_NAME or reference_bag.bag.

    --topic <TOPIC>
        ROS topic containing driving model steering target values.
        Default: /cmd/steering_target.

Usage:
    python 1_process_datasets/1E_model_output.py --bag-name snowy.bag

    python 1_process_datasets/1E_model_output.py \
        --bag-name snowy.bag \
        --topic /cmd/steering_target
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
DEFAULT_TOPIC = "/cmd/steering_target"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract driving model steering predictions from a ROS bag."
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
        help=f"Model-output steering topic to extract. Default: {DEFAULT_TOPIC}",
    )

    return parser.parse_args()


# =============================================================================
# MODEL OUTPUT EXTRACTION
# =============================================================================

def read_model_outputs_from_bag(bag_path, topic):
    """
    Read driving model steering predictions from a ROS bag.

    Returns:
        list of tuples:
            timestamp, steering_target
    """
    output_data = []

    with AnyReader([bag_path]) as reader:
        conns = [
            conn for conn in reader.connections
            if conn.topic == topic
        ]

        if not conns:
            raise RuntimeError(f"Model-output topic not found: {topic}")

        print(f"[INFO] Reading model-output topic: {topic}")

        for conn in conns:
            for connection, ts, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, connection.msgtype)

                timestamp = ts * 1e-9
                value = msg.data

                output_data.append((
                    float(timestamp),
                    value,
                ))

                if len(output_data) % 1000 == 0:
                    print(f"[INFO] Extracted {len(output_data)} model-output messages...")

    output_data.sort(key=lambda item: item[0])

    if not output_data:
        raise RuntimeError(f"No model-output messages found on topic: {topic}")

    print(f"[OK] Loaded {len(output_data)} model-output samples.")

    return output_data


# =============================================================================
# OUTPUT WRITING
# =============================================================================

def write_model_outputs_txt(output_path, output_data):
    """
    Write model steering predictions to steering_predictions.txt.
    """
    print(f"[INFO] Writing model-output data: {output_path}")

    with open(output_path, "w") as file:
        file.write("# timestamp, steering_target\n")

        for timestamp, value in output_data:
            file.write(f"{timestamp:.9f}, {value}\n")

    print(f"[OK] Wrote model-output data: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    bag_name = args.bag_name
    bag_stem = Path(bag_name).stem

    bag_path = PROJECT_ROOT / "data" / "raw_ros_data" / bag_name
    dataset_dir = PROJECT_ROOT / "data" / "raw_dataset" / bag_stem
    output_path = dataset_dir / "steering_predictions.txt"

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL OUTPUT EXTRACTION")
    print("=" * 70)
    print(f"[INFO] Project root:        {PROJECT_ROOT}")
    print(f"[INFO] Bag:                 {bag_name}")
    print(f"[INFO] Bag path:            {bag_path}")
    print(f"[INFO] Output directory:    {dataset_dir}")
    print(f"[INFO] Model-output topic:  {args.topic}")
    print("=" * 70)

    output_data = read_model_outputs_from_bag(
        bag_path=bag_path,
        topic=args.topic,
    )

    write_model_outputs_txt(
        output_path=output_path,
        output_data=output_data,
    )

    print("=" * 70)
    print("[OK] Model-output extraction completed.")
    print(f"[OK] Samples: {len(output_data)}")
    print(f"[OK] Output:  {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()