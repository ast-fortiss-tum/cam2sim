#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1UTIL_print_bag_info.py

Inspect a ROS bag: list all topics with type and message count, and the
total recording time range.

Reads from (project root):
    data/raw_ros_data/<BAG>.bag
"""

import os
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader


def main():
    # ---------------- CONFIG ----------------
    DEFAULT_BAG_NAME = "reference_bag.bag"

    parser = argparse.ArgumentParser(
        description="Inspect a ROS bag: topics, types, message counts, time range."
    )
    parser.add_argument(
        "--bag-name",
        default=os.environ.get("BAG_NAME", DEFAULT_BAG_NAME),
        help="Bag filename including .bag extension (default: env BAG_NAME or 'reference_bag.bag').",
    )
    args = parser.parse_args()

    bag_name = args.bag_name
    bag_path = Path("data") / "raw_ros_data" / bag_name

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    print(f"Reading: {bag_path}\n")
    # ----------------------------------------

    with AnyReader([bag_path]) as reader:
        # ---------------- TOPICS ----------------
        print("=== TOPICS ===\n")
        topic_info = {}
        for c in reader.connections:
            topic = c.topic
            msgtype = c.msgtype
            if topic not in topic_info:
                topic_info[topic] = {
                    "type": msgtype,
                    "count": 0,
                }

        # count messages + time range
        start_time = None
        end_time = None
        for conn, ts, raw in reader.messages():
            topic = conn.topic
            if topic in topic_info:
                topic_info[topic]["count"] += 1
            if start_time is None:
                start_time = ts
            end_time = ts

        # print topics
        for topic, info in sorted(topic_info.items()):
            print(f"Topic: {topic}")
            print(f"Type : {info['type']}")
            print(f"Msgs : {info['count']}\n")

        # ---------------- TIME RANGE ----------------
        print("=== TIME RANGE ===")
        print(f"Start: {start_time}")
        print(f"End  : {end_time}")


if __name__ == "__main__":
    main()