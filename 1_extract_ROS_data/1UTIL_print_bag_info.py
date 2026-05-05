import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
bag_path = Path('data/raw_ros_data/reference_bag.bag')
# ---------------------------------------


def main():

    if not bag_path.exists():
        print("Bag not found")
        return

    print(f"Reading: {bag_path}\n")

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
                    "count": 0
                }

        # count messages + time range
        start_time = None
        end_time = None

        for conn, ts, raw in reader.messages():

            topic = conn.topic

            if topic in topic_info:
                topic_info[topic]["count"] += 1

            t_sec = ts * 1e-9

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