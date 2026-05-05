import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
import os

# ---------------- CONFIG ----------------
bag_path = Path('data/raw_ros_data/reference_bag.bag')
topic = '/vehicle/steering_pct'

bag_name = bag_path.stem
dataset_dir = os.path.join(os.getcwd(), "data", "extracted_ros_data", bag_name)
os.makedirs(dataset_dir, exist_ok=True)

output_path = os.path.join(dataset_dir, "steering_pct.txt")
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