import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
bag_path = Path('/media/davidejannussi/New Volume/02-03-2026 cloudy/2026-02-16-23-07-19.bag')
topic = '/vehicle/steering_pct'

output_path = Path.cwd() / "datasets" / bag_path.stem / "steering_pct.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)
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