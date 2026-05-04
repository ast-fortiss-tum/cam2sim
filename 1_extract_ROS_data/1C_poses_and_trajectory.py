import os
import math
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
bag_path = Path('/media/davidejannussi/New Volume/02-03-2026 cloudy/2026-02-16-23-07-19.bag')
odom_topic = '/odom'

dataset_dir = Path.cwd() / "datasets" / bag_path.stem
dataset_dir.mkdir(parents=True, exist_ok=True)

odom_path = dataset_dir / "odometry.txt"
traj_path = dataset_dir / "trajectory.txt"

print(f"Output: {dataset_dir}")
# ----------------------------------------


def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def main():

    if not bag_path.exists():
        print("Bag file not found")
        return

    print(f"Reading: {bag_path}")

    odom_data = []

    with AnyReader([bag_path]) as reader:

        conns = {}
        for c in reader.connections:
            conns.setdefault(c.topic, []).append(c)

        if odom_topic not in conns:
            raise RuntimeError("Odom topic not found")

        print("Reading odometry...")

        for c in conns[odom_topic]:
            for conn, ts, raw in reader.messages(connections=[c]):

                msg = reader.deserialize(raw, conn.msgtype)

                p = msg.pose.pose.position
                q = msg.pose.pose.orientation

                t = ts * 1e-9
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

                odom_data.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w, yaw))

                if len(odom_data) % 1000 == 0:
                    print(f"Extracted {len(odom_data)} odometry messages...")

    # sort by time (important)
    odom_data.sort(key=lambda x: x[0])

    # =========================
    # SAVE FULL ODOMETRY
    # =========================
    print(f"Writing: {odom_path}")

    with open(odom_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw yaw\n")

        for t, x, y, z, qx, qy, qz, qw, yaw in odom_data:
            f.write(
                f"{t:.9f}, "
                f"{x:.6f}, {y:.6f}, {z:.6f}, "
                f"{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}, "
                f"{yaw:.6f}\n"
            )

    # =========================
    # SAVE TRAJECTORY ONLY
    # =========================
    print(f"Writing: {traj_path}")

    with open(traj_path, "w") as f:
        f.write("# timestamp x y z yaw\n")

        for t, x, y, z, _, _, _, _, yaw in odom_data:
            f.write(f"{t:.9f}, {x:.6f}, {y:.6f}, {z:.6f}, {yaw:.6f}\n")

    print("Done.")
    print(f"Saved {len(odom_data)} odometry samples")


if __name__ == "__main__":
    main()