import os
import math
import csv
from pathlib import Path
from rosbags.highlevel import AnyReader

# ---------------- CONFIG ----------------
bag_path = Path("data/raw_ros_data/reference_bag.bag")
odom_topic = "/odom"

bag_name = bag_path.stem

dataset_dir = os.path.join(os.getcwd(), "data", "raw_dataset", bag_name)
os.makedirs(dataset_dir, exist_ok=True)

odom_path = os.path.join(dataset_dir, "odometry.csv")
traj_path = os.path.join(dataset_dir, "trajectory.csv")

print(f"Output: {dataset_dir}")
# ----------------------------------------


def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def main():
    if not bag_path.exists():
        print(f"Bag file not found: {bag_path}")
        return

    print(f"Reading: {bag_path}")

    odom_data = []

    with AnyReader([bag_path]) as reader:
        conns = {}
        for c in reader.connections:
            conns.setdefault(c.topic, []).append(c)

        if odom_topic not in conns:
            raise RuntimeError(f"Odom topic not found: {odom_topic}")

        print("Reading odometry...")

        for c in conns[odom_topic]:
            for conn, ts, raw in reader.messages(connections=[c]):
                msg = reader.deserialize(raw, conn.msgtype)

                p = msg.pose.pose.position
                q = msg.pose.pose.orientation

                timestamp = ts * 1e-9
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

                odom_data.append((
                    timestamp,
                    p.x,
                    p.y,
                    p.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                    yaw,
                ))

                if len(odom_data) % 1000 == 0:
                    print(f"Extracted {len(odom_data)} odometry messages...")

    odom_data.sort(key=lambda x: x[0])

    # =========================
    # SAVE FULL ODOMETRY CSV
    # =========================
    print(f"Writing: {odom_path}")

    with open(odom_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
            "yaw",
        ])

        for t, x, y, z, qx, qy, qz, qw, yaw in odom_data:
            writer.writerow([
                f"{t:.9f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{qx:.6f}",
                f"{qy:.6f}",
                f"{qz:.6f}",
                f"{qw:.6f}",
                f"{yaw:.6f}",
            ])

    # =========================
    # SAVE TRAJECTORY CSV
    # =========================
    print(f"Writing: {traj_path}")

    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "x",
            "y",
            "z",
            "yaw",
        ])

        for t, x, y, z, _, _, _, _, yaw in odom_data:
            writer.writerow([
                f"{t:.9f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{yaw:.6f}",
            ])

    print("Done.")
    print(f"Saved {len(odom_data)} odometry samples")
    print(f"Odometry CSV: {odom_path}")
    print(f"Trajectory CSV: {traj_path}")


if __name__ == "__main__":
    main()