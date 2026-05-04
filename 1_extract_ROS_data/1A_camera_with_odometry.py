import os
import math
import numpy as np
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader

# --------------------- USER SETTINGS ---------------------
bag_path = Path('/media/davidejannussi/New Volume/02-03-2026 cloudy/2026-02-16-23-07-19.bag')
odom_topic = "/odom"
cam_topic  = "/gmsl_camera/front_narrow/image_raw" 

bag_name = bag_path.stem 

dataset_dir = os.path.join(os.getcwd(), "datasets", bag_name)
images_dir  = os.path.join(dataset_dir, "images")

os.makedirs(images_dir, exist_ok=True)

camera_sync_path = os.path.join(dataset_dir, "images_positions.txt")

print(f"Output directory set to: {dataset_dir}")
# --------------------------------------------------------


def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w*z + x*y)
    c = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(s, c)


def decode_image_msg(msg):
    dtype = np.uint8
    img_data = np.frombuffer(msg.data, dtype=dtype)

    if "bayer" in msg.encoding or msg.encoding == "mono8":
        img = img_data.reshape(msg.height, msg.width)
        if "bayer" in msg.encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img_data.reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


print(f"Reading bag: {bag_path} ...")

# 1. READ DATA
odoms = []
cam_data = []

with AnyReader([bag_path]) as reader:

    conns = {}
    for c in reader.connections:
        conns.setdefault(c.topic, []).append(c)

    if odom_topic not in conns:
        raise RuntimeError(f"Odom topic {odom_topic} not found.")
    if cam_topic not in conns:
        print(f"[WARNING] Camera topic {cam_topic} not found.")

    # --- ODOM ---
    if odom_topic in conns:
        for c in conns[odom_topic]:
            for connection, ts, raw in reader.messages(connections=[c]):
                m = reader.deserialize(raw, connection.msgtype)
                p = m.pose.pose.position
                q = m.pose.pose.orientation
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
                odoms.append((ts * 1e-9, float(p.x), float(p.y), yaw))

    # --- IMAGES ---
    if cam_topic in conns:
        frame_idx = 0
        print("Extracting images...")
        for c in conns[cam_topic]:
            for connection, ts, raw in reader.messages(connections=[c]):
                msg = reader.deserialize(raw, connection.msgtype)
                timestamp_sec = ts * 1e-9

                try:
                    img = decode_image_msg(msg)
                    filename = f"frame_{frame_idx:06d}.png"
                    save_path = os.path.join(images_dir, filename)
                    cv2.imwrite(save_path, img)

                    cam_data.append({
                        'ts': timestamp_sec,
                        'id': frame_idx,
                        'filename': filename
                    })

                    frame_idx += 1

                    if frame_idx % 10 == 0:
                        print(f"Saved {frame_idx} images...", end='\r')

                except Exception as e:
                    print(f"Error decoding frame {frame_idx}: {e}")

        print(f"\nFinished extracting {frame_idx} images.")

# Sort data
odoms.sort(key=lambda x: x[0])
cam_data.sort(key=lambda x: x['ts'])

if not odoms:
    raise RuntimeError("No odometry messages found.")

odom_arr = np.array(odoms)
odom_t_all = odom_arr[:, 0]
odom_x_all = odom_arr[:, 1]
odom_y_all = odom_arr[:, 2]
odom_yaw_all = odom_arr[:, 3]

# =========================================================================
# 2. Export Camera Sync
# =========================================================================
print(f"\nProcessing Sync for {len(cam_data)} camera frames...")

if len(cam_data) > 0:
    cam_timestamps = np.array([c['ts'] for c in cam_data])
    cam_sync_x = np.interp(cam_timestamps, odom_t_all, odom_x_all)
    cam_sync_y = np.interp(cam_timestamps, odom_t_all, odom_y_all)
    cam_sync_yaw = np.interp(cam_timestamps, odom_t_all, np.unwrap(odom_yaw_all))

    print(f"Exporting synchronized camera data to: {camera_sync_path}")

    try:
        with open(camera_sync_path, 'w') as f:
            f.write("# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, ImageFile\n")

            for i, c in enumerate(cam_data):
                f.write(
                    f"{c['id']}, {c['ts']:.6f}, "
                    f"{cam_sync_x[i]:.4f}, {cam_sync_y[i]:.4f}, {cam_sync_yaw[i]:.4f}, "
                    f"{c['filename']}\n"
                )

        print(f"Successfully wrote {len(cam_data)} synchronized frames.")

    except Exception as e:
        print(f"Error writing sync file: {e}")