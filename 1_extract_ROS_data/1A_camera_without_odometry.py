import os
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# --------------------- USER SETTINGS ---------------------
bag_path = Path('data/raw_ros_data/reference_bag.bag')
cam_topic  = "/gmsl_camera/front_narrow/image_raw" 

bag_name = bag_path.stem
dataset_dir = os.path.join(os.getcwd(), "data", "raw_dataset", bag_name)
images_dir = os.path.join(dataset_dir, "images")

os.makedirs(images_dir, exist_ok=True)

print(f"Output directory: {dataset_dir}")
# --------------------------------------------------------

def decode_image_msg(msg):
    """Decode ROS image message to OpenCV (BGR)."""
    img_data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "mono8":
        img = img_data.reshape(msg.height, msg.width)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif msg.encoding == "rgb8":
        img = img_data.reshape(msg.height, msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif msg.encoding == "bgr8":
        img = img_data.reshape(msg.height, msg.width, 3)

    elif "bayer" in msg.encoding:
        img = img_data.reshape(msg.height, msg.width)
        img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)

    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    return img


print(f"Reading bag: {bag_path}")

with AnyReader([bag_path]) as reader:

    # Find camera connections
    cam_conns = [c for c in reader.connections if c.topic == cam_topic]

    if not cam_conns:
        raise RuntimeError(f"Camera topic not found: {cam_topic}")

    frame_idx = 0

    print("Extracting camera frames...")

    for conn, ts, raw in reader.messages(connections=cam_conns):
        msg = reader.deserialize(raw, conn.msgtype)

        try:
            img = decode_image_msg(msg)

            filename = f"frame_{frame_idx:06d}.png"
            save_path = os.path.join(images_dir, filename)

            cv2.imwrite(save_path, img)

            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"Saved {frame_idx} frames...", end="\r")

        except Exception as e:
            print(f"Error at frame {frame_idx}: {e}")

print(f"\nDone. Saved {frame_idx} images to:")
print(images_dir)