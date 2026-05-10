import socket
import struct
import io
from PIL import Image
import numpy as np

import tensorflow as tf
from dave2 import Dave2Model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]  # 5 GB
        )
    except RuntimeError as e:
        print(e)

# Load model once at startup
dave2_model = Dave2Model("final.h5")

HOST = "localhost"   # Listen on all interfaces
PORT = 5090        # You can change this port

def handle_client(conn):
    try:
        while True:
            # Step 1: receive the image size (4 bytes)
            raw_size = conn.recv(4)
            if not raw_size:
                break  # client closed connection
            img_size = struct.unpack(">I", raw_size)[0]

            # Step 2: receive the image bytes
            img_bytes = b""
            while len(img_bytes) < img_size:
                chunk = conn.recv(min(4096, img_size - len(img_bytes)))
                if not chunk:
                    break
                img_bytes += chunk

            if not img_bytes:
                break

            # Step 3: decode image
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Step 4: run inference
            steering, throttle = dave2_model.calculate_dave2_image(image)

            # Step 5: send back results
            response = f"{steering},{throttle}".encode("utf-8")
            conn.sendall(struct.pack(">I", len(response)) + response)

    except Exception as e:
        print("Error handling client:", e)
    finally:
        conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        print(f"Dave2 server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server.accept()
            print("Connected by", addr)
            handle_client(conn)

if __name__ == "__main__":
    start_server()