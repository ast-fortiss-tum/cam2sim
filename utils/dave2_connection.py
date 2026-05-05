import socket
import struct
import io
from PIL import Image

SERVER_HOST = "localhost"  # Change if server is remote
SERVER_PORT = 5090

def connect_to_dave2_server(host=SERVER_HOST, port=SERVER_PORT):
    """
    Establish a persistent connection to the Dave2 server.
    
    Returns:
        socket.socket: connected TCP socket
    """
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((host, port))
    return conn

def send_image_over_connection(conn, pil_image: Image.Image):
    """
    Send a PIL image over an existing connection and receive steering/throttle.
    
    Args:
        conn (socket.socket): persistent connection to server
        pil_image (PIL.Image.Image): input image
    
    Returns:
        tuple: (steering: float, throttle: float)
    """
    # Convert PIL image to JPEG bytes
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Send image length + image bytes
    conn.sendall(struct.pack(">I", len(img_bytes)))
    conn.sendall(img_bytes)

    # Receive response length
    raw_len = conn.recv(4)
    if not raw_len:
        raise RuntimeError("No response from server")
    resp_len = struct.unpack(">I", raw_len)[0]

    # Receive response string
    resp_bytes = b""
    while len(resp_bytes) < resp_len:
        chunk = conn.recv(resp_len - len(resp_bytes))
        if not chunk:
            break
        resp_bytes += chunk

    response = resp_bytes.decode("utf-8")
    steering, throttle = map(float, response.split(","))

    return steering, throttle
