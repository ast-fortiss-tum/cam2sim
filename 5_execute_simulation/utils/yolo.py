import cv2
from ultralytics import YOLO
from utils.stable_diffusion import get_device
from PIL import Image

def load_yolo_model():
    device = get_device()
    return YOLO("yolo11n.pt", verbose=False).to(device)

def calculate_yolo_image(model, pil_image):
    results = model(pil_image, verbose = False)
    annotated_bgr = results[0].plot()

    # Convert back to RGB PIL for display
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

    return results, annotated_pil