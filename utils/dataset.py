import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from datasets import Features, Value, Image as DSImage
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


from config import CARLA_MODE, SEGFORMER_MODEL


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')




def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def get_dataset_features():
    return Features({
        "image": DSImage(),
        "segmentation": DSImage(),
        "depth": DSImage(),
        "previous": DSImage(),
        "caption": Value("string")
    })

def decode_cityscapes_mask(mask):
    segmentation_colors = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    if CARLA_MODE:
        segmentation_colors = [
            (128, 64, 128), (128, 64, 128),
            (0,0,0), (0,0,0), (0,0,0), (0,0,0),
            (250, 170, 30), (220, 220, 0),
            (0,0,0), (0,0,0), (0,0,0),
            (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
        ]

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(segmentation_colors):
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask)

def add_temporal_links(dataset):
    """
    Aggiunge la colonna 'previous' per la consistenza temporale.
    Frame N -> Previous = Frame N-1
    """
    print("⏳ Adding 'previous' column for Temporal Consistency...")
    for i in range(len(dataset)):
        if i == 0:
            # Il primo frame ha se stesso come precedente
            dataset[i]['previous'] = dataset[i]['image']
        else:
            dataset[i]['previous'] = dataset[i-1]['image']
    return dataset

def load_frames_from_folder(images_root):
    """
    Scans the folder for .png images and extracts the frame ID directly 
    from the filename (assuming format 'frame_000123.png').
    """
    if not os.path.exists(images_root):
        raise FileNotFoundError(f"Images folder not found: {images_root}")

    dataset_list = []
    print(f"📂 Scanning images in: {images_root}")
    
    # Get all png files
    image_files = glob.glob(os.path.join(images_root, "*.png"))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Try to extract ID from filename (e.g., "frame_000123.png")
        try:
            # Splits "frame_000123.png" -> ["frame", "000123.png"] -> "000123"
            frame_id_str = filename.split('_')[-1].split('.')[0]
            frame_id = int(frame_id_str)
            
            dataset_list.append({
                'image': img_path,
                'frame_id': frame_id
            })
        except ValueError:
            print(f"⚠️ Skipping file with unexpected format: {filename}")
            continue

    # Sort by Frame ID to ensure temporal order
    dataset_list.sort(key=lambda x: x['frame_id'])
    
    return dataset_list

def add_existing_data(current_dataset, dataset_path, image_type):
    folder = os.path.join(dataset_path, image_type)
    os.makedirs(folder, exist_ok=True)

    existing_depth = len([f for f in os.listdir(folder) if f.endswith(".png")])

    if existing_depth == len(current_dataset):
        print(f"The Dataset-Folder already contains {existing_depth} {folder} images. Skipping {folder}-image creation.")
        for entry in current_dataset:
            image_path = entry["image"]
            image_name = os.path.basename(image_path)
            entry[image_type] = os.path.join(folder, image_name)
        return True
    return False


def create_segmentation_data(current_dataset, dataset_path):
    segmentation_folder = os.path.join(dataset_path, "segmentation")
    
    if add_existing_data(current_dataset, dataset_path, "segmentation"):
        return

    device = get_device()

    image_extractor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.eval().to(device)

    for entry in tqdm(current_dataset, desc="Creating Segmentation-Map for each frame"):
        image_path = entry["image"]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        inputs = image_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [B, num_classes, H, W]
            upsampled = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            predicted = upsampled.argmax(1)[0].cpu().numpy()

        segmentation_path = os.path.join(segmentation_folder, image_name)
        segmentation_image = decode_cityscapes_mask(predicted)
        segmentation_image.save(segmentation_path)

        entry["segmentation"] = segmentation_path
