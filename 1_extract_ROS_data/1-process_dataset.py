import os
import sys
import argparse
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
from datasets import Dataset, Image
from utils.argparser import parse_dataset_args
from utils.save_data import save_dataset, create_dotenv
from utils.dataset import (
    create_segmentation_data, 
    load_frames_from_folder, 
    add_temporal_links
)

current_dir = os.path.dirname(os.path.abspath(__file__))
DATASETS_ROOT = os.path.join(current_dir, "datasets")



def main():
    args = parse_dataset_args()
    
    # Define paths
    data_dir = os.path.join(DATASETS_ROOT, args.input_folder)
    images_dir = os.path.join(data_dir, "images")
    dataset_name = args.name if args.name else args.input_folder

    # Auth logic...
    try: create_dotenv()
    except: pass
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if args.upload and token: login(token=token)

    print(f"📂 Processing data in: {data_dir}")
    
    # --- CHANGED: Load directly from folder ---
    try:
        dataset = load_frames_from_folder(images_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    if not dataset:
        print("❌ No images found.")
        return
    print(f"✅ Loaded {len(dataset)} frames directly from folder.")

    # ... (The rest of the script: segmentation, saving, upload stays exactly the same) ...
    create_segmentation_data(dataset, data_dir)
    dataset = add_temporal_links(dataset)
    
    # Packaging
    hf_dataset = Dataset.from_list(dataset)
    hf_dataset = hf_dataset.cast_column("image", Image())
    hf_dataset = hf_dataset.cast_column("previous", Image())
    if "segmentation" in hf_dataset.column_names:
        hf_dataset = hf_dataset.cast_column("segmentation", Image())

    save_dataset(data_dir, hf_dataset)
    
    if args.upload and token:
        hf_dataset.push_to_hub(f"{HfApi().whoami()['name']}/{dataset_name}", private=True, token=token)
        print("✅ Upload successful!")

if __name__ == "__main__":
    main()