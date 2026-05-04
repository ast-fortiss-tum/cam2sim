import os
import json
import glob
import re
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Image as HFImage

# ================= CONFIGURATION =================
# 1. WORK IN THIS FOLDER ONLY
DATASET_ROOT = os.path.join("datasets", "2025-11-07-11-34-46_fixed_512_png")

# 2. Coordinate JSON Path
COORD_JSON_PATH = "/home/davidejannussi/Sim2Diff/maps/guerickestrae_alte_heide_munich_25_12_17/trajectory_positions_rear.json"

# 3. Output for the BINARY data (Saved INSIDE the existing folder)
COMPILED_OUTPUT = os.path.join(DATASET_ROOT, "hf_binary")

TARGET_SIZE = (512, 512)
MASK_KEYWORDS = ["segmentation", "instance", "mask", "label"]
VALID_EXTENSIONS = ('.png',)
# =================================================

def load_coords(json_path):
    print(f"📖 Loading coordinates from: {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ Error: JSON file not found at {json_path}")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    for entry in data:
        fid = entry.get('frame_id')
        loc = entry.get('transform', {}).get('location', {})
        mapping[fid] = f"pos x: {loc.get('x', 0):.2f}, y: {loc.get('y', 0):.2f}"
    return mapping

def get_frame_id(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Error: Dataset folder not found: {DATASET_ROOT}")
        return

    # 1. Load Coordinates
    coords_map = load_coords(COORD_JSON_PATH)

    # 2. Verify / Resize Images (In-Place)
    print(f"\n⚙️  Verifying images in: {DATASET_ROOT}")
    
    files_to_check = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                # Skip checking inside the binary folder if it already exists
                if "hf_binary" in root:
                    continue
                files_to_check.append(os.path.join(root, file))

    processed_cnt = 0
    skipped_cnt = 0

    for img_path in tqdm(files_to_check, desc="Checking"):
        try:
            # We assume files existing here are "correct" unless size is wrong
            # This is fast.
            with Image.open(img_path) as img:
                if img.size != TARGET_SIZE:
                    # Resize logic
                    path_parts = img_path.lower().split(os.sep)
                    is_mask = any(k in part for part in path_parts for k in MASK_KEYWORDS)
                    method = Image.NEAREST if is_mask else Image.LANCZOS
                    img.resize(TARGET_SIZE, method).save(img_path, format='PNG')
                    processed_cnt += 1
                else:
                    skipped_cnt += 1
        except Exception as e:
            print(f"Error checking {img_path}: {e}")

    print(f"✅ Verified. Resized: {processed_cnt}, Skipped: {skipped_cnt}")

    # 3. Generate Metadata AND Prepare Data List with ABSOLUTE PATHS
    # We do this manually to avoid 'FileNotFound' errors during save_to_disk
    print("\n📝 Preparing Data & Metadata...")
    
    # Find main images folder
    subfolders = [f.name for f in os.scandir(DATASET_ROOT) if f.is_dir()]
    img_folder_name = next((f for f in subfolders if not any(k in f.lower() for k in MASK_KEYWORDS) and "previous" not in f.lower() and "hf_binary" not in f.lower()), None)
    
    if not img_folder_name:
         for name in ["images", "image", "rgb"]:
             if os.path.exists(os.path.join(DATASET_ROOT, name)):
                 img_folder_name = name
                 break
    
    if not img_folder_name:
        print("❌ Error: Could not find 'images' folder.")
        return

    image_files = sorted(glob.glob(os.path.join(DATASET_ROOT, img_folder_name, "*.png")))
    
    dataset_entries = [] # This list will hold the final data for HF
    metadata_lines = []  # This list is just for the JSONL file (backup)

    for img_path in tqdm(image_files, desc="Matching"):
        fname = os.path.basename(img_path)
        fid = get_frame_id(fname)
        
        # 1. Coordinates
        caption = coords_map.get(fid, "pos x: 0.00, y: 0.00")
        
        # 2. Paths
        # For Metadata JSONL: We use Relative paths (Standard practice)
        rel_img = os.path.join(img_folder_name, fname)
        
        # For HF Dataset Builder: We use ABSOLUTE paths (Fixes FileNotFoundError)
        abs_img = os.path.abspath(img_path)

        entry_json = {"file_name": rel_img, "image": rel_img, "text": caption}
        entry_hf = {"image": abs_img, "text": caption}

        # 3. Siblings
        for key, aliases in {
            "segmentation": ["segmentation", "seg"],
            "instance": ["instance", "inst"],
            "previous": ["previous", "prev"]
        }.items():
            for alias in aliases:
                match_folder = next((f for f in subfolders if alias in f.lower()), None)
                if match_folder:
                    sibling_abs = os.path.join(DATASET_ROOT, match_folder, fname)
                    if os.path.exists(sibling_abs):
                        # Relative for JSONL
                        entry_json[key] = os.path.join(match_folder, fname)
                        # Absolute for HF
                        entry_hf[key] = os.path.abspath(sibling_abs)
                        break
        
        metadata_lines.append(entry_json)
        dataset_entries.append(entry_hf)

    # Save JSONL (Standard Backup)
    json_path = os.path.join(DATASET_ROOT, "metadata.jsonl")
    with open(json_path, 'w') as f:
        for line in metadata_lines:
            f.write(json.dumps(line) + "\n")
    print(f"✅ metadata.jsonl saved.")

    # 4. Create Binary Dataset (Fixing 'AttributeError' and 'FileNotFoundError')
    print("\n🚀 Compiling Binary Dataset...")
    
    # Create dataset directly from memory list (using Absolute Paths)
    ds = Dataset.from_list(dataset_entries)
    
    # Define features to Cast
    new_features = ds.features.copy()
    new_features["image"] = HFImage()
    if "segmentation" in ds.column_names:
        new_features["segmentation"] = HFImage()
    if "instance" in ds.column_names:
        new_features["instance"] = HFImage()
    if "previous" in ds.column_names:
        new_features["previous"] = HFImage()

    print("⚙️  Casting columns...")
    ds = ds.cast(new_features)

    print(f"💾 Saving binary data to: {COMPILED_OUTPUT}")
    ds.save_to_disk(COMPILED_OUTPUT)
    
    print("\n🎉 SUCCESS!")
    print(f"👉 Now update train_models.sh with:")
    print(f'   DATASET_DIR="{COMPILED_OUTPUT}"')

if __name__ == "__main__":
    main()