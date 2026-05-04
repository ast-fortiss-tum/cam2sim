import os
import json
import re
import glob
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Image as HFImage

# Import your existing logic to stay consistent with the "old" way
from utils.dataset import create_segmentation_data, add_temporal_links

# ================= CONFIGURATION =================
# Raw data from your capture/extraction scripts
INPUT_DIR = "/home/davidejannussi/Sim2Diff/datasets/snowy"
# Where the processed 512x512 data and binary files will go
OUTPUT_DIR = "/home/davidejannussi/Sim2Diff/datasets/snowy_512"
# JSON with exact X, Y coordinates
COORD_JSON = os.path.join(INPUT_DIR, "trajectory_positions_rear_odom_yaw.json")
# Square resolution for Stable Diffusion 1.5
TARGET_SIZE = (512, 512)
# =================================================

def get_frame_id(filename):
    """Extracts integer ID from filenames like frame_000000.png."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def main():
    # Ensure all subdirectories exist
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "instance"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "segmentation"), exist_ok=True)
    
    # 1. Load Coordinates for Captions
    print(f"📖 Loading coordinates from: {COORD_JSON}")
    if not os.path.exists(COORD_JSON):
        print(f"❌ Error: {COORD_JSON} not found!")
        return

    with open(COORD_JSON, 'r') as f:
        traj_data = json.load(f)
    
    # Create lookup map for fast access during the loop
    coords_map = {entry['frame_id']: entry['transform']['location'] for entry in traj_data}

    # 2. Collect RGB files
    img_files = sorted(glob.glob(os.path.join(INPUT_DIR, "images", "*.png")))
    raw_dataset_list = []

    print(f"⚙️  Preparing frames (Resizing and Path Mapping)...")
    for i, img_path in enumerate(tqdm(img_files)):
        fname = os.path.basename(img_path)
        fid = get_frame_id(fname)
        
        # Check if coordinates exist for this frame
        if fid not in coords_map:
            continue

        # Check if instance map exists (required for seg-maps and instance training)
        inst_path = os.path.join(INPUT_DIR, "instance", fname)
        if not os.path.exists(inst_path):
            continue

        # Define output paths
        rgb_out = os.path.join(OUTPUT_DIR, "images", fname)
        inst_out = os.path.join(OUTPUT_DIR, "instance", fname)
        seg_out = os.path.join(OUTPUT_DIR, "segmentation", fname) # Pre-defined!

        # --- PROCESS & RESIZE (if missing) ---
        if not os.path.exists(rgb_out):
            rgb_tmp = Image.open(img_path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
            rgb_tmp.save(rgb_out)
        
        if not os.path.exists(inst_out):
            inst_tmp = Image.open(inst_path).resize(TARGET_SIZE, Image.NEAREST)
            inst_tmp.save(inst_out)

        # Build entry with ALL keys present from the start
        loc = coords_map.get(fid)
        raw_dataset_list.append({
            "image": os.path.abspath(rgb_out),
            "instance": os.path.abspath(inst_out),
            "segmentation": os.path.abspath(seg_out), # Path exists even if file doesn't yet
            "text": f"pos x: {loc['x']:.2f}, y: {loc['y']:.2f}",
            "frame_id": fid
        })

    # 3. Apply Segmentation and Temporal logic
    # This function uses SegFormer to fill the 'segmentation' folder
    print("\n🎨 Generating segmentation maps via create_segmentation_data...")
    create_segmentation_data(raw_dataset_list, OUTPUT_DIR)

    # This function creates the 'previous' column by linking N to N-1
    print("🔗 Adding temporal links (previous frames)...")
    final_list = add_temporal_links(raw_dataset_list)

    # 4. Save as Hugging Face Binary (Arrow)
    print("\n🚀 Compiling Binary Dataset (Apache Arrow)...")
    
    # Clean up any potential missing entries before compiling
    final_list = [e for e in final_list if os.path.exists(e["segmentation"])]
    
    ds = Dataset.from_list(final_list)
    
    # Cast paths to proper Image types to embed pixels into the binary
    ds = ds.cast_column("image", HFImage())
    ds = ds.cast_column("previous", HFImage())
    if "segmentation" in ds.column_names:
        ds = ds.cast_column("segmentation", HFImage())
    if "instance" in ds.column_names:
        ds = ds.cast_column("instance", HFImage())

    compiled_path = os.path.join(OUTPUT_DIR, "hf_binary")
    ds.save_to_disk(compiled_path)
    
    print(f"\n" + "="*50)
    print(f"✅ SUCCESS!")
    print(f"Binary dataset ready: {compiled_path}")
    print(f"Update your Training Script LORA_DATASET_NAME to this path.")
    print("="*50)

if __name__ == "__main__":
    main()