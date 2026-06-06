#!/bin/bash
#
# step2_sd.sh
#
# Step 2 - Stable Diffusion branch
# Runs the SD-specific data processing pipeline.
#
# Uses 2A_sd (camera detection with YOLO instance maps + per-car RGB
# colors) instead of the standard 2A, then runs 2C (map), 2F (semantic),
# 2G (sidewalk fix), 3A (CARLA trajectory needed by 2H), and finally 2H
# to package the dataset for SD training.
#
# Prerequisites (run BEFORE this script):
#   - Step 1 (1_extract_ROS_data/step1.sh <bag>)
#   - Conda envs: data_extraction (for 2A_sd/2C/2F/3A) and stable_diff (for 2H)

set -e

# ---------- Conda init ----------
# Required for `conda activate` to work in non-interactive scripts.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- Usage ----------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <bag_name.bag>"
    echo "Example: $0 reference_bag.bag"
    exit 1
fi

BAG_NAME="$1"
BAG_STEM="${BAG_NAME%.bag}"

echo "=========================================="
echo "Step 2 (SD branch) for bag: $BAG_NAME"
echo "=========================================="

# ---------- Activate data_extraction env ----------
echo ""
echo "--- Activating conda env: data_extraction ---"
conda activate data_extraction

# ---------- Model downloads (infrastructure, not bag-specific) ----------

# YOLO segmentation model (needed by 2A_sd)
YOLO_FILE="2_process_datasets/utils/yolov8n-seg.pt"
if [ ! -f "$YOLO_FILE" ]; then
    echo "Downloading YOLOv8n-seg"
    wget -O "$YOLO_FILE" \
        https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
fi

# FCOS3D model (needed by 2A_sd via 2A base logic)
FCOS3D_FILE="2_process_datasets/utils/fcos3d.pth"
if [ ! -f "$FCOS3D_FILE" ]; then
    echo "Downloading FCOS3D"
    gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O "$FCOS3D_FILE"
fi

# ---------- Run 2A_sd, 2C, 2F ----------

SCRIPTS=(
    "2_process_datasets/2A_sd_camera_parked_cars_detection.py"
    "2_process_datasets/2C_create_map_from_coordinates_auto.py"
    "2_process_datasets/2F_extract_semantic_maps.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    echo ""
    echo "--- Running $SCRIPT ---"
    python3 "$SCRIPT" --bag-name "$BAG_NAME"
done

# ---------- 2G: sidewalk fix on OpenDRIVE map ----------

echo ""
echo "--- Running 2G_OPT_fix_sidewalk.sh ---"
chmod +x 2_process_datasets/2G_OPT_fix_sidewalk.sh
bash 2_process_datasets/2G_OPT_fix_sidewalk.sh "$BAG_NAME"

# ---------- 3A: transform coordinates to CARLA (produces trajectory needed by 2H) ----------

SCRIPT_3A="3_generate_simulation_data/3A_transform_coordinates_to_carla.py"
echo ""
echo "--- Running $SCRIPT_3A ---"
python3 "$SCRIPT_3A" --bag-name "$BAG_NAME"

# ---------- Sanity checks before 2H ----------

TRAJ_FILE="data/data_for_carla/${BAG_STEM}/trajectory_positions_rear_odom_yaw.json"
if [ ! -f "$TRAJ_FILE" ]; then
    echo ""
    echo "ERROR: $TRAJ_FILE not found after 3A ran. Something went wrong."
    exit 1
fi

INSTANCE_MAPS_DIR="data/processed_dataset/${BAG_STEM}/camera_detections/instance_maps"
if [ ! -d "$INSTANCE_MAPS_DIR" ]; then
    echo ""
    echo "ERROR: $INSTANCE_MAPS_DIR not found."
    echo "       2A_sd did not produce instance maps. Check the script output above."
    exit 1
fi

# ---------- Switch to stable_diff env for 2H ----------

echo ""
echo "--- Switching conda env: data_extraction -> stable_diff ---"
conda deactivate
conda activate stable_diff

# ---------- 2H: prepare HuggingFace Arrow dataset for SD training ----------
# 2H requires the `stable_diff` env (HuggingFace `datasets` is not in `data_extraction`).

SCRIPT_2H="2_process_datasets/2H_prepare_dataset_for_stable_diffusion.py"
echo ""
echo "--- Running $SCRIPT_2H ---"
python3 "$SCRIPT_2H" --bag-name "$BAG_NAME"

conda deactivate

echo ""
echo "=========================================="
echo "Step 2 (SD branch) completed for $BAG_NAME"
echo "=========================================="