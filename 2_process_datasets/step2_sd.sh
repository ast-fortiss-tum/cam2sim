#!/bin/bash
# Step 2 - Stable Diffusion branch
# Runs the SD-specific data processing pipeline.
#
# Uses 2AS (camera detection with YOLO instance maps + per-car RGB
# colors) instead of the standard 2A, then runs 2C (map), 2F (semantic),
# 2G (sidewalk fix), 3A (CARLA trajectory needed by 2H), and finally 2H
# to package the dataset for SD training.
#
# Prerequisites (run BEFORE this script):
#   - Step 1 (1_extract_ROS_data/step1.sh)

set -e

SCRIPTS=(
    "2_process_datasets/2A_sd_camera_parked_cars_detection.py"
    "2_process_datasets/2C_create_map_from_coordinates_auto.py"
    "2_process_datasets/2F_extract_semantic_maps.py"
)

# --- YOLO segmentation model (needed by 2AS) ---
YOLO_FILE="2_process_datasets/utils/yolov8n-seg.pt"
if [ ! -f "$YOLO_FILE" ]; then
    echo "Downloading YOLOv8n-seg"
    wget -O "$YOLO_FILE" \
        https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
fi

# --- FCOS3D model (needed by 2AS via 2A base logic) ---
PTH_FILE_1="2_process_datasets/utils/fcos3d.pth"
if [ ! -f "$PTH_FILE_1" ]; then
    echo "Downloading FCOS3D"
    gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O "$PTH_FILE_1"
fi

# --- Run 2AS, 2C, 2F ---
for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running Script $SCRIPT"
    python3 "$SCRIPT"
    if [ $? -ne 0 ]; then
        echo "Error in $SCRIPT. Aborting."
        exit 1
    fi
done

# --- 2G: sidewalk fix on semantic maps ---
chmod +x 2_process_datasets/2G_OPT_fix_sidewalk.sh
bash 2_process_datasets/2G_OPT_fix_sidewalk.sh

# --- 3A: transform coordinates to CARLA (produces trajectory needed by 2H) ---
SCRIPT_3A="3_generate_simulation_data/3A_transform_coordinates_to_carla.py"
echo "Running Script $SCRIPT_3A"
python3 "$SCRIPT_3A"
if [ $? -ne 0 ]; then
    echo "Error in $SCRIPT_3A. Aborting."
    exit 1
fi

# --- Sanity checks before 2H ---
TRAJ_FILE="data/data_for_carla/reference_bag/trajectory_positions_rear_odom_yaw.json"
if [ ! -f "$TRAJ_FILE" ]; then
    echo ""
    echo "ERROR: $TRAJ_FILE not found after 3A ran. Something went wrong."
    exit 1
fi

INSTANCE_MAPS_DIR="data/processed_dataset/reference_bag/camera_detections/instance_maps"
if [ ! -d "$INSTANCE_MAPS_DIR" ]; then
    echo ""
    echo "ERROR: $INSTANCE_MAPS_DIR not found."
    echo "       2AS did not produce instance maps. Check the script output above."
    exit 1
fi

# --- 2H: prepare HuggingFace Arrow dataset for SD training ---
SCRIPT_2H="2_process_datasets/2H_prepare_dataset_for_stable_diffusion.py"
echo "Running Script $SCRIPT_2H"
python3 "$SCRIPT_2H"
if [ $? -ne 0 ]; then
    echo "Error in $SCRIPT_2H. Aborting."
    exit 1
fi

echo ""
echo "All Scripts for Step 2 (Stable Diffusion branch) completed successfully"