#!/bin/bash
set -e

# ---------- Usage ----------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <bag_name.bag>"
    echo "Example: $0 reference_bag.bag"
    exit 1
fi

BAG_NAME="$1"
BAG_PATH="data/raw_ros_data/$BAG_NAME"

if [ ! -f "$BAG_PATH" ]; then
    echo "ERROR: Bag file not found: $BAG_PATH"
    echo "       Place the bag manually under data/raw_ros_data/ before running."
    exit 1
fi

# ---------- Run all scripts ----------
SCRIPTS=(
    "1_extract_ROS_data/1A_camera_with_odometry.py"
    "1_extract_ROS_data/1B_lidar_with_odometry.py"
    "1_extract_ROS_data/1C_poses_and_trajectory.py"
    "1_extract_ROS_data/1D_steering_status.py"
    "1_extract_ROS_data/1E_model_output.py"
)

echo "=== Step 1: extracting data from bag: $BAG_NAME ==="

for SCRIPT in "${SCRIPTS[@]}"; do
    echo ""
    echo "--- Running $SCRIPT ---"
    python3 "$SCRIPT" --bag-name "$BAG_NAME"
done

echo ""
echo "=== Step 1 completed successfully for $BAG_NAME ==="