SCRIPTS=(
    "1_extract_ROS_data/1A_camera_with_odometry.py"
    "1_extract_ROS_data/1B_lidar_with_odometry.py"
    "1_extract_ROS_data/1C_poses_and_trajectory.py"
    "1_extract_ROS_data/1D_steering_status.py"
    "1_extract_ROS_data/1E_model_output.py"
)

REFERENCE_BAG="data/raw_ros_data/reference_bag.bag" 

if [ ! -f "$REFERENCE_BAG" ]; then  
    echo "Downloading Reference Bag"
    gdown 1ka4dqG83aprB6FWjd0W0mWxyPZHsRfj9 -O "$REFERENCE_BAG"
fi

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running Script $SCRIPT"
    python3 "$SCRIPT"

    if [ $? -ne 0 ]; then
        echo "Error in $SCRIPT. Aborting."
        exit 1
    fi
done

echo ""
echo "All Scripts for Step 1 completed successfully"