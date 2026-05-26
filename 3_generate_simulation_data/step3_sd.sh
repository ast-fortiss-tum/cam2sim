#!/bin/bash
# Step 3 - Stable Diffusion branch
# Runs the SD-specific scenario generation pipeline.
#
# Uses 3B_sd (camera centroids with RGB colors) instead of the standard 3B,
# and uses 3F_sd (scenario with instance color mapping) instead of the
# standard 3F.
#
# Note: 3A is NOT run here, it was already run by step2_sd.sh.
#
# Prerequisites (run BEFORE this script):
#   - Step 1 (1_extract_ROS_data/step1.sh)
#   - Step 2 SD (2_process_datasets/step2_sd.sh)
#     which also runs 3A and produces trajectory_positions_rear_odom_yaw.json

set -e

SCRIPTS=(
    "3_generate_simulation_data/3B_sd_transform_parked_vehicles_to_carla.py"
    "3_generate_simulation_data/3C_setup_carla.py"
    "3_generate_simulation_data/3F_sd_generate_carla_scenario.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running Script $SCRIPT"
    if [ "$SCRIPT" != "3_generate_simulation_data/3C_setup_carla.py" ]; then
        python3 "$SCRIPT"
        if [ $? -ne 0 ]; then
            echo "Error in $SCRIPT. Aborting."
            exit 1
        fi
    else
        python3 "$SCRIPT" &
        PID=$!
        sleep 5
    fi
done

echo ""
echo "All Scripts for Step 3 (Stable Diffusion branch) completed successfully"