#!/bin/bash
#
# step3_sd.sh
#
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
#   - Step 1 (1_extract_ROS_data/step1.sh <bag>)
#   - Step 2 SD (2_process_datasets/step2_sd.sh <bag>)
#     which also runs 3A and produces trajectory_positions_rear_odom_yaw.json
#   - Conda env: data_extraction

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
echo "Step 3 (SD branch) for bag: $BAG_NAME"
echo "=========================================="

# ---------- Activate data_extraction env ----------
echo ""
echo "--- Activating conda env: data_extraction ---"
conda activate data_extraction

# ---------- CARLA RPC settings ----------

CARLA_HOST="127.0.0.1"
CARLA_PORT="2000"
CARLA_WAIT_TIMEOUT=120

wait_for_carla() {
    echo "Waiting for CARLA RPC on ${CARLA_HOST}:${CARLA_PORT} ..."
    local elapsed=0

    while ! (echo > /dev/tcp/${CARLA_HOST}/${CARLA_PORT}) 2>/dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))

        if [ $elapsed -ge $CARLA_WAIT_TIMEOUT ]; then
            echo "Timed out waiting for CARLA after ${CARLA_WAIT_TIMEOUT}s"
            return 1
        fi

        echo "  ... still waiting (${elapsed}s)"
    done

    echo "CARLA is up (RPC port reachable)."
    # Small cushion to let CARLA finish initialization after port opens
    sleep 3
    return 0
}

# ---------- 3B_sd: parked vehicles with RGB colors -> CARLA ----------

SCRIPT_3B_SD="3_generate_simulation_data/3B_sd_transform_parked_vehicles_to_carla.py"
echo ""
echo "--- Running $SCRIPT_3B_SD ---"
python3 "$SCRIPT_3B_SD" --bag-name "$BAG_NAME"

# ---------- 3C_sd: launch CARLA with custom Glass override ----------
# Infrastructure script (no bag dependency). Runs in background; we wait until
# the RPC port becomes reachable before proceeding to 3F_sd.

SCRIPT_3C_SD="3_generate_simulation_data/3C_sd_setup_carla.py"
echo ""
echo "--- Running $SCRIPT_3C_SD (background) ---"
python3 "$SCRIPT_3C_SD" &
CARLA_PID=$!

if ! wait_for_carla; then
    echo "CARLA never became ready. Aborting."
    kill $CARLA_PID 2>/dev/null
    exit 1
fi

# ---------- 3F_sd: spawn hero + parked cars + build instance color map ----------

SCRIPT_3F_SD="3_generate_simulation_data/3F_sd_generate_carla_scenario.py"
echo ""
echo "--- Running $SCRIPT_3F_SD ---"
python3 "$SCRIPT_3F_SD" --bag-name "$BAG_NAME"

# ---------- Done ----------

conda deactivate

echo ""
echo "=========================================="
echo "Step 3 (SD branch) completed for $BAG_NAME"
echo "=========================================="
echo ""
echo "[NOTE] CARLA is still running in background (PID $CARLA_PID)."
echo "       Kill it manually when you're done, or use:"
echo "       kill $CARLA_PID"