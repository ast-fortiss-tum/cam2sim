#!/usr/bin/env bash
set -e

# =============================================================================
# step3.sh
#
# Generate one Gaussian Splatting CARLA scenario.
#
# Reads from:
#   data/raw_dataset/<BAG>/images_positions.txt
#   data/processed_dataset/<BAG>/maps/
#   data/processed_dataset/<BAG>/lidar_detections/unified_clusters.txt
#
# Writes to:
#   data/data_for_carla/<BAG>/
#   CARLA world state (map, hero vehicle, and parked vehicles)
#
# Parameters:
#   <bag_name.bag>
#       Bag filename whose scenario should be generated.
#
# Required Conda environment:
#   data_extraction
#
# Usage:
#   bash 3_generate_simulation_data/step3.sh snowy.bag
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "$PROJECT_ROOT"

# ---------- Usage ----------
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <bag_name.bag>"
    echo "Example: $0 reference_bag.bag"
    exit 1
fi

BAG_NAME="$1"
DATASET_DIR="${PROJECT_ROOT}/data/raw_dataset/${BAG_NAME%.bag}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[ERROR] Extracted dataset not found: $DATASET_DIR"
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "=========================================="
echo "Step 3 (GS branch) for bag: $BAG_NAME"
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

# ---------- 3A: transform coordinates to CARLA ----------

SCRIPT_3A="3_generate_simulation_data/3A_transform_coordinates_to_carla.py"
echo ""
echo "--- Running $SCRIPT_3A ---"
python3 "$SCRIPT_3A" --bag-name "$BAG_NAME"

# ---------- 3B: parked vehicles -> CARLA ----------

SCRIPT_3B="3_generate_simulation_data/3B_transform_parked_vehicles_to_carla.py"
echo ""
echo "--- Running $SCRIPT_3B ---"
python3 "$SCRIPT_3B" --bag-name "$BAG_NAME"

# ---------- 3C: launch CARLA ----------
# Infrastructure script (no bag dependency). Runs in background; we wait until
# the RPC port becomes reachable before proceeding to 3F.

SCRIPT_3C="3_generate_simulation_data/3C_setup_carla.py"
echo ""
echo "--- Running $SCRIPT_3C (background) ---"
python3 "$SCRIPT_3C" &
CARLA_PID=$!

if ! wait_for_carla; then
    echo "CARLA never became ready. Aborting."
    kill $CARLA_PID 2>/dev/null
    exit 1
fi

# ---------- 3F: spawn hero + parked cars in CARLA ----------

SCRIPT_3F="3_generate_simulation_data/3F_generate_carla_scenario.py"
echo ""
echo "--- Running $SCRIPT_3F ---"
python3 "$SCRIPT_3F" --bag-name "$BAG_NAME"

# ---------- Done ----------

conda deactivate

echo ""
echo "=========================================="
echo "Step 3 (GS branch) completed for $BAG_NAME"
echo "=========================================="
echo ""
echo "[NOTE] CARLA is still running in background (PID $CARLA_PID)."
echo "       Kill it manually when you're done, or use:"
echo "       kill $CARLA_PID"