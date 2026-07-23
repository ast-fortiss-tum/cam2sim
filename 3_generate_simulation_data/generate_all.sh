#!/usr/bin/env bash
set -e

# =============================================================================
# generate_all.sh
#
# Generate standard and Stable Diffusion parked-vehicle data for every paper
# scenario. Stage 3A is not repeated because process_all.sh already runs it.
#
# Reads from:
#   data/processed_dataset/{sunny,snowy,cloudy}/maps/
#   data/processed_dataset/{sunny,snowy,cloudy}/lidar_detections/
#   data/processed_dataset/{sunny,snowy,cloudy}/camera_detections/
#   data/data_for_carla/{sunny,snowy,cloudy}/vehicle_data.json
#
# Writes to:
#   data/data_for_carla/{sunny,snowy,cloudy}/vehicle_data.json
#
# Parameters:
#   None.
#
# Required Conda environment:
#   data_extraction
#
# Usage:
#   bash 3_generate_simulation_data/generate_all.sh
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0"
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate data_extraction

BAGS=("sunny.bag" "snowy.bag" "cloudy.bag")

for BAG_NAME in "${BAGS[@]}"; do
    BAG_STEM="${BAG_NAME%.bag}"
    if [[ ! -d "data/processed_dataset/${BAG_STEM}" || ! -f "data/data_for_carla/${BAG_STEM}/vehicle_data.json" ]]; then
        echo "[ERROR] Required processed data not found for $BAG_NAME."
        echo "        Run 2_process_datasets/process_all.sh first."
        exit 1
    fi
done

for BAG_NAME in "${BAGS[@]}"; do
    echo "[INFO] Generating standard parked-vehicle data: $BAG_NAME"
    python3 "${SCRIPT_DIR}/3B_transform_parked_vehicles_to_carla.py" --bag-name "$BAG_NAME"

    echo "[INFO] Generating SD parked-vehicle data: $BAG_NAME"
    python3 "${SCRIPT_DIR}/3B_sd_transform_parked_vehicles_to_carla.py" --bag-name "$BAG_NAME"
done

conda deactivate
echo "[OK] Stage 3 data generated for all paper scenarios."
