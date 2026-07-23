#!/usr/bin/env bash
set -e

# =============================================================================
# train_all.sh
#
# Run the complete stage-4 preparation for all paper scenarios.
#
# Reads from:
#   data/data_for_gaussian_splatting/{sunny,snowy,cloudy}/
#   data/data_for_stable_diffusion/{sunny,snowy,cloudy}/hf_binary/
#
# Writes to:
#   data/data_for_gaussian_splatting/{sunny,snowy,cloudy}/colmap/
#   data/data_for_gaussian_splatting/{sunny,snowy,cloudy}/outputs/
#   data/stable_diff_models/ by default
#
# Parameters:
#   -big, --big
#       Train splatfacto-big and nerfacto-big instead of base variants.
#
# Required Conda environments:
#   nerfstudio
#   stable_diff
#
# Usage:
#   bash 4_gaussian_splatting_preparation/train_all.sh
#   bash 4_gaussian_splatting_preparation/train_all.sh --big
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "$PROJECT_ROOT"

BIG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -big|--big)
            BIG=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-big|--big]"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$BIG" == true ]]; then
    METHODS=("splatfacto-big" "nerfacto-big")
else
    METHODS=("splatfacto" "nerfacto")
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

BAGS=("sunny.bag" "snowy.bag" "cloudy.bag")
for BAG_NAME in "${BAGS[@]}"; do
    BAG_STEM="${BAG_NAME%.bag}"
    [[ -d "data/data_for_gaussian_splatting/${BAG_STEM}" ]] || { echo "[ERROR] Missing GS data for $BAG_NAME"; exit 1; }
    [[ -d "data/data_for_stable_diffusion/${BAG_STEM}/hf_binary" ]] || { echo "[ERROR] Missing SD data for $BAG_NAME"; exit 1; }
done

for BAG_NAME in "${BAGS[@]}"; do
    if [[ "$BAG_NAME" == "snowy.bag" ]]; then NUM_PARTS=2; else NUM_PARTS=3; fi

    bash "${SCRIPT_DIR}/4A_colmap_auto.sh" "$BAG_NAME"
    for METHOD in "${METHODS[@]}"; do
        echo "[INFO] Training $METHOD models for $BAG_NAME"
        bash "${SCRIPT_DIR}/4B_train_gaussian_splatting.sh" "$BAG_NAME" --method "$METHOD"
    done

    conda activate stable_diff
    python3 "${SCRIPT_DIR}/4B_train_stable_diff.py" --bag-name "$BAG_NAME" --num-parts "$NUM_PARTS"
    conda deactivate
done

echo "[OK] Stage 4 completed for all paper scenarios."
