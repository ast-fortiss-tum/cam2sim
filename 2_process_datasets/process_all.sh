#!/usr/bin/env bash
set -e

# =============================================================================
# process_all.sh
#
# Run the Gaussian Splatting and Stable Diffusion processing pipelines for all
# datasets used in the paper. Stable Diffusion cluster matching runs without
# opening the 2D refinement window.
#
# Reads from:
#   data/raw_dataset/sunny/
#   data/raw_dataset/snowy/
#   data/raw_dataset/cloudy/
#
# Writes to:
#   data/processed_dataset/{sunny,snowy,cloudy}/
#   data/data_for_gaussian_splatting/{sunny,snowy,cloudy}/
#   data/data_for_carla/{sunny,snowy,cloudy}/
#   data/data_for_stable_diffusion/{sunny,snowy,cloudy}/
#
# Parameters:
#   --conda-env <ENV>
#       Conda environment used by the processing scripts.
#       Default: data_extraction.
#
# Usage:
#   bash 2_process_datasets/process_all.sh
#   bash 2_process_datasets/process_all.sh --conda-env data_extraction
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

cd "$PROJECT_ROOT"

CONDA_ENV="data_extraction"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --conda-env ENV         Conda environment to use (default: $CONDA_ENV)
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "[ERROR] --conda-env requires an environment name."
                usage
                exit 1
            fi
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            echo "[ERROR] Unexpected argument: $1"
            usage
            exit 1
            ;;
    esac
done

BAGS=(
    "sunny.bag"
    "snowy.bag"
    "cloudy.bag"
)

for BAG_NAME in "${BAGS[@]}"; do
    BAG_STEM="${BAG_NAME%.bag}"
    DATASET_DIR="${PROJECT_ROOT}/data/raw_dataset/${BAG_STEM}"

    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "[ERROR] Extracted dataset not found: $DATASET_DIR"
        echo "        Run Step 1 for $BAG_NAME before running this script."
        exit 1
    fi
done

echo "=========================================="
echo "STEP 2: PROCESS ALL PAPER DATASETS"
echo "=========================================="
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Bags:         ${BAGS[*]}"
echo "[INFO] Conda env:    $CONDA_ENV"
echo "=========================================="

for BAG_NAME in "${BAGS[@]}"; do
    echo ""
    if [[ "$BAG_NAME" == "snowy.bag" ]]; then NUM_SPLITS=2; else NUM_SPLITS=3; fi
    echo "=========================================="
    echo "[INFO] Processing Gaussian Splatting data: $BAG_NAME"
    echo "=========================================="

    bash "${SCRIPT_DIR}/step2.sh" \
        "$BAG_NAME" \
        --conda-env "$CONDA_ENV" \
        --frame-skip 2 \
        --num-splits "$NUM_SPLITS" \
        --overlap-frames 100

    echo ""
    echo "=========================================="
    echo "[INFO] Processing Stable Diffusion data: $BAG_NAME"
    echo "=========================================="

    bash "${SCRIPT_DIR}/step2_sd.sh" \
        "$BAG_NAME" \
        --conda-env "$CONDA_ENV" \
        --no-refinement-window
done

echo ""
echo "=========================================="
echo "[OK] Step 2 completed successfully for all paper datasets."
echo "=========================================="
