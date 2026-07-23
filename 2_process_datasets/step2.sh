#!/usr/bin/env bash
set -e

# =============================================================================
# step2.sh
#
# Process one extracted dataset for the Gaussian Splatting branch.
#
# Reads from:
#   data/raw_dataset/<BAG>/
#
# Writes to:
#   data/processed_dataset/<BAG>/
#   data/data_for_gaussian_splatting/<BAG>/
#
# Existing complete map and semantic-map outputs are reused. Map generation is
# retried up to 10 times when the Overpass service returns a transient error.
#
# Parameters:
#   <bag_name.bag>
#       Bag filename whose extracted dataset should be processed.
#
#   -r, --refinement
#       Open the interactive LiDAR refinement window in 2B.
#
#   --conda-env <ENV>
#       Conda environment to activate.
#       Default: data_extraction.
#
# Usage:
#   bash 2_process_datasets/step2.sh snowy.bag
#   bash 2_process_datasets/step2.sh snowy.bag --refinement
#   bash 2_process_datasets/step2.sh snowy.bag --conda-env data_extraction
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

cd "$PROJECT_ROOT"

# ---------- Parse args ----------

REFINEMENT=false
CONDA_ENV="data_extraction"
FRAME_SKIP=2
NUM_SPLITS=3
OVERLAP_FRAMES=100
BAG_NAME=""

usage() {
    echo "Usage: $0 <bag_name.bag> [options]"
    echo ""
    echo "Arguments:"
    echo "  <bag_name.bag>      Bag filename including .bag extension"
    echo ""
    echo "Options:"
    echo "  -r, --refinement    Open the interactive LiDAR refinement window"
    echo "  --conda-env ENV     Conda environment to activate (default: $CONDA_ENV)"
    echo "  --frame-skip N      2E frame skip (default: $FRAME_SKIP)"
    echo "  --num-splits N      2E split count (default: $NUM_SPLITS)"
    echo "  --overlap-frames N  2E overlap (default: $OVERLAP_FRAMES)"
    echo "  -h, --help          Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--refinement)
            REFINEMENT=true
            shift
            ;;
        --conda-env)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "[ERROR] --conda-env requires an environment name."
                usage
                exit 1
            fi
            CONDA_ENV="$2"
            shift 2
            ;;
        --frame-skip) FRAME_SKIP="$2"; shift 2 ;;
        --num-splits) NUM_SPLITS="$2"; shift 2 ;;
        --overlap-frames) OVERLAP_FRAMES="$2"; shift 2 ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [ -z "$BAG_NAME" ]; then
                BAG_NAME="$1"
            else
                echo "ERROR: Unexpected extra argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$BAG_NAME" ]; then
    echo "ERROR: Missing bag name."
    usage
    exit 1
fi

BAG_STEM="${BAG_NAME%.bag}"
DATASET_DIR="${PROJECT_ROOT}/data/raw_dataset/${BAG_STEM}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "[ERROR] Extracted dataset not found: $DATASET_DIR"
    echo "        Run Step 1 for $BAG_NAME before running this script."
    exit 1
fi

# ---------- Conda init ----------

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    echo "        Initialize conda first or run this script from a shell where conda is available."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "=========================================="
echo "Step 2 (GS branch) for bag: $BAG_NAME"
echo "Refinement mode: $REFINEMENT"
echo "=========================================="

# ---------- Activate data_extraction env ----------
echo ""
echo "--- Activating conda env: data_extraction ---"
conda activate "$CONDA_ENV"

# ---------- Pick 2B variant ----------

SCRIPT_2B="2_process_datasets/2B_lidar_parked_cars_detection.py"

# ---------- Model downloads (infrastructure, not bag-specific) ----------

FCOS3D_FILE="2_process_datasets/utils/fcos3d.pth"
POINTPILLARS_FILE="2_process_datasets/utils/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"

if [ ! -f "$FCOS3D_FILE" ]; then
    echo "Downloading FCOS3D"
    gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O "$FCOS3D_FILE"
fi

if [ ! -f "$POINTPILLARS_FILE" ]; then
    echo "Downloading PointPillars"
    gdown 1AGOR8C0tDUsWSSWTEc0fA7kysIE9-iol -O "$POINTPILLARS_FILE"
fi

# ---------- Run 2A, 2B, 2C, 2E, 2F ----------

run_2c() {
    local attempt
    local map_osm="data/processed_dataset/${BAG_STEM}/maps/map.osm"
    local map_xodr="data/processed_dataset/${BAG_STEM}/maps/map.xodr"
    local map_args=(--bag-name "$BAG_NAME")

    if [[ -s "$map_osm" && -s "$map_xodr" ]]; then
        echo "[INFO] OSM and OpenDRIVE maps already exist. Skipping 2C."
        return 0
    fi

    if [[ -s "$map_osm" ]]; then
        echo "[INFO] Existing OSM map found. Reusing it without downloading."
        map_args+=(--skip-fetch)
    fi

    for attempt in {1..10}; do
        echo "[INFO] 2C attempt $attempt of 10."

        if python3 "2_process_datasets/2C_create_map.py" "${map_args[@]}"; then
            return 0
        fi

        if [[ "$attempt" -eq 10 ]]; then
            echo "[ERROR] 2C_create_map.py failed after 10 attempts for $BAG_NAME."
            echo "        Check the Overpass service and try again later."
            return 1
        fi

        echo "[WARN] 2C failed. Retrying in 10 seconds."
        sleep 10
    done
}

run_2f() {
    local images_dir="data/raw_dataset/${BAG_STEM}/images"
    local semantic_dir="data/processed_dataset/${BAG_STEM}/semantic_maps"
    local image_count
    local semantic_count

    image_count="$(find "$images_dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
    semantic_count=0

    if [[ -d "$semantic_dir" ]]; then
        semantic_count="$(find "$semantic_dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
    fi

    if [[ "$image_count" -gt 0 && "$semantic_count" -eq "$image_count" ]]; then
        echo "[INFO] All $semantic_count segmentation maps already exist. Skipping 2F."
        return 0
    fi

    echo "[INFO] Semantic maps are missing or incomplete ($semantic_count/$image_count)."
    python3 "2_process_datasets/2F_extract_semantic_maps.py" --bag-name "$BAG_NAME"
}

SCRIPTS=(
    "2_process_datasets/2A_camera_parked_cars_detection.py"
    "$SCRIPT_2B"
    "2_process_datasets/2C_create_map.py"
    "2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py"
    "2_process_datasets/2F_extract_semantic_maps.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    echo ""
    echo "--- Running $SCRIPT ---"
    if [[ "$SCRIPT" == "2_process_datasets/2C_create_map.py" ]]; then
        run_2c
    elif [[ "$SCRIPT" == "2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py" ]]; then
        python3 "$SCRIPT" --bag-name "$BAG_NAME" --frame-skip "$FRAME_SKIP" --num-splits "$NUM_SPLITS" --overlap-frames "$OVERLAP_FRAMES"
    elif [[ "$SCRIPT" == "2_process_datasets/2F_extract_semantic_maps.py" ]]; then
        run_2f
    elif [[ "$SCRIPT" == "$SCRIPT_2B" && "$REFINEMENT" == true ]]; then
        python3 "$SCRIPT" --bag-name "$BAG_NAME" --vis
    else
        python3 "$SCRIPT" --bag-name "$BAG_NAME"
    fi
done

# ---------- 2G: sidewalk fix on OpenDRIVE map ----------

echo ""
echo "--- Running 2G_OPT_fix_sidewalk.sh ---"
bash 2_process_datasets/2G_OPT_fix_sidewalk.sh "$BAG_NAME"

# ---------- Done ----------

conda deactivate

echo ""
echo "=========================================="
echo "Step 2 (GS branch) completed for $BAG_NAME"
echo "=========================================="