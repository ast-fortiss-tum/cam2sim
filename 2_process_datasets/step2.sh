#!/bin/bash
#
# step2.sh
#
# Step 2 - Gaussian Splatting branch
# Runs the GS-specific data processing pipeline (2A + 2B + 2C + 2E + 2F + 2G).
#
# Prerequisites (run BEFORE this script):
#   - Step 1 (1_extract_ROS_data/step1.sh <bag_name.bag>)
#   - Conda env: data_extraction

set -e

# ---------- Conda init ----------
# Required for `conda activate` to work in non-interactive scripts.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- Parse args ----------

REFINEMENT=false
BAG_NAME=""

usage() {
    echo "Usage: $0 <bag_name.bag> [-r|--refinement]"
    echo ""
    echo "Arguments:"
    echo "  <bag_name.bag>      Bag filename including .bag extension"
    echo ""
    echo "Options:"
    echo "  -r, --refinement    Use 2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py"
    echo "                      instead of 2B_lidar_parked_cars_detection.py"
    echo "  -h, --help          Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--refinement)
            REFINEMENT=true
            shift
            ;;
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

echo "=========================================="
echo "Step 2 (GS branch) for bag: $BAG_NAME"
echo "Refinement mode: $REFINEMENT"
echo "=========================================="

# ---------- Activate data_extraction env ----------
echo ""
echo "--- Activating conda env: data_extraction ---"
conda activate data_extraction

# ---------- Pick 2B variant ----------

if [ "$REFINEMENT" = true ]; then
    SCRIPT_2B="2_process_datasets/2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py"
    echo "[INFO] Refinement mode: using 2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py"
else
    SCRIPT_2B="2_process_datasets/2B_lidar_parked_cars_detection.py"
    echo "[INFO] Standard mode: using 2B_lidar_parked_cars_detection.py"
fi

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

SCRIPTS=(
    "2_process_datasets/2A_camera_parked_cars_detection.py"
    "$SCRIPT_2B"
    "2_process_datasets/2C_create_map_from_coordinates_auto.py"
    "2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py"
    "2_process_datasets/2F_extract_semantic_maps.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    echo ""
    echo "--- Running $SCRIPT ---"
    python3 "$SCRIPT" --bag-name "$BAG_NAME"
done

# ---------- 2G: sidewalk fix on OpenDRIVE map ----------

echo ""
echo "--- Running 2G_OPT_fix_sidewalk.sh ---"
chmod +x 2_process_datasets/2G_OPT_fix_sidewalk.sh
bash 2_process_datasets/2G_OPT_fix_sidewalk.sh "$BAG_NAME"

# ---------- Done ----------

conda deactivate

echo ""
echo "=========================================="
echo "Step 2 (GS branch) completed for $BAG_NAME"
echo "=========================================="