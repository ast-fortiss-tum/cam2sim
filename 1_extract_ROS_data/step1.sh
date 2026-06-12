#!/usr/bin/env bash
set -e

# =============================================================================
# step1.sh
#
# Extract raw sensor and vehicle data from a ROS bag.
#
# Reads from:
#   data/raw_ros_data/<BAG>.bag
#
# Writes to:
#   data/raw_dataset/<BAG>/
#
# Parameters:
#   <bag_name.bag>
#       ROS bag filename to process.
#
#   --no-odom
#       Extract camera and LiDAR without odometry synchronization.
#       1C_poses_and_trajectory.py is skipped.
#
#   --conda-env <ENV>
#       Conda environment to activate.
#       Default: data_extraction.
#
# Usage:
#   bash 1_extract_ROS_data/step1.sh snowy.bag
#   bash 1_extract_ROS_data/step1.sh snowy.bag --no-odom
#   bash 1_extract_ROS_data/step1.sh snowy.bag --conda-env data_extraction
# =============================================================================

# ---------- Paths ----------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

cd "$PROJECT_ROOT"

# ---------- Defaults ----------

CONDA_ENV="data_extraction"
NO_ODOM=false
BAG_NAME=""

# ---------- Usage ----------

usage() {
    cat <<EOF
Usage: $0 <bag_name.bag> [options]

Arguments:
  <bag_name.bag>          Bag filename including .bag extension

Options:
  --no-odom               Extract camera/LiDAR without odometry sync and skip 1C
  --conda-env ENV         Conda environment to activate (default: $CONDA_ENV)
  -h, --help              Show this help message
EOF
}

# ---------- Args ----------

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-odom)
            NO_ODOM=true
            shift
            ;;
        --conda-env)
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
            if [[ -z "$BAG_NAME" ]]; then
                BAG_NAME="$1"
            else
                echo "[ERROR] Unexpected extra argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$BAG_NAME" ]]; then
    echo "[ERROR] Missing bag name."
    usage
    exit 1
fi

BAG_PATH="${PROJECT_ROOT}/data/raw_ros_data/${BAG_NAME}"

if [[ ! -f "$BAG_PATH" ]]; then
    echo "[ERROR] Bag file not found: $BAG_PATH"
    echo "        Place the bag manually under data/raw_ros_data/ before running."
    exit 1
fi

# ---------- Conda activation ----------

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    echo "        Initialize conda first or run this script from a shell where conda is available."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[INFO] Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

# ---------- Banner ----------

echo "=========================================="
echo "STEP 1: ROS DATA EXTRACTION"
echo "=========================================="
echo "[INFO] Project root:  $PROJECT_ROOT"
echo "[INFO] Bag name:      $BAG_NAME"
echo "[INFO] Bag path:      $BAG_PATH"
echo "[INFO] Conda env:     $CONDA_ENV"
echo "[INFO] No odom mode:  $NO_ODOM"
echo "=========================================="

# ---------- Run scripts ----------

if [[ "$NO_ODOM" == true ]]; then
    echo ""
    echo "--- Running 1A_extract_camera.py without odometry ---"
    python "1_extract_ROS_data/1A_extract_camera.py" \
        --bag-name "$BAG_NAME" \
        --no-odom

    echo ""
    echo "--- Running 1B_extract_lidar.py without odometry ---"
    python "1_extract_ROS_data/1B_extract_lidar.py" \
        --bag-name "$BAG_NAME" \
        --no-odom

    echo ""
    echo "[INFO] Skipping 1C_poses_and_trajectory.py because --no-odom is enabled."

else
    echo ""
    echo "--- Running 1A_extract_camera.py ---"
    python "1_extract_ROS_data/1A_extract_camera.py" \
        --bag-name "$BAG_NAME"

    echo ""
    echo "--- Running 1B_extract_lidar.py ---"
    python "1_extract_ROS_data/1B_extract_lidar.py" \
        --bag-name "$BAG_NAME"

    echo ""
    echo "--- Running 1C_poses_and_trajectory.py ---"
    python "1_extract_ROS_data/1C_poses_and_trajectory.py" \
        --bag-name "$BAG_NAME"
fi

echo ""
echo "--- Running 1D_steering_status.py ---"
python "1_extract_ROS_data/1D_steering_status.py" \
    --bag-name "$BAG_NAME"

echo ""
echo "--- Running 1E_model_output.py ---"
python "1_extract_ROS_data/1E_model_output.py" \
    --bag-name "$BAG_NAME"

# ---------- Done ----------

echo ""
echo "=========================================="
echo "[OK] Step 1 completed successfully for $BAG_NAME"
echo "=========================================="

conda deactivate