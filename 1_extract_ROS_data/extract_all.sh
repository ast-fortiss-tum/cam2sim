#!/usr/bin/env bash
set -e

# =============================================================================
# extract_all.sh
#
# Run step1.sh for all ROS bags used in the paper.
#
# Expected input bags:
#   data/raw_ros_data/sunny.bag
#   data/raw_ros_data/snowy.bag
#   data/raw_ros_data/cloudy.bag
#
# All bags are expected to contain odometry, so step1.sh is called without
# --no-odom.
#
# Usage:
#   bash 1_extract_ROS_data/extract_all.sh
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

cd "$PROJECT_ROOT"

BAGS=(
    "sunny.bag"
    "snowy.bag"
    "cloudy.bag"
)

echo "=========================================="
echo "STEP 1: EXTRACT ALL PAPER BAGS"
echo "=========================================="
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Bags:         ${BAGS[*]}"
echo "=========================================="

for BAG_NAME in "${BAGS[@]}"; do
    BAG_PATH="${PROJECT_ROOT}/data/raw_ros_data/${BAG_NAME}"

    echo ""
    echo "=========================================="
    echo "[INFO] Processing bag: $BAG_NAME"
    echo "=========================================="

    if [[ ! -f "$BAG_PATH" ]]; then
        echo "[ERROR] Bag file not found: $BAG_PATH"
        echo "        Place the bag under data/raw_ros_data/ before running."
        exit 1
    fi

    bash "${SCRIPT_DIR}/step1.sh" "$BAG_NAME"
done

echo ""
echo "=========================================="
echo "[OK] Step 1 completed successfully for all paper bags."
echo "=========================================="