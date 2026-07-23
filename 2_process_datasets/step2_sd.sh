#!/usr/bin/env bash
set -e

# =============================================================================
# step2_sd.sh
#
# Process one extracted dataset for the Stable Diffusion branch.
#
# Reads from:
#   data/raw_dataset/<BAG>/
#   data/processed_dataset/<BAG>/lidar_detections/  # when available
#
# Writes to:
#   data/processed_dataset/<BAG>/
#   data/data_for_carla/<BAG>/
#   data/data_for_stable_diffusion/<BAG>/
#
# Existing complete map and semantic-map outputs are reused. Map generation is
# retried up to 10 times when the Overpass service returns a transient error.
# When camera and LiDAR clusters both exist, 2D matches them and opens its GUI
# unless --no-refinement-window is supplied.
#
# Parameters:
#   <bag_name.bag>
#       Bag filename whose extracted dataset should be processed.
#
#   --conda-env <ENV>
#       Conda environment used by 2A_sd, 2C, 2D, 2F, and 3A.
#       Default: data_extraction.
#
#   --no-refinement-window
#       Save the automatic 2D matching result without opening the GUI.
#
# Usage:
#   bash 2_process_datasets/step2_sd.sh snowy.bag
#   bash 2_process_datasets/step2_sd.sh snowy.bag --no-refinement-window
#   bash 2_process_datasets/step2_sd.sh snowy.bag --conda-env data_extraction
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

cd "$PROJECT_ROOT"

# ---------- Defaults ----------
CONDA_ENV="data_extraction"
NO_REFINEMENT_WINDOW=false
BAG_NAME=""

# ---------- Usage ----------
usage() {
    echo "Usage: $0 <bag_name.bag> [options]"
    echo ""
    echo "Arguments:"
    echo "  <bag_name.bag>      Bag filename including .bag extension"
    echo ""
    echo "Options:"
    echo "  --conda-env ENV     Conda environment to activate (default: $CONDA_ENV)"
    echo "  --no-refinement-window"
    echo "                      Save automatic 2D matching without opening the GUI"
    echo "  -h, --help          Show this help message"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

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
        --no-refinement-window)
            NO_REFINEMENT_WINDOW=true
            shift
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
echo "Step 2 (SD branch) for bag: $BAG_NAME"
echo "=========================================="

# ---------- Activate data_extraction env ----------
echo ""
echo "--- Activating conda env: data_extraction ---"
conda activate "$CONDA_ENV"

# ---------- Model downloads (infrastructure, not bag-specific) ----------

# YOLO segmentation model (needed by 2A_sd)
YOLO_FILE="2_process_datasets/utils/yolov8n-seg.pt"
if [ ! -f "$YOLO_FILE" ]; then
    echo "Downloading YOLOv8n-seg"
    wget -O "$YOLO_FILE" \
        https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
fi

# FCOS3D model (needed by 2A_sd via 2A base logic)
FCOS3D_FILE="2_process_datasets/utils/fcos3d.pth"
if [ ! -f "$FCOS3D_FILE" ]; then
    echo "Downloading FCOS3D"
    gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O "$FCOS3D_FILE"
fi

# ---------- Run 2A_sd, 2C, 2F ----------

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
    "2_process_datasets/2A_sd_camera_parked_cars_detection.py"
    "2_process_datasets/2C_create_map.py"
    "2_process_datasets/2F_extract_semantic_maps.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    echo ""
    echo "--- Running $SCRIPT ---"
    if [[ "$SCRIPT" == "2_process_datasets/2C_create_map.py" ]]; then
        run_2c
    elif [[ "$SCRIPT" == "2_process_datasets/2F_extract_semantic_maps.py" ]]; then
        run_2f
    else
        python3 "$SCRIPT" --bag-name "$BAG_NAME"
    fi
done

# ---------- 2D: camera-LiDAR cluster matching ----------

CAMERA_CLUSTERS="data/processed_dataset/${BAG_STEM}/camera_detections/unified_clusters.txt"
LIDAR_CLUSTERS="data/processed_dataset/${BAG_STEM}/lidar_detections/unified_clusters.txt"

if [[ -s "$CAMERA_CLUSTERS" && -s "$LIDAR_CLUSTERS" ]]; then
    echo ""
    echo "--- Running 2D_refine_clusters.py ---"

    REFINEMENT_ARGS=(
        --bag-name "$BAG_NAME"
        --source camera
        --match
    )

    if [[ "$NO_REFINEMENT_WINDOW" == true ]]; then
        REFINEMENT_ARGS+=(--no-window)
    fi

    python3 "2_process_datasets/2D_refine_clusters.py" "${REFINEMENT_ARGS[@]}"
else
    echo "[INFO] Skipping 2D because camera and LiDAR clusters are not both available."
fi

# ---------- 2G: sidewalk fix on OpenDRIVE map ----------

echo ""
echo "--- Running 2G_OPT_fix_sidewalk.sh ---"
bash 2_process_datasets/2G_OPT_fix_sidewalk.sh "$BAG_NAME"

# ---------- 3A: transform coordinates to CARLA (produces trajectory needed by 2H) ----------

SCRIPT_3A="3_generate_simulation_data/3A_transform_coordinates_to_carla.py"
echo ""
echo "--- Running $SCRIPT_3A ---"
python3 "$SCRIPT_3A" --bag-name "$BAG_NAME"

# ---------- Sanity checks before 2H ----------

TRAJ_FILE="data/data_for_carla/${BAG_STEM}/trajectory_positions_rear_odom_yaw.json"
if [ ! -f "$TRAJ_FILE" ]; then
    echo ""
    echo "ERROR: $TRAJ_FILE not found after 3A ran. Something went wrong."
    exit 1
fi

INSTANCE_MAPS_DIR="data/processed_dataset/${BAG_STEM}/camera_detections/instance_maps"
if [ ! -d "$INSTANCE_MAPS_DIR" ]; then
    echo ""
    echo "ERROR: $INSTANCE_MAPS_DIR not found."
    echo "       2A_sd did not produce instance maps. Check the script output above."
    exit 1
fi

# ---------- Switch to stable_diff env for 2H ----------

echo ""
echo "--- Switching conda env: data_extraction -> stable_diff ---"
conda deactivate
conda activate stable_diff

# ---------- 2H: prepare HuggingFace Arrow dataset for SD training ----------
# 2H requires the `stable_diff` env (HuggingFace `datasets` is not in `data_extraction`).

SCRIPT_2H="2_process_datasets/2H_prepare_dataset_for_stable_diffusion.py"
echo ""
echo "--- Running $SCRIPT_2H ---"
python3 "$SCRIPT_2H" --bag-name "$BAG_NAME"

conda deactivate

echo ""
echo "=========================================="
echo "Step 2 (SD branch) completed for $BAG_NAME"
echo "=========================================="