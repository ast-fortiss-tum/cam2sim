#!/usr/bin/env bash
# =============================================================================
# 4A_colmap_auto.sh
#
# Reconstruct COLMAP models for every Gaussian Splatting split.
#
# Reads from:
#   data/data_for_gaussian_splatting/<BAG>/images_gs_split_*_1_of_*/
#   data/data_for_gaussian_splatting/<BAG>/sky_masks_gs_split_*_1_of_*/
#
# Writes to:
#   data/data_for_gaussian_splatting/<BAG>/colmap/
#
# Parameters:
#   <bag_name.bag>
#       Bag filename whose splits should be reconstructed.
#   --camera-model <MODEL>
#       COLMAP camera model. Default: OPENCV.
#   --camera-params <VALUES>
#       Comma-separated camera intrinsics.
#   --no-masks
#       Run COLMAP without sky masks.
#
# Required Conda environment:
#   nerfstudio
#
# Usage:
#   bash 4_gaussian_splatting_preparation/4A_colmap_auto.sh snowy.bag
#   bash 4_gaussian_splatting_preparation/4A_colmap_auto.sh snowy.bag --no-masks
# =============================================================================

set -e

# ---------- Paths ----------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

cd "$PROJECT_ROOT"

# ---------- Conda init ----------

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- Defaults ----------

SEQUENTIAL_OVERLAP=10
CONDA_ENV="nerfstudio"

# OPENCV intrinsics for front narrow camera.
# Order: fx, fy, cx, cy, k1, k2, p1, p2
CAMERA_MODEL="OPENCV"
CAMERA_PARAMS="772.906855,777.596896,424.980372,258.452509,-0.274231,0.034838,0.00226,-0.000972"

# Use sky masks if present.
USE_SKY_MASKS=true

# Mapper: keep intrinsics fixed.
BA_REFINE_FOCAL_LENGTH=0
BA_REFINE_PRINCIPAL_POINT=0
BA_REFINE_EXTRA_PARAMS=0

# ---------- Usage / args ----------

usage() {
    cat <<EOF
Usage: $0 <bag_name.bag> [options]

Arguments:
  <bag_name.bag>            Bag filename including .bag extension

Options:
  --camera-model MODEL      COLMAP camera model (default: $CAMERA_MODEL)
  --camera-params "p1,..."  Camera intrinsics
                            (default: front narrow camera params)
  --no-masks                Skip sky masks during feature extraction
  -h, --help                Show this help message

Split detection:
  The script auto-detects split folders from:
      data/data_for_gaussian_splatting/<BAG>/images_gs_split_*_1_of_<FRAME_SKIP>/

EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

BAG_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera-model)
            [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] --camera-model requires a value."; exit 1; }
            CAMERA_MODEL="$2"
            shift 2
            ;;
        --camera-params)
            [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] --camera-params requires a value."; exit 1; }
            CAMERA_PARAMS="$2"
            shift 2
            ;;
        --no-masks)
            USE_SKY_MASKS=false
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
            if [ -z "$BAG_NAME" ]; then
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

if [ -z "$BAG_NAME" ]; then
    echo "[ERROR] Missing bag name."
    usage
    exit 1
fi

BAG_STEM="${BAG_NAME%.bag}"

# ---------- Paths ----------

GS_DATA_ROOT="${PROJECT_ROOT}/data/data_for_gaussian_splatting/${BAG_STEM}"
COLMAP_ROOT="${GS_DATA_ROOT}/colmap"

if [ ! -d "$GS_DATA_ROOT" ]; then
    echo "[ERROR] GS data folder not found: $GS_DATA_ROOT"
    echo "        Run the GS dataset preparation step first."
    exit 1
fi

mapfile -t SKIP_VALUES < <(
    find "$GS_DATA_ROOT" -maxdepth 1 -type d -printf '%f\n' |
        sed -n 's/^images_gs_split_[0-9]\+_1_of_\([0-9]\+\)$/\1/p' | sort -nu
)

if [[ "${#SKIP_VALUES[@]}" -ne 1 ]]; then
    echo "[ERROR] Expected exactly one frame-skip value, found: ${SKIP_VALUES[*]:-(none)}"
    exit 1
fi

FRAME_SKIP="${SKIP_VALUES[0]}"
SPLIT_LABEL="1_of_${FRAME_SKIP}"

mkdir -p "$COLMAP_ROOT"

# ---------- Auto-detect split ids ----------

mapfile -t SPLIT_DIRS < <(
    find "$GS_DATA_ROOT" -maxdepth 1 -type d \
        -name "images_gs_split_*_${SPLIT_LABEL}" | sort -V
)

if [[ "${#SPLIT_DIRS[@]}" -eq 0 ]]; then
    echo "[ERROR] No image split folders found under $GS_DATA_ROOT"
    echo "        Expected pattern: images_gs_split_*_${SPLIT_LABEL}/"
    echo "        Did you run the GS dataset preparation step with FRAME_SKIP=${FRAME_SKIP}?"
    exit 1
fi

SPLIT_IDS=()

for split_dir in "${SPLIT_DIRS[@]}"; do
    split_base="$(basename "$split_dir")"
    split_id="$(echo "$split_base" | sed -n "s/^images_gs_split_\([0-9]\+\)_${SPLIT_LABEL}$/\1/p")"

    if [[ -n "$split_id" ]]; then
        SPLIT_IDS+=("$split_id")
    fi
done

if [[ "${#SPLIT_IDS[@]}" -eq 0 ]]; then
    echo "[ERROR] Could not parse split ids from detected folders."
    echo "        Detected folders:"
    for split_dir in "${SPLIT_DIRS[@]}"; do
        echo "        - $(basename "$split_dir")"
    done
    exit 1
fi

NUM_SPLITS="${#SPLIT_IDS[@]}"

# ---------- Activate conda env ----------

echo ""
echo "--- Activating conda env: $CONDA_ENV ---"
conda activate "$CONDA_ENV"

# ---------- Sanity check: colmap available ----------

if ! command -v colmap >/dev/null 2>&1; then
    echo "[ERROR] 'colmap' not found in PATH. Install with:"
    echo "        sudo apt install colmap"
    echo "        or activate the right conda env"
    exit 1
fi

# ---------- Banner ----------

echo "=========================================="
echo "COLMAP RECONSTRUCTION"
echo "=========================================="
echo "Bag:                 $BAG_NAME"
echo "Bag stem:            $BAG_STEM"
echo "Frame skip:          $FRAME_SKIP"
echo "Split label:         $SPLIT_LABEL"
echo "Splits detected:     $NUM_SPLITS"
echo "Split ids:           ${SPLIT_IDS[*]}"
echo "Camera model:        $CAMERA_MODEL"
echo "Camera params:       $CAMERA_PARAMS"
echo "Sequential overlap:  $SEQUENTIAL_OVERLAP"
echo "Use sky masks:       $USE_SKY_MASKS"
echo "Project root:        $PROJECT_ROOT"
echo "GS data root:        $GS_DATA_ROOT"
echo "COLMAP root:         $COLMAP_ROOT"
echo "=========================================="

# ---------- Per-split loop ----------

for SPLIT_IDX in "${SPLIT_IDS[@]}"; do
    IMAGE_DIR="${GS_DATA_ROOT}/images_gs_split_${SPLIT_IDX}_${SPLIT_LABEL}"
    MASK_DIR="${GS_DATA_ROOT}/sky_masks_gs_split_${SPLIT_IDX}_${SPLIT_LABEL}"

    DATABASE_PATH="${COLMAP_ROOT}/database_split_${SPLIT_IDX}.db"
    SPARSE_OUT="${COLMAP_ROOT}/split_${SPLIT_IDX}/sparse"

    echo ""
    echo "=========================================="
    echo "SPLIT ${SPLIT_IDX}"
    echo "=========================================="
    echo "Images:    $IMAGE_DIR"
    echo "Masks:     $MASK_DIR"
    echo "Database:  $DATABASE_PATH"
    echo "Output:    $SPARSE_OUT"
    echo "=========================================="

    if [ ! -d "$IMAGE_DIR" ]; then
        echo "[ERROR] Image folder not found for split $SPLIT_IDX:"
        echo "        $IMAGE_DIR"
        echo "        Detected split folders:"
        for split_dir in "${SPLIT_DIRS[@]}"; do
            echo "        - $(basename "$split_dir")"
        done
        exit 1
    fi

    # Clean up old DB / sparse so reconstruction is reproducible.
    rm -f "$DATABASE_PATH"
    rm -rf "$SPARSE_OUT"
    mkdir -p "$SPARSE_OUT"

    # ---------- 1. Feature extraction ----------

    echo ""
    echo "--- [Split ${SPLIT_IDX}] 1/3 Feature extraction ---"

    FEAT_ARGS=(
        "feature_extractor"
        "--database_path" "$DATABASE_PATH"
        "--image_path" "$IMAGE_DIR"
        "--ImageReader.single_camera" "1"
        "--ImageReader.camera_model" "$CAMERA_MODEL"
        "--ImageReader.camera_params" "$CAMERA_PARAMS"
    )

    if [ "$USE_SKY_MASKS" = true ] && [ -d "$MASK_DIR" ]; then
        FEAT_ARGS+=("--ImageReader.mask_path" "$MASK_DIR")
    elif [ "$USE_SKY_MASKS" = true ]; then
        echo "[WARN] Sky masks requested but not found: $MASK_DIR"
        echo "       Proceeding without masks for this split."
    fi

    colmap "${FEAT_ARGS[@]}"

    # ---------- 2. Sequential matching ----------

    echo ""
    echo "--- [Split ${SPLIT_IDX}] 2/3 Sequential matching ---"

    colmap sequential_matcher \
        --database_path "$DATABASE_PATH" \
        --SequentialMatching.overlap "$SEQUENTIAL_OVERLAP"

    # ---------- 3. Mapper ----------

    echo ""
    echo "--- [Split ${SPLIT_IDX}] 3/3 Mapper (sparse reconstruction) ---"

    colmap mapper \
        --database_path "$DATABASE_PATH" \
        --image_path "$IMAGE_DIR" \
        --output_path "$SPARSE_OUT" \
        --Mapper.ba_refine_focal_length "$BA_REFINE_FOCAL_LENGTH" \
        --Mapper.ba_refine_principal_point "$BA_REFINE_PRINCIPAL_POINT" \
        --Mapper.ba_refine_extra_params "$BA_REFINE_EXTRA_PARAMS"

    # ---------- Verify ----------

    if [ ! -f "${SPARSE_OUT}/0/cameras.bin" ] || \
       [ ! -f "${SPARSE_OUT}/0/images.bin" ] || \
       [ ! -f "${SPARSE_OUT}/0/points3D.bin" ]; then
        echo "[ERROR] Split ${SPLIT_IDX} reconstruction incomplete."
        echo "        Missing cameras.bin / images.bin / points3D.bin in:"
        echo "        ${SPARSE_OUT}/0/"
        exit 1
    fi

    echo ""
    echo "--- [Split ${SPLIT_IDX}] OK: sparse model at ${SPARSE_OUT}/0/ ---"
done

# ---------- Done ----------

conda deactivate

echo ""
echo "=========================================="
echo "COLMAP reconstruction completed for $BAG_NAME"
echo "=========================================="
echo ""
echo "Sparse models:"
for SPLIT_IDX in "${SPLIT_IDS[@]}"; do
    echo "  Split $SPLIT_IDX: $COLMAP_ROOT/split_${SPLIT_IDX}/sparse/0/"
done
echo ""
echo "Next step: 4B_train_gaussian_splatting.sh"