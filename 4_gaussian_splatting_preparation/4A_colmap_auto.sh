#!/usr/bin/env bash
# =============================================================================
# 4A_colmap_reconstruction.sh
#
# Automated COLMAP reconstruction for all GS splits.
#
# Per split, runs:
#   1. feature_extractor   (camera model + intrinsics, single camera, optional masks)
#   2. sequential_matcher  (overlap=10)
#   3. mapper              (intrinsics refinement DISABLED to keep calibration)
#
# Sparse model is exported automatically by `mapper` into:
#   data/data_for_gaussian_splatting/<BAG>/colmap/split_<N>/sparse/0/
#       {cameras,images,points3D}.bin
#
# Replaces the manual GUI steps documented in 4A_colmap_guide.md.
#
# Prerequisites:
#   - Step 2 GS (2_process_datasets/step2.sh <bag>) which produces:
#       data/data_for_gaussian_splatting/<bag>/images_gs_split_*_1_of_<FRAME_SKIP>/
#       data/data_for_gaussian_splatting/<bag>/sky_masks_gs_split_*_1_of_<FRAME_SKIP>/
#   - Conda env: nerfstudio (with COLMAP available in PATH)
#
# Usage:
#   bash 4_gaussian_splatting_preparation/4A_colmap_reconstruction.sh <bag_name.bag>
#   bash 4_gaussian_splatting_preparation/4A_colmap_reconstruction.sh <bag_name.bag> --num-splits 4
#   bash 4_gaussian_splatting_preparation/4A_colmap_reconstruction.sh <bag_name.bag> \
#       --camera-model PINHOLE --camera-params "800,800,400,250"
# =============================================================================

set -e

# ---------- Conda init ----------
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- Defaults ----------

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

NUM_SPLITS=3
FRAME_SKIP=2
SEQUENTIAL_OVERLAP=10
CONDA_ENV="nerfstudio"

# OPENCV intrinsics for reference_bag's front narrow camera.
# Order: fx, fy, cx, cy, k1, k2, p1, p2
CAMERA_MODEL="OPENCV"
CAMERA_PARAMS="772.906855,777.596896,424.980372,258.452509,-0.274231,0.034838,0.00226,-0.000972"

# Use sky masks (produced by 2E) to mask out the sky during feature extraction.
USE_SKY_MASKS=true

# Mapper: keep intrinsics fixed (do NOT re-optimize during BA)
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
  --num-splits N            Number of image splits (default: $NUM_SPLITS)
  --frame-skip N            Frame skip used in step 2E (default: $FRAME_SKIP)
  --camera-model MODEL      COLMAP camera model (default: $CAMERA_MODEL)
  --camera-params "p1,..."  Camera intrinsics (default: reference_bag front cam)
  --no-masks                Skip sky masks during feature extraction
  -h, --help                Show this help message
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

BAG_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-splits)     NUM_SPLITS="$2"; shift 2 ;;
        --frame-skip)     FRAME_SKIP="$2"; shift 2 ;;
        --camera-model)   CAMERA_MODEL="$2"; shift 2 ;;
        --camera-params)  CAMERA_PARAMS="$2"; shift 2 ;;
        --no-masks)       USE_SKY_MASKS=false; shift ;;
        -h|--help)        usage; exit 0 ;;
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

# ---------- Activate conda env ----------

echo ""
echo "--- Activating conda env: $CONDA_ENV ---"
conda activate "$CONDA_ENV"

# ---------- Sanity check: colmap available ----------

if ! command -v colmap >/dev/null 2>&1; then
    echo "[ERROR] 'colmap' not found in PATH. Install with:"
    echo "        sudo apt install colmap"
    echo "        (or activate the right conda env)"
    exit 1
fi

# ---------- Paths ----------

GS_DATA_ROOT="${PROJECT_ROOT}/data/data_for_gaussian_splatting/${BAG_STEM}"
COLMAP_ROOT="${GS_DATA_ROOT}/colmap"

if [ ! -d "$GS_DATA_ROOT" ]; then
    echo "[ERROR] GS data folder not found: $GS_DATA_ROOT"
    echo "        Run 2_process_datasets/step2.sh first."
    exit 1
fi

mkdir -p "$COLMAP_ROOT"

# ---------- Banner ----------

echo "=========================================="
echo "COLMAP RECONSTRUCTION"
echo "=========================================="
echo "Bag:               $BAG_NAME"
echo "Bag stem:          $BAG_STEM"
echo "Splits:            $NUM_SPLITS"
echo "Frame skip:        $FRAME_SKIP"
echo "Camera model:      $CAMERA_MODEL"
echo "Camera params:     $CAMERA_PARAMS"
echo "Sequential overlap: $SEQUENTIAL_OVERLAP"
echo "Use sky masks:     $USE_SKY_MASKS"
echo "GS data root:      $GS_DATA_ROOT"
echo "COLMAP root:       $COLMAP_ROOT"
echo "=========================================="

# ---------- Per-split loop ----------

for SPLIT_IDX in $(seq 1 "$NUM_SPLITS"); do

    SPLIT_LABEL="1_of_${FRAME_SKIP}"

    IMAGE_DIR="${GS_DATA_ROOT}/images_gs_split_${SPLIT_IDX}_${SPLIT_LABEL}"
    MASK_DIR="${GS_DATA_ROOT}/sky_masks_gs_split_${SPLIT_IDX}_${SPLIT_LABEL}"

    DATABASE_PATH="${COLMAP_ROOT}/database_split_${SPLIT_IDX}.db"
    SPARSE_OUT="${COLMAP_ROOT}/split_${SPLIT_IDX}/sparse"

    echo ""
    echo "=========================================="
    echo "SPLIT ${SPLIT_IDX}/${NUM_SPLITS}"
    echo "=========================================="
    echo "Images:    $IMAGE_DIR"
    echo "Masks:     $MASK_DIR"
    echo "Database:  $DATABASE_PATH"
    echo "Output:    $SPARSE_OUT"
    echo "=========================================="

    if [ ! -d "$IMAGE_DIR" ]; then
        echo "[ERROR] Image folder not found for split $SPLIT_IDX: $IMAGE_DIR"
        echo "        Did you run step 2E with NUM_SPLITS=$NUM_SPLITS, FRAME_SKIP=$FRAME_SKIP?"
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
        "--database_path"  "$DATABASE_PATH"
        "--image_path"     "$IMAGE_DIR"
        "--ImageReader.single_camera"  "1"
        "--ImageReader.camera_model"   "$CAMERA_MODEL"
        "--ImageReader.camera_params"  "$CAMERA_PARAMS"
    )

    if [ "$USE_SKY_MASKS" = true ] && [ -d "$MASK_DIR" ]; then
        FEAT_ARGS+=( "--ImageReader.mask_path" "$MASK_DIR" )
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
        --image_path    "$IMAGE_DIR" \
        --output_path   "$SPARSE_OUT" \
        --Mapper.ba_refine_focal_length     "$BA_REFINE_FOCAL_LENGTH" \
        --Mapper.ba_refine_principal_point  "$BA_REFINE_PRINCIPAL_POINT" \
        --Mapper.ba_refine_extra_params     "$BA_REFINE_EXTRA_PARAMS"

    # ---------- Verify ----------

    if [ ! -f "${SPARSE_OUT}/0/cameras.bin" ] || \
       [ ! -f "${SPARSE_OUT}/0/images.bin" ] || \
       [ ! -f "${SPARSE_OUT}/0/points3D.bin" ]; then
        echo "[ERROR] Split ${SPLIT_IDX} reconstruction incomplete."
        echo "        Missing cameras.bin / images.bin / points3D.bin in ${SPARSE_OUT}/0/"
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
for SPLIT_IDX in $(seq 1 "$NUM_SPLITS"); do
    echo "  Split $SPLIT_IDX: $COLMAP_ROOT/split_${SPLIT_IDX}/sparse/0/"
done
echo ""
echo "Next step: 4B_train_gaussian_splatting.sh"