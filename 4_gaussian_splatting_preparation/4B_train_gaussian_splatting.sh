#!/usr/bin/env bash
# =============================================================================
# 4B_train_gaussian_splatting.sh
#
# Train one nerfstudio model per split (splatfacto, splatfacto-big, nerfacto,
# nerfacto-big).
#
# IDEMPOTENT: if a split already has a trained checkpoint, skip it.
# This means re-running the script only trains the splits that are
# missing or incomplete.
#
# Uses --viewer.quit-on-train-completion so the viewer auto-closes when
# training ends and the script moves to the next split without manual Ctrl+C.
#
# The number of splits is auto-detected from the folder names produced by 2E.
# Sky masks are used automatically if present; pass --no-masks to skip them.
#
# Method-family-aware "trained" threshold:
#   splatfacto / splatfacto-big -> default 30000 steps  (threshold 29000)
#   nerfacto   / nerfacto-big   -> default 100000 steps (threshold 99000)
#
# Method-family-aware extra training flags:
#   nerfacto / nerfacto-big -> --pipeline.model.camera-optimizer.mode off
#                              (keeps extrinsics aligned with COLMAP poses so
#                               4C_utm_yaw_to_nerfstudio.py stays consistent)
#   splatfacto*             -> none (poses are fixed by default)
#
# Nerfstudio internal method folders:
#   splatfacto-big -> outputs/splatfacto-big_split_N/splatfacto/<TIMESTAMP>/
#   nerfacto-big   -> outputs/nerfacto-big_split_N/nerfacto/<TIMESTAMP>/
#
# Prerequisites:
#   - Step 2 GS  (produces images_gs_split_*_1_of_<SKIP>/ folders)
#   - Step 4A    (produces colmap/split_<N>/sparse/0/*.bin)
#   - Conda env: nerfstudio
#
# Usage:
#   bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh <bag_name.bag>
#   bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh <bag_name.bag> --method splatfacto-big
#   bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh <bag_name.bag> --method nerfacto
#   bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh <bag_name.bag> --no-masks
#   bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh <bag_name.bag> --max-jobs 2
# =============================================================================

set +e   # do NOT exit on error: keep going if one split fails

# ---------- Defaults ----------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SCRIPT_4C="${PROJECT_ROOT}/4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py"

SPLIT_SKIP=2
METHOD="splatfacto"
CONDA_ENV="nerfstudio"
USE_SKY_MASKS=true

# Parallel CUDA build jobs for gsplat / tinycudann JIT compilation.
# Empty = let ninja use the system default (typically nproc). Lower if you
# hit RAM OOM during JIT compilation of CUDA kernels.
MAX_JOBS=""

# Methods supported by this script. ns-train accepts more, but we want to
# fail fast on typos.
ALLOWED_METHODS=("splatfacto" "splatfacto-big" "nerfacto" "nerfacto-big")

# ---------- Usage / args ----------

usage() {
    cat <<EOF
Usage: $0 <bag_name.bag> [options]

Arguments:
  <bag_name.bag>            Bag filename including .bag extension

Options:
  --split-skip N            Skip value used by 2E in folder names
                            (matches FRAME_SKIP in 2E; default: $SPLIT_SKIP)
  --method NAME             nerfstudio method (default: $METHOD)
                            Allowed: ${ALLOWED_METHODS[*]}
  --no-masks                Do not pass sky masks to ns-train even if present
  --max-jobs N              Limit parallel CUDA build jobs (MAX_JOBS env var)
                            for gsplat/tinycudann JIT compilation. Lower this
                            if you hit RAM OOM during build (default: unset).
  -h, --help                Show this help message

The number of splits is auto-detected from how many
images_gs_split_*_1_of_<SPLIT_SKIP>/ folders exist.
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

BAG_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split-skip)
            SPLIT_SKIP="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --no-masks)
            USE_SKY_MASKS=false
            shift
            ;;
        --max-jobs)
            MAX_JOBS="$2"
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

# ---------- Validate method ----------

METHOD_OK=false
for m in "${ALLOWED_METHODS[@]}"; do
    if [[ "$m" == "$METHOD" ]]; then
        METHOD_OK=true
        break
    fi
done

if [ "$METHOD_OK" != true ]; then
    echo "[ERROR] Unsupported method: $METHOD"
    echo "        Allowed: ${ALLOWED_METHODS[*]}"
    exit 1
fi

# ---------- Method-family-aware checkpoint threshold ----------
# splatfacto* trains to 30000 steps by default; nerfacto* to 100000.

case "$METHOD" in
    splatfacto|splatfacto-big)
        MIN_CHECKPOINT_STEP=29000
        EXPECTED_FINAL_STEP=30000
        ;;
    nerfacto|nerfacto-big)
        MIN_CHECKPOINT_STEP=99000
        EXPECTED_FINAL_STEP=100000
        ;;
    *)
        MIN_CHECKPOINT_STEP=29000
        EXPECTED_FINAL_STEP=30000
        ;;
esac

# ---------- Method-specific extra training flags ----------
# For nerfacto/nerfacto-big, disable the camera-pose optimizer so that the
# extrinsics stay aligned with what COLMAP / our transforms.json declared.
# Otherwise 4C_utm_yaw_to_nerfstudio.py would compute the UTM->Nerfstudio
# transform from pre-optimization poses while the model is rendering with
# post-optimization ones, introducing a small offset/rotation drift.
# Splatfacto* keeps poses fixed by default, so no flag is needed there.

EXTRA_NS_TRAIN_FLAGS=()
case "$METHOD" in
    nerfacto|nerfacto-big)
        EXTRA_NS_TRAIN_FLAGS+=(--pipeline.model.camera-optimizer.mode off)
        ;;
esac

# ---------- Nerfstudio internal method folder ----------
# Nerfstudio stores some variant methods under the base method family:
#   splatfacto-big -> splatfacto
#   nerfacto-big   -> nerfacto
#
# The outer experiment folder still keeps the full cam2sim method name:
#   outputs/splatfacto-big_split_1/splatfacto/<TIMESTAMP>/
#   outputs/nerfacto-big_split_1/nerfacto/<TIMESTAMP>/

case "$METHOD" in
    splatfacto-big)
        NS_METHOD_DIR_NAME="splatfacto"
        ;;
    nerfacto-big)
        NS_METHOD_DIR_NAME="nerfacto"
        ;;
    *)
        NS_METHOD_DIR_NAME="$METHOD"
        ;;
esac

BAG_STEM="${BAG_NAME%.bag}"
SPLIT_LABEL="1_of_${SPLIT_SKIP}"

# ---------- Paths ----------

DATA_ROOT="${PROJECT_ROOT}/data/data_for_gaussian_splatting/${BAG_STEM}"
OUTPUT_ROOT="${DATA_ROOT}/outputs"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] Data folder not found: ${DATA_ROOT}"
    exit 1
fi

# ---------- Conda init + activate ----------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---------- Force gcc-11 for CUDA compilation if available ----------
# Required on systems whose default gcc is too new for the installed CUDA
# (e.g. Ubuntu 24 with gcc-13). On systems where gcc-11 is the default or
# is not installed, this block does nothing.

if command -v gcc-11 &>/dev/null && command -v g++-11 &>/dev/null; then
    export CC="$(command -v gcc-11)"
    export CXX="$(command -v g++-11)"
    export CUDA_HOST_COMPILER="$(command -v g++-11)"
    echo "[INFO] Using CC=${CC}, CXX=${CXX}, CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}"
else
    echo "[INFO] gcc-11/g++-11 not found in PATH, leaving compiler env vars untouched."
fi

# ---------- Limit parallel CUDA build jobs ----------

if [[ -n "${MAX_JOBS}" ]]; then
    export MAX_JOBS="${MAX_JOBS}"
    echo "[INFO] MAX_JOBS=${MAX_JOBS} (limits ninja parallel jobs for gsplat/tinycudann build)"
else
    echo "[INFO] MAX_JOBS not set; ninja will use the system default."
fi

if ! command -v ns-train &>/dev/null; then
    echo "[ERROR] ns-train not found. Is the '${CONDA_ENV}' env correct?"
    exit 1
fi

# ---------- Auto-detect splits ----------

mapfile -t SPLIT_DIRS < <(
    find "$DATA_ROOT" -maxdepth 1 -type d \
        -name "images_gs_split_*_${SPLIT_LABEL}" | sort -V
)

NUM_SPLITS="${#SPLIT_DIRS[@]}"

if [ "$NUM_SPLITS" -eq 0 ]; then
    echo "[ERROR] No split folders found under $DATA_ROOT"
    echo "        Expected pattern: images_gs_split_*_${SPLIT_LABEL}/"
    echo "        Did you run step 2E with FRAME_SKIP=${SPLIT_SKIP}?"
    exit 1
fi

# ---------- Banner ----------

echo "=========================================="
echo "NERFSTUDIO TRAINING"
echo "=========================================="
echo "Bag:                  $BAG_NAME"
echo "Bag stem:             $BAG_STEM"
echo "Split skip label:     $SPLIT_SKIP  (folder suffix: $SPLIT_LABEL)"
echo "Splits detected:      $NUM_SPLITS"
echo "Method:               $METHOD"
echo "Nerfstudio dir name:  $NS_METHOD_DIR_NAME"
echo "Use sky masks:        $USE_SKY_MASKS"
echo "MAX_JOBS:             ${MAX_JOBS:-(default)}"
echo "Expected final step:  $EXPECTED_FINAL_STEP"
echo "Min ckpt to skip:     $MIN_CHECKPOINT_STEP"
echo "Extra ns-train flags: ${EXTRA_NS_TRAIN_FLAGS[*]:-(none)}"
echo "Data root:            $DATA_ROOT"
echo "Output root:          $OUTPUT_ROOT"
echo "=========================================="

# =============================================================================
# Helper: check whether a given split already has a complete enough model.
#
# A split is considered already trained if there exists at least one checkpoint
# file step-NNNNNNNNN.ckpt under the experiment directory with
# NNN >= MIN_CHECKPOINT_STEP.
#
# The search starts from EXP_DIR instead of METHOD_DIR on purpose, so it works
# for methods whose internal Nerfstudio folder differs from the cam2sim method.
# =============================================================================

get_best_checkpoint_step() {
    local exp_dir="$1"
    local best_step=0

    if [[ ! -d "${exp_dir}" ]]; then
        echo ""
        return
    fi

    while IFS= read -r ckpt; do
        local fname
        fname=$(basename "${ckpt}")
        local step
        step=$(echo "${fname}" | sed -n 's/^step-0*\([0-9]\+\)\.ckpt$/\1/p')
        if [[ -n "${step}" ]] && (( step > best_step )); then
            best_step=${step}
        fi
    done < <(find "${exp_dir}" -type f -name "step-*.ckpt" 2>/dev/null)

    if (( best_step > 0 )); then
        echo "${best_step}"
    else
        echo ""
    fi
}

# =============================================================================
# Helper: check whether the UTM-to-Nerfstudio transform JSON exists for a run.
# Returns the run dir if a JSON is found, empty otherwise.
# =============================================================================

find_run_with_transform() {
    local method_dir="$1"

    if [[ ! -d "${method_dir}" ]]; then
        echo ""
        return
    fi

    while IFS= read -r run_dir; do
        if [[ -f "${run_dir}/utm_to_nerfstudio_transform.json" ]]; then
            echo "${run_dir}"
            return
        fi
    done < <(ls -td "${method_dir}"/*/ 2>/dev/null)

    echo ""
}

# =============================================================================
# Helper: find newest timestamp directory inside a Nerfstudio method dir.
# =============================================================================

find_newest_timestamp_dir() {
    local method_dir="$1"

    if [[ ! -d "${method_dir}" ]]; then
        echo ""
        return
    fi

    local timestamp_dir
    timestamp_dir=$(ls -td "${method_dir}/"*/ 2>/dev/null | head -n 1)

    if [[ -z "${timestamp_dir}" ]]; then
        echo ""
        return
    fi

    basename "${timestamp_dir}"
}

# ---------- Per-split loop ----------

for SPLIT in $(seq 1 "${NUM_SPLITS}"); do
    COLMAP_PATH="colmap/split_${SPLIT}/sparse/0"
    IMAGES_PATH="images_gs_split_${SPLIT}_${SPLIT_LABEL}"
    MASKS_PATH="sky_masks_gs_split_${SPLIT}_${SPLIT_LABEL}"

    EXP_NAME="${METHOD}_split_${SPLIT}"
    EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    METHOD_DIR="${EXP_DIR}/${NS_METHOD_DIR_NAME}"

    echo ""
    echo "============================================================"
    echo "Split ${SPLIT}/${NUM_SPLITS}"
    echo "  Method:              ${METHOD}"
    echo "  Nerfstudio dir name: ${NS_METHOD_DIR_NAME}"
    echo "  Experiment:          ${EXP_NAME}"
    echo "  Experiment dir:      ${EXP_DIR}"
    echo "  Method dir:          ${METHOD_DIR}"
    echo "============================================================"

    # ---- IDEMPOTENCY CHECK: skip if already trained ----

    BEST_STEP=$(get_best_checkpoint_step "${EXP_DIR}")

    if [[ -n "${BEST_STEP}" ]] && (( BEST_STEP >= MIN_CHECKPOINT_STEP )); then
        echo "[SKIP] Split ${SPLIT} already trained"
        echo "       Best checkpoint = step ${BEST_STEP} >= ${MIN_CHECKPOINT_STEP}"
        echo "       Output: ${EXP_DIR}"

        # Even if training is done, still make sure the UTM transform JSON
        # exists. If not, run the conversion step.
        RUN_WITH_TF=$(find_run_with_transform "${METHOD_DIR}")

        if [[ -z "${RUN_WITH_TF}" ]]; then
            TIMESTAMP_DIR=$(find_newest_timestamp_dir "${METHOD_DIR}")

            if [[ -n "${TIMESTAMP_DIR}" ]]; then
                echo "[INFO] Existing model has no utm_to_nerfstudio_transform.json,"
                echo "       running conversion now..."
                echo "[INFO] Config:"
                echo "       ${METHOD_DIR}/${TIMESTAMP_DIR}/config.yml"

                python "${SCRIPT_4C}" \
                    --gs_config "${METHOD_DIR}/${TIMESTAMP_DIR}/config.yml" \
                    --utm_file "${DATA_ROOT}/frame_positions_split_${SPLIT}_${SPLIT_LABEL}.txt" \
                    --data_root "${DATA_ROOT}"
            else
                echo "[ERROR] No timestamp folder found in ${METHOD_DIR}"
                echo "        Cannot run 4C conversion for split ${SPLIT}."
            fi
        else
            echo "[INFO] utm_to_nerfstudio_transform.json already present in"
            echo "       ${RUN_WITH_TF}"
        fi

        continue
    fi

    if [[ -n "${BEST_STEP}" ]]; then
        echo "[WARN] Split ${SPLIT} has only a partial checkpoint"
        echo "       Best step = ${BEST_STEP}, below threshold ${MIN_CHECKPOINT_STEP}"
        echo "       Retraining from scratch."
    fi

    # ---- Input checks ----

    if [[ ! -f "${DATA_ROOT}/${COLMAP_PATH}/cameras.bin" ]] \
       || [[ ! -f "${DATA_ROOT}/${COLMAP_PATH}/images.bin" ]] \
       || [[ ! -f "${DATA_ROOT}/${COLMAP_PATH}/points3D.bin" ]]; then
        echo "[WARN] Skipping split ${SPLIT}: missing COLMAP reconstruction"
        echo "       Expected files in:"
        echo "       ${DATA_ROOT}/${COLMAP_PATH}/"
        continue
    fi

    if [[ ! -d "${DATA_ROOT}/${IMAGES_PATH}" ]]; then
        echo "[WARN] Skipping split ${SPLIT}: images folder missing:"
        echo "       ${DATA_ROOT}/${IMAGES_PATH}"
        continue
    fi

    # ---- Training ----

    echo "Training split ${SPLIT}/${NUM_SPLITS}"
    echo "  Method:              ${METHOD}"
    echo "  Nerfstudio dir name: ${NS_METHOD_DIR_NAME}"
    echo "  Experiment:          ${EXP_NAME}"
    echo "  Data root:           ${DATA_ROOT}"
    echo "  COLMAP path:         ${COLMAP_PATH}"
    echo "  Images path:         ${IMAGES_PATH}"
    echo "  Masks path:          ${MASKS_PATH}"
    echo "  Output dir:          ${OUTPUT_ROOT}"
    echo "  Extra flags:         ${EXTRA_NS_TRAIN_FLAGS[*]:-(none)}"

    if [[ "${USE_SKY_MASKS}" = true ]] && [[ -d "${DATA_ROOT}/${MASKS_PATH}" ]]; then
        ns-train "${METHOD}" \
            --data "${DATA_ROOT}" \
            --output-dir "${OUTPUT_ROOT}" \
            --experiment-name "${EXP_NAME}" \
            --viewer.quit-on-train-completion True \
            "${EXTRA_NS_TRAIN_FLAGS[@]}" \
            colmap \
            --colmap-path "${COLMAP_PATH}" \
            --images-path "${IMAGES_PATH}" \
            --masks-path "${MASKS_PATH}"
    else
        if [[ "${USE_SKY_MASKS}" = true ]]; then
            echo "[WARN] Sky masks requested but not found: ${DATA_ROOT}/${MASKS_PATH}"
            echo "       Training without masks for this split."
        fi

        ns-train "${METHOD}" \
            --data "${DATA_ROOT}" \
            --output-dir "${OUTPUT_ROOT}" \
            --experiment-name "${EXP_NAME}" \
            --viewer.quit-on-train-completion True \
            "${EXTRA_NS_TRAIN_FLAGS[@]}" \
            colmap \
            --colmap-path "${COLMAP_PATH}" \
            --images-path "${IMAGES_PATH}"
    fi

    if [[ $? -ne 0 ]]; then
        echo "[WARN] Split ${SPLIT} returned non-zero exit code, continuing..."
    fi

    # ---- Run 4C conversion for newest timestamp ----

    TIMESTAMP_DIR=$(find_newest_timestamp_dir "${METHOD_DIR}")

    if [[ -z "${TIMESTAMP_DIR}" ]]; then
        echo "[ERROR] No timestamp folder found for ${EXP_NAME} in ${METHOD_DIR}"
        continue
    fi

    if [[ ! -f "${METHOD_DIR}/${TIMESTAMP_DIR}/config.yml" ]]; then
        echo "[ERROR] config.yml not found:"
        echo "        ${METHOD_DIR}/${TIMESTAMP_DIR}/config.yml"
        continue
    fi

    echo "[INFO] Running 4C conversion"
    echo "       Config: ${METHOD_DIR}/${TIMESTAMP_DIR}/config.yml"
    echo "       UTM:    ${DATA_ROOT}/frame_positions_split_${SPLIT}_${SPLIT_LABEL}.txt"

    python "${SCRIPT_4C}" \
        --gs_config "${METHOD_DIR}/${TIMESTAMP_DIR}/config.yml" \
        --utm_file "${DATA_ROOT}/frame_positions_split_${SPLIT}_${SPLIT_LABEL}.txt" \
        --data_root "${DATA_ROOT}"
done

conda deactivate

echo ""
echo "============================================================"
echo "All splits processed."
echo "Method:               ${METHOD}"
echo "Nerfstudio dir name:  ${NS_METHOD_DIR_NAME}"
echo "Outputs in:           ${OUTPUT_ROOT}"
echo "============================================================"