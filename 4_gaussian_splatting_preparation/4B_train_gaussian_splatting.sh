#!/usr/bin/env bash
# =============================================================================
# Train one Gaussian Splatting model per split with nerfstudio.
#
# IDEMPOTENT: if a split already has a trained checkpoint, skip it.
# This means re-running the script only trains the splits that are
# missing or incomplete.
#
# Uses --viewer.quit-on-train-completion so the viewer auto-closes when
# training ends and the script moves to the next split without manual Ctrl+C.
# =============================================================================

set +e   # do NOT exit on error: keep going if one split fails

BAG_NAME="reference_bag"
NUM_SPLITS=3
FRAME_SKIP=2

METHOD="splatfacto"
CONDA_ENV="nerfstudio"

# A checkpoint is considered "complete enough" to skip retraining if its
# step number is at least this many steps. Splatfacto's default training
# is 30000 steps, so we set the bar reasonably close to it.
MIN_CHECKPOINT_STEP=29000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data/data_for_gaussian_splatting/${BAG_NAME}"
OUTPUT_ROOT="${DATA_ROOT}/outputs"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] Data folder not found: ${DATA_ROOT}"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! command -v ns-train &>/dev/null; then
    echo "[ERROR] ns-train not found. Is the '${CONDA_ENV}' env correct?"
    exit 1
fi


# =============================================================================
# Helper: check whether a given split already has a "complete enough" model.
#
# A split is considered already trained if there exists at least one run
# under outputs/<EXP_NAME>/splatfacto/<TIMESTAMP>/nerfstudio_models/
# containing a checkpoint file step-NNNNNNNNN.ckpt with NNN >= MIN_CHECKPOINT_STEP.
#
# Echoes the step number of the best existing checkpoint on success,
# or empty string if no acceptable checkpoint exists.
# =============================================================================
get_best_checkpoint_step() {
    local exp_dir="$1"
    local best_step=0

    if [[ ! -d "${exp_dir}" ]]; then
        echo ""
        return
    fi

    # Iterate over each run timestamp dir
    while IFS= read -r ckpt; do
        # Extract step number from filename: step-000029999.ckpt -> 29999
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
    local splatfacto_dir="$1"
    if [[ ! -d "${splatfacto_dir}" ]]; then
        echo ""
        return
    fi
    # Take the newest run dir that has the transform JSON
    while IFS= read -r run_dir; do
        if [[ -f "${run_dir}/utm_to_nerfstudio_transform.json" ]]; then
            echo "${run_dir}"
            return
        fi
    done < <(ls -td "${splatfacto_dir}"/*/ 2>/dev/null)
    echo ""
}


for SPLIT in $(seq 1 "${NUM_SPLITS}"); do
    COLMAP_PATH="colmap/split_${SPLIT}/sparse/0"
    IMAGES_PATH="images_gs_split_${SPLIT}_1_of_${FRAME_SKIP}"
    MASKS_PATH="sky_masks_gs_split_${SPLIT}_1_of_${FRAME_SKIP}"
    EXP_NAME="${METHOD}_split_${SPLIT}"
    EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    SPLATFACTO_DIR="${EXP_DIR}/${METHOD}"

    # ---- IDEMPOTENCY CHECK: skip if already trained ----
    BEST_STEP=$(get_best_checkpoint_step "${EXP_DIR}")
    if [[ -n "${BEST_STEP}" ]] && (( BEST_STEP >= MIN_CHECKPOINT_STEP )); then
        echo ""
        echo "============================================================"
        echo "[SKIP] Split ${SPLIT} already trained "
        echo "       (best checkpoint = step ${BEST_STEP} >= ${MIN_CHECKPOINT_STEP})"
        echo "       Output: ${EXP_DIR}"
        echo "============================================================"

        # Even if training is done, still make sure the UTM transform JSON
        # exists. If not, run the conversion step.
        RUN_WITH_TF=$(find_run_with_transform "${SPLATFACTO_DIR}")
        if [[ -z "${RUN_WITH_TF}" ]]; then
            TIMESTAMP_DIR=$(ls -td "${SPLATFACTO_DIR}/"*/ 2>/dev/null | head -n 1)
            TIMESTAMP_DIR=$(basename "${TIMESTAMP_DIR}")
            if [[ -n "${TIMESTAMP_DIR}" ]]; then
                echo "[INFO] Existing model has no utm_to_nerfstudio_transform.json,"
                echo "       running conversion now..."
                python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
                    --gs_config "${SPLATFACTO_DIR}/${TIMESTAMP_DIR}/config.yml" \
                    --utm_file "${DATA_ROOT}/frame_positions_split_${SPLIT}_1_of_${FRAME_SKIP}.txt" \
                    --data_root "${DATA_ROOT}"
            fi
        else
            echo "[INFO] utm_to_nerfstudio_transform.json already present in"
            echo "       ${RUN_WITH_TF}"
        fi

        continue
    fi

    if [[ -n "${BEST_STEP}" ]]; then
        echo ""
        echo "============================================================"
        echo "[WARN] Split ${SPLIT} has only a partial checkpoint "
        echo "       (best step = ${BEST_STEP}, below threshold ${MIN_CHECKPOINT_STEP})."
        echo "       Retraining from scratch."
        echo "============================================================"
    fi

    # ---- INPUT CHECKS ----
    if [[ ! -f "${DATA_ROOT}/${COLMAP_PATH}/cameras.bin" ]] \
       || [[ ! -f "${DATA_ROOT}/${COLMAP_PATH}/images.bin" ]] \
       || [[ ! -f "${DATA_ROOT}/${COLMAP_PATH}/points3D.bin" ]]; then
        echo ""
        echo "============================================================"
        echo "[WARN] Skipping split ${SPLIT}: missing COLMAP reconstruction"
        echo "       Expected files in:"
        echo "       ${DATA_ROOT}/${COLMAP_PATH}/"
        echo "============================================================"
        continue
    fi

    if [[ ! -d "${DATA_ROOT}/${IMAGES_PATH}" ]]; then
        echo "[WARN] Skipping split ${SPLIT}: images folder missing:"
        echo "       ${DATA_ROOT}/${IMAGES_PATH}"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Training split ${SPLIT}/${NUM_SPLITS}"
    echo "  Experiment:   ${EXP_NAME}"
    echo "  Data root:    ${DATA_ROOT}"
    echo "  COLMAP path:  ${COLMAP_PATH}"
    echo "  Images path:  ${IMAGES_PATH}"
    echo "  Masks path:   ${MASKS_PATH}"
    echo "  Output dir:   ${OUTPUT_ROOT}"
    echo "============================================================"

    if [[ -d "${DATA_ROOT}/${MASKS_PATH}" ]]; then
        ns-train "${METHOD}" \
            --data "${DATA_ROOT}" \
            --output-dir "${OUTPUT_ROOT}" \
            --experiment-name "${EXP_NAME}" \
            --viewer.quit-on-train-completion True \
            colmap \
            --colmap-path "${COLMAP_PATH}" \
            --images-path "${IMAGES_PATH}" \
            --masks-path "${MASKS_PATH}"
    else
        ns-train "${METHOD}" \
            --data "${DATA_ROOT}" \
            --output-dir "${OUTPUT_ROOT}" \
            --experiment-name "${EXP_NAME}" \
            --viewer.quit-on-train-completion True \
            colmap \
            --colmap-path "${COLMAP_PATH}" \
            --images-path "${IMAGES_PATH}"
    fi

    if [[ $? -ne 0 ]]; then
        echo "!!! Split ${SPLIT} returned non-zero exit code, continuing..."
    fi

    # Find the newest timestamp folder for this split
    TIMESTAMP_DIR=$(ls -td "${SPLATFACTO_DIR}/"*/ 2>/dev/null | head -n 1)
    TIMESTAMP_DIR=$(basename "${TIMESTAMP_DIR}")

    if [[ -z "${TIMESTAMP_DIR}" ]]; then
        echo "[ERROR] No timestamp folder found for ${EXP_NAME} in ${SPLATFACTO_DIR}"
        continue
    fi

    python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
        --gs_config "${SPLATFACTO_DIR}/${TIMESTAMP_DIR}/config.yml" \
        --utm_file "${DATA_ROOT}/frame_positions_split_${SPLIT}_1_of_${FRAME_SKIP}.txt" \
        --data_root "${DATA_ROOT}"
done

echo ""
echo "============================================================"
echo "All splits processed."
echo "Outputs are in: ${OUTPUT_ROOT}"
echo "============================================================"