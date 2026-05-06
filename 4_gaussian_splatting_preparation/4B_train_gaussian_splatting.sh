#!/usr/bin/env bash
# =============================================================================
# Train one Gaussian Splatting model per split with nerfstudio.
# Uses --viewer.quit-on-train-completion so the viewer auto-closes when
# training ends and the script moves to the next split without manual Ctrl+C.
# =============================================================================

set +e   # do NOT exit on error: keep going if one split fails

BAG_NAME="reference_bag"
NUM_SPLITS=1
FRAME_SKIP=3

METHOD="splatfacto"
CONDA_ENV="nerfstudio"

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

for SPLIT in $(seq 1 "${NUM_SPLITS}"); do
    COLMAP_PATH="colmap/split_${SPLIT}/sparse/0"
    IMAGES_PATH="images_gs_split_${SPLIT}_1_of_${FRAME_SKIP}"
    MASKS_PATH="sky_masks_gs_split_${SPLIT}_1_of_${FRAME_SKIP}"
    EXP_NAME="${METHOD}_split_${SPLIT}"

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

    echo ""
    echo "[INFO] Split ${SPLIT} done."
done

echo ""
echo "============================================================"
echo "All splits processed."
echo "Outputs are in: ${OUTPUT_ROOT}"
echo "============================================================"