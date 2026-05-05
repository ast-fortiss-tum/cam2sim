#!/usr/bin/env bash
set -euo pipefail

BAG_NAME="reference_bag"
NUM_SPLITS=3
FRAME_SKIP=3

# Use splatfacto for Gaussian Splatting.
# Change to "splatfacto-big" if that is what you want.
METHOD="nerfacto"

DISABLE_TORCH_COMPILE=1
CONDA_ENV="nerfstudio"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

DATA_ROOT="${PROJECT_ROOT}/data/data_for_gaussian_splatting/${BAG_NAME}"

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

if [[ "${DISABLE_TORCH_COMPILE}" == "1" ]]; then
    export TORCH_COMPILE_DISABLE=1
fi

for SPLIT in $(seq 1 "${NUM_SPLITS}"); do
    COLMAP_PATH="colmap/split_${SPLIT}/sparse/0"
    IMAGES_PATH="images_gs_split_${SPLIT}_1_of_${FRAME_SKIP}"
    MASKS_PATH="sky_masks_gs_split_${SPLIT}_1_of_${FRAME_SKIP}"
    OUTPUT_DIR="${DATA_ROOT}/outputs/split_${SPLIT}"

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
    echo "  Data root:    ${DATA_ROOT}"
    echo "  COLMAP path:  ${COLMAP_PATH}"
    echo "  Images path:  ${IMAGES_PATH}"
    echo "  Masks path:   ${MASKS_PATH}"
    echo "  Output dir:   ${OUTPUT_DIR}"
    echo "============================================================"

    if [[ -d "${DATA_ROOT}/${MASKS_PATH}" ]]; then
        ns-train "${METHOD}" \
            --data "${DATA_ROOT}" \
            --output-dir "${OUTPUT_DIR}" \
            colmap \
            --colmap-path "${COLMAP_PATH}" \
            --images-path "${IMAGES_PATH}" \
            --masks-path "${MASKS_PATH}"
    else
        ns-train "${METHOD}" \
            --data "${DATA_ROOT}" \
            --output-dir "${OUTPUT_DIR}" \
            colmap \
            --colmap-path "${COLMAP_PATH}" \
            --images-path "${IMAGES_PATH}"
    fi

    echo ""
    echo "[INFO] Split ${SPLIT} done."
done

echo ""
echo "============================================================"
echo "All splits processed."
echo "Outputs are in: ${DATA_ROOT}/outputs/"
echo "============================================================"