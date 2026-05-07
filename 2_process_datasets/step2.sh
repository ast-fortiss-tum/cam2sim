SCRIPTS=(
    "2_process_datasets/2A_camera_parked_cars_detection.py"
    "2_process_datasets/2B_lidar_parked_cars_detection.py"
    "2_process_datasets/2C_create_map_from_coordinates_auto.py"
    "2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py"
    "2_process_datasets/2F_extract_semantic_maps.py"
)

PTH_FILE_1="2_process_datasets/utils/fcos3d.pth"
PTH_FILE_2="2_process_datasets/utils/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"

if [ ! -f "$PTH_FILE_1" ]; then  
    echo "Downloading FCOS3D"
    gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O "$PTH_FILE_1"
fi

if [ ! -f "$PTH_FILE_2" ]; then  
    echo "Downloading PointPillars"
    gdown 1AGOR8C0tDUsWSSWTEc0fA7kysIE9-iol -O "$PTH_FILE_2"
fi

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running Script $SCRIPT"
    python3 "$SCRIPT"

    if [ $? -ne 0 ]; then
        echo "Error in $SCRIPT. Aborting."
        exit 1
    fi
done

echo ""
echo "All Scripts for Step 2 completed successfully"