# Install

## Create conda env
conda create -n repl python=3.10 
conda activate repl  

NEW 
pip install -U pip setuptools wheel
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" scipy pandas opencv-python rosbags
pip install -U openmim
mim install "mmengine"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmdet3d==1.4.0"
pip install osmnx
pip install pyrender
pip install carla==0.9.15
pip install contextily
python -m pip install "transformers==4.36.2"
sudo apt install colmap 
pip install pygame


conda create -n dave_2 python=3.8 
conda activate dave_2 
pip install tensorflow==2.13.1
pip install pillow
pip install opencv-python

<details>
<summary><code>1_extract_ROS_data</code></summary>

# 1_extract_ROS_data

This folder contains the first step of the data-processing pipeline.

The scripts read data from a ROS bag file and export selected camera, LiDAR, odometry, steering, and model-output data into a structured dataset folder.

Each script is intended to be run from the project root:

```bash
python 1_extract_ROS_data/<script_name>.py
```

For example:

```bash
python 1_extract_ROS_data/1A_camera_with_odometry.py
```

---

## Purpose

The goal of this step is to convert raw ROS bag messages into files that are easier to use in later stages of the pipeline.

The scripts can extract:

- Camera images
- Camera images synchronized with odometry
- LiDAR point clouds
- LiDAR point clouds synchronized with odometry
- Full odometry and simplified trajectory
- Vehicle steering status
- Steering target / model output
- ROS bag topic information

---

## Expected project structure

```text
project_root/
├── 1_extract_ROS_data/
│   ├── 1UTIL_print_bag_info.py
│   ├── 1A_camera_without_odometry.py
│   ├── 1A_camera_with_odometry.py
│   ├── 1B_lidar_without_odometry.py
│   ├── 1B_lidar_with_odometry.py
│   ├── 1C_poses_and_trajectory.py
│   ├── 1D_steering_status.py
│   └── 1E_model_output.py
│
├── data/
│   ├── raw_ros_data/
│   │   └── reference_bag.bag
│   │
│   └── raw_dataset/
│       └── <bag_name>/
│           ├── images/
│           ├── point_clouds/
│           ├── images_positions.txt
│           ├── lidar_positions.txt
│           ├── odometry.csv
│           ├── trajectory.csv
│           ├── steering_pct.txt
│           └── steering_predictions.txt
```

All scripts write their output to:

```text
data/raw_dataset/<bag_name>/
```

where `<bag_name>` is automatically taken from the ROS bag filename.

For example, if the bag file is:

```text
data/raw_ros_data/reference_bag.bag
```

then the output folder will be:

```text
data/raw_dataset/reference_bag/
```

---

## Requirements

Use the existing Conda environment named `data_extraction`.

Activate it before running any extraction script:

```bash
conda activate data_extraction
```

The scripts use `rosbags`, `numpy`, and, for camera extraction, `opencv-python`.

---

## Configuration

Each script has a configuration section near the top.

Typical configuration values include:

```python
bag_path = Path("data/raw_ros_data/reference_bag.bag")
```

and one or more ROS topic names, for example:

```python
cam_topic = "/gmsl_camera/front_narrow/image_raw"
lidar_topic = "/velodyne_points"
odom_topic = "/odom"
topic = "/vehicle/steering_pct"
```

Before running a script, check that:

1. The ROS bag exists at the configured `bag_path`.
2. The topic names match the topics in your ROS bag.
3. You are running the command from the project root.

To inspect the available topics in a bag, run:

```bash
python 1_extract_ROS_data/1UTIL_print_bag_info.py
```

---

## Scripts

### 1UTIL_print_bag_info.py

Utility script for inspecting a ROS bag.

It prints information such as:

- Available ROS topics
- Message type for each topic
- Number of messages per topic
- Start and end timestamps of the bag

Run:

```bash
python 1_extract_ROS_data/1UTIL_print_bag_info.py
```

Use this script before extraction to confirm the correct topic names for camera, LiDAR, odometry, steering status, and steering target messages.

---

### 1A_camera_without_odometry.py

Extracts camera frames from the ROS bag and saves them as PNG images.

Default camera topic:

```text
/gmsl_camera/front_narrow/image_raw
```

Output:

```text
data/raw_dataset/<bag_name>/images/frame_000000.png
data/raw_dataset/<bag_name>/images/frame_000001.png
...
```

Run:

```bash
python 1_extract_ROS_data/1A_camera_without_odometry.py
```

This script only extracts images. It does not associate camera frames with odometry.

Supported image encodings include:

- `mono8`
- `rgb8`
- `bgr8`
- Bayer encodings

Unsupported image encodings raise an error for the affected frame.

---

### 1A_camera_with_odometry.py

Extracts camera frames and synchronizes each image timestamp with odometry.

Default topics:

```text
/gmsl_camera/front_narrow/image_raw
/odom
```

Outputs:

```text
data/raw_dataset/<bag_name>/images/
data/raw_dataset/<bag_name>/images_positions.txt
```

Image files are saved as:

```text
frame_000000.png
frame_000001.png
...
```

The synchronized metadata file contains:

```text
# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile
```

Run:

```bash
python 1_extract_ROS_data/1A_camera_with_odometry.py
```

This script interpolates odometry values at each camera timestamp.

Position values are linearly interpolated. Quaternion components are also interpolated component-by-component, normalized, and then used to compute yaw.

---

### 1B_lidar_without_odometry.py

Extracts LiDAR point clouds from the ROS bag and saves each scan as a binary `.bin` file.

Default LiDAR topic:

```text
/velodyne_points
```

Output:

```text
data/raw_dataset/<bag_name>/point_clouds/point_cloud_000000.bin
data/raw_dataset/<bag_name>/point_clouds/point_cloud_000001.bin
...
```

Each `.bin` file stores points as float32 values in the following order:

```text
x, y, z, intensity
```

Run:

```bash
python 1_extract_ROS_data/1B_lidar_without_odometry.py
```

This script only extracts point clouds. It does not associate LiDAR scans with odometry.

If the point cloud message does not contain an `intensity` field, intensity is written as `0.0`.

---

### 1B_lidar_with_odometry.py

Extracts LiDAR point clouds and synchronizes each scan timestamp with odometry.

Default topics:

```text
/velodyne_points
/odom
```

Outputs:

```text
data/raw_dataset/<bag_name>/point_clouds/
data/raw_dataset/<bag_name>/lidar_positions.txt
```

Point clouds are saved as:

```text
point_cloud_000000.bin
point_cloud_000001.bin
...
```

Each `.bin` file stores points as float32 values in the following order:

```text
x, y, z, intensity
```

The synchronized metadata file contains:

```text
# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, PointCloudFile
```

Run:

```bash
python 1_extract_ROS_data/1B_lidar_with_odometry.py
```

This script interpolates odometry position and yaw at each LiDAR timestamp.

---

### 1C_poses_and_trajectory.py

Extracts odometry from the ROS bag and exports both full odometry and a simplified trajectory.

Default odometry topic:

```text
/odom
```

Outputs:

```text
data/raw_dataset/<bag_name>/odometry.csv
data/raw_dataset/<bag_name>/trajectory.csv
```

`odometry.csv` contains:

```text
timestamp, tx, ty, tz, qx, qy, qz, qw, yaw
```

`trajectory.csv` contains:

```text
timestamp, x, y, z, yaw
```

Run:

```bash
python 1_extract_ROS_data/1C_poses_and_trajectory.py
```

This script converts quaternion orientation to yaw and sorts all odometry samples by timestamp before writing the output files.

---

### 1D_steering_status.py

Extracts the actual vehicle steering status from the ROS bag.

Default topic:

```text
/vehicle/steering_pct
```

Output:

```text
data/raw_dataset/<bag_name>/steering_pct.txt
```

The output file contains:

```text
# timestamp value
```

Each data row is written as:

```text
timestamp, value
```

Run:

```bash
python 1_extract_ROS_data/1D_steering_status.py
```

Use this file to compare actual steering status against predicted or commanded steering values.

---

### 1E_model_output.py

Extracts steering target / model-output values from the ROS bag.

Default topic:

```text
/cmd/steering_target
```

Output:

```text
data/raw_dataset/<bag_name>/steering_predictions.txt
```

The output file contains:

```text
# timestamp steering_target
```

Each data row is written as:

```text
timestamp, steering_target
```

Run:

```bash
python 1_extract_ROS_data/1E_model_output.py
```

Use this file as the model prediction or command signal for later comparison with the vehicle steering status.

---

## Suggested execution order

A typical workflow is:

```bash
python 1_extract_ROS_data/1UTIL_print_bag_info.py
python 1_extract_ROS_data/1A_camera_with_odometry.py
python 1_extract_ROS_data/1B_lidar_with_odometry.py
python 1_extract_ROS_data/1C_poses_and_trajectory.py
python 1_extract_ROS_data/1D_steering_status.py
python 1_extract_ROS_data/1E_model_output.py
```

If odometry synchronization is not needed, use the scripts without odometry:

```bash
python 1_extract_ROS_data/1A_camera_without_odometry.py
python 1_extract_ROS_data/1B_lidar_without_odometry.py
```

---

## Output files summary

| Script | Main output |
|---|---|
| `1UTIL_print_bag_info.py` | Console printout of bag topics, message counts, message types, and time range |
| `1A_camera_without_odometry.py` | `images/frame_XXXXXX.png` |
| `1A_camera_with_odometry.py` | `images/frame_XXXXXX.png`, `images_positions.txt` |
| `1B_lidar_without_odometry.py` | `point_clouds/point_cloud_XXXXXX.bin` |
| `1B_lidar_with_odometry.py` | `point_clouds/point_cloud_XXXXXX.bin`, `lidar_positions.txt` |
| `1C_poses_and_trajectory.py` | `odometry.csv`, `trajectory.csv` |
| `1D_steering_status.py` | `steering_pct.txt` |
| `1E_model_output.py` | `steering_predictions.txt` |

---

## Timestamp format

All timestamps are converted from nanoseconds to seconds using:

```python
timestamp = ts * 1e-9
```

The exported timestamps are written in seconds.

---

## Notes

- Output folders are created automatically if they do not already exist.
- The scripts assume a single ROS bag file configured by `bag_path`.
- The output folder name is derived from `bag_path.stem`.
- Make sure the topic names in each script match the topics in your ROS bag.
- Run `1UTIL_print_bag_info.py` first when working with a new bag.
- If the configured bag file is not found, the script prints an error message and stops.
- If a required topic is missing, the corresponding script raises an error.
- Camera outputs are saved as `.png`.
- LiDAR outputs are saved as raw float32 `.bin` files containing `x, y, z, intensity`.

</details>

<details>
<summary><code>2_process_datasets</code></summary>

# 2_process_datasets

This folder contains the second step of the data-processing pipeline.

The scripts take the extracted data from step 1 and run computer vision, 3D detection, map-generation, manual-cleanup, and Gaussian Splatting preparation steps. They produce per-dataset detection files, cleaned object lists, OSM-derived map data, and image/mask splits used by later stages of the pipeline.

Each script is intended to be run from the project root:

```bash
python 2_process_datasets/<script_name>.py
```

For example:

```bash
python 2_process_datasets/2A_camera_parked_cars_detection.py
```

---

## Purpose

The goal of this step is to convert the raw extracted dataset into processed assets for detection, mapping, simulation, and Gaussian Splatting.

The scripts can produce:

- 3D bounding boxes of parked vehicles from camera images
- 3D bounding boxes of parked vehicles from LiDAR point clouds
- A manually refined LiDAR ground truth
- Cleaned camera and/or LiDAR centroid files edited interactively
- OSM map data for the recorded trajectory area
- A placeholder `vehicle_data.json` file for CARLA simulation
- Cropped images, sky masks, and overlapping image splits for Gaussian Splatting training

---

## Expected project structure

```text
project_root/
├── 2_process_datasets/
│   ├── 2A_camera_parked_cars_detection.py
│   ├── 2B_lidar_parked_cars_detection.py
│   ├── 2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py
│   ├── 2C_create_map_from_coordinates_auto.py
│   ├── 2C_create_map_from_coordinates_manual.py
│   ├── 2D_manual_refinment_parked_cars_camera.py
│   ├── 2D_manual_refinment_parked_cars_lidar.py
│   ├── 2E_prepare_dataset_for_gaussian_splatting.py
│   ├── 2F_TODO_extract_semantic_maps.py
│   └── utils/
│       ├── fcos3d_config.py
│       ├── fcos3d.pth
│       ├── my_pointpillars_config.py
│       ├── hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_*.pth
│       ├── coordinates.py
│       ├── map_data.py
│       ├── save_data.py
│       ├── plotting.py
│       └── other.py
│
├── data/
│   ├── raw_dataset/
│   │   └── <bag_name>/
│   │       ├── images/
│   │       ├── point_clouds/
│   │       ├── images_positions.txt
│   │       ├── lidar_positions.txt
│   │       ├── odometry.csv
│   │       └── trajectory.csv
│   │
│   ├── processed_dataset/
│   │   └── <bag_name>/
│   │       ├── camera_detections/
│   │       ├── lidar_detections/
│   │       ├── lidar_refinement/
│   │       └── maps/
│   │
│   └── data_for_gaussian_splatting/
│       └── <bag_name>/
```

Scripts generally read input from:

```text
data/raw_dataset/<bag_name>/
```

and write output to:

```text
data/processed_dataset/<bag_name>/
```

The Gaussian Splatting preparation script writes to:

```text
data/data_for_gaussian_splatting/<bag_name>/
```

because that data is intended for an external training pipeline.

---

## Requirements

Use the existing Conda environment named `data_extraction`.

Activate it before running any script in this folder:

```bash
conda activate data_extraction
```

The scripts use packages for ROS bag processing, numerical computation, computer vision, 3D detection, map processing, visualization, and Gaussian Splatting preparation.

The camera and LiDAR detection scripts require pretrained model files in:

```text
2_process_datasets/utils/
├── fcos3d_config.py
├── fcos3d.pth
├── my_pointpillars_config.py
└── hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_*.pth
```

If a required model or config file is missing, the corresponding script will fail at startup with a `FileNotFoundError`.

The map scripts and Gaussian Splatting preparation script do not require the FCOS3D or PointPillars model files.

---

## The `utils/` subfolder

The `utils/` folder contains shared helper code, model configuration files, and pretrained model checkpoints used by the processing scripts.

Typical contents include:

```text
2_process_datasets/utils/
├── fcos3d_config.py
├── fcos3d.pth
├── my_pointpillars_config.py
├── hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_*.pth
├── coordinates.py
├── map_data.py
├── save_data.py
├── plotting.py
└── other.py
```

The files are used as follows:

| File | Purpose |
|---|---|
| `fcos3d_config.py` | FCOS3D configuration for camera-based 3D detection |
| `fcos3d.pth` | FCOS3D pretrained checkpoint |
| `my_pointpillars_config.py` | PointPillars configuration for LiDAR-based 3D detection |
| `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_*.pth` | PointPillars pretrained checkpoint |
| `coordinates.py` | Coordinate conversion utilities |
| `map_data.py` | OSM/map processing utilities |
| `save_data.py` | Helpers for writing map and vehicle data |
| `plotting.py` | Plotting and visualization helpers |
| `other.py` | Additional shared helper functions |

Do not move or rename files in `utils/` unless the corresponding import paths and model paths in the scripts are also updated.

---

## Configuration

Each script has a configuration section near the top.

The most important value is the dataset name:

```python
DATASET_NAME = "reference_bag"
```

This must match the folder name produced by step 1 inside:

```text
data/raw_dataset/
```

For example:

```text
data/raw_dataset/reference_bag/
```

Other commonly edited values include:

```python
SKIP_FRAMES = 5            # Process one frame every N frames
CONFIDENCE_THRESH = 0.50   # Detection score threshold
DIST = 200                 # OSM extraction radius in meters
ADDRESS = "..."            # Used by the manual map script
```

Before running any script, check that:

1. `data/raw_dataset/<DATASET_NAME>/` exists.
2. The required input files for the script exist.
3. The required files in `2_process_datasets/utils/` exist.
4. You are running the command from the project root.

---

## Scripts

### 2A_camera_parked_cars_detection.py

Detects parked cars from camera images using FCOS3D.

The script tracks detections in world coordinates across frames and clusters duplicate detections into one cluster per parked car.

Default input:

```text
data/raw_dataset/<bag_name>/images/
data/raw_dataset/<bag_name>/images_positions.txt
```

Outputs:

```text
data/processed_dataset/<bag_name>/camera_detections/
├── camera_detections.json
├── unified_clusters.txt
└── unified_bbox_overlays/
    ├── bbox_000000.png
    ├── bbox_000005.png
    └── ...
```

`camera_detections.json` contains full per-cluster information:

```text
id, x, y, z, length, width, height, yaw, count, conf, side, orient
```

`unified_clusters.txt` contains a simplified text representation:

```text
# cluster_id, x, y, z, count, conf, orientation, side
```

`unified_bbox_overlays/` contains one PNG per processed frame, with 3D detections drawn on the image. Use these overlays to visually check detection quality.

Run:

```bash
python 2_process_datasets/2A_camera_parked_cars_detection.py
```

GPU is used automatically if available.

---

### 2B_lidar_parked_cars_detection.py

Detects parked cars from LiDAR point clouds using PointPillars.

The script transforms detections into world coordinates, clusters duplicate detections, saves the resulting parked-car clusters, and opens an interactive Open3D viewer.

Default input:

```text
data/raw_dataset/<bag_name>/odometry.csv
data/raw_dataset/<bag_name>/lidar_positions.txt
data/raw_dataset/<bag_name>/point_clouds/*.bin
```

Outputs:

```text
data/processed_dataset/<bag_name>/lidar_detections/
├── lidar_detections.json
├── unified_clusters.txt
├── lidar_bboxes.txt
└── screenshots/
```

Run:

```bash
python 2_process_datasets/2B_lidar_parked_cars_detection.py
```

Viewer controls:

| Key | Action |
|---|---|
| `S` | Save screenshot to `screenshots/` |
| `Q` | Exit viewer |

You can also close the viewer window to exit.

---

### 2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py

Runs the same LiDAR detection pipeline as `2B_lidar_parked_cars_detection.py`, then opens an interactive editor for manual refinement.

This script is optional. Use it when you need a manually refined LiDAR ground truth, for example to evaluate detection quality.

Default input:

```text
data/raw_dataset/<bag_name>/odometry.csv
data/raw_dataset/<bag_name>/lidar_positions.txt
data/raw_dataset/<bag_name>/point_clouds/*.bin
```

Outputs:

```text
data/processed_dataset/<bag_name>/lidar_refinement/
├── detections_raw.json
├── ground_truth_refined.json
├── final_clusters.txt
├── ground_truth_bboxes.txt
└── screenshots/
```

Run:

```bash
python 2_process_datasets/2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py
```

Editor controls:

| Key | Action |
|---|---|
| `LEFT` / `RIGHT` | Select previous / next box |
| `C` | Select the box closest to the camera |
| `W` / `S` | Move selected box forward / backward |
| `A` / `D` | Move selected box left / right |
| `R` / `F` | Move selected box up / down |
| `Q` / `E` | Rotate selected box |
| `X` | Delete selected box |
| `I` | Insert a new box |
| `U` | Recompute side and orientation for the selected box |
| `P` | Save screenshot |
| `SPACE` | Confirm and save refined boxes |
| `ESC` | Cancel without saving refined boxes |

If the editor is cancelled with `ESC`, only the raw detections are kept.

---

### 2C_create_map_from_coordinates_auto.py

Downloads OSM data for the area covered by the recorded trajectory and prepares the `maps/` folder used by step 3.

The script reads the trajectory, converts the first UTM pose to latitude/longitude, reverse-geocodes the location, downloads OSM data, and creates a placeholder `vehicle_data.json`.

Default input:

```text
data/raw_dataset/<bag_name>/trajectory.csv
```

Outputs:

```text
data/processed_dataset/<bag_name>/maps/
├── map.osm
└── vehicle_data.json
```

Run:

```bash
python 2_process_datasets/2C_create_map_from_coordinates_auto.py
```

The generated `vehicle_data.json` contains a placeholder `hero_car` entry. The real CARLA hero position is filled in later by the simulation step.

---

### 2C_create_map_from_coordinates_manual.py

Creates the same map output as the automatic map script, but uses a hardcoded address instead of deriving the area from the trajectory.

Use this script when:

- The trajectory does not exist yet.
- The trajectory is unreliable.
- You want to download a specific area manually.
- You want to manually click the hero car position on the map.

Example configuration:

```python
ADDRESS = "Guerickestraße, Alte Heide, Munich"
MAP_NAME = "reference_bag"
DIST = 200
```

Outputs:

```text
data/processed_dataset/<map_name>/maps/
├── map.osm
└── vehicle_data.json
```

Run:

```bash
python 2_process_datasets/2C_create_map_from_coordinates_manual.py
```

Depending on the selected mode, the GUI can be used to click the hero car position and, if needed, parking areas.

---

### 2D_manual_refinment_parked_cars_camera.py

Interactive cleaner for the camera-based cluster file.

The script loads camera clusters, plots them together with the recorded trajectory, and lets the user delete, move, rotate, change side, or insert clusters.

Default input:

```text
data/raw_dataset/<bag_name>/trajectory.csv
data/processed_dataset/<bag_name>/camera_detections/unified_clusters.txt
```

Output:

```text
data/processed_dataset/<bag_name>/camera_detections/unified_clusters_filtered.txt
```

Run:

```bash
python 2_process_datasets/2D_manual_refinment_parked_cars_camera.py
```

GUI controls:

| Mode / Button | Action |
|---|---|
| `DELETE` | Click a cluster to remove it |
| `ROTATE` | Click a cluster to toggle parallel / perpendicular |
| `SIDE` | Click a cluster to toggle left / right |
| `MOVE` | Click and drag a cluster to a new position |
| `INSERT` | Click on the map to add a new cluster |
| `Save Filtered` | Save the cleaned cluster file |

An optional ground-truth file can be overlaid for visual reference by setting `GROUND_TRUTH_FILE` near the top of the script.

---

### 2D_manual_refinment_parked_cars_lidar.py

Interactive cleaner for the LiDAR-based cluster file.

This script works like `2D_manual_refinment_parked_cars_camera.py`, but it loads the LiDAR clusters produced by `2B_lidar_parked_cars_detection.py`.

Default input:

```text
data/raw_dataset/<bag_name>/trajectory.csv
data/processed_dataset/<bag_name>/lidar_detections/unified_clusters.txt
```

Output:

```text
data/processed_dataset/<bag_name>/lidar_detections/unified_clusters_filtered.txt
```

Run:

```bash
python 2_process_datasets/2D_manual_refinment_parked_cars_lidar.py
```

The GUI controls are the same as in the camera refinement script.

---

### 2E_prepare_dataset_for_gaussian_splatting.py

Prepares images, sky masks, and overlapping splits for Gaussian Splatting training.

The script selects frames, crops the bottom of each image, generates sky masks, and creates overlapping image/mask/pose splits.

Default input:

```text
data/raw_dataset/<bag_name>/images/
data/raw_dataset/<bag_name>/images_positions.txt
```

Outputs:

```text
data/data_for_gaussian_splatting/<bag_name>/
├── _tmp_images_gs_1_of_<FRAME_SKIP>/
├── _tmp_sky_masks_gs_1_of_<FRAME_SKIP>/
├── images_gs_split_1_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_1_1_of_<FRAME_SKIP>/
├── frame_positions_split_1_1_of_<FRAME_SKIP>.txt
├── images_gs_split_2_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_2_1_of_<FRAME_SKIP>/
└── frame_positions_split_2_1_of_<FRAME_SKIP>.txt
```

Run:

```bash
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py
```

The first run may download SegFormer weights from the Hugging Face Hub and cache them locally. Later runs reuse the cached weights.

---

### 2F_TODO_extract_semantic_maps.py

Placeholder script for semantic-map extraction.

This script is marked `TODO` and is not part of the current standard pipeline.

Do not run this script unless it has been implemented.

---

## Suggested execution order

A typical workflow is:

```bash
# 3D detections
python 2_process_datasets/2A_camera_parked_cars_detection.py
python 2_process_datasets/2B_lidar_parked_cars_detection.py

# Optional LiDAR ground-truth refinement
python 2_process_datasets/2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py

# OSM map preparation: choose one
python 2_process_datasets/2C_create_map_from_coordinates_auto.py
# or
python 2_process_datasets/2C_create_map_from_coordinates_manual.py

# Optional cluster cleanup
python 2_process_datasets/2D_manual_refinment_parked_cars_camera.py
python 2_process_datasets/2D_manual_refinment_parked_cars_lidar.py

# Gaussian Splatting input preparation
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py
```

If you only need camera-based detections, you can skip the `2B*` scripts and `2D_manual_refinment_parked_cars_lidar.py`.

If you only need LiDAR-based detections, you can skip `2A_camera_parked_cars_detection.py` and `2D_manual_refinment_parked_cars_camera.py`.

---

## Output files summary

| Script | Main output |
|---|---|
| `2A_camera_parked_cars_detection.py` | `camera_detections/camera_detections.json`, `camera_detections/unified_clusters.txt`, `camera_detections/unified_bbox_overlays/` |
| `2B_lidar_parked_cars_detection.py` | `lidar_detections/lidar_detections.json`, `lidar_detections/unified_clusters.txt`, `lidar_detections/lidar_bboxes.txt`, `lidar_detections/screenshots/` |
| `2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py` | `lidar_refinement/detections_raw.json`, `lidar_refinement/ground_truth_refined.json`, `lidar_refinement/final_clusters.txt`, `lidar_refinement/ground_truth_bboxes.txt` |
| `2C_create_map_from_coordinates_auto.py` | `maps/map.osm`, `maps/vehicle_data.json` |
| `2C_create_map_from_coordinates_manual.py` | `maps/map.osm`, `maps/vehicle_data.json` |
| `2D_manual_refinment_parked_cars_camera.py` | `camera_detections/unified_clusters_filtered.txt` |
| `2D_manual_refinment_parked_cars_lidar.py` | `lidar_detections/unified_clusters_filtered.txt` |
| `2E_prepare_dataset_for_gaussian_splatting.py` | `data/data_for_gaussian_splatting/<bag_name>/` |
| `2F_TODO_extract_semantic_maps.py` | Not implemented |

---

## Notes

- Output folders for detection scripts are deleted and recreated at the start of each run. Move or rename old results before rerunning if you want to keep them.
- `SKIP_FRAMES` controls detection density. Lower values process more frames but increase runtime.
- The camera detection script uses empirical correction constants because the FCOS3D weights were trained on KITTI, while this project uses a different camera setup.
- The map scripts can optionally check whether CARLA is reachable. Set `NO_CARLA = True` near the top of the map scripts to skip this check when CARLA is not running.
- The Gaussian Splatting script uses hard links when possible to avoid duplicating image files. If hard links are not supported, it falls back to regular file copies.
- Do not edit generated files manually unless you are intentionally refining outputs. Prefer rerunning the relevant script from the project root.

</details>

