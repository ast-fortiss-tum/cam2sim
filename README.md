# Setup

To run this pipeline you need:

1. **Conda** (Miniconda or Anaconda) — used to manage the Python environments.
2. **CARLA 0.9.15** — used by stages 3 and 5.

## Install Conda

Follow the official installation guide:

- Miniconda (recommended, lightweight): <https://www.anaconda.com/docs/getting-started/miniconda/install/overview>

After installation, restart the terminal and verify:

```bash
conda --version
```

## Install CARLA

Download and extract CARLA 0.9.15 following the official guide:

- CARLA 0.9.15 quick start: <https://carla.readthedocs.io/en/0.9.15/start_quickstart/>

After extracting, set the CARLA installation path inside the project. Open:

```text
3_generate_simulation_data/utils/config.py
```

and edit:

```python
CARLA_INSTALLATION_PATH = "/absolute/path/to/CARLA_0.9.15"
```

## Clone the repository

```bash
git clone <repo-url> cam2sim
cd cam2sim
```

All commands below assume the project root is the current directory.

---

## Conda environments

This pipeline uses two separate Conda environments. Each one isolates dependencies that would otherwise conflict.

### `data_extraction` (main environment)

Used by stages 1, 2, 3, and 5.

Create the environment with Python 3.10 and activate it:

```bash
conda create -n data_extraction python=3.10 -y
conda activate data_extraction
```

Install the Python packages listed in `data_extraction_requirements.txt`:

```bash
pip install -U pip setuptools wheel
pip install -r data_extraction_requirements.txt
```

Install the OpenMMLab packages with `mim` (this installs them inside the active environment):

```bash
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmdet3d==1.4.0
```

Verify the installation:

```bash
python -c "import torch, mmcv, mmdet, mmdet3d; print('OK')"
```

### `nerfstudio` (Gaussian Splatting / Nerfstudio training)

Used by stage 4.

Setup instructions for this environment are provided in section 4.

---

## Download pretrained model checkpoints

Stage 2 uses two pretrained 3D detection models that are too large to ship inside the Git repository:

- **FCOS3D** (camera-based 3D detection, ~1 GB)
- **PointPillars** (LiDAR-based 3D detection, ~20 MB)

Both must be placed inside `2_process_datasets/utils/` with their original filenames.

Make sure the `data_extraction` environment is active (so `gdown` is available), then run:

```bash
conda activate data_extraction

# Make sure gdown is available
pip install -U gdown

cd 2_process_datasets/utils

# FCOS3D
gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O fcos3d.pth

# PointPillars
gdown 1AGOR8C0tDUsWSSWTEc0fA7kysIE9-iol -O hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

cd ../..
```

Alternatively, download the files manually from these links and put them in `2_process_datasets/utils/`:

- FCOS3D: <https://drive.google.com/file/d/1JIKRFQQI9CmQARk21Q619TPkdS49Voel/view?usp=sharing>
- PointPillars: <https://drive.google.com/file/d/1AGOR8C0tDUsWSSWTEc0fA7kysIE9-iol/view?usp=sharing>

After downloading, the folder should look like:

```text
2_process_datasets/utils/
├── fcos3d_config.py
├── fcos3d.pth
├── my_pointpillars_config.py
├── hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
└── ...
```

Verify that both files exist:

```bash
ls -lh 2_process_datasets/utils/fcos3d.pth \
       2_process_datasets/utils/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
```

# Detailed description and usage of each script

<details>
<summary><code>1_extract_ROS_data</code></summary>

---

## Example ROS bag (optional)

To run this pipeline without recording your own data, an example ROS bag(~6 GB) is provided on Google Drive. It corresponds to the dataset referenced as `reference_bag` throughout the pipeline.

Make sure the `data_extraction` environment is active, then run:

```bash
conda activate data_extraction

gdown 1ka4dqG83aprB6FWjd0W0mWxyPZHsRfj9 -O data/raw_ros_data/reference_bag.bag
```

Alternatively, download the file manually from this link and place it in `data/raw_ros_data/`:

- Example bag: <https://drive.google.com/file/d/1ka4dqG83aprB6FWjd0W0mWxyPZHsRfj9/view?usp=sharing>

After downloading, the folder should look like:

```text
data/raw_ros_data/
└── reference_bag.bag
```

Verify the file:

```bash
ls -lh data/raw_ros_data/reference_bag.bag
```

If you use your own bag instead of the example one, update the `bag_path` value at the top of each script to point to it.

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

<details>
<summary><code>3_generate_simulation_data</code></summary>

# 3_generate_simulation_data

This folder contains the third step of the data-processing pipeline.

The scripts convert the processed dataset from step 2 into CARLA-ready simulation inputs. They transform the recorded trajectory and detected parked vehicles into CARLA coordinates, prepare `vehicle_data.json`, start CARLA if needed, visualize the generated positions, and set up a CARLA scene with the hero vehicle and parked cars.

Each script is intended to be run from the project root:

```bash
python 3_generate_simulation_data/<script_name>.py
```

For example:

```bash
python 3_generate_simulation_data/3A_transform_coordinates_to_carla.py
```

---

## Purpose

The goal of this step is to prepare all data needed to recreate the recorded scene inside CARLA.

The scripts can produce or use:

- CARLA-coordinate hero trajectory files
- Rear-axle trajectory files for accurate hero spawning
- CARLA-ready `vehicle_data.json`
- Parked-vehicle spawn positions from detected clusters
- OpenDRIVE map loading inside CARLA
- Visual checks for parked cars and trajectory alignment
- A prepared CARLA world containing the hero vehicle and parked cars

---

## Expected project structure

```text
project_root/
├── 3_generate_simulation_data/
│   ├── 3A_transform_coordinates_to_carla.py
│   ├── 3B_transform_parked_vehicles_to_carla.py
│   ├── 3C_setup_carla.py
│   ├── 3D_visualize_parked_spawn_positions.py
│   ├── 3E_visualize_trajectory.py
│   ├── 3F_generate_carla_scenario.py
│   └── utils/
│       ├── carla_simulator.py
│       ├── config.py
│       ├── coordinates.py
│       ├── map_data.py
│       └── other helper files
│
├── data/
│   ├── raw_dataset/
│   │   └── <bag_name>/
│   │       └── images_positions.txt
│   │
│   ├── processed_dataset/
│   │   └── <bag_name>/
│   │       ├── lidar_detections/
│   │       │   └── unified_clusters.txt
│   │       └── maps/
│   │           ├── map.osm
│   │           ├── map.xodr
│   │           └── vehicle_data.json
│   │
│   └── data_for_carla/
│       └── <bag_name>/
│           ├── vehicle_data.json
│           ├── trajectory_positions.json
│           ├── trajectory_positions_rear.json
│           ├── trajectory_positions_odom_yaw.json
│           └── trajectory_positions_rear_odom_yaw.json
```

Scripts read input from:

```text
data/raw_dataset/<bag_name>/
data/processed_dataset/<bag_name>/
```

and write CARLA-ready output to:

```text
data/data_for_carla/<bag_name>/
```

---

## Requirements

Use the existing Conda environment named `data_extraction`.

Activate it before running any script in this folder:

```bash
conda activate data_extraction
```

The scripts use packages for coordinate conversion, map processing, JSON generation, CARLA control, and visualization.

CARLA-related scripts also require:

- A working CARLA installation
- The Python `carla` package available in the active environment
- A valid CARLA installation path set in `3_generate_simulation_data/utils/config.py`
- A generated `map.xodr` file inside `data/processed_dataset/<bag_name>/maps/`

---

## The `utils/` subfolder

The `utils/` folder contains shared helper code and configuration used by the simulation-generation scripts.

Typical contents include:

```text
3_generate_simulation_data/utils/
├── carla_simulator.py
├── config.py
├── coordinates.py
├── map_data.py
└── other helper files
```

The files are used as follows:

| File | Purpose |
|---|---|
| `config.py` | CARLA connection settings, CARLA installation path, spawn offsets, hero vehicle type, and other shared constants |
| `carla_simulator.py` | CARLA helper functions for loading OpenDRIVE maps, setting synchronous mode, filtering vehicle blueprints, and reading XODR projection information |
| `coordinates.py` | Coordinate conversion utilities, including odometry / UTM to WGS84 conversion |
| `map_data.py` | Map and road-edge helper functions used to generate parked-car spawn positions |
| Other helper files | Additional shared utilities used by the scripts |

Do not move or rename files in `utils/` unless the corresponding import paths in the scripts are also updated.

Important values to check in `utils/config.py` include:

```python
CARLA_INSTALLATION_PATH = "..."
CARLA_IP = "127.0.0.1"
CARLA_PORT = 2000
HERO_VEHICLE_TYPE = "vehicle.tesla.model3"
```

Some scripts also use spawn-offset and heading constants from `utils/config.py`, for example:

```python
SPAWN_OFFSET_METERS
SPAWN_OFFSET_METERS_LEFT
SPAWN_OFFSET_METERS_RIGHT
CARLA_OFFSET_X
CARLA_OFFSET_Y
ROTATION_DEGREES
```

---

## Configuration

Each script has a configuration section near the top.

The most important value is the bag name:

```python
BAG_NAME = "reference_bag"
```

This must match the dataset folder created by the previous steps.

For example:

```text
data/raw_dataset/reference_bag/
data/processed_dataset/reference_bag/
```

The CARLA output will be written to:

```text
data/data_for_carla/reference_bag/
```

Before running the scripts, check that:

1. `data/raw_dataset/<BAG_NAME>/images_positions.txt` exists.
2. `data/processed_dataset/<BAG_NAME>/maps/map.osm` exists.
3. `data/processed_dataset/<BAG_NAME>/maps/map.xodr` exists.
4. `data/processed_dataset/<BAG_NAME>/maps/vehicle_data.json` exists.
5. If parked cars are needed, `data/processed_dataset/<BAG_NAME>/lidar_detections/unified_clusters.txt` exists.
6. `3_generate_simulation_data/utils/config.py` contains the correct CARLA paths and connection settings.
7. You are running the command from the project root.

---

## Scripts

### 3A_transform_coordinates_to_carla.py

Converts the recorded trajectory from odometry coordinates into CARLA coordinates.

This script reads `images_positions.txt`, uses the generated map projection from the OpenDRIVE map, converts the trajectory to CARLA coordinates, and writes multiple trajectory JSON files for later simulation.

Default input:

```text
data/raw_dataset/<bag_name>/images_positions.txt
data/processed_dataset/<bag_name>/maps/
data/processed_dataset/<bag_name>/maps/vehicle_data.json
```

Outputs:

```text
data/data_for_carla/<bag_name>/
├── vehicle_data.json
├── trajectory_positions.json
├── trajectory_positions_rear.json
├── trajectory_positions_odom_yaw.json
└── trajectory_positions_rear_odom_yaw.json
```

The output trajectory files contain CARLA transform entries with location and rotation data.

The four trajectory files are:

| File | Meaning |
|---|---|
| `trajectory_positions.json` | Center trajectory using kinematic yaw |
| `trajectory_positions_rear.json` | Rear-offset trajectory using kinematic yaw |
| `trajectory_positions_odom_yaw.json` | Center trajectory using odometry yaw from `images_positions.txt` |
| `trajectory_positions_rear_odom_yaw.json` | Rear-offset trajectory using odometry yaw |

The script also updates:

```text
data/data_for_carla/<bag_name>/vehicle_data.json
```

with the initial `hero_car` position and heading.

Run:

```bash
python 3_generate_simulation_data/3A_transform_coordinates_to_carla.py
```

---

### 3B_transform_parked_vehicles_to_carla.py

Converts parked-vehicle centroids from the processed LiDAR detections into CARLA spawn positions.

This script reads detected parked-car clusters, projects them into the CARLA/OpenDRIVE coordinate system, snaps them to generated parking lines, assigns heading based on side and orientation, and writes the result into the final CARLA `vehicle_data.json`.

Default input:

```text
data/processed_dataset/<bag_name>/maps/
data/processed_dataset/<bag_name>/maps/map.xodr
data/processed_dataset/<bag_name>/lidar_detections/unified_clusters.txt
```

Output:

```text
data/data_for_carla/<bag_name>/vehicle_data.json
```

The script preserves the existing `hero_car` entry if it was already written by `3A_transform_coordinates_to_carla.py`.

It overwrites or updates the `spawn_positions` field in `vehicle_data.json`.

The final `vehicle_data.json` has the following structure:

```text
{
  "offset": {
    "x": 0.0,
    "y": 0.0,
    "heading": 0.0
  },
  "dist": 200,
  "hero_car": {
    "position": [x, y, z],
    "heading": yaw
  },
  "spawn_positions": [...]
}
```

Run:

```bash
python 3_generate_simulation_data/3B_transform_parked_vehicles_to_carla.py
```

---

### 3C_setup_carla.py

Starts CARLA from the installation path configured in `utils/config.py`.

The script reads:

```python
CARLA_INSTALLATION_PATH
```

from:

```text
3_generate_simulation_data/utils/config.py
```

and runs:

```text
<Carla installation path>/CarlaUE4.sh
```

Run:

```bash
python 3_generate_simulation_data/3C_setup_carla.py
```

If CARLA is already running, this script is not needed.

If the CARLA path is missing or invalid, the script raises an error and prints the path that failed.

---

### 3D_visualize_parked_spawn_positions.py

Visualizes the generated parked-car spawn positions inside CARLA.

This script loads the OpenDRIVE map into CARLA, reads `spawn_positions` from `vehicle_data.json`, spawns static parked vehicles at those locations, and moves the spectator above the first parked car.

Default input:

```text
data/processed_dataset/<bag_name>/maps/map.xodr
data/data_for_carla/<bag_name>/vehicle_data.json
```

Run:

```bash
python 3_generate_simulation_data/3D_visualize_parked_spawn_positions.py
```

This script keeps running so the scene remains visible.

To exit:

```text
Ctrl+C
```

When interrupted, the script destroys the parked vehicles it spawned.

Use this script to check that parked-car positions and headings look correct before generating the final CARLA scenario.

---

### 3E_visualize_trajectory.py

Visualizes the converted hero trajectory inside CARLA.

This script loads the OpenDRIVE map, reads the converted trajectory, and spawns static “ghost” hero vehicles along the path. The cars are frozen in place so the trajectory alignment can be inspected visually.

Default input:

```text
data/processed_dataset/<bag_name>/maps/map.xodr
data/data_for_carla/<bag_name>/trajectory_positions_odom_yaw.json
data/data_for_carla/<bag_name>/vehicle_data.json
```

Run:

```bash
python 3_generate_simulation_data/3E_visualize_trajectory.py
```

Useful configuration values near the top of the script include:

```python
SPAWN_STEP = 5
SPAWN_Z_OFFSET = 0.1
MANUAL_OFFSET_X = 0.0
MANUAL_OFFSET_Y = 0.0
MANUAL_Z_OFFSET = 0.0
MANUAL_YAW_OFFSET = 0.0
```

This script keeps running so the trajectory remains visible.

To exit:

```text
Ctrl+C
```

When interrupted, the script destroys the ghost vehicles it spawned.

---

### 3F_generate_carla_scenario.py

Prepares the CARLA world with the OpenDRIVE map, parked vehicles, and the hero vehicle.

This script is used after the CARLA-ready trajectory and parked-car spawn positions have been generated. It loads the map, spawns parked vehicles, spawns the hero car at the first rear-trajectory pose, freezes all spawned actors, disables synchronous mode, and exits while leaving the actors alive in CARLA.

Default input:

```text
data/processed_dataset/<bag_name>/maps/map.xodr
data/data_for_carla/<bag_name>/vehicle_data.json
data/data_for_carla/<bag_name>/trajectory_positions_rear_odom_yaw.json
```

Run:

```bash
python 3_generate_simulation_data/3F_generate_carla_scenario.py
```

The script chooses the hero start in this order:

1. `trajectory_positions_rear_odom_yaw.json`
2. `trajectory_positions_rear.json`
3. `hero_car` from `vehicle_data.json`

By default, this script leaves the actors alive in CARLA and exits. This allows a later replay or execution script to take over the simulation without this setup script continuing to tick the world.

Important configuration values near the top of the script include:

```python
DESTROY_EXISTING_VEHICLES = True
DESTROY_ACTORS_ON_EXIT = False
LEAVE_WORLD_READY_AND_EXIT = True
USE_SYNCHRONOUS_MODE_DURING_PREP = True
MAX_PARKED_CARS = None
```

---

## Suggested execution order

A typical workflow is:

```bash
# Convert trajectory to CARLA coordinates
python 3_generate_simulation_data/3A_transform_coordinates_to_carla.py

# Convert detected parked vehicles to CARLA spawn positions
python 3_generate_simulation_data/3B_transform_parked_vehicles_to_carla.py

# Start CARLA if it is not already running
python 3_generate_simulation_data/3C_setup_carla.py

# Optional visual checks
python 3_generate_simulation_data/3D_visualize_parked_spawn_positions.py
python 3_generate_simulation_data/3E_visualize_trajectory.py

# Prepare final CARLA scenario
python 3_generate_simulation_data/3F_generate_carla_scenario.py
```

If CARLA is already running, skip `3C_setup_carla.py`.

If you only want to generate CARLA JSON files and do not need to inspect or spawn the scene yet, run only:

```bash
python 3_generate_simulation_data/3A_transform_coordinates_to_carla.py
python 3_generate_simulation_data/3B_transform_parked_vehicles_to_carla.py
```

---

## Output files summary

| Script | Main output |
|---|---|
| `3A_transform_coordinates_to_carla.py` | `data/data_for_carla/<bag_name>/trajectory_positions*.json`, `vehicle_data.json` with `hero_car` |
| `3B_transform_parked_vehicles_to_carla.py` | `data/data_for_carla/<bag_name>/vehicle_data.json` with `spawn_positions` |
| `3C_setup_carla.py` | Starts CARLA from `CARLA_INSTALLATION_PATH` |
| `3D_visualize_parked_spawn_positions.py` | Temporary parked vehicles spawned in CARLA for visual inspection |
| `3E_visualize_trajectory.py` | Temporary ghost hero vehicles spawned in CARLA for visual inspection |
| `3F_generate_carla_scenario.py` | CARLA world prepared with map, hero vehicle, and parked vehicles |

---

## Notes

- The scripts use a hardcoded `BAG_NAME`; change it near the top of each script before running a different dataset.
- Run the scripts from the project root so relative paths resolve correctly.
- `3A_transform_coordinates_to_carla.py` should usually be run before `3B_transform_parked_vehicles_to_carla.py`, because it writes the `hero_car` entry that `3B` preserves.
- `3B_transform_parked_vehicles_to_carla.py` requires `map.xodr`; this file is expected to exist in `data/processed_dataset/<bag_name>/maps/`.
- `3D_visualize_parked_spawn_positions.py`, `3E_visualize_trajectory.py`, and `3F_generate_carla_scenario.py` require a running CARLA server.
- The visualization scripts spawn temporary actors and clean them up when interrupted.
- `3F_generate_carla_scenario.py` is designed to leave the prepared actors alive in CARLA and then exit.
- If positions look slightly shifted in CARLA, check the offset constants in `utils/config.py` and the manual offset constants near the top of the visualization scripts.

</details>

<details>
<summary><code>4_gaussian_splatting_preparation</code></summary>

# 4_gaussian_splatting_preparation

This folder contains the fourth step of the data-processing pipeline.

The files in this folder prepare and train Gaussian Splatting models from the image splits created in step 2. The trained models are later used during simulation to render camera views that resemble the original real-world recording.

This step uses the data produced by:

```bash
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py
```

That script creates image splits, sky-mask splits, and filtered pose files under:

```text
data/data_for_gaussian_splatting/<bag_name>/
```

---

## Purpose

The goal of this step is to train one Gaussian Splatting model per route segment and compute the coordinate alignment needed to query the trained models from simulation.

This step can produce:

- COLMAP sparse reconstructions for each image split
- One Nerfstudio / Gaussian Splatting model per split
- A transform from dataset coordinates to Nerfstudio coordinates
- Model outputs used later for GS-rendered simulation views

---

## Expected project structure

```text
project_root/
├── 4_gaussian_splatting_preparation/
│   ├── 4A_colmap_guide.md
│   ├── 4B_train_gaussian_splatting.sh
│   └── 4C_utm_yaw_to_nerfstudio.py
│
├── data/
│   └── data_for_gaussian_splatting/
│       └── <bag_name>/
│           ├── images_gs_split_1_1_of_<FRAME_SKIP>/
│           ├── sky_masks_gs_split_1_1_of_<FRAME_SKIP>/
│           ├── frame_positions_split_1_1_of_<FRAME_SKIP>.txt
│           ├── images_gs_split_2_1_of_<FRAME_SKIP>/
│           ├── sky_masks_gs_split_2_1_of_<FRAME_SKIP>/
│           ├── frame_positions_split_2_1_of_<FRAME_SKIP>.txt
│           ├── images_gs_split_3_1_of_<FRAME_SKIP>/
│           ├── sky_masks_gs_split_3_1_of_<FRAME_SKIP>/
│           ├── frame_positions_split_3_1_of_<FRAME_SKIP>.txt
│           ├── colmap/
│           │   ├── database_split_1.db
│           │   ├── database_split_2.db
│           │   ├── database_split_3.db
│           │   ├── split_1/
│           │   │   └── sparse/
│           │   │       └── 0/
│           │   │           ├── cameras.bin
│           │   │           ├── images.bin
│           │   │           └── points3D.bin
│           │   ├── split_2/
│           │   │   └── sparse/
│           │   │       └── 0/
│           │   │           ├── cameras.bin
│           │   │           ├── images.bin
│           │   │           └── points3D.bin
│           │   └── split_3/
│           │       └── sparse/
│           │           └── 0/
│           │               ├── cameras.bin
│           │               ├── images.bin
│           │               └── points3D.bin
│           └── outputs/
│               ├── split_1/
│               ├── split_2/
│               └── split_3/
```

The number of splits depends on the configuration used in step 2.

---

## Requirements

Use the existing Conda environment named `nerfstudio`.

Activate it before running the training or alignment scripts:

```bash
conda activate nerfstudio
```

This step requires:

- `nerfstudio`
- `torch`
- `numpy`
- `COLMAP`
- A CUDA-capable GPU for practical training

Check that Nerfstudio is available with:

```bash
ns-train --help
```

Check that COLMAP is available with:

```bash
colmap -h
```

To open the COLMAP graphical interface, run:

```bash
colmap gui
```

---

## Input from step 2

Before running this step, make sure step 2 has produced the Gaussian Splatting input data:

```text
data/data_for_gaussian_splatting/<bag_name>/
├── images_gs_split_1_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_1_1_of_<FRAME_SKIP>/
├── frame_positions_split_1_1_of_<FRAME_SKIP>.txt
├── images_gs_split_2_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_2_1_of_<FRAME_SKIP>/
├── frame_positions_split_2_1_of_<FRAME_SKIP>.txt
├── images_gs_split_3_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_3_1_of_<FRAME_SKIP>/
└── frame_positions_split_3_1_of_<FRAME_SKIP>.txt
```

If these folders do not exist, run:

```bash
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py
```

---

## Configuration

The training script has a configuration section near the top:

```bash
BAG_NAME="reference_bag"
NUM_SPLITS=3
FRAME_SKIP=3
METHOD="splatfacto"
CONDA_ENV="nerfstudio"
```

Set `BAG_NAME` to the dataset name under:

```text
data/data_for_gaussian_splatting/
```

Set `NUM_SPLITS` and `FRAME_SKIP` to match the values used in step 2.

For Gaussian Splatting, use:

```bash
METHOD="splatfacto"
```

or, for a larger model:

```bash
METHOD="splatfacto-big"
```

---

## Files

### 4A_colmap_guide.md

Guide for manually running COLMAP reconstruction for each image split.

COLMAP must be run once for each split produced by step 2. For example, if step 2 was configured with:

```bash
NUM_SPLITS=3
FRAME_SKIP=3
```

then the Gaussian Splatting data folder contains:

```text
data/data_for_gaussian_splatting/reference_bag/
├── images_gs_split_1_1_of_3/
├── sky_masks_gs_split_1_1_of_3/
├── images_gs_split_2_1_of_3/
├── sky_masks_gs_split_2_1_of_3/
├── images_gs_split_3_1_of_3/
└── sky_masks_gs_split_3_1_of_3/
```

For each split, follow the procedure below.

#### Step 1: Open COLMAP

Run:

```bash
colmap gui
```

#### Step 2: Create a new COLMAP project

In the COLMAP GUI:

```text
File -> New Project
```

Create a new database for the current split.

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/colmap/database_split_1.db
```

Then select the image folder of the current split.

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/images_gs_split_1_1_of_3
```

Use the same naming pattern for the other splits:

```text
data/data_for_gaussian_splatting/reference_bag/colmap/database_split_2.db
data/data_for_gaussian_splatting/reference_bag/images_gs_split_2_1_of_3

data/data_for_gaussian_splatting/reference_bag/colmap/database_split_3.db
data/data_for_gaussian_splatting/reference_bag/images_gs_split_3_1_of_3
```

#### Step 3: Run feature extraction

In the COLMAP GUI, go to:

```text
Processing -> Feature Extraction
```

Use the camera model:

```text
OPENCV
```

Enable:

```text
Single camera
```

Set the camera parameters according to the dataset calibration.

The COLMAP `OPENCV` camera parameter order is:

```text
fx, fy, cx, cy, k1, k2, p1, p2
```

For the front narrow camera, use:

```text
785.34926249, 784.07587341, 406.50794975, 249.45341029, -0.42020115, 0.64296938, -0.00531934, -0.00215015
```

Optional but recommended: select the sky-mask folder for the same split.

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/sky_masks_gs_split_1_1_of_3
```

Then run feature extraction.

#### Step 4: Run feature matching

In the COLMAP GUI, go to:

```text
Processing -> Feature Matching
```

Select:

```text
Sequential matching
```

Set:

```text
Sequential overlap: 10
```

Then run matching.

#### Step 5: Configure reconstruction options

In the COLMAP GUI, go to:

```text
Reconstruction -> Reconstruction Options
```

Open the `Bundle Adjustment` options.

To keep the calibrated camera parameters fixed, disable intrinsic refinement options such as:

```text
Refine focal length
Refine principal point
Refine extra parameters
```

Then start reconstruction:

```text
Reconstruction -> Start Reconstruction
```

Wait until COLMAP finishes.

#### Step 6: Export the sparse model

After reconstruction finishes, export the sparse model for the current split into the matching split folder.

Use this export layout:

```text
Split 1 -> data/data_for_gaussian_splatting/reference_bag/colmap/split_1/sparse/0
Split 2 -> data/data_for_gaussian_splatting/reference_bag/colmap/split_2/sparse/0
Split 3 -> data/data_for_gaussian_splatting/reference_bag/colmap/split_3/sparse/0
```

Each exported sparse model folder must contain:

```text
cameras.bin
images.bin
points3D.bin
```

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/colmap/split_1/sparse/0/
├── cameras.bin
├── images.bin
└── points3D.bin
```

#### Step 7: Repeat for every split

Repeat the full COLMAP procedure for each split.

At the end, the expected folder structure is:

```text
data/data_for_gaussian_splatting/reference_bag/
├── images_gs_split_1_1_of_3/
├── images_gs_split_2_1_of_3/
├── images_gs_split_3_1_of_3/
├── frame_positions_split_1_1_of_3.txt
├── frame_positions_split_2_1_of_3.txt
├── frame_positions_split_3_1_of_3.txt
└── colmap/
    ├── split_1/
    │   └── sparse/
    │       └── 0/
    │           ├── cameras.bin
    │           ├── images.bin
    │           └── points3D.bin
    ├── split_2/
    │   └── sparse/
    │       └── 0/
    │           ├── cameras.bin
    │           ├── images.bin
    │           └── points3D.bin
    └── split_3/
        └── sparse/
            └── 0/
                ├── cameras.bin
                ├── images.bin
                └── points3D.bin
```

---

### 4B_train_gaussian_splatting.sh

Trains one Nerfstudio model per split.

Run:

```bash
bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh
```

For each split, the script checks that the COLMAP reconstruction exists:

```text
colmap/split_<N>/sparse/0/cameras.bin
colmap/split_<N>/sparse/0/images.bin
colmap/split_<N>/sparse/0/points3D.bin
```

It then runs `ns-train` using:

- the split image folder
- the split sky-mask folder, if available
- the split COLMAP reconstruction

Outputs are written to:

```text
data/data_for_gaussian_splatting/<bag_name>/outputs/
```

Example output folders:

```text
data/data_for_gaussian_splatting/<bag_name>/outputs/split_1/
data/data_for_gaussian_splatting/<bag_name>/outputs/split_2/
data/data_for_gaussian_splatting/<bag_name>/outputs/split_3/
```

If one split is missing a reconstruction, the script skips it and continues with the remaining splits.

---

### 4C_utm_yaw_to_nerfstudio.py

Computes the alignment between the dataset coordinate system and the trained Nerfstudio model coordinate system.

This is needed because Nerfstudio applies internal transformations to the COLMAP reconstruction, including centering, scaling, and axis changes. Therefore, raw dataset coordinates cannot be used directly to query the trained GS model.

The script:

1. Loads a trained Nerfstudio model from its `config.yml`.
2. Extracts the camera poses used internally by Nerfstudio.
3. Matches those camera poses with the original frame-position file.
4. Computes a 2D similarity transform from dataset coordinates to Nerfstudio coordinates.
5. Computes the yaw mapping between dataset yaw and Nerfstudio yaw.
6. Saves the transform as a JSON file for later inference.

Example run:

```bash
python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
  --gs_config data/data_for_gaussian_splatting/reference_bag/outputs/split_1/<run_name>/config.yml \
  --utm_file data/data_for_gaussian_splatting/reference_bag/frame_positions_split_1_1_of_3.txt \
  --data_root data/data_for_gaussian_splatting/reference_bag \
  --output data/data_for_gaussian_splatting/reference_bag/outputs/split_1/utm_to_nerfstudio_transform.json
```

The output JSON contains:

- scale
- rotation
- translation
- yaw sign and yaw offset
- camera pitch and roll estimates
- matching and alignment error statistics

This file is used later to convert simulation poses into the coordinate system expected by the GS renderer.

---

## Suggested execution order

A typical workflow is:

```bash
# 1. Prepare GS image splits in step 2
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py

# 2. Run COLMAP once for each split
# Follow the COLMAP instructions above.

# 3. Train one GS model per split
bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh

# 4. Compute dataset-to-Nerfstudio alignment for each trained split
python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
  --gs_config <path_to_split_config.yml> \
  --utm_file <path_to_frame_positions_split.txt> \
  --data_root data/data_for_gaussian_splatting/<bag_name> \
  --output <path_to_output_transform.json>
```

---

## Output files summary

| File / script | Main output |
|---|---|
| `4A_colmap_guide.md` | COLMAP sparse reconstructions under `colmap/split_<N>/sparse/0/` |
| `4B_train_gaussian_splatting.sh` | Trained Nerfstudio / GS models under `outputs/` |
| `4C_utm_yaw_to_nerfstudio.py` | `utm_to_nerfstudio_transform.json` alignment files |

---

## Notes

- COLMAP must be run once per split before training.
- The COLMAP export folder for each split must contain `cameras.bin`, `images.bin`, and `points3D.bin`.
- The values of `BAG_NAME`, `NUM_SPLITS`, and `FRAME_SKIP` in `4B_train_gaussian_splatting.sh` must match the output produced by step 2.
- Use `METHOD="splatfacto"` or `METHOD="splatfacto-big"` for Gaussian Splatting.
- Sky masks are optional, but recommended because they prevent sky regions from influencing the reconstruction.
- The front narrow camera parameters shown above are dataset-specific. Replace them if using another camera.
- Keep COLMAP intrinsic refinement disabled if you want to preserve the calibrated camera parameters.
- The alignment script should be run after training, because it loads the final Nerfstudio model coordinate system.
- One alignment file should be generated for each trained split.
- The trained GS models and alignment files are used by the simulation step to render camera observations from simulated poses.

</details>

