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

# 1_extract_ROS_data

This folder contains the first step of the data-processing pipeline.  
The scripts read data from a ROS bag file and export selected sensor, odometry, steering, and model-output data into a structured dataset folder.

Each script is intended to be run from the project root as:

```bash
python 1_extract_ROS_data/<script_name>.py
```

For example:

```bash
python 1_extract_ROS_data/1A_camera_with_odometry.py
```

---

## Purpose

The goal of this step is to convert raw ROS bag messages into files that are easier to use in the following stages of the pipeline.

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
│           ├── odometry.txt
│           ├── trajectory.txt
│           ├── steering_pct.txt
│           └── steering_predictions.txt
```

Scripts write their output to:

```text
data/raw_dataset/<bag_name>/
```

where `<bag_name>` is automatically taken from the ROS bag filename.

## Requirements

Use the existing Conda environment named `data_extraction`.

Activate it before running any extraction script:

```bash
conda activate data_extraction
```

## Configuration

Each script has a configuration section near the top.

Typical configuration values include:

```python
bag_path = Path('data/raw_ros_data/reference_bag.bag')
```

and one or more ROS topic names, for example:

```python
cam_topic = "/gmsl_camera/front_narrow/image_raw"
lidar_topic = "/velodyne_points"
odom_topic = "/odom"
topic = "/vehicle/steering_pct"
```

Before running the scripts, check that:

1. The ROS bag exists at the configured `bag_path`.
2. The topic names match the topics in your bag.
3. You are running the command from the project root.

To inspect the available topics in a bag, run:

```bash
python 1_extract_ROS_data/1UTIL_print_bag_info.py
```

---


## Scripts

### 1UTIL_print_bag_info.py

Utility script for inspecting a ROS bag.

It prints:

- All available ROS topics
- Message type for each topic
- Number of messages per topic
- Start and end timestamps of the bag

Run:

```bash
python 1_extract_ROS_data/1UTIL_print_bag_info.py
```

Use this script before extraction to confirm the correct topic names for camera, LiDAR, odometry, and steering messages.

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

This script only extracts images. It does not associate the camera frames with odometry.

Supported image encodings include:

- `mono8`
- `rgb8`
- `bgr8`
- Bayer encodings

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
FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile
```

Run:

```bash
python 1_extract_ROS_data/1A_camera_with_odometry.py
```

This script interpolates odometry values at each camera timestamp.

Position values are linearly interpolated. Quaternion components are also interpolated and then normalized before yaw is computed.

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

Each `.bin` file stores points in the following format:

```text
x, y, z, intensity
```

Run:

```bash
python 1_extract_ROS_data/1B_lidar_without_odometry.py
```

This script only extracts point clouds. It does not associate the LiDAR scans with odometry.

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

The synchronized metadata file contains:

```text
FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, PointCloudFile
```

Run:

```bash
python 1_extract_ROS_data/1B_lidar_with_odometry.py
```

This script interpolates odometry position and yaw at each LiDAR timestamp.

Each point cloud is saved as a `.bin` file containing:

```text
x, y, z, intensity
```

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
| `1UTIL_print_bag_info.py` | Console printout of topics, message counts, and bag time range |
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
- If the configured bag file is not found, the scripts print an error message and stop.
- If a required topic is missing, the corresponding script raises an error or prints a warning.
- The scripts assume a single ROS bag file configured by `bag_path`.
- Make sure the topic names in each script match the topics in your ROS bag.
- Run `1UTIL_print_bag_info.py` first when working with a new bag.


# 2_process_datasets

This folder contains the second step of the data-processing pipeline.
The scripts take the data extracted in step 1 and run computer vision and
3D detection algorithms on it. They produce per-dataset detection files
and the OSM-derived map data that the simulation in step 3 will need.

Each script is intended to be run from the project root as:

```bash
python 2_process_datasets/<script_name>.py
```

For example:

```bash
python 2_process_datasets/2A_camera_parked_cars_detection.py
```

---

## Purpose

The goal of this step is to take the raw extracted data from step 1 and produce:

- 3D bounding boxes of parked vehicles, detected from camera images
- 3D bounding boxes of parked vehicles, detected from LiDAR point clouds
- A manually refined LiDAR ground truth, built on top of the LiDAR detections
- A cleaned set of camera and/or LiDAR centroids, edited interactively
- An OSM map of the area covered by the trajectory, plus a placeholder
  `vehicle_data.json` file for the CARLA simulation
- Cropped images, sky masks, and overlapping splits ready for Gaussian
  Splatting training

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
│   │       └── trajectory.txt
│   │
│   └── processed_dataset/
│       └── <bag_name>/
│           ├── camera_detections/
│           ├── lidar_detections/
│           ├── lidar_refinement/
│           ├── maps/
│           └── (data_for_gaussian_splatting/<bag_name>/ at project root)
```

Scripts read input from:

```text
data/raw_dataset/<bag_name>/
```

and write their output to:

```text
data/processed_dataset/<bag_name>/
```

The Gaussian Splatting preparation script writes to a separate folder
under `data/data_for_gaussian_splatting/<bag_name>/` because the data
will be used by an external training pipeline.

---

## Requirements

Use the existing Conda environment named `repl`.

Activate it before running any script in this folder:

```bash
conda activate repl
```

The detection scripts also require pretrained model files placed in:

```text
2_process_datasets/utils/
├── fcos3d_config.py
├── fcos3d.pth
├── my_pointpillars_config.py
└── hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_*.pth
```

If any of these files is missing, the corresponding script will fail at
startup with a clear `FileNotFoundError`.

The map and Gaussian Splatting scripts do not need any model files.

---

## Configuration

Each script has a configuration section near the top.

The most important value to set is the dataset name:

```python
DATASET_NAME = "reference_bag"
```

This must match the name of the folder produced by step 1 inside
`data/raw_dataset/`.

Other values that may be edited from time to time are:

```python
SKIP_FRAMES = 5            # Process one frame every N
CONFIDENCE_THRESH = 0.50   # Detection score threshold
DIST = 200                 # OSM extraction radius, meters
ADDRESS = "..."            # Used only by the manual map script
```

Before running any script, check that:

1. The folder `data/raw_dataset/<DATASET_NAME>/` exists.
2. It contains the input files required by that specific script
   (for example, `images/` and `images_positions.txt` for camera scripts,
   or `point_clouds/`, `lidar_positions.txt` and `odometry.csv` for
   LiDAR scripts).
3. You are running the command from the project root.

---

## Scripts

### 2A_camera_parked_cars_detection.py

Detects parked cars from camera images using FCOS3D, tracks them in world
coordinates across frames, and clusters duplicate detections into a single
cluster per car.

The pipeline performs the following steps:

1. Loads the per-frame poses from `images_positions.txt`.
2. Loads the FCOS3D 3D detector.
3. Fits a smooth trajectory through the ego positions.
4. Iterates over the camera frames (one every `SKIP_FRAMES`), runs FCOS3D,
   transforms each detection from camera coordinates to world coordinates,
   and feeds it to a world tracker.
5. Runs a two-stage clustering on the confirmed tracks to merge duplicate
   detections of the same car.
6. Saves the cluster positions, sides, and orientations.

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

`camera_detections.json` contains the full per-cluster information:

```text
id, x, y, z, length, width, height, yaw, count, conf, side, orient
```

`unified_clusters.txt` contains a simplified textual version:

```text
# cluster_id, x, y, z, count, conf, orientation, side
```

`unified_bbox_overlays/` contains one PNG per processed frame, with the 3D
detected bounding boxes drawn in green and the projected centroid drawn as
a red dot. Use these images to visually validate the detection quality.

Run:

```bash
python 2_process_datasets/2A_camera_parked_cars_detection.py
```

GPU is used automatically if available. The first frame triggers a CUDA
warm-up, so it is slower than the others.

---

### 2B_lidar_parked_cars_detection.py

Detects parked cars from LiDAR point clouds using PointPillars, transforms
the detections into world coordinates, clusters duplicates, and shows the
result in an interactive Open3D viewer.

The pipeline performs the following steps:

1. Loads odometry from `odometry.csv` and the LiDAR index from
   `lidar_positions.txt`.
2. Loads the PointPillars 3D detector.
3. Fits a smooth trajectory through the ego positions.
4. Iterates over the LiDAR scans (one every `SKIP_FRAMES`), runs the
   detector, transforms each detected box into world coordinates, and
   collects the detections.
5. Clusters detections that are spatially close, computes a confidence-
   weighted average box per cluster, and assigns side and orientation
   relative to the trajectory.
6. Saves the clusters and shows them in a 3D viewer where you can save
   screenshots.

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

In the 3D viewer:

- Press `S` to save a screenshot to the `screenshots/` folder.
- Press `Q` or close the window to exit.

---

### 2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py

Same LiDAR detection pipeline as `2B_lidar_parked_cars_detection.py`, plus
an interactive editor used to manually refine the bounding boxes and
produce a clean ground truth.

This script is marked `OPTIONAL` because it is only needed when a refined
LiDAR ground truth is required, for example to evaluate detection quality.
The CARLA simulation in step 3 does not depend on it.

The pipeline performs the following steps:

1. Runs the full LiDAR detection and clustering, exactly like
   `2B_lidar_parked_cars_detection.py`.
2. Saves the raw clusters before any manual editing.
3. Opens an interactive Open3D editor where each detected box can be
   moved, rotated, deleted, or replaced. New boxes can also be inserted.
4. After the editor is confirmed, recomputes side and orientation for
   every box and saves a refined ground truth.

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
| `LEFT` / `RIGHT` | Select previous / next box (sorted spatially) |
| `C` | Select the box closest to the camera |
| `W` / `S` | Move selected box forward / backward |
| `A` / `D` | Move selected box left / right |
| `R` / `F` | Move selected box up / down |
| `Q` / `E` | Rotate selected box |
| `X` | Delete selected box |
| `I` | Insert a new box at the current selection |
| `U` | Recompute side and orientation for the selected box |
| `P` | Save a screenshot |
| `SPACE` | Confirm and save refined boxes |
| `ESC` | Cancel without saving |

If the editor is cancelled with `ESC`, only the raw detections are kept.

---

### 2C_create_map_from_coordinates_auto.py

Downloads OSM data for the area covered by the recorded trajectory and
prepares the `maps/` folder used by step 3.

The pipeline performs the following steps:

1. Reads the first pose from `trajectory.txt` and converts the UTM
   coordinates into latitude / longitude.
2. Reverse-geocodes the coordinates with Nominatim to obtain a road-level
   address (street, neighbourhood, city).
3. Downloads OSM data for that address using `get_street_data`.
4. Creates the `maps/<bag_name>/` folder structure and saves the OSM data,
   the processed map geometry, and a placeholder `vehicle_data.json`.

The placeholder `vehicle_data.json` contains a dummy `hero_car` entry.
The real CARLA hero position is filled in later by the simulation step.

Default input:

```text
data/raw_dataset/<bag_name>/trajectory.txt
```

Outputs:

```text
data/processed_dataset/<bag_name>/maps/
├── map.osm
├── map.xodr        (created by step 3, not by this script)
└── vehicle_data.json
```

Run:

```bash
python 2_process_datasets/2C_create_map_from_coordinates_auto.py
```

This script does not open a GUI when `MODE = "auto"`. Use it when the
trajectory uniquely identifies the area and no manual selection is needed.

---

### 2C_create_map_from_coordinates_manual.py

Same map-preparation script as `2C_create_map_from_coordinates_auto.py`,
but the OSM area is selected from a hardcoded address instead of from the
trajectory.

Use this script when:

- The trajectory does not yet exist or is unreliable.
- You want to download a specific area (for example, only one street).
- You want to manually click the hero car position on the map.

Set the address near the top of the script:

```python
ADDRESS = "Guerickestraße, Alte Heide, Munich"
MAP_NAME = "reference_bag"
DIST = 200
```

The script always opens the GUI:

- In `MODE = "manual"`, click the hero car and the parking areas.
- In `MODE = "auto"`, click only the hero car. Parking areas will be
  injected later from cluster files.

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

---

### 2D_manual_refinment_parked_cars_camera.py

Interactive cleaner for the camera-based cluster file.

Loads `unified_clusters.txt` from `camera_detections/`, plots all centroids
on a Web Mercator basemap together with the recorded trajectory, and lets
the user delete, move, rotate, or insert clusters.

Default input:

```text
data/raw_dataset/<bag_name>/trajectory.txt
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

| Mode | Action |
|---|---|
| `DELETE` | Click a cluster to remove it |
| `ROTATE` | Click a cluster to toggle parallel / perpendicular |
| `SIDE`   | Click a cluster to toggle left / right |
| `MOVE`   | Click and drag a cluster to a new position |
| `INSERT` | Click on the map to add a new cluster (defaults to right / parallel) |
| `Save Filtered` | Write the cleaned file to disk |

An optional ground-truth file can be overlaid for visual reference by
setting `GROUND_TRUTH_FILE` near the top of the script.

---

### 2D_manual_refinment_parked_cars_lidar.py

Same interactive cleaner as `2D_manual_refinment_parked_cars_camera.py`,
but loads the LiDAR cluster file produced by
`2B_lidar_parked_cars_detection.py`.

Default input:

```text
data/raw_dataset/<bag_name>/trajectory.txt
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

The GUI controls are identical to those of
`2D_manual_refinment_parked_cars_camera.py`.

---

### 2E_prepare_dataset_for_gaussian_splatting.py

Prepares images, sky masks, and overlapping splits for Gaussian Splatting
training.

The pipeline performs the following steps:

1. Selects one image every `FRAME_SKIP` frames from the original
   `images_positions.txt`.
2. Crops `CROP_BOTTOM` pixels from the bottom of each image (used to
   remove the visible part of the ego vehicle).
3. Generates a sky mask for each cropped image using a SegFormer model
   pretrained on Cityscapes. Sky pixels are set to 0, all other pixels
   are set to 255.
4. Splits the resulting frames into `NUM_SPLITS` overlapping chunks, with
   `OVERLAP_FRAMES` original frames of overlap between consecutive splits.
5. For each split, writes a folder of images, a folder of sky masks, and a
   filtered position file.

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

The first run downloads the SegFormer weights from the Hugging Face Hub
and caches them locally. Subsequent runs reuse the cached weights.

---

### 2F_TODO_extract_semantic_maps.py

Placeholder script for semantic-map extraction.

This script is marked `TODO` and is not part of the current replication
package. It will be filled in in a future revision.

Do not run this script as part of the standard pipeline.

---

## Suggested execution order

A typical workflow is:

```bash
# 3D detections
python 2_process_datasets/2A_camera_parked_cars_detection.py
python 2_process_datasets/2B_lidar_parked_cars_detection.py

# Optional manual refinement of LiDAR ground truth
python 2_process_datasets/2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py

# OSM map preparation (pick one of the two)
python 2_process_datasets/2C_create_map_from_coordinates_auto.py
# or
python 2_process_datasets/2C_create_map_from_coordinates_manual.py

# Optional cluster cleanup
python 2_process_datasets/2D_manual_refinment_parked_cars_camera.py
python 2_process_datasets/2D_manual_refinment_parked_cars_lidar.py

# Gaussian Splatting input preparation
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py
```

If you only need camera-based detections, you can skip the `2B*` and
`2D_manual_refinment_parked_cars_lidar.py` steps. If you only need
LiDAR-based detections, you can skip `2A` and
`2D_manual_refinment_parked_cars_camera.py`.

---

## Output files summary

| Script | Main output |
|---|---|
| `2A_camera_parked_cars_detection.py` | `camera_detections/camera_detections.json`, `unified_clusters.txt`, `unified_bbox_overlays/` |
| `2B_lidar_parked_cars_detection.py` | `lidar_detections/lidar_detections.json`, `unified_clusters.txt`, `lidar_bboxes.txt`, `screenshots/` |
| `2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py` | `lidar_refinement/detections_raw.json`, `ground_truth_refined.json`, `final_clusters.txt`, `ground_truth_bboxes.txt`, `screenshots/` |
| `2C_create_map_from_coordinates_auto.py` | `maps/<bag_name>/map.osm`, `vehicle_data.json` |
| `2C_create_map_from_coordinates_manual.py` | `maps/<map_name>/map.osm`, `vehicle_data.json` |
| `2D_manual_refinment_parked_cars_camera.py` | `camera_detections/unified_clusters_filtered.txt` |
| `2D_manual_refinment_parked_cars_lidar.py` | `lidar_detections/unified_clusters_filtered.txt` |
| `2E_prepare_dataset_for_gaussian_splatting.py` | `data/data_for_gaussian_splatting/<bag_name>/` (cropped images, sky masks, splits, position files) |
| `2F_TODO_extract_semantic_maps.py` | (not implemented yet) |

---

## Notes

- Output folders for the detection scripts are wiped and recreated at
  the start of every run. Move or rename them if you want to keep
  results from previous runs.
- `SKIP_FRAMES` controls how dense the detection pass is: with
  `SKIP_FRAMES = 5` one frame out of five is processed. Lower values
  give more detections but slower runtime.
- The position-correction constants in `2A`
  (`DEPTH_SCALE`, `DEPTH_OFFSET`, `X_SCALE`, `X_OFFSET`, `Y_OFFSET`)
  compensate for the fact that the pretrained FCOS3D weights come from
  KITTI, while the camera used in this project has different intrinsics
  and a different mounting position. They are tuned empirically on the
  reference dataset.
- The map scripts can optionally check that CARLA is reachable.
  Set `NO_CARLA = True` near the top of each map script to skip that
  check while CARLA is not running.
- The Gaussian Splatting script uses hard links when possible to avoid
  duplicating image files between the `_tmp_*` folders and the per-split
  folders. On filesystems that do not support hard links, regular file
  copies are used instead.