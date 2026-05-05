# Install

## Create conda env
conda create -n repl python=3.10 
conda activate repl  

OLD REQUIREMENTS
pip install numpy
pip install opencv-python
pip install rosbags
pip install pandas
pip install torch
pip install scipy
pip install mmengine
pip install mmdet3d
pip install -U openmim
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"

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
│   └── extracted_ros_data/
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
data/extracted_ros_data/<bag_name>/
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
data/extracted_ros_data/<bag_name>/images/frame_000000.png
data/extracted_ros_data/<bag_name>/images/frame_000001.png
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
data/extracted_ros_data/<bag_name>/images/
data/extracted_ros_data/<bag_name>/images_positions.txt
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
data/extracted_ros_data/<bag_name>/point_clouds/point_cloud_000000.bin
data/extracted_ros_data/<bag_name>/point_clouds/point_cloud_000001.bin
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
data/extracted_ros_data/<bag_name>/point_clouds/
data/extracted_ros_data/<bag_name>/lidar_positions.txt
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
data/extracted_ros_data/<bag_name>/odometry.csv
data/extracted_ros_data/<bag_name>/trajectory.csv
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
data/extracted_ros_data/<bag_name>/steering_pct.txt
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
data/extracted_ros_data/<bag_name>/steering_predictions.txt
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