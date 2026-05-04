# Install

## Create conda env
conda create -n repl python=3.10 
conda activate repl  
pip install numpy
pip install opencv-python
pip install rosbags

# ROS data Extraction

## `1A_extract_camera_and_odometry.py` — Odometry, Camera Sync & Trajectory Extraction

This script extracts **odometry and camera data** from a ROS bag using `rosbags`, and builds a **synchronized dataset** that aligns robot motion with camera frames.

- Reads a ROS bag using `AnyReader`
- Extracts data from:
  - `/odom` (robot pose)
  - `/gmsl_camera/front_narrow/image_raw` (camera frames)
- Converts and synchronizes data into a structured dataset
- Saves images and aligned trajectory information

### Outputs

All extracted data is saved in a structured dataset folder:
datasets/<bag_name>/
├── images/ # Extracted camera frames (PNG)
├── trajectory.txt # Full odometry trajectory
└── frame_positions.txt # Camera frames + synchronized odometry

## `1Aalt_extract_camera_only.py` — Camera Frame Extraction from ROS Bag

This script extracts **camera images** from a ROS bag file and saves them as a structured image dataset.

- Reads a ROS bag using `rosbags`
- Subscribes to the camera topic:
  - `/gmsl_camera/front_narrow/image_raw`
- Decodes ROS image messages into OpenCV format
- Saves each frame as a `.png` image

### Outputs

All extracted frames are saved in the following structure:
datasets/<bag_name>/
└── images/ # Extracted camera frames (PNG)

## `1B_extract_lidar.py` — Odometry Extraction from ROS Bag

This script extracts **odometry data** from a ROS bag file and saves it into a structured dataset for further processing.

- Reads a ROS bag file using `rosbag`
- Subscribes to the `/odom` topic (robot pose)
- Extracts position and orientation data from each message
- Stores data in a timestamped CSV file for synchronization

### Outputs

All extracted data is saved in the following structure:
data_extraction/
└──  poses.csv # Odometry data (position + orientation)

