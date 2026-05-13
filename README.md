![Cam2Sim demo video](https://img.youtube.com/vi/KmZ74l1__lI/maxresdefault.jpg)

[▶ Watch the Cam2Sim screencast on YouTube](https://youtu.be/KmZ74l1__lI)

# Cam2Sim

**Neural Scenario Reconstruction for Closed-Loop Autonomous Driving Simulation**

Cam2Sim is a pipeline for transforming real-world driving recordings into executable **CARLA** simulation scenarios. Starting from camera images, ego poses, and optional LiDAR / steering signals, it reconstructs the road layout, ego trajectory, parked vehicles, and simulation-ready assets, then augments the scenario with **Gaussian Splatting** to render camera observations that resemble the original recording.

Cam2Sim is designed for researchers and practitioners working on simulation-based testing of autonomous driving systems, especially when the visual gap between real-world camera observations and simulator-rendered images matters for closed-loop behavior.

---

## Overview

<!--

Replace this placeholder with the workflow figure from the paper.

Suggested path:

docs/assets/cam2sim_overview.png

Example:

![Cam2Sim workflow overview](docs/assets/cam2sim_overview.png)

-->

![Cam2Sim workflow overview](docs/assets/ase_tool_schema.png)

Cam2Sim turns a real-world driving recording into a playable simulation scenario through six main stages:

1. **Data extraction**  

   Extract camera frames, ego poses, LiDAR point clouds, odometry, steering status, and model outputs from ROS bags.

2. **Dataset processing**  

   Detect parked vehicles from camera or LiDAR data, refine detections, prepare OpenStreetMap-based map data, and generate image splits for Gaussian Splatting.

3. **Simulation data generation**  

   Convert trajectories and parked-vehicle positions into CARLA coordinates, generate CARLA-ready JSON files, and prepare the OpenDRIVE scenario.

4. **Gaussian Splatting preparation**  

   Run COLMAP, train one Nerfstudio / Splatfacto model per route segment, and compute the coordinate alignment between dataset poses and Nerfstudio poses.

5. **Driving simulation**  

   Replay the recorded trajectory or run closed-loop DAVE-2 experiments in CARLA. The system under test can receive either raw CARLA images or GS-rendered camera observations.

6. **Validation**  

   Compare real and simulated semantic maps, replay outputs, steering behavior, and trajectory traces.

At the end of the pipeline, Cam2Sim produces CARLA-ready scenarios, trained Gaussian Splatting models, replay images, closed-loop driving traces, and validation outputs.

---

---

# Quick Start

This Quick Start lets you reproduce a single Gaussian Splatting trajectory replay (Step 5C) without running the full data-processing pipeline. It downloads a precomputed dataset and runs the GS replay directly. For the full pipeline (ROS bag extraction, COLMAP, Gaussian Splatting training, DAVE-2 closed-loop), see the [Setup](#setup) and [Detailed description](#detailed-description-and-usage-of-each-script) sections below.

## What it does

1. Loads a precomputed dataset of a real driving recording converted into a CARLA scenario.
2. Loads the Gaussian Splatting models trained on the same recording.
3. Starts CARLA, loads the map and parked vehicles, and replays the recorded trajectory.
4. Renders both the CARLA view and the Gaussian-Splatted view side by side.


## Setup

### Clone the repository

```bash
git clone <repo-url> cam2sim
cd cam2sim
```

All commands below assume the project root is the current directory.

### Install Conda

Follow the official guide:

- Miniconda (recommended): <https://www.anaconda.com/docs/getting-started/miniconda/install/overview>

After installation, restart the terminal and verify:

```bash
conda --version
```

### Install CARLA 0.9.15

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

### Install Nerfstudio

Follow the official guide:

- Nerfstudio installation: <https://docs.nerf.studio/quickstart/installation.html>

The guide walks you through creating a Conda environment named `nerfstudio` with the correct CUDA toolkit, PyTorch, `tinycudann`, `gsplat`, and Nerfstudio itself. **Keep the default environment name** so the rest of the pipeline can find it.

Verify the environment exists and works:

```bash
conda activate nerfstudio
ns-train --help
```

> Splatfacto (the GS model used here) requires a CUDA-capable GPU with compute capability 7.5 or higher (RTX 20-series or newer).

### Create the `data_extraction` environment

Needed by the CARLA-side scripts (3C, 3F) launched by Step 5.

```bash
conda create -n data_extraction python=3.10 -y
conda activate data_extraction

pip install -U pip setuptools wheel
pip install -r data_extraction_requirements.txt
```

### Download the precomputed dataset

The precomputed dataset contains everything Steps 1-4 would normally produce: the CARLA-ready trajectory, the OpenDRIVE map, the parked-vehicle JSON, and the trained Gaussian Splatting models with their alignment files.

From the project root, with the `data_extraction` environment active (so `gdown` is available):

```bash
conda activate data_extraction
pip install -U gdown

gdown 1MmAYlxy67F1oxDKADHl3yUZochmifV1Q -O data.zip
unzip -o data.zip
rm data.zip
```

Manual link: <https://drive.google.com/file/d/1MmAYlxy67F1oxDKADHl3yUZochmifV1Q/view?usp=sharing>

After extracting, the project should contain:

```text
cam2sim/
├── data/
│   ├── data_for_carla/reference_bag/
│   │   ├── camera.json
│   │   ├── trajectory_positions_rear_odom_yaw.json
│   │   └── vehicle_data.json
│   ├── processed_dataset/reference_bag/maps/
│   │   └── map.xodr
│   └── data_for_gaussian_splatting/reference_bag/
│       ├── frame_positions_split_*_1_of_2.txt
│       ├── images_gs_split_*_1_of_2/
│       └── outputs/splatfacto_split_*/splatfacto/<timestamp>/
│           ├── config.yml
│           ├── nerfstudio_models/
│           └── utm_to_nerfstudio_transform.json
```

## Run it

From the project root:

```bash
bash 5_execute_simulation/step5.sh 
```

`step5.sh` opens three terminals in sequence:

1. **CARLA server** (`3C_setup_carla.py`, env `data_extraction`)
2. **Map + parked vehicles** (`3F_generate_carla_scenario.py`, env `data_extraction`)
3. **Gaussian Splatting replay** (`5C_trajectory_replay.py`, env `nerfstudio`)

The replay opens a window showing CARLA on the left and the Gaussian-Splatted view on the right, frame by frame along the recorded trajectory. Output frames are saved under:

```text
data/data_for_carla/reference_bag/replay_results/reference_bag_replay/
├── carla/
├── gs/
└── combined/
```

## Tested setup

This pipeline has been tested on:

- **OS**: Ubuntu 20.04 LTS
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **NVIDIA driver**: 565.57.01 (required for CUDA 12.x)
- **CUDA**: 12.7 (compatible toolkit installed by Nerfstudio inside its Conda env)
- **CPU**: Intel Core Ultra 9
- **RAM**: 32 GB

Other Ubuntu versions (22.04, 24.04) and other CUDA-capable GPUs with compute capability ≥7.5 should also work, but have not been validated.

---
---

# Replication (full pipeline)

This section describes how to set up the **full pipeline**, from raw ROS bag through closed-loop DAVE-2 driving on Gaussian-Splatted views. It covers all six stages of Cam2Sim, including parked-vehicle detection, COLMAP, Gaussian Splatting training, and DAVE-2 closed-loop experiments.

If you only want to reproduce the trajectory replay with Gaussian Splatting, see the [Quick Start](#quick-start) above instead — it skips Steps 1-4 by downloading a precomputed dataset.

Sub-sections marked **[Quick Start already done]** can be skipped if you have already followed the Quick Start.


## Clone the repository  *[Quick Start already done]*

```bash
git clone <repo-url> cam2sim
cd cam2sim
```

All commands below assume the project root is the current directory.

---

## Software prerequisites

You need all of the following:

1. **Conda** (Miniconda or Anaconda) — manages the Python environments.
2. **CARLA 0.9.15** — used by stages 3 and 5.
3. **NVIDIA GPU** with a driver supporting CUDA 11.8 or higher.
4. **Nerfstudio** — used by stages 4 and 5. Installs the `nerfstudio` Conda environment.
5. **COLMAP** — used by stage 4 (sparse reconstruction before Gaussian Splatting training).

### Install Conda  *[Quick Start already done]*

- Miniconda (recommended, lightweight): <https://www.anaconda.com/docs/getting-started/miniconda/install/overview>

After installation, restart the terminal and verify:

```bash
conda --version
```

### Install CARLA 0.9.15  *[Quick Start already done]*

- CARLA 0.9.15 quick start: <https://carla.readthedocs.io/en/0.9.15/start_quickstart/>

After extracting, set the CARLA installation path inside `3_generate_simulation_data/utils/config.py`:

```python
CARLA_INSTALLATION_PATH = "/absolute/path/to/CARLA_0.9.15"
```

### Install Nerfstudio  *[Quick Start already done]*

- Nerfstudio installation: <https://docs.nerf.studio/quickstart/installation.html>

Use the default environment name (`nerfstudio`). Verify:

```bash
conda activate nerfstudio
ns-train --help
```

> Splatfacto requires a CUDA-capable GPU with compute capability 7.5 or higher (RTX 20-series or newer).

### Install COLMAP  *(new, not in Quick Start)*

Stage 4 uses COLMAP to compute a sparse reconstruction of the recorded images.

- COLMAP installation: <https://colmap.github.io/install.html>

On Ubuntu:

```bash
sudo apt install colmap
```

Verify:

```bash
colmap -h
```



## Conda environments

The pipeline uses three separate Conda environments. Each isolates dependencies that would otherwise conflict.

### `data_extraction` (main environment)  *[Quick Start already done]*

Used by stages 1, 2, 3, and 5 (for the CARLA-side scripts).

```bash
conda create -n data_extraction python=3.10 -y
conda activate data_extraction

pip install -U pip setuptools wheel
pip install -r data_extraction_requirements.txt
```

Stage 2 also requires the OpenMMLab packages installed with `mim`:

```bash
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmdet3d==1.4.0
```

Verify:

```bash
python -c "import torch, mmcv, mmdet, mmdet3d; print('OK')"
```

### `dave_2` (autonomous driving model)  *(new, not in Quick Start)*

Used by stage 5 to run the DAVE-2 steering model as a standalone TCP server (modes 5B and 5D). This environment is independent because TensorFlow 2.13 requires Python 3.8.

```bash
conda create -n dave_2 python=3.8 -y
conda activate dave_2

pip install -U pip setuptools wheel
pip install tensorflow==2.13.1
pip install pillow
pip install opencv-python
```

### `nerfstudio` (Gaussian Splatting)  *[Quick Start already done]*

Already created by the Nerfstudio installation step above.

```bash
conda activate nerfstudio
```

> **Troubleshooting — CUDA compilation errors.** On systems where the default GCC is too new for the installed CUDA toolkit (e.g. Ubuntu 24 with GCC 13), Nerfstudio or `gsplat` may fail to compile CUDA extensions. The scripts in this pipeline already export a compatible compiler automatically when `gcc-11` is available on the system. If you still hit a compiler error, install GCC 11:
>
> ```bash
> sudo apt install gcc-11 g++-11
> ```

---

## Download required assets

The full pipeline requires several files that are too large to ship in the Git repository: pretrained detection models, the DAVE-2 weights, and the example ROS bag.

### Pretrained 3D detection models  *(new, not in Quick Start)*

Stage 2 uses two pretrained 3D detection models:

- **FCOS3D** (camera-based 3D detection, ~1 GB)
- **PointPillars** (LiDAR-based 3D detection, ~20 MB)

Both must be placed inside `2_process_datasets/utils/` with their original filenames.

```bash
conda activate data_extraction
pip install -U gdown

cd 2_process_datasets/utils

# FCOS3D
gdown 1JIKRFQQI9CmQARk21Q619TPkdS49Voel -O fcos3d.pth

# PointPillars
gdown 1AGOR8C0tDUsWSSWTEc0fA7kysIE9-iol -O hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

cd ../..
```

Manual download links:
- FCOS3D: <https://drive.google.com/file/d/1JIKRFQQI9CmQARk21Q619TPkdS49Voel/view?usp=sharing>
- PointPillars: <https://drive.google.com/file/d/1AGOR8C0tDUsWSSWTEc0fA7kysIE9-iol/view?usp=sharing>

Verify both files exist:

```bash
ls -lh 2_process_datasets/utils/fcos3d.pth \
       2_process_datasets/utils/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
```

### DAVE-2 model weights  *(new, not in Quick Start)*

Stage 5 modes 5B and 5D drive a self-driving model. The DAVE-2 model weights file (~50 MB) must be placed inside `system_under_test/`.

```bash
conda activate data_extraction
cd system_under_test
gdown 1_pJHuvU4386FOYrF_B0ETIZGmShObhIF -O final.h5
cd ..
```

Manual link: <https://drive.google.com/file/d/1_pJHuvU4386FOYrF_B0ETIZGmShObhIF/view?usp=sharing>

Verify:

```bash
ls -lh system_under_test/final.h5
```


## Step 1 — Extract data from the ROS bag

This step reads the raw ROS bag and exports its contents into structured per-frame files: camera images synchronized with odometry, LiDAR point clouds, the full odometry stream, a simplified trajectory, the recorded vehicle steering status, and the steering target (model output). These files are the input for every subsequent stage.

### Input: the ROS bag

Step 1 reads a single ROS bag file from:

```text
data/raw_ros_data/reference_bag.bag
```

Download the example bag (~6 GB) corresponding to `reference_bag`:

```bash
conda activate data_extraction
pip install -U gdown   # only if not already installed

gdown 1ijhejhNO19jvrb3BUkEKRlxw-SkfZRhu -O data/raw_ros_data/reference_bag.bag
```

Manual link: <https://drive.google.com/file/d/1ijhejhNO19jvrb3BUkEKRlxw-SkfZRhu/view?usp=sharing>

Verify the file is in place:

```bash
ls -lh data/raw_ros_data/reference_bag.bag
```

> If you record your own bag, place it under `data/raw_ros_data/` and update the `bag_path` value at the top of the Step 1 scripts to point to it.

### Run Step 1

From the project root, with the `data_extraction` environment active:

```bash
conda activate data_extraction
bash 1_extract_ROS_data/step1.sh
```

`step1.sh` runs all extraction scripts in order:

1. `1A_camera_with_odometry.py` — camera frames + per-frame odometry-synchronized poses
2. `1B_lidar_with_odometry.py` — LiDAR point clouds + per-scan odometry-synchronized poses
3. `1C_poses_and_trajectory.py` — full odometry CSV + simplified trajectory CSV
4. `1D_steering_status.py` — actual vehicle steering values
5. `1E_model_output.py` — steering target / model-output values

### Output

All output is written under:

```text
data/raw_dataset/reference_bag/
├── images/                       # frame_000000.png, frame_000001.png, ...
├── images_positions.txt          # camera odometry-synchronized poses
├── point_clouds/                 # point_cloud_000000.bin, ...
├── lidar_positions.txt           # LiDAR odometry-synchronized poses
├── odometry.csv                  # full odometry stream
├── trajectory.csv                # simplified trajectory
├── steering_pct.txt              # actual steering values
└── steering_predictions.txt      # commanded / model-output steering
```

> **Skip Step 1 entirely.** If you only want to reproduce later stages and don't need to re-extract the bag, skip this step and rely on the precomputed dataset downloaded as part of the [Quick Start](#quick-start) (it contains the outputs of Steps 1–4 already).

For details on each individual script in Step 1, see the [`1_extract_ROS_data`](#1_extract_ros_data) section further below.

---


---

## Step 2 — Process the extracted dataset

This step processes the data extracted in Step 1 into assets used by all later stages: 3D bounding boxes of parked vehicles (from both camera and LiDAR), an OpenStreetMap-based road map for the recorded area, and image / mask splits ready for Gaussian Splatting training.

### Input

Step 2 reads from the output of Step 1:

```text
data/raw_dataset/reference_bag/
├── images/
├── images_positions.txt
├── point_clouds/
├── lidar_positions.txt
└── trajectory.csv
```

If you don't have this folder yet, run [Step 1](#step-1--extract-data-from-the-ros-bag) first.

Step 2 also requires the pretrained FCOS3D and PointPillars checkpoints inside `2_process_datasets/utils/` — these are downloaded as part of [Download required assets](#download-required-assets) at the top of this Replication guide. If `step2.sh` doesn't find them, it will try to download them automatically (so make sure `gdown` is installed: `pip install -U gdown`).

### Run Step 2

From the project root, with the `data_extraction` environment active:

```bash
conda activate data_extraction
bash 2_process_datasets/step2.sh
```

`step2.sh` runs the processing scripts in order:

1. `2A_camera_parked_cars_detection.py` — detects parked cars from camera images using FCOS3D
2. `2B_lidar_parked_cars_detection.py` — detects parked cars from LiDAR point clouds using PointPillars
3. `2C_create_map_from_coordinates_auto.py` — downloads OSM data and prepares the road map for the recorded area
4. `2E_prepare_dataset_for_gaussian_splatting.py` — crops images, generates sky masks, and creates overlapping image splits for Gaussian Splatting
5. `2G_OPT_fix_sidewalk.sh` — optional sidewalk fix on the generated map

#### Optional: refinement mode for LiDAR detections

The default Step 2 uses the standard PointPillars LiDAR detection. If you want to manually refine LiDAR detections through an interactive editor (move, rotate, insert, or delete bounding boxes), use the `-r` / `--refinement` flag:

```bash
bash 2_process_datasets/step2.sh --refinement
```

This replaces `2B_lidar_parked_cars_detection.py` with `2B_OPTIONAL_lidar_parked_cars_detection_with_refinement.py`. See the [`2_process_datasets`](#2_process_datasets) section below for editor controls.

### Output

All output is written under:

```text
data/processed_dataset/reference_bag/
├── camera_detections/
│   ├── camera_detections.json
│   ├── unified_clusters.txt
│   └── unified_bbox_overlays/
├── lidar_detections/
│   ├── lidar_detections.json
│   ├── unified_clusters.txt
│   ├── lidar_bboxes.txt
│   └── screenshots/
└── maps/
    ├── map.osm
    └── vehicle_data.json   (placeholder, filled in by Step 3)

data/data_for_gaussian_splatting/reference_bag/
├── _tmp_images_gs_1_of_<FRAME_SKIP>/
├── _tmp_sky_masks_gs_1_of_<FRAME_SKIP>/
├── images_gs_split_<N>_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_<N>_1_of_<FRAME_SKIP>/
└── frame_positions_split_<N>_1_of_<FRAME_SKIP>.txt
```

For details on each individual script in Step 2, see the [`2_process_datasets`](#2_process_datasets) section further below.

---

---

## Step 3 — Generate CARLA simulation data

This step converts the processed dataset into CARLA-ready simulation inputs: the recorded trajectory and detected parked vehicles are transformed into CARLA coordinates, the OpenDRIVE map is loaded into CARLA, and the scene is prepared with the hero vehicle and parked cars in place.

### Input

Step 3 reads from the outputs of Steps 1 and 2:

```text
data/raw_dataset/reference_bag/
└── images_positions.txt

data/processed_dataset/reference_bag/
├── lidar_detections/
│   └── unified_clusters.txt
└── maps/
    ├── map.osm
    ├── map.xodr
    └── vehicle_data.json
```

If you don't have these folders, run [Step 1](#step-1--extract-data-from-the-ros-bag) and [Step 2](#step-2--process-the-extracted-dataset) first.

### Camera configuration

Step 3 also reads a per-bag camera configuration file at:

```text
data/data_for_carla/reference_bag/camera.json
```

For `reference_bag`, this file is committed in the repository, so no action is needed. If you record your own bag, copy `camera.json` from `reference_bag` to your own bag folder and edit it to match your camera setup. See the [`3_generate_simulation_data`](#3_generate_simulation_data) section below for the file structure and field meaning.

### Run Step 3

From the project root, with the `data_extraction` environment active:

```bash
conda activate data_extraction
bash 3_generate_simulation_data/step3.sh
```

`step3.sh` runs the simulation-data scripts in order, and starts CARLA in the background between them:

1. `3A_transform_coordinates_to_carla.py` — converts the recorded trajectory into CARLA coordinates and writes the trajectory JSONs and the initial `hero_car` entry
2. `3B_transform_parked_vehicles_to_carla.py` — projects the detected parked-vehicle centroids into CARLA / OpenDRIVE coordinates and writes the final `vehicle_data.json`
3. `3C_setup_carla.py` — starts CARLA (in the background) using the `CARLA_INSTALLATION_PATH` set in `3_generate_simulation_data/utils/config.py`
4. `3F_generate_carla_scenario.py` — loads the OpenDRIVE map into CARLA, spawns the hero vehicle and the parked cars, freezes them, and exits leaving the prepared scene alive in CARLA

> CARLA must be installed and `CARLA_INSTALLATION_PATH` must be set inside `3_generate_simulation_data/utils/config.py` before running this step. If CARLA is already running, the launcher will reuse the existing instance.

### Output

```text
data/data_for_carla/reference_bag/
├── camera.json                                  (already in repo)
├── vehicle_data.json                            (with hero + parked spawn_positions)
├── trajectory_positions.json
├── trajectory_positions_rear.json
├── trajectory_positions_odom_yaw.json
└── trajectory_positions_rear_odom_yaw.json
```

At the end of Step 3 the CARLA world is loaded, the parked vehicles are placed, and the hero vehicle is spawned at the first trajectory pose. The world is left ready for Step 5 (`step5.sh`).

For details on each individual script in Step 3, see the [`3_generate_simulation_data`](#3_generate_simulation_data) section further below.

---
---

## Step 4 — Gaussian Splatting preparation

This step produces the Gaussian Splatting assets needed by Step 5: a COLMAP sparse reconstruction for each route segment, one trained Splatfacto model per segment, and the coordinate alignment between dataset poses and Nerfstudio's internal coordinate system.

Step 4 has two phases:

1. **COLMAP reconstruction** — manual, run once per split through the COLMAP GUI
2. **Splatfacto training + alignment** — automatic, all splits in one shell script

### Input

Step 4 reads from the output of Step 2:

```text
data/data_for_gaussian_splatting/reference_bag/
├── images_gs_split_1_1_of_2/
├── sky_masks_gs_split_1_1_of_2/
├── frame_positions_split_1_1_of_2.txt
├── images_gs_split_2_1_of_2/
├── sky_masks_gs_split_2_1_of_2/
├── frame_positions_split_2_1_of_2.txt
├── images_gs_split_3_1_of_2/
├── sky_masks_gs_split_3_1_of_2/
└── frame_positions_split_3_1_of_2.txt
```

If you don't have these folders, run [Step 2](#step-2--process-the-extracted-dataset) first.

> The exact split count and the `_1_of_<FRAME_SKIP>` suffix depend on the configuration set inside `2E_prepare_dataset_for_gaussian_splatting.py`. The defaults for `reference_bag` are `NUM_SPLITS=3` and `FRAME_SKIP=2`.

### Phase 1 — COLMAP reconstruction (manual, per split)

You must run COLMAP **once per split** using the COLMAP GUI. The full step-by-step procedure (open project, feature extraction with `OPENCV` model + dataset-specific intrinsics, sequential matching, bundle adjustment with intrinsics fixed, sparse export) is documented in:

```text
4_gaussian_splatting_preparation/4A_colmap_guide.md
```

At the end of Phase 1, the expected folder structure is:

```text
data/data_for_gaussian_splatting/reference_bag/colmap/
├── split_1/sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── split_2/sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── split_3/sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

> **Why manual?** COLMAP is run from the GUI because it requires interactive choices (camera model, intrinsic refinement options) that depend on the specific camera used to record the bag. The guide in `4A_colmap_guide.md` documents the exact settings used for `reference_bag`.

### Phase 2 — Splatfacto training + UTM alignment

Once every split has a COLMAP sparse reconstruction, train all Splatfacto models and compute the dataset-to-Nerfstudio alignment with a single command, from the project root, with the `nerfstudio` environment active:

```bash
conda activate nerfstudio
bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh
```

`4B_train_gaussian_splatting.sh` is **idempotent**: if a split already has a complete checkpoint (≥29000 steps), it is skipped automatically; if its `utm_to_nerfstudio_transform.json` is missing, the alignment step is run on the existing model. Re-running the script only retrains what is missing.

For each split, the script:

1. Runs `ns-train splatfacto` on the COLMAP reconstruction and the corresponding image / sky-mask folders (sky masks are included automatically when present)
2. Runs `4C_utm_yaw_to_nerfstudio.py` on the trained model to compute the 2D similarity transform and yaw offset from dataset coordinates to Nerfstudio coordinates, and saves the result as `utm_to_nerfstudio_transform.json` next to the model `config.yml`

Configuration values at the top of the script (`BAG_NAME`, `NUM_SPLITS`, `FRAME_SKIP`, `METHOD`) must match the values used in Step 2. The defaults match `reference_bag`.

### Output

```text
data/data_for_gaussian_splatting/reference_bag/outputs/
├── splatfacto_split_1/splatfacto/<timestamp>/
│   ├── config.yml
│   ├── nerfstudio_models/step-000029999.ckpt
│   └── utm_to_nerfstudio_transform.json
├── splatfacto_split_2/splatfacto/<timestamp>/
│   ├── config.yml
│   ├── nerfstudio_models/step-000029999.ckpt
│   └── utm_to_nerfstudio_transform.json
└── splatfacto_split_3/splatfacto/<timestamp>/
    ├── config.yml
    ├── nerfstudio_models/step-000029999.ckpt
    └── utm_to_nerfstudio_transform.json
```

Verify the alignment files are in place before moving on:

```bash
find data/data_for_gaussian_splatting/reference_bag/outputs -name "utm_to_nerfstudio_transform.json"
```

You should see one file per split.

For details on the COLMAP guide, the training script, and the alignment script, see the [`4_gaussian_splatting_preparation`](#4_gaussian_splatting_preparation) section further below.

---


---

## Step 5 — Run the simulation

This step runs the actual driving simulation. Step 5 has four independent modes that answer different questions:

- **5A** — trajectory replay in CARLA only (no GS, no DAVE-2). Simplest sanity check.
- **5B** — DAVE-2 closed-loop drive on raw CARLA RGB images.
- **5C** — trajectory replay rendered through Gaussian Splatting (CARLA + GS side-by-side).
- **5D** — DAVE-2 closed-loop drive on Gaussian-Splatted views. The main experiment of the pipeline.

The modes are not alternatives to each other — you may run any subset depending on what you want to evaluate.

### Input

Step 5 reads from the outputs of Steps 1–4:

```text
data/processed_dataset/reference_bag/maps/
└── map.xodr

data/data_for_carla/reference_bag/
├── camera.json
├── trajectory_positions_rear_odom_yaw.json
└── vehicle_data.json

# 5C / 5D only:
data/data_for_gaussian_splatting/reference_bag/
├── frame_positions_split_<N>_1_of_<FRAME_SKIP>.txt
└── outputs/splatfacto_split_<N>/splatfacto/<timestamp>/
    ├── config.yml
    ├── nerfstudio_models/step-000029999.ckpt
    └── utm_to_nerfstudio_transform.json

# 5B / 5D only:
system_under_test/
├── communicator.py
├── dave2.py
└── final.h5
```

If any of these are missing, see [Step 1](#step-1--extract-data-from-the-ros-bag) through [Step 4](#step-4--gaussian-splatting-preparation). DAVE-2 weights (`final.h5`) are downloaded as part of [Download required assets](#download-required-assets) at the top of this Replication guide.

> **Important**: CARLA must NOT be already running. The launcher starts CARLA itself.

### Run Step 5

A convenience launcher script orchestrates the entire Step 5 pipeline in separate terminals (CARLA, scenario loading, DAVE-2 server when needed, and the chosen step-5 script). Pick the mode that matches the experiment you want to run:

| Mode | What it runs                                                        | Conda env  | DAVE-2 server |
|------|---------------------------------------------------------------------|------------|---------------|
| 5A   | Trajectory replay in CARLA only                                     | `data_extraction` | no  |
| 5B   | DAVE-2 closed-loop on raw CARLA images                              | `data_extraction` | yes |
| 5C   | Trajectory replay with Gaussian Splatting (CARLA + GS side-by-side) | `nerfstudio`      | no  |
| 5D   | DAVE-2 closed-loop on Gaussian-Splatted views                       | `nerfstudio`      | yes |

From the project root, run any one of:

```bash
bash 5_execute_simulation/step5.sh --mode 5A
bash 5_execute_simulation/step5.sh --mode 5B
bash 5_execute_simulation/step5.sh --mode 5C
bash 5_execute_simulation/step5.sh --mode 5D
```

The launcher opens the required terminals in sequence:

1. **CARLA** (`3C_setup_carla.py`) and waits for it to be reachable on port 2000
2. **Map + parked vehicles** (`3F_generate_carla_scenario.py`) and waits for it to finish
3. **DAVE-2 server** (`system_under_test/communicator.py`), only for 5B / 5D, and waits for port 5090 to be reachable
4. **The chosen step-5 script**

Each terminal stays open after its command finishes so you can read logs or errors.

### Output

Each mode writes to its own folder:

```text
# 5A
data/processed_dataset/reference_bag/carla_replay_dataset/
├── rgb/
├── semantic/
├── instance/
└── data/all_frame_data.json

# 5B
data/processed_dataset/reference_bag/dave2_runs/only_carla_run<RUN_NUMBER>/
├── rgb/
├── semantic/
├── instance/
├── depth/
└── data/trajectory.json

# 5C
data/data_for_carla/reference_bag/replay_results/reference_bag_replay/
├── carla/
├── gs/
└── combined/

# 5D
data/data_for_carla/reference_bag/drive_results/reference_bag_drive_<timestamp>/
├── rgb_gt/
├── generated_gs/
└── trajectory.json
```

For details on each individual script in Step 5 and on the launcher, see the [`5_execute_simulation`](#5_execute_simulation) section further below.

---


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

Alternatively the script step1.sh runs all scripts in order
```bash
bash 1_extract_ROS_data/step1.sh
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

Alternatively the script step2.sh runs all scripts in order
```bash
bash 2_process_datasets/step2.sh
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

This script is optional. Use it when you need to manually refined LiDAR detections to increase precision of the parked cars.

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


Alternatively the script step3.sh runs all scripts in order
```bash
bash 3_generate_simulation_data/step3.sh
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

---

## Camera configuration file

The scripts in this folder rely on a per-bag camera configuration file:

```text
data/data_for_carla/<bag_name>/camera.json
```

For the example dataset (`reference_bag`), this file is committed in the repository, so no action is required if you only intend to run the pipeline on the example bag.

If you record your own ROS bag, you must create a new `camera.json` for it before running the scripts in this folder. Copy the existing file from the reference bag as a starting point:

```bash
mkdir -p data/data_for_carla/<your_bag_name>
cp data/data_for_carla/reference_bag/camera.json \
   data/data_for_carla/<your_bag_name>/camera.json
```

Then edit the new file to match your own camera setup.

The structure of `camera.json` is:

```json
{
  "image_size": {
    "x": 512,
    "y": 512
  },
  "camera": {
    "fov": 54.7,
    "fps": 30,
    "original_size": {
      "x": 800,
      "y": 503
    },
    "position": {
      "x": 0.762,
      "y": -0.015,
      "z": 1.21
    },
    "pitch": 0.6,
    "calibration": {
      "K": [
        [772.906855, 0.0, 424.980372],
        [0.0, 777.596896, 258.452509],
        [0.0, 0.0, 1.0]
      ],
      "distortion": [-0.274231, 0.034838, 0.00226, -0.000972, 0.0],
      "checkerboard_size": [10, 7],
      "square_size": 24.0
    }
  }
}
```

The fields are:

| Field | Meaning |
|---|---|
| `image_size.x`, `image_size.y` | Final image size (after crop / resize) used by the simulation step |
| `camera.fov` | Camera horizontal field of view in degrees |
| `camera.fps` | Frame rate of the camera topic in the ROS bag |
| `camera.original_size.x`, `camera.original_size.y` | Original image size produced by the camera, before any crop or resize |
| `camera.position.x`, `camera.position.y`, `camera.position.z` | Camera mounting position relative to the vehicle base, in meters (CARLA frame: x forward, y left, z up) |
| `camera.pitch` | Camera mounting pitch in degrees |
| `camera.calibration.K` | 3×3 intrinsic matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` |
| `camera.calibration.distortion` | Lens distortion coefficients `[k1, k2, p1, p2, k3]` |
| `camera.calibration.checkerboard_size` | Checkerboard size used during calibration (informational) |
| `camera.calibration.square_size` | Checkerboard square size in millimeters used during calibration (informational) |

If the file is missing or any required field is invalid, the affected scripts will fail at startup.

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

> ⚠️ **The camera model and parameters below are valid only for the example `reference_bag`** (front narrow camera used during the original recording). If you are running this pipeline on your own ROS bag, you need to:
>
> 1. **Choose the COLMAP camera model that matches your camera lens.** The most common choices are:
>    - `SIMPLE_PINHOLE` — ideal pinhole, no distortion (`f, cx, cy`)
>    - `PINHOLE` — pinhole with separate fx/fy, no distortion (`fx, fy, cx, cy`)
>    - `OPENCV` — pinhole + radial-tangential distortion (`fx, fy, cx, cy, k1, k2, p1, p2`) — used for `reference_bag`
>    - `OPENCV_FISHEYE` — fisheye lens (`fx, fy, cx, cy, k1, k2, k3, k4`)
>    - `FULL_OPENCV` — pinhole + extended distortion (`fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6`)
>
>    The full list and parameter order is documented in the COLMAP source: <https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h>
>
> 2. **Provide the calibration values for your camera.** You can typically read them from the `K` matrix and `distortion` array in your `data/data_for_carla/<your_bag>/camera.json` file (see Section 3 of this README), where:
>    - `fx, fy, cx, cy` come from the intrinsic matrix `K`
>    - `k1, k2, p1, p2` are the first four entries of the `distortion` array (only if your model includes them)

For `reference_bag`, use the camera model:

```text
OPENCV
```

Enable:

```text
Single camera
```

The COLMAP `OPENCV` camera parameter order is:

```text
fx, fy, cx, cy, k1, k2, p1, p2
```

For the front narrow camera of `reference_bag`, use:

```text
772.906855, 777.596896, 424.980372, 258.452509, -0.274231, 0.034838, 0.00226, -0.000972
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

Trains one Gaussian Splatting model per split using Nerfstudio.

For each split listed in the configuration, the script checks that the COLMAP sparse reconstruction exists:

```text
colmap/split_<N>/sparse/0/cameras.bin
colmap/split_<N>/sparse/0/images.bin
colmap/split_<N>/sparse/0/points3D.bin
```

and that the corresponding image folder is present:

```text
images_gs_split_<N>_1_of_<FRAME_SKIP>/
```

If either check fails for a given split, that split is skipped and the script continues with the remaining ones.

When all required files are present, the script runs `ns-train` with the `colmap` dataparser and the following inputs:

- the split image folder (`images_gs_split_<N>_1_of_<FRAME_SKIP>`)
- the split sky-mask folder (`sky_masks_gs_split_<N>_1_of_<FRAME_SKIP>`), if it exists
- the split COLMAP reconstruction (`colmap/split_<N>/sparse/0`)

The script uses `--viewer.quit-on-train-completion True` so the Nerfstudio viewer closes automatically when training ends, and the loop moves on to the next split without manual intervention.

Configuration values at the top of the script:

```bash
BAG_NAME="reference_bag"
NUM_SPLITS=3
FRAME_SKIP=3
METHOD="splatfacto"
CONDA_ENV="nerfstudio"
```

`BAG_NAME` must match the dataset folder under `data/data_for_gaussian_splatting/`. `NUM_SPLITS` and `FRAME_SKIP` must match the values used in step 2 when generating the image splits. `METHOD` selects the Nerfstudio model (`splatfacto` or `splatfacto-big`). `CONDA_ENV` is the Conda environment with Nerfstudio installed.

Default input:

```text
data/data_for_gaussian_splatting/<bag_name>/
├── images_gs_split_<N>_1_of_<FRAME_SKIP>/
├── sky_masks_gs_split_<N>_1_of_<FRAME_SKIP>/    (optional)
└── colmap/split_<N>/sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

Output:

```text
data/data_for_gaussian_splatting/<bag_name>/outputs/
├── splatfacto_split_1/splatfacto/<timestamp>/
│   ├── config.yml
│   └── nerfstudio_models/step-000029999.ckpt
├── splatfacto_split_2/splatfacto/<timestamp>/
│   └── ...
└── splatfacto_split_3/splatfacto/<timestamp>/
    └── ...
```

Each trained run contains the full Nerfstudio output folder, including the final checkpoint and the `config.yml` file required by step 5.

Run:

```bash
bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh
```

The script activates the configured Conda environment automatically, so it does not require the environment to be active before launching it.

If a split fails during training, the error is reported but the loop continues. Use the console output to identify which splits completed successfully.

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

This script must be run **once per split**, after step 4B has finished training.

Default input (per split):

```text
data/data_for_gaussian_splatting/<bag_name>/
├── outputs/splatfacto_split_<N>/splatfacto/<timestamp>/config.yml
└── frame_positions_split_<N>_1_of_<FRAME_SKIP>.txt
```

Default output (per split), saved next to the model `config.yml`:

```text
data/data_for_gaussian_splatting/<bag_name>/outputs/splatfacto_split_<N>/splatfacto/<timestamp>/
└── utm_to_nerfstudio_transform.json
```

The alignment file is saved next to `config.yml` because that is where step 5 looks for it. Do not move it.

CLI arguments:

| Argument | Required | Default |
|---|---|---|
| `--gs_config` | Yes | — |
| `--utm_file` | Yes | — |
| `--data_root` | No | `"."` |
| `--output` | No | `<config.yml folder>/utm_to_nerfstudio_transform.json` |

The `--output` argument is rarely needed. Keep the default so step 5 can find the file automatically.

The `<timestamp>` value changes at every training run. List the trained model folders for each split with:

```bash
ls data/data_for_gaussian_splatting/reference_bag/outputs/splatfacto_split_1/splatfacto/
ls data/data_for_gaussian_splatting/reference_bag/outputs/splatfacto_split_2/splatfacto/
ls data/data_for_gaussian_splatting/reference_bag/outputs/splatfacto_split_3/splatfacto/
```

Each folder contains one `<timestamp>` subfolder with the trained model.

Run, once per split (replace `<TIMESTAMP_N>` with the actual timestamp from the listing above):

```bash
# Split 1
python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
  --gs_config data/data_for_gaussian_splatting/reference_bag/outputs/splatfacto_split_1/splatfacto/<TIMESTAMP_1>/config.yml \
  --utm_file  data/data_for_gaussian_splatting/reference_bag/frame_positions_split_1_1_of_3.txt \
  --data_root data/data_for_gaussian_splatting/reference_bag

# Split 2
python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
  --gs_config data/data_for_gaussian_splatting/reference_bag/outputs/splatfacto_split_2/splatfacto/<TIMESTAMP_2>/config.yml \
  --utm_file  data/data_for_gaussian_splatting/reference_bag/frame_positions_split_2_1_of_3.txt \
  --data_root data/data_for_gaussian_splatting/reference_bag

# Split 3
python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
  --gs_config data/data_for_gaussian_splatting/reference_bag/outputs/splatfacto_split_3/splatfacto/<TIMESTAMP_3>/config.yml \
  --utm_file  data/data_for_gaussian_splatting/reference_bag/frame_positions_split_3_1_of_3.txt \
  --data_root data/data_for_gaussian_splatting/reference_bag
```

The output JSON contains:

- scale, rotation, translation (2D similarity transform)
- yaw sign and yaw offset (orientation alignment)
- camera pitch and roll estimates (median over training cameras)
- matching and alignment error statistics

After running the script for every split, verify that all alignment files exist:

```bash
find data/data_for_gaussian_splatting/reference_bag/outputs -name "utm_to_nerfstudio_transform.json"
```

You should see one file per split (three for `reference_bag`).

The alignment files are used later by step 5 to convert simulation poses into the coordinate system expected by the GS renderer.

---

## Suggested execution order

A typical workflow is:

```bash
# 1. Prepare GS image splits in step 2 (already done in step 2)
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py

# 2. Run COLMAP once for each split
# Follow the COLMAP instructions in 4A_colmap_guide.md.

# 3. Train one GS model per split
bash 4_gaussian_splatting_preparation/4B_train_gaussian_splatting.sh

# 4. Compute dataset-to-Nerfstudio alignment, once per trained split
python 4_gaussian_splatting_preparation/4C_utm_yaw_to_nerfstudio.py \
  --gs_config <path_to_split_N_config.yml> \
  --utm_file  <path_to_frame_positions_split_N.txt> \
  --data_root data/data_for_gaussian_splatting/<bag_name>
```

---

## Output files summary

| File / script | Main output |
|---|---|
| `4A_colmap_guide.md` | COLMAP sparse reconstructions under `colmap/split_<N>/sparse/0/` |
| `4B_train_gaussian_splatting.sh` | Trained Gaussian Splatting models under `outputs/splatfacto_split_<N>/splatfacto/<timestamp>/` |
| `4C_utm_yaw_to_nerfstudio.py` | One `utm_to_nerfstudio_transform.json` per split, saved next to each `config.yml` |

---

## Notes

- COLMAP must be run once per split before training.
- The COLMAP export folder for each split must contain `cameras.bin`, `images.bin`, and `points3D.bin`.
- The values of `BAG_NAME`, `NUM_SPLITS`, and `FRAME_SKIP` in `4B_train_gaussian_splatting.sh` must match the output produced by step 2.
- Use `METHOD="splatfacto"` or `METHOD="splatfacto-big"` for Gaussian Splatting. Splatfacto requires a CUDA-capable GPU with compute capability 7.5 or higher (RTX 20-series and newer); older GPUs are not supported by `gsplat`.
- Sky masks are optional, but recommended because they prevent sky regions from influencing the reconstruction.
- The front narrow camera parameters shown above are dataset-specific. Replace them if using another camera.
- Keep COLMAP intrinsic refinement disabled if you want to preserve the calibrated camera parameters.
- The alignment script (`4C`) must be run **after** training, because it loads the final Nerfstudio model coordinate system.
- One alignment file must be generated for each trained split. If any is missing, step 5 will warn and skip that split.
- By default, `4C` saves the alignment file next to the model `config.yml`, which is where step 5 expects it. Avoid passing `--output` unless you need to override that location.
- The `<timestamp>` value changes at every training run, so the path to `config.yml` changes too. There is no way to hardcode it in advance; list the trained folder before running `4C`.
- The trained GS models and alignment files are used by step 5 to render camera observations from simulated poses.

</details>

<details>
<summary><code>5_execute_simulation</code></summary>

# 5_execute_simulation

This folder contains the fifth and final step of the data-processing pipeline.

The scripts in this folder execute different kinds of simulation runs inside CARLA, using the data prepared in steps 1–4. Each script is **independent** and corresponds to a different experiment:

- Replay the recorded trajectory in CARLA (no model)
- Run the DAVE-2 self-driving model with raw CARLA images
- Replay the recorded trajectory in CARLA + Gaussian Splatting side by side
- Run the DAVE-2 self-driving model on Gaussian-Splatted views

The scripts are not alternatives to each other — they answer different questions, and you may run any subset depending on what you want to evaluate.

Each script is intended to be run from the project root:

```bash
python 5_execute_simulation/<script_name>.py
```

For example:

```bash
python 5_execute_simulation/5A_trajectory_only_carla.py
```

---

## Purpose

The goal of this step is to execute the prepared CARLA scenario in different configurations and to record the resulting sensor data, vehicle trajectories, and steering predictions.

The scripts can produce or use:

- A frame-by-frame replay of the recorded ego-vehicle trajectory inside CARLA
- A closed-loop DAVE-2 driving run on raw CARLA RGB images
- A frame-by-frame replay with side-by-side CARLA and Gaussian Splatting renders
- A closed-loop DAVE-2 driving run on Gaussian-Splatted views

---

## Expected project structure

```text
project_root/
├── 5_execute_simulation/
│   ├── 5A_trajectory_only_carla.py
│   ├── 5B_dave2_only_carla.py
│   ├── 5C_trajectory_replay.py
│   ├── 5D_dave2.py
│   └── utils/
│       ├── carla_simulator.py
│       ├── config.py
│       ├── dave2_connection.py
│       └── other helper files
│
├── system_under_test/
│   ├── communicator.py
│   ├── dave2.py
│   ├── final.h5            (DAVE-2 model weights, see below)
│   └── OPTIONAL.txt
│
├── data/
│   ├── data_for_carla/
│   │   └── <bag_name>/
│   │       ├── camera.json
│   │       ├── trajectory_positions_rear_odom_yaw.json
│   │       ├── vehicle_data.json
│   │       └── drive_results/      (created by 5D)
│   │
│   ├── data_for_gaussian_splatting/
│   │   └── <bag_name>/
│   │       └── outputs/
│   │           └── splatfacto_split_<N>/splatfacto/<timestamp>/
│   │               ├── config.yml
│   │               └── utm_to_nerfstudio_transform.json
│   │
│   └── processed_dataset/
│       └── <bag_name>/
│           ├── carla_replay_dataset/   (created by 5A)
│           ├── dave2_runs/             (created by 5B)
│           └── maps/
│               └── map.xodr
```

---

## Requirements

The Conda environment to use depends on the script:

| Script | Conda environment | Why |
|---|---|---|
| `5A_trajectory_only_carla.py` | `data_extraction` | CARLA only, no GS rendering |
| `5B_dave2_only_carla.py` | `data_extraction` | CARLA only, talks to a separate DAVE-2 server |
| `5C_trajectory_replay.py` | `nerfstudio` | Renders Gaussian Splatting models with `gsplat` |
| `5D_dave2.py` | `nerfstudio` | Renders Gaussian Splatting models with `gsplat`, talks to DAVE-2 server |
| DAVE-2 server (`system_under_test/communicator.py`) | `dave_2` | Loads TensorFlow 2.13 + DAVE-2 model |

> The `nerfstudio` environment is required for any script that loads a `splatfacto` model, because Gaussian Splatting rasterization depends on `gsplat`, which is a CUDA extension installed in that environment. Splatfacto models also require a CUDA-capable GPU with compute capability 7.5 or higher (RTX 20-series or newer).

Always activate the correct environment before running a script:

```bash
conda activate <environment>
```

---

## Common prerequisites for all scripts

Before running any script in this folder, you need:

1. **CARLA server running**, listening on the IP/port set in `5_execute_simulation/utils/config.py` (defaults: `127.0.0.1:2000`).

2. **The CARLA scenario already prepared** by step 3, that is, you must have run:

```bash
   python 3_generate_simulation_data/3F_generate_carla_scenario.py
```

   This script loads the OpenDRIVE map, spawns the parked vehicles and the hero vehicle, and leaves them alive in the CARLA world. All scripts in step 5 reuse the existing hero vehicle and do not load the map themselves.

3. **The data files referenced by the scripts**:
   - `data/processed_dataset/<bag_name>/maps/map.xodr`
   - `data/data_for_carla/<bag_name>/camera.json`
   - `data/data_for_carla/<bag_name>/trajectory_positions_rear_odom_yaw.json`
   - `data/data_for_carla/<bag_name>/vehicle_data.json`

4. For Gaussian-Splatting scripts (`5C`, `5D`) only, you also need:
   - One trained `splatfacto_split_<N>/splatfacto/<timestamp>/config.yml` per split
   - One `utm_to_nerfstudio_transform.json` next to each `config.yml`
   - One `frame_positions_split_<N>_1_of_<FRAME_SKIP>.txt` in `data/data_for_gaussian_splatting/<bag_name>/`

   These are produced by step 4.

5. For DAVE-2 scripts (`5B`, `5D`) only, you also need the **DAVE-2 server running** (see "Starting the DAVE-2 server" below).

---

## Configuration

Each script has a configuration section near the top.

The most important value is the bag name:

```python
BAG_NAME = "reference_bag"
```

This must match the dataset folder used by the previous steps. Other typical values include the camera resolution, the drive speed, the maximum number of frames, and termination thresholds.

`5_execute_simulation/utils/config.py` contains shared settings such as:

```python
CARLA_IP = "127.0.0.1"
CARLA_PORT = 2000
HERO_VEHICLE_TYPE = "vehicle.tesla.model3"
ROTATION_DEGREES
```

---

## Starting the DAVE-2 server

Scripts `5B_dave2_only_carla.py` and `5D_dave2.py` rely on a separate **DAVE-2 inference server** that listens on a TCP socket. The server runs in its own Conda environment (`dave_2`), because TensorFlow 2.13 requires Python 3.8 and conflicts with the `data_extraction` and `nerfstudio` environments.

The server lives in:

```text
project_root/
└── system_under_test/
    ├── communicator.py
    ├── dave2.py
    └── final.h5            (model weights, must be downloaded)
```

### Step 1: download the DAVE-2 model

The model weights file `final.h5` is not shipped in this repository (~50 MB).

Make sure the `data_extraction` environment is active (so `gdown` is available), then run:

```bash
conda activate data_extraction
pip install -U gdown   # only if not already installed

cd system_under_test

gdown 1_pJHuvU4386FOYrF_B0ETIZGmShObhIF -O final.h5

cd ..
```

Alternatively, download the file manually from this link and place it in `system_under_test/`:

- DAVE-2 weights: <https://drive.google.com/file/d/1_pJHuvU4386FOYrF_B0ETIZGmShObhIF/view?usp=sharing>

Verify:

```bash
ls -lh system_under_test/final.h5
```

### Step 2: start the server

In a **separate terminal**, activate the `dave_2` environment (TensorFlow 2.13 + Python 3.8, defined in the install guide):

```bash
conda activate dave_2
cd system_under_test
python communicator.py
```

When the server is ready, it prints:

```text
🚗 Dave2 server listening on localhost:5090
```

Leave this terminal open during the entire `5B` or `5D` run. The CARLA-side scripts in step 5 connect automatically to `localhost:5090` using `utils/dave2_connection.py`.

To stop the server, press `Ctrl+C` in its terminal.

---

## Scripts

Ensure CARLA is running and the Scenario is loaded. You will need two separate Terminals. Running 3F requires the conda environment data_extraction
```bash
python 3_generate_simulation_data/3C_setup_carla.py
```

```bash
python 3_generate_simulation_data/3F_generate_carla_scenario.py
```

### 5A_trajectory_only_carla.py

Replays the recorded ego-vehicle trajectory inside CARLA, frame by frame, and saves CARLA RGB, semantic, and instance segmentation images.

The script does **not** load the map and does **not** drive the car: it teleports the hero vehicle along the recorded trajectory, ticks the world, and records sensor outputs.

Default input:

```text
data/processed_dataset/<bag_name>/maps/map.xodr      (loaded by 3F)
data/data_for_carla/<bag_name>/camera.json
data/data_for_carla/<bag_name>/trajectory_positions_rear_odom_yaw.json
```

Output:

```text
data/processed_dataset/<bag_name>/carla_replay_dataset/
├── rgb/
├── semantic/
├── instance/
└── data/
    └── all_frame_data.json
```

Run:

```bash
conda activate data_extraction
python 5_execute_simulation/5A_trajectory_only_carla.py
```

This script is the simplest one and is useful as a sanity check that the prepared CARLA scenario, the camera configuration, and the trajectory are all consistent.

---

### 5B_dave2_only_carla.py

Runs a closed-loop DAVE-2 driving session inside CARLA using **raw CARLA RGB images** as the input to DAVE-2 (no Gaussian Splatting).

The script teleports the hero vehicle to the start of the recorded trajectory, applies a stabilization sequence and a launch warmup to the vehicle physics, and then enters a closed-loop control loop where every camera frame is sent to the DAVE-2 server, the predicted steering is applied to the vehicle, and the vehicle keeps moving forward at a constant speed (`DRIVE_SPEED_KMH = 10.0` by default).

Termination conditions:

- Maximum frame count reached (`MAX_FRAMES`)
- Vehicle falls below `MIN_Z_THRESHOLD`
- Vehicle moves less than `STUCK_THRESHOLD` for `STUCK_FRAME_LIMIT` consecutive frames
- User presses `Q` or closes the pygame window

Default input:

```text
data/data_for_carla/<bag_name>/camera.json
data/data_for_carla/<bag_name>/trajectory_positions_rear_odom_yaw.json
```

Output:

```text
data/processed_dataset/<bag_name>/dave2_runs/only_carla_run<RUN_NUMBER>/
├── rgb/
├── semantic/
├── instance/
├── depth/
└── data/
    └── trajectory.json
```

Run, in **two separate terminals**:

Terminal 1 (DAVE-2 server):

```bash
conda activate dave_2
cd system_under_test
python communicator.py
```

Terminal 2 (driving script):

```bash
conda activate data_extraction
python 5_execute_simulation/5B_dave2_only_carla.py
```

This script is useful as the **baseline DAVE-2 closed-loop run** without any neural rendering between the simulator and the model.

---

### 5C_trajectory_replay.py

Replays the recorded ego-vehicle trajectory inside CARLA and, in parallel, renders the same poses through one or more trained Gaussian Splatting models. The result is a side-by-side video of CARLA versus GS for every frame of the trajectory.

This script does **not** drive the car and does **not** use DAVE-2: it teleports the hero vehicle along the recorded trajectory and samples both CARLA and the GS renderer at every step.

Two phases:

- **Phase 1 — calibration GUI** (4 panels: CARLA, GS free camera, original training image, GS rendered from the same training pose). Sliders allow per-split position and orientation offsets to be applied. Skip with `--skip_calibration`.
- **Phase 2 — replay**: the script teleports along the trajectory and saves CARLA frames, GS frames, and side-by-side combined frames.

Splits and trained models are auto-detected from:

```text
data/data_for_gaussian_splatting/<bag_name>/outputs/splatfacto_split_<N>/splatfacto/<timestamp>/config.yml
```

Default input:

```text
data/processed_dataset/<bag_name>/maps/map.xodr
data/data_for_carla/<bag_name>/camera.json
data/data_for_carla/<bag_name>/trajectory_positions_rear_odom_yaw.json
data/data_for_gaussian_splatting/<bag_name>/outputs/splatfacto_split_<N>/splatfacto/<timestamp>/config.yml
data/data_for_gaussian_splatting/<bag_name>/outputs/splatfacto_split_<N>/splatfacto/<timestamp>/utm_to_nerfstudio_transform.json
data/data_for_gaussian_splatting/<bag_name>/frame_positions_split_<N>_1_of_<FRAME_SKIP>.txt
```

Output:

```text
data/data_for_carla/<bag_name>/replay_results/<bag_name>_replay/
├── carla/
├── gs/
└── combined/
```

Run:

```bash
conda activate nerfstudio
python 5_execute_simulation/5C_trajectory_replay.py
```

Useful CLI arguments:

| Argument | Effect |
|---|---|
| `--only_split N` | Load only split `N` (useful with limited VRAM) |
| `--skip_calibration` | Skip Phase 1 and use auto-computed offsets |
| `--max_frames N` | Stop after `N` frames |
| `--no_save` | Disable frame saving |
| `--only_carla` | Run without GS (CARLA-only replay) |

This script is the right choice to **compare CARLA against GS visually**, frame by frame, on the recorded trajectory.

---

### 5D_dave2.py

Runs a closed-loop DAVE-2 driving session inside CARLA, but DAVE-2 receives **Gaussian-Splatted views** instead of raw CARLA frames.

Two phases:

- **Phase 1 — calibration GUI**: the same 4-panel calibration used by `5C` is run once per split, so that the GS coordinate frames are correctly aligned with CARLA before driving starts. Skip with `--skip_calibration`.
- **Phase 2 — closed-loop drive**: the hero vehicle is teleported to the first training camera, then stabilized (physics off → on with 10-tick settle) and given a 100-tick warmup launch (`LAUNCH_SPEED_KMH = 12.0`) to overcome ackermann startup inertia. After warmup, every CARLA pose is rendered through the active GS split, the rendered image is sent to the DAVE-2 server, and the predicted steering is applied to the vehicle.

When several splits are loaded, the script automatically switches to the split whose training cameras' centroid is closest in nerfstudio space, with a `SWITCH_DELAY` of 50 frames to avoid rapid flicker between splits.

Termination conditions:

- Maximum frame count reached (`--max_frames`)
- Vehicle falls below `MIN_Z_THRESHOLD`
- Vehicle moves less than `STUCK_THRESHOLD` for `STUCK_FRAME_LIMIT` consecutive frames
- Vehicle out of training coverage for `COVERAGE_FRAME_LIMIT` consecutive frames
- User presses `Q` or closes the pygame window

Default input: same as `5C`, plus a running DAVE-2 server.

Output:

```text
data/data_for_carla/<bag_name>/drive_results/<bag_name>_drive_<timestamp>/
├── rgb_gt/
├── generated_gs/
└── trajectory.json
```

Run, in **two separate terminals**:

Terminal 1 (DAVE-2 server):

```bash
conda activate dave_2
cd system_under_test
python communicator.py
```

Terminal 2 (driving script):

```bash
conda activate nerfstudio
python 5_execute_simulation/5D_dave2.py
```

Useful CLI arguments:

| Argument | Effect |
|---|---|
| `--only_split N` | Load only split `N` (useful with limited VRAM) |
| `--skip_calibration` | Skip Phase 1 and use auto-computed offsets |
| `--max_frames N` | Stop after `N` frames |
| `--no_save` | Disable frame saving |
| `--only_carla` | Fall back to raw CARLA images (equivalent to `5B`) |

This script is the **main experiment** of the pipeline: it answers whether a self-driving model trained on real images can drive in a CARLA scenario when the input is reconstructed from the same real images via Gaussian Splatting.

---

## Suggested execution order

The four scripts are **independent** and serve different purposes. Run any subset depending on what you want to measure.

If you want to run all of them in turn (each preceded by a fresh `3F` to reset the world):

```bash
# Common prerequisite: CARLA running + scenario prepared
python 3_generate_simulation_data/3F_generate_carla_scenario.py

# Trajectory replay in CARLA only
conda activate data_extraction
python 5_execute_simulation/5A_trajectory_only_carla.py

# DAVE-2 closed-loop on raw CARLA images
# (DAVE-2 server must be running in another terminal)
python 5_execute_simulation/5B_dave2_only_carla.py

# Trajectory replay with CARLA + GS side by side
conda activate nerfstudio
python 5_execute_simulation/5C_trajectory_replay.py

# DAVE-2 closed-loop on GS-rendered views
# (DAVE-2 server must be running in another terminal)
python 5_execute_simulation/5D_dave2.py
```

If a previous run left the world in a strange state (vehicles destroyed, sensors leaked, etc.), re-run `3F_generate_carla_scenario.py` to restore a clean scenario before launching the next script.

---

## Output files summary

| Script | Main output |
|---|---|
| `5A_trajectory_only_carla.py` | `data/processed_dataset/<bag_name>/carla_replay_dataset/{rgb,semantic,instance}/`, `data/all_frame_data.json` |
| `5B_dave2_only_carla.py` | `data/processed_dataset/<bag_name>/dave2_runs/only_carla_run<RUN_NUMBER>/{rgb,semantic,instance,depth}/`, `data/trajectory.json` |
| `5C_trajectory_replay.py` | `data/data_for_carla/<bag_name>/replay_results/<bag_name>_replay/{carla,gs,combined}/` |
| `5D_dave2.py` | `data/data_for_carla/<bag_name>/drive_results/<bag_name>_drive_<timestamp>/{rgb_gt,generated_gs}/`, `trajectory.json` |

---

## Notes

- All four scripts are **independent**. They are not alternatives to each other and you can run any subset.
- All four scripts require CARLA running and the scenario prepared by `3F_generate_carla_scenario.py`. They reuse the existing hero vehicle and do not load the map themselves.
- Scripts that use Gaussian Splatting (`5C`, `5D`) must be run from the `nerfstudio` environment because `gsplat` is only installed there.
- Scripts that use DAVE-2 (`5B`, `5D`) require the DAVE-2 server (`system_under_test/communicator.py`) to be running in a separate terminal, in the `dave_2` environment, with `final.h5` present.
- The `final.h5` model weights file must be obtained separately and placed inside `system_under_test/`. The DAVE-2 server will fail to start otherwise.
- `5B` and `5D` use a stabilization sequence and a 100-tick launch warmup before the closed-loop drive starts. This is required to overcome the ackermann controller's startup inertia after teleporting the hero vehicle. Without the warmup, the stuck-detector trips around frame 50 and the run ends prematurely.
- `5D` spawns the hero vehicle using the road waypoint z (`waypoint.z + 0.10`) instead of the raw trajectory z, because the trajectory's z is the LiDAR/base_link altitude and tends to be a few centimeters below the CARLA road mesh. Spawning at the raw z makes physics expel the car upward, which prevents it from driving forward.
- All four scripts honor `--max_frames` (or `MAX_FRAMES` in the configuration section) to limit how long they run, which is useful for quick smoke tests.
- After running `3F_generate_carla_scenario.py`, the parked vehicles and hero remain alive in CARLA between step-5 scripts, so you can launch them back-to-back without re-running `3F` every time.

</details>

<details>
<summary><code>6_validation</code></summary>

# 6_validation

This folder contains the sixth step of the data-processing pipeline.

The scripts in this folder compare the output of the simulation step against ground-truth produced from the real recording, and quantify how close the simulated views are to the real ones.

The first validation script compares semantic segmentation maps frame by frame, computing per-class Intersection-over-Union (IoU) and mean IoU between:

- the **real** semantic maps, produced upstream by running SegFormer on the recorded RGB frames, and
- the **simulated** semantic maps, produced by the CARLA trajectory replay in step 5A.

More validation scripts (e.g. trajectory error, steering jitter, completion rate) can be added in the same folder.

Each script is intended to be run from the project root:

```bash
python 6_validation/<script_name>.py
```

For example:

```bash
python 6_validation/6-semantic_map_comparision.py
```

---

## Purpose

The goal of this step is to evaluate the simulation against the real recording with reproducible metrics.

This step can produce:

- Per-frame IoU (Background, Car, Road) between real and simulated semantic maps
- Per-frame mean IoU
- Per-frame visual diagnostics (GT | Diff | Pred panel)
- An aggregated summary across all processed frames

---

## Expected project structure

```text
project_root/
├── 6_validation/
│   └── 6-semantic_map_comparision.py
│
├── data/
│   └── processed_dataset/
│       └── <bag_name>/
│           ├── semantic_maps/                       (input: GT from SegFormer)
│           ├── carla_replay_dataset/
│           │   └── semantic/                        (input: simulated semantic from 5A)
│           └── iou_results_semantic/                (output of this step)
│               ├── <frame>_iou.json
│               └── <frame>_vis.png
```

Inputs are read from:

```text
data/processed_dataset/<bag_name>/semantic_maps/
data/processed_dataset/<bag_name>/carla_replay_dataset/semantic/
```

Output is written to:

```text
data/processed_dataset/<bag_name>/iou_results_semantic/
```

---

## Requirements

Use the existing Conda environment named `data_extraction`.

Activate it before running any script in this folder:

```bash
conda activate data_extraction
```

The scripts in this folder use only `numpy`, `Pillow`, and `matplotlib`. They do not need CARLA, Nerfstudio, or the DAVE-2 server to be running.

---

## Inputs from previous steps

Before running this step, the two input directories must already exist for the same `<bag_name>`.

1. **Real semantic maps** (ground truth), produced by running SegFormer on the raw recorded images:

```text
data/processed_dataset/<bag_name>/semantic_maps/
```

   Each PNG uses only three RGB colors:

   - Background: `(0, 0, 0)`
   - Car:        `(0, 0, 142)`
   - Road:       `(128, 64, 128)`

   These files are produced upstream by the SegFormer-based semantic map generator (see the corresponding script in step 2 or its dedicated helper).

2. **Simulated semantic maps**, produced by the CARLA trajectory replay in step 5A:

```text
data/processed_dataset/<bag_name>/carla_replay_dataset/semantic/
```

   These are saved by `5_execute_simulation/5A_trajectory_only_carla.py`. They are already 512x512, use the CityScapes palette after cleanup, and are named `{frame_id:06d}.png`.

If either directory is missing, the validation script prints an error and exits.

---

## Configuration

The script does not take any command-line arguments. All paths are hardcoded and derived from a single configuration value at the top of the file:

```python
BAG_NAME = "reference_bag"
```

This must match the dataset folder used by the previous steps. Other values that can be edited at the top of the script:

```python
TARGET_SIZE = (512, 512)   # (W, H) used for both GT and Pred
NUM_CLASSES = 3            # Background, Car, Road
COLOR_TOLERANCE = 15       # RGB tolerance to absorb resize / compression noise
```

The fixed color-to-class mapping is also defined at the top of the script:

```python
COLOR_TO_CLASS = {
    (0, 0, 0):       0,   # Background
    (0, 0, 142):     1,   # Car
    (128, 64, 128):  2,   # Road
}
```

---

## Scripts

### 6-semantic_map_comparision.py

Compares the real semantic maps against the simulated semantic maps frame by frame and computes per-class IoU.

What the script does:

1. Lists every PNG in `semantic_maps/` (the ground truth).
2. For each GT file, extracts the integer frame id from the filename (the script strips optional `seg_` or `frame_` prefixes), then looks for the matching `{frame_id:06d}.png` in `carla_replay_dataset/semantic/`.
3. Resizes the GT to 512x512 with NEAREST interpolation so the two grids match. Aspect ratio is intentionally not preserved.
4. Maps both RGB images to class IDs via the fixed `COLOR_TO_CLASS` lookup with a tolerance of `COLOR_TOLERANCE` to absorb resize and compression artifacts.
5. Computes the 3x3 confusion matrix and per-class IoU (Background, Car, Road), plus the mean IoU over the classes that actually appear in the GT.
6. Skips any frame that already has a result file in the output directory, so the script is resumable.
7. After processing, re-reads every `*_iou.json` in the output directory and prints an aggregated summary (mean IoU per class across all frames).

Default input:

```text
data/processed_dataset/<bag_name>/semantic_maps/
data/processed_dataset/<bag_name>/carla_replay_dataset/semantic/
```

Output:

```text
data/processed_dataset/<bag_name>/iou_results_semantic/
├── <frame>_iou.json
└── <frame>_vis.png
```

Per frame, the script writes:

- `<frame>_iou.json` with fields:

```text
gt_file, sim_file, iou_background, iou_car, iou_road, mean_iou
```

- `<frame>_vis.png`, a three-panel figure: GT | Diff | Pred.

Run:

```bash
conda activate data_extraction
python 6_validation/6-semantic_map_comparision.py
```

The script does not take any arguments.

---

## Suggested execution order

This step is meant to be run **after** step 5A has produced the CARLA replay semantic maps, and after the SegFormer-based real semantic maps have been generated for the same bag.

A typical workflow is:

```bash
# 1. Generate real semantic maps (SegFormer on raw recorded images)
#    (upstream script; not part of this folder)

# 2. Generate simulated semantic maps via CARLA replay
conda activate data_extraction
python 5_execute_simulation/5A_trajectory_only_carla.py

# 3. Compare real vs simulated semantic maps
python 6_validation/6-semantic_map_comparision.py
```

---

## Output files summary

| Script | Main output |
|---|---|
| `6-semantic_map_comparision.py` | `data/processed_dataset/<bag_name>/iou_results_semantic/<frame>_iou.json`, `data/processed_dataset/<bag_name>/iou_results_semantic/<frame>_vis.png` |

---

## Notes

- The script does not take any CLI argument. To run on a different bag, edit `BAG_NAME` at the top of the file.
- The script is resumable: any frame that already has a `<frame>_iou.json` in the output directory is skipped on subsequent runs. Delete the output directory to recompute from scratch.
- The GT is resized to 512x512 with NEAREST interpolation. If the real semantic maps were generated at a different aspect ratio, this resize intentionally distorts them so that the GT and the simulated maps share the same pixel grid.
- The color tolerance of 15 absorbs small RGB differences introduced by image saving and resizing. If the GT and the simulated maps use different palettes, edit `COLOR_TO_CLASS` and `COLOR_TOLERANCE` accordingly.
- The aggregated summary printed at the end of the run is computed by re-reading every `*_iou.json` in the output directory, so it always reflects the full set of processed frames, not only the ones produced in the latest run.
- This is the first validation script. Additional validation scripts (trajectory error, steering jitter, completion rate, etc.) belong in the same `6_validation/` folder and follow the same conventions.

</details>