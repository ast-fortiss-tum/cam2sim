# Manual COLMAP Reconstruction for Gaussian Splatting

After running:

```bash
python 2_process_datasets/2E_prepare_dataset_for_gaussian_splatting.py
```

the Gaussian Splatting data is saved in:

```text
data/data_for_gaussian_splatting/<BAG_NAME>/
```

For example:

```text
data/data_for_gaussian_splatting/reference_bag/
```

The script creates one image folder and one sky-mask folder for each split, for example:

```text
images_gs_split_1_1_of_3/
sky_masks_gs_split_1_1_of_3/
images_gs_split_2_1_of_3/
sky_masks_gs_split_2_1_of_3/
images_gs_split_3_1_of_3/
sky_masks_gs_split_3_1_of_3/
```

You must run COLMAP once for each split.

---

## 1. Open COLMAP GUI

In a terminal, run:

```bash
colmap gui
```

---

## 2. Create a new COLMAP project

In the COLMAP GUI:

- `File` → `New Project`
- Create a new database for the current split.

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/colmap/database_split_1.db
```

Then select the image folder of the current split.

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/images_gs_split_1_1_of_3
```

---

## 3. Feature extraction

Go to:

`Processing` → `Feature Extraction`

Use the camera model:

```text
OPENCV
```

Enable:

- `Single camera`

Set the camera parameters according to the dataset calibration.

The COLMAP `OPENCV` camera parameter order is:

```text
fx, fy, cx, cy, k1, k2, p1, p2
```

For the front narrow camera, use:

```text
785.34926249, 784.07587341, 406.50794975, 249.45341029, -0.42020115, 0.64296938, -0.00531934, -0.00215015
```

Optional: select the sky-mask folder for the same split.

Example for split 1:

```text
data/data_for_gaussian_splatting/reference_bag/sky_masks_gs_split_1_1_of_3
```

Then run feature extraction.

---

## 4. Feature matching

Go to:

`Processing` → `Feature Matching`

Select:

- `Sequential matching`

Set:

- `Sequential overlap`: `10`

Then run matching.

---

## 5. Reconstruction options

Go to:

`Reconstruction` → `Reconstruction Options`

Then open the `Bundle Adjustment` options.

To keep the calibrated camera parameters fixed, uncheck the camera intrinsic refinement options such as:

- `Refine focal length`
- `Refine principal point`
- `Refine extra parameters`

Then start reconstruction:

`Reconstruction` → `Start Reconstruction`

Wait until COLMAP finishes.

---

## 6. Export the sparse model

After reconstruction finishes in the COLMAP GUI, export the sparse model for the current split into the matching split folder created by `2E`.

The current layout is:

- Split 1 → `data/data_for_gaussian_splatting/reference_bag/colmap/split_1/sparse/0`
- Split 2 → `data/data_for_gaussian_splatting/reference_bag/colmap/split_2/sparse/0`
- Split 3 → `data/data_for_gaussian_splatting/reference_bag/colmap/split_3/sparse/0`

Each exported sparse model folder must contain:

```text
cameras.bin
images.bin
points3D.bin
```

---

## 7. Repeat for each split

Repeat the full COLMAP procedure for every split.

Use this export layout:

- Split 1 → `colmap/split_1/sparse/0`
- Split 2 → `colmap/split_2/sparse/0`
- Split 3 → `colmap/split_3/sparse/0`

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