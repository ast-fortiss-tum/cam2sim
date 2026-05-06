import os
import json
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import torch

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from mmdet3d.apis import init_model, inference_detector


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Dataset name must match the folder name inside data/raw_dataset/
DATASET_NAME = "reference_bag"

# Input folder from step 1: ROS extraction
EXTRACTED_ROOT = "data/raw_dataset"
DATASET_DIR = os.path.join(EXTRACTED_ROOT, DATASET_NAME)

ODOMETRY_FILE = os.path.join(DATASET_DIR, "odometry.csv")
LIDAR_POSITIONS_FILE = os.path.join(DATASET_DIR, "lidar_positions.txt")
POINT_CLOUD_DIR = os.path.join(DATASET_DIR, "point_clouds")

# Output folder for step 2: processed datasets
PROCESSED_ROOT = "data/processed_dataset"
OUTPUT_DATASET_DIR = os.path.join(PROCESSED_ROOT, DATASET_NAME)
OUTPUT_DIR = os.path.join(OUTPUT_DATASET_DIR, "lidar_detections")

OUTPUT_JSON = os.path.join(OUTPUT_DIR, "lidar_detections.json")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "unified_clusters.txt")
OUTPUT_BB_TXT = os.path.join(OUTPUT_DIR, "lidar_bboxes.txt")
OUTPUT_SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
TEMP_BIN_FILE = os.path.join(OUTPUT_DIR, "_temp_calc.bin")

# Model files
CONFIG_FILE = "2_process_datasets/utils/my_pointpillars_config.py"
CHECKPOINT_FILE = (
    "2_process_datasets/utils/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.pth"
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Logic settings
CONFIDENCE_THRESH = 0.50
CLUSTER_DIST_THRESH = 2.0
MIN_HITS = 3
MAX_DETECTION_RANGE = 40.0

# Visualization settings
VOXEL_SIZE = 0.1
SKIP_FRAMES = 5
REMOVE_GROUND_BELOW_Z = -1.3
BOX_LINE_RADIUS = 0.05

# Normalization
TARGET_GROUND_Z = -1.73


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_static_chain_matrix():
    r1 = R.from_quat([0, 0, 0.7071, 0.7071]).as_matrix()
    T_loc_imar = np.eye(4)
    T_loc_imar[:3, :3] = r1

    r2 = R.from_quat([1.0, 0, 0, 0]).as_matrix()
    t2 = np.array([0.62, 0, 1.43])
    T_imar_velo = np.eye(4)
    T_imar_velo[:3, :3] = r2
    T_imar_velo[:3, 3] = t2

    return T_loc_imar @ T_imar_velo


def get_interpolated_pose_matrix(target_time, pose_df, global_origin):
    idx = pose_df["timestamp"].searchsorted(target_time)

    if idx == 0:
        row = pose_df.iloc[0]
    elif idx == len(pose_df):
        row = pose_df.iloc[-1]
    else:
        prev = pose_df.iloc[idx - 1]
        curr = pose_df.iloc[idx]

        if abs(target_time - prev["timestamp"]) < abs(target_time - curr["timestamp"]):
            row = prev
        else:
            row = curr

    tx = row["tx"] - global_origin[0]
    ty = row["ty"] - global_origin[1]
    tz = row["tz"] - global_origin[2]

    rot = R.from_quat([
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    ]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [tx, ty, tz]

    return T


def normalize_data(bin_path):
    raw = np.fromfile(bin_path, dtype=np.float32)

    if len(raw) == 0:
        return None, 0.0

    if len(raw) % 4 == 0:
        points = raw.reshape(-1, 4)
    elif len(raw) % 3 == 0:
        points_xyz = raw.reshape(-1, 3)
        intensity = np.zeros((len(points_xyz), 1), dtype=np.float32)
        points = np.hstack((points_xyz, intensity))
    else:
        return None, 0.0

    if len(points) == 0:
        return None, 0.0

    if points[:, 3].max() > 1.0:
        points[:, 3] /= 255.0

    ground_level = np.percentile(points[:, 2], 5)
    z_shift = TARGET_GROUND_Z - ground_level

    if abs(z_shift) > 0.2:
        points[:, 2] += z_shift
    else:
        z_shift = 0.0

    return points, z_shift


def transform_box_to_global(box_local, T_total, z_correction):
    x, y, z_bottom, length, width, height, yaw = box_local

    real_z = z_bottom - z_correction

    local_center = np.array([
        x,
        y,
        real_z + height / 2.0,
        1.0,
    ])

    global_center_h = T_total @ local_center
    gx, gy, gz = global_center_h[:3]

    local_vec = np.array([
        np.cos(yaw),
        np.sin(yaw),
        0.0,
        0.0,
    ])

    global_vec = T_total @ local_vec
    global_yaw = np.arctan2(global_vec[1], global_vec[0])

    return [
        gx,
        gy,
        gz - height / 2.0,
        length,
        width,
        height,
        global_yaw,
    ]


def create_thick_box(center, length, width, height, rot_mat, color):
    x = length / 2.0
    y = width / 2.0
    z = height / 2.0

    corners = np.array([
        [-x, -y, -z],
        [x, -y, -z],
        [-x, y, -z],
        [x, y, -z],
        [-x, -y, z],
        [x, -y, z],
        [-x, y, z],
        [x, y, z],
    ])

    corners = corners @ rot_mat.T + center

    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    meshes = []

    for line in lines:
        p1 = corners[line[0]]
        p2 = corners[line[1]]

        vec = p2 - p1
        length_line = np.linalg.norm(vec)

        if length_line == 0:
            continue

        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=BOX_LINE_RADIUS,
            height=length_line,
        )
        cyl.paint_uniform_color(color)

        z_axis = np.array([0, 0, 1])
        vec_norm = vec / length_line

        axis = np.cross(z_axis, vec_norm)
        dot = np.clip(np.dot(z_axis, vec_norm), -1.0, 1.0)
        angle = np.arccos(dot)

        if np.linalg.norm(axis) < 0.001:
            if vec_norm[2] < 0:
                R_cyl = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ])
            else:
                R_cyl = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            R_cyl = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        cyl.rotate(R_cyl, center=[0, 0, 0])
        cyl.translate((p1 + p2) / 2.0)

        meshes.append(cyl)

    return meshes


# ==========================================
# 3. TRAJECTORY-BASED CLASSIFICATION
# ==========================================

def fit_trajectory(positions, smoothing=50.0):
    """Fit smooth spline through ego positions."""
    step = max(1, len(positions) // 500)
    pts = positions[::step]

    if len(pts) < 4:
        raise RuntimeError("Not enough pose samples to fit trajectory spline.")

    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, k=3)

    u = np.linspace(0, 1, 2000)

    traj = np.array(splev(u, tck)).T
    tang = np.array(splev(u, tck, der=1)).T

    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tang = tang / norms

    return traj, tang


def get_side_from_trajectory(pos, traj, tang):
    """Determine if car is on left or right side of trajectory."""
    tree = cKDTree(traj)
    _, idx = tree.query(pos[:2])

    to_car = pos[:2] - traj[idx]
    cross = tang[idx, 0] * to_car[1] - tang[idx, 1] * to_car[0]

    return "left" if cross > 0 else "right"


def get_orientation_from_trajectory(yaw, pos, traj, tang):
    """
    Determine if car is parallel or perpendicular to the trajectory.

    This compares the car orientation with the local trajectory direction
    at the nearest point on the trajectory.
    """
    tree = cKDTree(traj)
    _, idx = tree.query(pos[:2])

    traj_angle = np.arctan2(tang[idx, 1], tang[idx, 0])

    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    traj_angle = (traj_angle + np.pi) % (2 * np.pi) - np.pi

    angle_diff = abs(yaw - traj_angle)

    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff

    if angle_diff < np.pi / 4 or angle_diff > 3 * np.pi / 4:
        return "parallel"

    return "perpendicular"


# ==========================================
# 4. LOAD LIDAR INDEX
# ==========================================

def load_lidar_positions():
    lidar_df = pd.read_csv(
        LIDAR_POSITIONS_FILE,
        comment="#",
        header=None,
        names=[
            "frame_id",
            "timestamp",
            "odom_x",
            "odom_y",
            "odom_yaw",
            "pointcloud_file",
        ],
        skipinitialspace=True,
    )

    lidar_df["pointcloud_file"] = lidar_df["pointcloud_file"].astype(str).str.strip()

    lidar_df["bin_path"] = lidar_df["pointcloud_file"].apply(
        lambda filename: os.path.join(POINT_CLOUD_DIR, filename)
    )

    missing = lidar_df[~lidar_df["bin_path"].apply(os.path.exists)]

    if len(missing) > 0:
        print(f"WARNING: {len(missing)} point cloud files listed in lidar_positions.txt were not found.")
        lidar_df = lidar_df[lidar_df["bin_path"].apply(os.path.exists)]

    lidar_df = lidar_df.sort_values("frame_id").reset_index(drop=True)

    if lidar_df.empty:
        raise RuntimeError("No valid LiDAR point clouds found.")

    return lidar_df


# ==========================================
# 5. SAVE FUNCTIONS
# ==========================================

def save_detections_json(cars, global_origin, filepath):
    """Save detections with full bounding-box info to JSON."""
    data = {
        "source": "lidar",
        "dataset_name": DATASET_NAME,
        "input_dataset": DATASET_DIR,
        "global_origin": global_origin.tolist(),
        "cars": [],
    }

    for i, car in enumerate(cars):
        box = car["box"]

        if hasattr(box, "tolist"):
            box = box.tolist()
        else:
            box = [float(v) for v in box]

        data["cars"].append({
            "id": i + 1,
            "x": float(box[0] + global_origin[0]),
            "y": float(box[1] + global_origin[1]),
            "z": float(box[2] + global_origin[2]),
            "length": float(box[3]),
            "width": float(box[4]),
            "height": float(box[5]),
            "yaw": float(box[6]),
            "count": int(car["count"]),
            "conf": float(car["conf"]),
            "side": car["side"],
            "orient": car["orient"],
        })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {len(cars)} detections to {filepath}")


def save_clusters_txt(cars, global_origin, filepath):
    """Save in legacy TXT format."""
    with open(filepath, "w") as f:
        f.write("# cluster_id, x, y, z, count, last_conf, orientation, side, rgb_color\n")

        for i, car in enumerate(cars):
            box = car["box"]

            final_x = box[0] + global_origin[0]
            final_y = box[1] + global_origin[1]
            final_z = box[2] + global_origin[2]

            line = (
                f"{i + 1}, "
                f"{final_x:.3f}, "
                f"{final_y:.3f}, "
                f"{final_z:.3f}, "
                f"{car['count']}, "
                f"{car['conf']:.3f}, "
                f"{car['orient']}, "
                f"{car['side']}, "
                f"0-0-0"
            )

            f.write(line + "\n")

    print(f"  Saved {len(cars)} clusters to {filepath}")


def save_bboxes_txt(cars, global_origin, filepath):
    """Save full bounding-box coordinates to TXT."""
    with open(filepath, "w") as f:
        f.write("# cluster_id, x, y, z, length, width, height, yaw, orientation, side\n")

        for i, car in enumerate(cars):
            box = car["box"]

            final_x = box[0] + global_origin[0]
            final_y = box[1] + global_origin[1]
            final_z = box[2] + global_origin[2]

            length, width, height, yaw = box[3], box[4], box[5], box[6]

            line = (
                f"{i + 1}, "
                f"{final_x:.3f}, "
                f"{final_y:.3f}, "
                f"{final_z:.3f}, "
                f"{length:.3f}, "
                f"{width:.3f}, "
                f"{height:.3f}, "
                f"{yaw:.4f}, "
                f"{car['orient']}, "
                f"{car['side']}"
            )

            f.write(line + "\n")

    print(f"  Saved {len(cars)} bounding boxes to {filepath}")


# ==========================================
# 6. MAIN LOOP
# ==========================================

def main():
    print("=" * 70)
    print("LIDAR PARKED CAR DETECTION PIPELINE")
    print("=" * 70)

    print(f"Dataset name: {DATASET_NAME}")
    print(f"Input dataset: {DATASET_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # Validate input files
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    if not os.path.exists(ODOMETRY_FILE):
        raise FileNotFoundError(f"Odometry file not found: {ODOMETRY_FILE}")

    if not os.path.exists(LIDAR_POSITIONS_FILE):
        raise FileNotFoundError(f"LiDAR positions file not found: {LIDAR_POSITIONS_FILE}")

    if not os.path.exists(POINT_CLOUD_DIR):
        raise FileNotFoundError(f"Point cloud folder not found: {POINT_CLOUD_DIR}")

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"PointPillars config file not found: {CONFIG_FILE}")

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"PointPillars checkpoint file not found: {CHECKPOINT_FILE}")

    # Setup output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_SCREENSHOT_DIR, exist_ok=True)

    print("\n1. Loading data...")

    pose_df = pd.read_csv(ODOMETRY_FILE).sort_values("timestamp").reset_index(drop=True)
    lidar_df = load_lidar_positions()

    if pose_df.empty:
        raise RuntimeError("Odometry file is empty.")

    first_pose = pose_df.iloc[0]
    global_origin = np.array([
        first_pose["tx"],
        first_pose["ty"],
        first_pose["tz"],
    ])

    print(f"   Global origin set: {global_origin}")
    print(f"   Odometry poses: {len(pose_df)}")
    print(f"   LiDAR scans: {len(lidar_df)}")

    print("\n2. Fitting trajectory...")

    positions = pose_df[["tx", "ty"]].values
    trajectory, tangents = fit_trajectory(positions)

    print(f"   Trajectory points: {len(trajectory)}")

    print("\n3. Loading PointPillars model...")

    T_static_chain = get_static_chain_matrix()
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

    print("   Model loaded.")

    vis_global_cloud = o3d.geometry.PointCloud()
    all_detections = []

    print(f"\n4. Processing {len(lidar_df)} scans...")

    for i, lidar_row in lidar_df.iterrows():
        if i % SKIP_FRAMES != 0:
            continue

        frame_id = int(lidar_row["frame_id"])
        timestamp = float(lidar_row["timestamp"])
        bin_file = lidar_row["bin_path"]

        points, z_shift = normalize_data(bin_file)

        if points is None:
            continue

        points.tofile(TEMP_BIN_FILE)

        T_dynamic = get_interpolated_pose_matrix(
            timestamp,
            pose_df,
            global_origin,
        )

        T_total = T_dynamic @ T_static_chain

        result, _ = inference_detector(model, TEMP_BIN_FILE)

        pred = result.pred_instances_3d
        bboxes = pred.bboxes_3d.tensor.cpu().numpy()
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()

        for box, score, label in zip(bboxes, scores, labels):
            if score < CONFIDENCE_THRESH:
                continue

            # KITTI 3-class PointPillars usually uses:
            # 0 = Pedestrian, 1 = Cyclist, 2 = Car
            if label != 2:
                continue

            if np.linalg.norm(box[:2]) > MAX_DETECTION_RANGE:
                continue

            g_box = transform_box_to_global(box, T_total, z_shift)

            all_detections.append({
                "box": g_box,
                "score": float(score),
            })

        # Visualization cloud update
        points_sensor = points[:, :3].copy()
        points_sensor[:, 2] -= z_shift

        mask = points_sensor[:, 2] > REMOVE_GROUND_BELOW_Z
        points_filtered = points_sensor[mask]

        if len(points_filtered) > 0:
            z_vals = points_filtered[:, 2]

            norm_z = np.clip((z_vals + 1.7) / 2.0, 0, 1)

            colors = np.zeros((len(z_vals), 3))
            colors[:, 0] = norm_z
            colors[:, 2] = 1 - norm_z

            R_total = T_total[:3, :3]
            t_total = T_total[:3, 3]

            points_world = points_filtered @ R_total.T + t_total

            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(points_world)
            pcd_frame.colors = o3d.utility.Vector3dVector(colors)

            pcd_frame = pcd_frame.voxel_down_sample(voxel_size=VOXEL_SIZE)
            vis_global_cloud += pcd_frame

            if i % (SKIP_FRAMES * 10) == 0:
                vis_global_cloud = vis_global_cloud.voxel_down_sample(
                    voxel_size=VOXEL_SIZE
                )
                print(f"   Frame {frame_id}... detections so far: {len(all_detections)}")

    if os.path.exists(TEMP_BIN_FILE):
        os.remove(TEMP_BIN_FILE)

    # ==========================================
    # 7. CLUSTERING WITH VOTING
    # ==========================================

    print("\n5. Clustering with metadata voting...")

    unique_cars = []

    for det in all_detections:
        box_arr = np.array(det["box"])
        score = det["score"]
        yaw = box_arr[6]

        matched = False

        for car in unique_cars:
            current_avg_box = car["sum_box"] / car["sum_score"]

            if np.linalg.norm(box_arr[:2] - current_avg_box[:2]) < CLUSTER_DIST_THRESH:
                car["sum_box"] += box_arr * score
                car["sum_score"] += score
                car["count"] += 1
                car["last_conf"] = score

                car["yaw_sin"] += np.sin(yaw) * score
                car["yaw_cos"] += np.cos(yaw) * score

                matched = True
                break

        if not matched:
            unique_cars.append({
                "sum_box": box_arr * score,
                "sum_score": score,
                "count": 1,
                "last_conf": score,
                "yaw_sin": np.sin(yaw) * score,
                "yaw_cos": np.cos(yaw) * score,
            })

    print(f"   Raw detections: {len(all_detections)}")
    print(f"   Initial clusters: {len(unique_cars)}")

    final_cars = []

    for c in unique_cars:
        if c["count"] < MIN_HITS:
            continue

        avg_box = c["sum_box"] / c["sum_score"]
        avg_box[6] = np.arctan2(c["yaw_sin"], c["yaw_cos"])

        global_pos = np.array([
            avg_box[0] + global_origin[0],
            avg_box[1] + global_origin[1],
            avg_box[2] + global_origin[2],
        ])

        final_side = get_side_from_trajectory(
            global_pos,
            trajectory,
            tangents,
        )

        final_orient = get_orientation_from_trajectory(
            avg_box[6],
            global_pos,
            trajectory,
            tangents,
        )

        final_cars.append({
            "box": avg_box.tolist(),
            "count": c["count"],
            "conf": c["last_conf"],
            "side": final_side,
            "orient": final_orient,
        })

    print(f"   Final cars: {len(final_cars)}")

    # ==========================================
    # 8. SAVE OUTPUTS
    # ==========================================

    print("\n6. Saving outputs...")

    save_detections_json(final_cars, global_origin, OUTPUT_JSON)
    save_clusters_txt(final_cars, global_origin, OUTPUT_TXT)
    save_bboxes_txt(final_cars, global_origin, OUTPUT_BB_TXT)

    # ==========================================
    # 9. VISUALIZATION
    # ==========================================

    print("\n7. Visualizing...")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Final LiDAR Clusters - Press 'S' to save screenshot",
        width=1280,
        height=720,
    )

    vis.add_geometry(vis_global_cloud)

    # Draw trajectory
    traj_points = []

    for idx in range(0, len(pose_df), 10):
        row = pose_df.iloc[idx]

        traj_points.append([
            row["tx"] - global_origin[0],
            row["ty"] - global_origin[1],
            row["tz"] - global_origin[2],
        ])

    if len(traj_points) > 1:
        lines = [[i, i + 1] for i in range(len(traj_points) - 1)]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_points),
            lines=o3d.utility.Vector2iVector(lines),
        )

        line_set.paint_uniform_color([0, 1, 0])
        vis.add_geometry(line_set)

    # Draw final car boxes
    for car in final_cars:
        box = car["box"]
        x, y, z_bottom, length, width, height, yaw = box

        center = [x, y, z_bottom + height / 2.0]

        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])

        meshes = create_thick_box(
            center,
            length,
            width,
            height,
            rot_mat,
            [1, 0, 0],
        )

        for mesh in meshes:
            vis.add_geometry(mesh)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sphere.translate(center)
        sphere.paint_uniform_color([0, 1, 0])
        vis.add_geometry(sphere)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])

    screenshot_counter = [0]

    def screenshot_callback(vis_obj):
        screenshot_counter[0] += 1

        filename = os.path.join(
            OUTPUT_SCREENSHOT_DIR,
            f"visualization_screenshot_{screenshot_counter[0]:03d}.png",
        )

        vis_obj.capture_screen_image(filename, do_render=True)

        print(f"\n   Screenshot saved: {filename}")

        return False

    vis.register_key_callback(ord("S"), screenshot_callback)
    vis.register_key_callback(ord("s"), screenshot_callback)

    print("   Press 'S' or 's' to save screenshot")
    print("   Press 'Q' or close window to exit")

    vis.run()
    vis.destroy_window()

    print("\nDONE.")
    print(f"  JSON:        {OUTPUT_JSON}")
    print(f"  Clusters:    {OUTPUT_TXT}")
    print(f"  BBoxes:      {OUTPUT_BB_TXT}")
    print(f"  Screenshots: {OUTPUT_SCREENSHOT_DIR}/")


if __name__ == "__main__":
    main()