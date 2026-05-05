#!/usr/bin/env python3
"""
UNIFIED PARKED CAR DETECTION PIPELINE

Combines:
- FCOS3D for accurate 3D detection
- SimpleWorldTracker for temporal tracking in world coordinates
- Trajectory-based left/right and parallel/perpendicular classification
- Two-stage clustering to reduce duplicates
- 3D bounding box visualization overlays

Output:
- camera_detections.json: Full detection info with BB (compatible with evaluation)
- unified_clusters.txt: Cluster positions with orientation, side
- unified_bbox_overlays/: Per-detection-frame images with detected 3D bounding boxes
"""
import os
import shutil
import json
import numpy as np
import pandas as pd
import cv2
import torch
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

# FCOS3D imports
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS
from mmengine.dataset import Compose
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import CameraInstance3DBoxes

# ==========================================
# CONFIGURATION
# ==========================================
# Single source of truth for the dataset path: it must point to a folder that
# contains both `images/` and `images_positions.txt` produced by extract_bag.py
DATASET_DIR = '/home/cam2sim/Documents/cam2sim/datasets/2026-02-16-23-07-19'

DATA_DIR = DATASET_DIR
POSES_FILE = os.path.join(DATASET_DIR, 'images_positions.txt')
OUTPUT_DIR = DATASET_DIR

FCOS3D_CONFIG = 'fcos3d_config.py'
FCOS3D_CHECKPOINT = 'fcos3d.pth'

# Output files (all under OUTPUT_DIR)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'camera_detections.json')
OUTPUT_CLUSTERS = os.path.join(OUTPUT_DIR, 'unified_clusters.txt')
OUTPUT_BBOX_DIR = os.path.join(OUTPUT_DIR, 'unified_bbox_overlays')

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Detection settings
FCOS3D_CONF_THRESH = 0.28
MAX_DETECTION_RANGE = 50.0

# Position corrections (for KITTI-trained model on different camera)
# Depth: z_corrected = z * DEPTH_SCALE - DEPTH_OFFSET
DEPTH_SCALE = 0.85     # Reduce depth by ~8%
DEPTH_OFFSET = 2.2     # Subtract 2 meters
# X scale: if cars shifted OUTWARD on both sides, reduce this (e.g. 0.85)
X_SCALE = 0.84         # Scale X towards center
X_OFFSET = 0.0         # Additional X offset if needed
# Y correction (usually not needed)
Y_OFFSET = 0.0

# Tracking settings
TRACK_MATCH_DIST = 2.0
TRACK_MAX_AGE = 15
TRACK_MIN_HITS = 4

# Clustering settings
CLUSTER_DIST_STAGE1 = 1.8
CLUSTER_DIST_STAGE2 = 2.5

# Frame processing
SKIP_FRAMES = 5  # For detection pass

# Camera intrinsics
CAM_INTRINSICS = np.array([
    [772.906855, 0.0, 424.980372],
    [0.0, 777.596896, 258.452509],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

CLASS_NAMES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']


# ==========================================
# WORLD TRACKER
# ==========================================
class SimpleWorldTracker:
    """Track static objects in world coordinates."""

    def __init__(self, match_dist=2.5, max_age=20, min_hits=3):
        self.match_dist = match_dist
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}
        self.archived = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections):
        """Update with new detections. Each detection has world_pos, box, score, etc."""
        self.frame_count += 1
        unmatched = list(range(len(detections)))

        for tid, track in list(self.tracks.items()):
            best_dist = self.match_dist
            best_idx = -1

            for idx in unmatched:
                det = detections[idx]
                dist = np.linalg.norm(
                    np.array(det['world_pos'][:2]) - np.array(track['pos'][:2])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                det = detections[best_idx]
                track['pos'] = det['world_pos']
                track['hits'] += 1
                track['age'] = 0
                track['detections'].append(det)
                unmatched.remove(best_idx)
            else:
                track['age'] += 1

        # Archive dead tracks
        for tid in [t for t, tr in self.tracks.items() if tr['age'] > self.max_age]:
            if self.tracks[tid]['hits'] >= self.min_hits:
                self.archived[tid] = self.tracks[tid]
            del self.tracks[tid]

        # New tracks
        for idx in unmatched:
            det = detections[idx]
            self.tracks[self.next_id] = {
                'pos': det['world_pos'],
                'hits': 1,
                'age': 0,
                'detections': [det],
                'first_frame': self.frame_count,
            }
            self.next_id += 1

    def get_confirmed(self):
        confirmed = {tid: t for tid, t in self.tracks.items() if t['hits'] >= self.min_hits}
        confirmed.update(self.archived)
        return confirmed


# ==========================================
# TRANSFORM FUNCTIONS
# ==========================================
def make_tf(trans, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = trans
    return T


def get_static_chain():
    T_loc_imar = make_tf([0, 0, 0], [0, 0, 0.7071, 0.7071])
    T_imar_velo = make_tf([0.62, 0, 1.43], [1, 0, 0, 0])
    T_velo_cam = make_tf([0.580393, 0.0727572, -0.211861],
                         [-0.502988, 0.496432, -0.507426, 0.493029])
    return T_loc_imar @ T_imar_velo @ T_velo_cam


def get_pose_matrix(row, origin):
    """Build 4x4 pose matrix directly from a row of the per-frame pose file."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']]).as_matrix()
    T[:3, 3] = [row['tx'] - origin[0], row['ty'] - origin[1], row['tz'] - origin[2]]
    return T


def get_T_cam_world(row, origin):
    return get_pose_matrix(row, origin) @ get_static_chain()


# ==========================================
# MODEL LOADING
# ==========================================
def load_fcos3d(config, checkpoint, device):
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    model.cfg = cfg
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.CLASSES = CLASS_NAMES
    model.to(device).eval()
    return model


def run_fcos3d(model, img_path, intrinsics, device):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    data = dict(
        images=dict(CAM2=dict(img_path=os.path.abspath(img_path), cam2img=intrinsics.tolist())),
        box_type_3d=CameraInstance3DBoxes, box_mode_3d=1,
    )
    pipeline = Compose([TRANSFORMS.build(t) for t in [
        dict(type='LoadImageFromFileMono3D'),
        dict(type='mmdet.Resize', scale_factor=1.0),
        dict(type='Pack3DDetInputs', keys=['img'])
    ]])
    data = pipeline(data)

    with torch.no_grad():
        results = model.test_step({
            'inputs': {'img': data['inputs']['img'].unsqueeze(0).to(device)},
            'data_samples': [data['data_samples']]
        })
    return results[0], img


# ==========================================
# BOX TRANSFORMS
# ==========================================
def box_to_world(box_cam, T):
    x, y, z = box_cam[:3]
    center = T @ np.array([x, y, z, 1.0])
    yaw_cam = box_cam[6] if len(box_cam) > 6 else 0
    # FCOS3D yaw rotation: forward direction is [cos, 0, -sin] based on rotation matrix convention
    dir_w = T @ np.array([np.cos(yaw_cam), 0, -np.sin(yaw_cam), 0])
    yaw_world = np.arctan2(dir_w[1], dir_w[0])
    return np.array([center[0], center[1], center[2], box_cam[3], box_cam[4], box_cam[5], yaw_world])


# ==========================================
# TRAJECTORY-BASED CLASSIFICATION (matching LiDAR)
# ==========================================
def fit_trajectory(positions, smoothing=50.0):
    """Fit smooth spline through ego positions."""
    step = max(1, len(positions) // 500)
    pts = positions[::step]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, k=3)
    u = np.linspace(0, 1, 2000)
    traj = np.array(splev(u, tck)).T
    tang = np.array(splev(u, tck, der=1)).T
    tang = tang / np.linalg.norm(tang, axis=1, keepdims=True)
    return traj, tang


def get_side_from_trajectory(pos, traj, tang):
    """Determine if car is on left or right side of trajectory."""
    tree = cKDTree(traj)
    _, idx = tree.query(pos[:2])
    to_car = pos[:2] - traj[idx]
    cross = tang[idx, 0] * to_car[1] - tang[idx, 1] * to_car[0]
    return 'left' if cross > 0 else 'right'


def get_orientation_from_trajectory(yaw, pos, traj, tang):
    """
    Determine if car is parallel or perpendicular to the trajectory.

    This compares the car's orientation (yaw) with the local trajectory direction
    at the nearest point on the trajectory. Same logic as LiDAR script.
    """
    # Find nearest point on trajectory
    tree = cKDTree(traj)
    _, idx = tree.query(pos[:2])

    # Get trajectory direction at that point
    traj_angle = np.arctan2(tang[idx, 1], tang[idx, 0])

    # Normalize both angles to [-pi, pi]
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    traj_angle = (traj_angle + np.pi) % (2 * np.pi) - np.pi

    # Calculate angular difference
    angle_diff = abs(yaw - traj_angle)

    # Wrap to [0, pi]
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff

    # If angle difference is close to 0 or pi, it's parallel
    # If angle difference is close to pi/2, it's perpendicular
    # Threshold = pi/4 (45 degrees)
    if angle_diff < np.pi/4 or angle_diff > 3*np.pi/4:
        return 'parallel'
    else:
        return 'perpendicular'


# ==========================================
# 3D BOUNDING BOX VISUALIZATION
# ==========================================
def draw_3d_box(img, box_cam, T_cam_world, intrinsics, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box on image."""
    x, y, z = box_cam[:3]
    l, w, h = box_cam[3:6]
    yaw = box_cam[6] if len(box_cam) > 6 else 0

    # Create 3D corners in camera frame
    corners_3d = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2], [l/2, w/2, h/2], [-l/2, w/2, h/2]
    ])

    # Rotation matrix from yaw
    rot_mat = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Rotate and translate
    corners_3d = np.dot(corners_3d, rot_mat.T)
    corners_3d += np.array([x, y, z])

    # Project to 2D
    corners_2d = []
    for corner in corners_3d:
        if corner[2] <= 0:  # Behind camera
            return
        px = intrinsics[0, 0] * corner[0] / corner[2] + intrinsics[0, 2]
        py = intrinsics[1, 1] * corner[1] / corner[2] + intrinsics[1, 2]
        corners_2d.append([int(px), int(py)])

    corners_2d = np.array(corners_2d)

    # Check if any corner is in image bounds
    H, W = img.shape[:2]
    if not any(0 <= x < W and 0 <= y < H for x, y in corners_2d):
        return

    # Draw edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]

    for edge in edges:
        pt1, pt2 = corners_2d[edge[0]], corners_2d[edge[1]]
        cv2.line(img, tuple(pt1), tuple(pt2), color, thickness)


# ==========================================
# CLUSTERING
# ==========================================
def cluster_tracks(confirmed, dist_thresh):
    clusters = []

    for tid, track in confirmed.items():
        # Weighted average
        sum_box = np.zeros(7)
        sum_w = 0
        orients = defaultdict(int)
        first_frame = track.get('first_frame', 0)

        for det in track['detections']:
            w = det['score']
            sum_box += np.array(det['box']) * w
            sum_w += w
            orients[det['orient']] += 1

        avg_box = sum_box / sum_w

        # Try merge
        merged = False
        for c in clusters:
            c_pos = c['sum_box'][:2] / c['sum_w']
            if np.linalg.norm(avg_box[:2] - c_pos) < dist_thresh:
                c['sum_box'] += sum_box
                c['sum_w'] += sum_w
                c['count'] += track['hits']
                c['first_frame'] = min(c['first_frame'], first_frame)
                for k, v in orients.items():
                    if k not in c['orients']:
                        c['orients'][k] = 0
                    c['orients'][k] += v
                merged = True
                break

        if not merged:
            clusters.append({
                'sum_box': sum_box.copy(),
                'sum_w': sum_w,
                'count': track['hits'],
                'orients': dict(orients),
                'first_frame': first_frame,
            })

    return clusters


def merge_clusters(clusters, dist_thresh):
    merged = []
    used = [False] * len(clusters)

    for i, c1 in enumerate(clusters):
        if used[i]:
            continue

        mc = {
            'sum_box': c1['sum_box'].copy(),
            'sum_w': c1['sum_w'],
            'count': c1['count'],
            'orients': dict(c1['orients']),
            'first_frame': c1.get('first_frame', 0),
        }
        used[i] = True

        for j, c2 in enumerate(clusters):
            if used[j]:
                continue
            p1 = mc['sum_box'][:2] / mc['sum_w']
            p2 = c2['sum_box'][:2] / c2['sum_w']
            if np.linalg.norm(p1 - p2) < dist_thresh:
                mc['sum_box'] += c2['sum_box']
                mc['sum_w'] += c2['sum_w']
                mc['count'] += c2['count']
                mc['first_frame'] = min(mc['first_frame'], c2.get('first_frame', 0))
                for k, v in c2['orients'].items():
                    mc['orients'][k] = mc['orients'].get(k, 0) + v
                used[j] = True

        merged.append(mc)

    return merged


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("UNIFIED PARKED CAR DETECTION PIPELINE")
    print("=" * 70)

    # Setup output directories
    if os.path.exists(OUTPUT_BBOX_DIR):
        shutil.rmtree(OUTPUT_BBOX_DIR)
    os.makedirs(OUTPUT_BBOX_DIR)

    # Load data
    print("\n[1/6] Loading data...", flush=True)
    # The pose file has one row per camera frame, with header:
    # FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Z, Qx, Qy, Qz, Qw, Odom_Yaw, ImageFile
    pose_df = pd.read_csv(
        POSES_FILE,
        comment='#',
        header=None,
        names=['frame_id', 'timestamp', 'tx', 'ty', 'tz',
               'qx', 'qy', 'qz', 'qw', 'yaw', 'image_file'],
        skipinitialspace=True,
    )
    pose_df = pose_df.sort_values('frame_id').reset_index(drop=True)

    img_files = [
        os.path.join(DATA_DIR, 'images', name)
        for name in pose_df['image_file']
    ]
    origin = np.array([pose_df.iloc[0]['tx'], pose_df.iloc[0]['ty'], pose_df.iloc[0]['tz']])
    print(f"  Frames (with pose): {len(pose_df)}")

    # Load models
    print("\n[2/6] Loading models...", flush=True)
    fcos3d = load_fcos3d(FCOS3D_CONFIG, FCOS3D_CHECKPOINT, DEVICE)
    print("  FCOS3D loaded")

    # Fit trajectory
    print("\n[3/6] Fitting trajectory...", flush=True)
    positions = pose_df[['tx', 'ty']].values
    trajectory, tangents = fit_trajectory(positions)
    print(f"  Trajectory points: {len(trajectory)}")

    # ========== PASS 1: DETECTION + TRACKING ==========
    print(f"\n[4/6] Detection pass (skip={SKIP_FRAMES})...", flush=True)
    tracker = SimpleWorldTracker(TRACK_MATCH_DIST, TRACK_MAX_AGE, TRACK_MIN_HITS)

    for i, img_file in enumerate(img_files):
        if i % SKIP_FRAMES != 0:
            continue

        row = pose_df.iloc[i]
        T = get_T_cam_world(row, origin)

        # FCOS3D detection
        result, img = run_fcos3d(fcos3d, img_file, CAM_INTRINSICS, DEVICE)
        if result is None:
            tracker.update([])
            continue

        pred = result.pred_instances_3d
        if len(pred) == 0:
            tracker.update([])
            continue

        bboxes = pred.bboxes_3d.tensor.cpu().numpy()
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()

        # Build detections
        frame_dets = []
        for box, score, label in zip(bboxes, scores, labels):
            if score < FCOS3D_CONF_THRESH or label != 0 or box[2] > MAX_DETECTION_RANGE:
                continue

            # Apply position corrections (KITTI-trained model on different camera)
            box = box.copy()
            box[0] = box[0] * X_SCALE + X_OFFSET  # X correction (scale towards center + offset)
            box[1] = box[1] + Y_OFFSET  # Y correction (up-down)
            box[2] = box[2] * DEPTH_SCALE - DEPTH_OFFSET  # Depth correction

            world_box = box_to_world(box, T)
            world_pos = world_box[:3] + origin

            # Get orientation relative to trajectory (same as LiDAR script)
            orient = get_orientation_from_trajectory(world_box[6], world_pos, trajectory, tangents)

            # Depth weighting
            depth = box[2]
            depth_weight = 1.0 / (1.0 + depth / 20.0)
            weighted_score = score * depth_weight

            frame_dets.append({
                'world_pos': world_pos,
                'box': world_box,
                'score': weighted_score,
                'orient': orient,
            })

        # Save detected bboxes overlaid on original image (only for processed frames)
        bbox_img = img.copy()
        for box, score, label in zip(bboxes, scores, labels):
            if score < FCOS3D_CONF_THRESH or label != 0 or box[2] > MAX_DETECTION_RANGE:
                continue
            # Adjust box center: shift up by h/2 (y points down in camera frame, so subtract)
            box_draw = box.copy()
            box_draw[1] = box[1] - box[4] / 2  # box[4] is height in FCOS3D format
            draw_3d_box(bbox_img, box_draw, T, CAM_INTRINSICS, color=(0, 255, 0), thickness=2)
            # Project 3D centroid to 2D
            if box[2] > 0:  # Ensure point is in front of camera
                # Project center point (x, y, z) to pixel coordinates (px, py)
                px = int(CAM_INTRINSICS[0, 0] * box[0] / box[2] + CAM_INTRINSICS[0, 2])

                # We use box_draw[1] because we shifted the center up previously to match the visual box
                py = int(CAM_INTRINSICS[1, 1] * box_draw[1] / box[2] + CAM_INTRINSICS[1, 2])

                # Draw a small red circle at the centroid
                cv2.circle(bbox_img, (px, py), 4, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(OUTPUT_BBOX_DIR, f"bbox_{i:06d}.png"), bbox_img)

        tracker.update(frame_dets)

        if i == 0:
            print(f"  First frame done, {len(frame_dets)} detections (CUDA warmup complete)", flush=True)
        elif i % 50 == 0:
            print(f"  Frame {i}/{len(img_files)}, detections: {len(frame_dets)}, tracks: {len(tracker.get_confirmed())}", flush=True)

    # ========== CLUSTERING ==========
    print("\n[5/6] Clustering...", flush=True)
    confirmed = tracker.get_confirmed()
    print(f"  Confirmed tracks: {len(confirmed)}")

    clusters = cluster_tracks(confirmed, CLUSTER_DIST_STAGE1)
    print(f"  After stage 1: {len(clusters)}")

    clusters = merge_clusters(clusters, CLUSTER_DIST_STAGE2)
    print(f"  After stage 2: {len(clusters)}")

    # Sort clusters by first appearance
    clusters.sort(key=lambda c: c.get('first_frame', 0))

    # Finalize clusters
    final_clusters = []
    for idx, c in enumerate(clusters, 1):
        avg_box = c['sum_box'] / c['sum_w']
        pos = avg_box[:3] + origin

        # Compute orientation from final yaw and position (same as LiDAR script)
        orient = get_orientation_from_trajectory(avg_box[6], pos, trajectory, tangents)

        # Side (already determined by trajectory)
        # Re-compute to ensure consistency
        side = get_side_from_trajectory(pos, trajectory, tangents)

        # Bounding box dimensions from avg_box
        length, width, height = avg_box[3], avg_box[4], avg_box[5]
        yaw = avg_box[6]

        final_clusters.append({
            'id': idx,
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'length': length,
            'width': width,
            'height': height,
            'yaw': yaw,
            'count': c['count'],
            'conf': c['sum_w'] / c['count'],
            'orient': orient,
            'side': side,
            'first_frame': c.get('first_frame', 0),
        })

    print(f"  Final clusters: {len(final_clusters)}")

    # ========== SAVE OUTPUTS ==========
    print("\n[6/6] Saving detection results...", flush=True)

    # 1. Save JSON (compatible with evaluation script)
    json_data = {
        'source': 'camera',
        'global_origin': origin.tolist(),
        'cars': []
    }
    for c in final_clusters:
        json_data['cars'].append({
            'id': c['id'],
            'x': float(c['x']),
            'y': float(c['y']),
            'z': float(c['z']),
            'length': float(c['length']),
            'width': float(c['width']),
            'height': float(c['height']),
            'yaw': float(c['yaw']),
            'count': int(c['count']),
            'conf': float(c['conf']),
            'side': c['side'],
            'orient': c['orient'],
        })

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved JSON to {OUTPUT_JSON}")

    # 2. Save TXT (legacy format)
    with open(OUTPUT_CLUSTERS, 'w') as f:
        f.write("# cluster_id, x, y, z, count, conf, orientation, side\n")
        for c in final_clusters:
            f.write(f"{c['id']}, {c['x']:.3f}, {c['y']:.3f}, {c['z']:.3f}, "
                    f"{c['count']}, {c['conf']:.3f}, {c['orient']}, {c['side']}\n")
    print(f"  Saved clusters to {OUTPUT_CLUSTERS}")

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    print(f"  JSON:        {OUTPUT_JSON} ({len(final_clusters)} cars)")
    print(f"  Clusters:    {OUTPUT_CLUSTERS}")
    print(f"  BB overlays: {OUTPUT_BBOX_DIR}/ (detection frames only)")


if __name__ == "__main__":
    main()