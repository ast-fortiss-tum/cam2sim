#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import contextily as cx
from pyproj import Transformer

# =======================
# PATH SETUP
# =======================
# Find the parent folder that contains utils/coordinates.py.
# This makes imports work even if the script is launched from anywhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEARCH_DIR = SCRIPT_DIR
while True:
    utils_coordinates = os.path.join(SEARCH_DIR, "utils", "coordinates.py")

    if os.path.exists(utils_coordinates):
        BASE_DIR = SEARCH_DIR
        break

    parent = os.path.dirname(SEARCH_DIR)

    if parent == SEARCH_DIR:
        raise RuntimeError("Could not find utils/coordinates.py in this script or any parent folder.")

    SEARCH_DIR = parent

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.coordinates import (
    ODOM0_X,
    ODOM0_Y,
    get_projected_coords,
)

# =======================
# CONFIGURATION
# =======================

# Dataset name must match the folder in data/extracted_ros_data/
DATASET_NAME = "reference_bag"

# Map name must match the folder created by the map-download script.
MAP_NAME = DATASET_NAME

# Input/output roots.
EXTRACTED_ROOT = os.path.join("data", "extracted_ros_data")
GENERATED_ROOT = os.path.join("data", "generated_data_from_extracted_data")
MAP_ROOT = os.path.join("data", "generated_data_from_extracted_data", MAP_NAME, "maps")

# Input files.
TRAJECTORY_PATH = os.path.join(
    EXTRACTED_ROOT,
    DATASET_NAME,
    "trajectory.txt",
)

CENTROID_FILE = os.path.join(
    GENERATED_ROOT,
    DATASET_NAME,
    "lidar_detections",
    "unified_clusters.txt",
)

# Optional ground-truth overlay file.
# Set to None to disable.
GROUND_TRUTH_FILE = None
# Example:
# GROUND_TRUTH_FILE = os.path.join(
#     GENERATED_ROOT,
#     DATASET_NAME,
#     "lidar_refinement",
#     "final_clusters.txt",
# )

# Optional calibration file.
# If the file does not exist, zero shift and zero yaw are used.
SHIFT_FILE = os.path.join(
    EXTRACTED_ROOT,
    DATASET_NAME,
    "shift.txt",
)

# Output file.
OUTPUT_FILTERED_FILE = os.path.splitext(CENTROID_FILE)[0] + "_filtered.txt"

# Map folder created by the map-download script.
MAP_FOLDER = os.path.join(MAP_ROOT)

# Plot settings.
BUFFER_M = 150
BAR_LENGTH = 4.0

# Coordinate transformers.
TRANSFORMER_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
TRANSFORMER_WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


# =======================
# MATH HELPERS
# =======================

def get_inverse_coords(proj_x, proj_y, shift_x, shift_y, yaw_offset):
    """
    Convert Web Mercator coordinates back to odometry / UTM coordinates.

    Input:
        proj_x, proj_y: EPSG:3857 map coordinates.

    Output:
        raw_x, raw_y: odometry / UTM coordinates.
    """
    lon, lat = TRANSFORMER_TO_4326.transform(proj_x, proj_y)

    utm_x, utm_y = TRANSFORMER_WGS84_TO_UTM.transform(lon, lat)

    if abs(yaw_offset) > 1e-9:
        dx = utm_x - ODOM0_X
        dy = utm_y - ODOM0_Y

        c = math.cos(-yaw_offset)
        s = math.sin(-yaw_offset)

        dx_rot = c * dx - s * dy
        dy_rot = s * dx + c * dy

        utm_x = ODOM0_X + dx_rot
        utm_y = ODOM0_Y + dy_rot

    raw_x = utm_x - shift_x
    raw_y = utm_y - shift_y

    return raw_x, raw_y


# =======================
# DATA LOADING
# =======================

def load_trajectory_from_txt(path):
    """
    Load trajectory file.

    Expected format:
    # timestamp x y z yaw
    timestamp, x, y, z, yaw
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    data = np.loadtxt(path, delimiter=",", comments="#", usecols=(1, 2))

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[:, 0], data[:, 1]


def load_calibration(path):
    """
    Load optional shift/yaw calibration file.

    Expected keys:
    SHIFT_X=
    SHIFT_Y=
    YAW_OFFSET=
    """
    shift_x = 0.0
    shift_y = 0.0
    yaw_offset = 0.0

    if not os.path.exists(path):
        print(f"Calibration file not found, using zero shift: {path}")
        return shift_x, shift_y, yaw_offset

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("SHIFT_X="):
                shift_x = float(line.split("=", 1)[1])
            elif line.startswith("SHIFT_Y="):
                shift_y = float(line.split("=", 1)[1])
            elif line.startswith("YAW_OFFSET="):
                yaw_offset = float(line.split("=", 1)[1])

    return shift_x, shift_y, yaw_offset


def load_ground_truth(path):
    """
    Load optional ground-truth file for reference display.

    Expected columns:
    cluster_id, x, y, z, ...
    """
    gt_list = []

    if path is None:
        return gt_list

    if not os.path.exists(path):
        print(f"Ground-truth file not found: {path}")
        return gt_list

    try:
        raw = np.genfromtxt(
            path,
            delimiter=",",
            comments="#",
            dtype=None,
            encoding="utf-8",
        )

        if raw.ndim == 0:
            raw = np.array([raw])

        for row in raw:
            gt_list.append({
                "id": int(float(row[0])),
                "x": float(row[1]),
                "y": float(row[2]),
                "z": float(row[3]),
            })

        print(f"Loaded {len(gt_list)} ground-truth points.")

    except Exception as exc:
        print(f"Error loading ground truth: {exc}")

    return gt_list


def load_centroids(path):
    """
    Load detected centroid file.

    Expected format:
    cluster_id, x, y, z, count, conf, orientation, side
    """
    centroid_list = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Centroid file not found: {path}")

    raw = np.genfromtxt(
        path,
        delimiter=",",
        comments="#",
        dtype=None,
        encoding="utf-8",
    )

    if raw.ndim == 0:
        raw = np.array([raw])

    for row in raw:
        centroid = {
            "id": int(float(row[0])),
            "x": float(row[1]),
            "y": float(row[2]),
            "z": float(row[3]),
            "count": int(float(row[4])),
            "conf": float(row[5]),
            "orient": str(row[6]).lower().strip(),
            "side": str(row[7]).lower().strip(),
        }

        if centroid["side"] == "left":
            centroid["color"] = "limegreen"
        elif centroid["side"] == "right":
            centroid["color"] = "hotpink"
        else:
            centroid["color"] = "gray"

        centroid_list.append(centroid)

    return centroid_list


# =======================
# PLOT HELPERS
# =======================

def calculate_bar_segments(cx, cy, tx, ty, orientations, length):
    segments = []

    for i in range(len(cx)):
        dists = (tx - cx[i]) ** 2 + (ty - cy[i]) ** 2
        idx = np.argmin(dists)

        if idx < len(tx) - 1:
            angle = np.arctan2(ty[idx + 1] - ty[idx], tx[idx + 1] - tx[idx])
        else:
            angle = np.arctan2(ty[idx] - ty[idx - 1], tx[idx] - tx[idx - 1])

        if orientations[i] == "perpendicular":
            angle += np.pi / 2.0

        dx = (length / 2.0) * np.cos(angle)
        dy = (length / 2.0) * np.sin(angle)

        segments.append([
            (cx[i] - dx, cy[i] - dy),
            (cx[i] + dx, cy[i] + dy),
        ])

    return segments


# =======================
# MAIN
# =======================

print("=" * 70)
print("INTERACTIVE CLUSTER CLEANER")
print("=" * 70)
print(f"Dataset name: {DATASET_NAME}")
print(f"Map name: {MAP_NAME}")
print(f"Map folder: {os.path.abspath(MAP_FOLDER)}")
print(f"Trajectory: {os.path.abspath(TRAJECTORY_PATH)}")
print(f"Centroids: {os.path.abspath(CENTROID_FILE)}")
print(f"Ground truth: {GROUND_TRUTH_FILE}")
print(f"Shift file: {os.path.abspath(SHIFT_FILE)}")
print(f"Output: {os.path.abspath(OUTPUT_FILTERED_FILE)}")

odom_x, odom_y = load_trajectory_from_txt(TRAJECTORY_PATH)
init_sx, init_sy, init_yaw = load_calibration(SHIFT_FILE)

gt_data_list = load_ground_truth(GROUND_TRUTH_FILE)
centroid_data_list = load_centroids(CENTROID_FILE)

print(f"Loaded {len(centroid_data_list)} centroids.")

proj_tx, proj_ty = get_projected_coords(odom_x, odom_y)

print(f"Projected trajectory x range: [{np.min(proj_tx):.2f}, {np.max(proj_tx):.2f}]")
print(f"Projected trajectory y range: [{np.min(proj_ty):.2f}, {np.max(proj_ty):.2f}]")

proj_gt_x = np.array([])
proj_gt_y = np.array([])

if gt_data_list:
    gt_x = np.array([g["x"] for g in gt_data_list])
    gt_y = np.array([g["y"] for g in gt_data_list])
    proj_gt_x, proj_gt_y = get_projected_coords(gt_x, gt_y)


# =======================
# FIGURE SETUP
# =======================

fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(bottom=0.20)

ax.set_xlim(np.min(proj_tx) - BUFFER_M, np.max(proj_tx) + BUFFER_M)
ax.set_ylim(np.min(proj_ty) - BUFFER_M, np.max(proj_ty) + BUFFER_M)

try:
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron)
except Exception as exc:
    print(f"Basemap failed: {type(exc).__name__}: {exc}")

ax.plot(proj_tx, proj_ty, "r-", alpha=0.5, label="Trajectory")

if len(proj_gt_x) > 0:
    ax.scatter(
        proj_gt_x,
        proj_gt_y,
        c="gold",
        s=100,
        marker="s",
        edgecolor="black",
        linewidth=1,
        zorder=4,
        label="Ground Truth",
    )

    for i, gt in enumerate(gt_data_list):
        ax.annotate(
            f"GT{gt['id']}",
            (proj_gt_x[i], proj_gt_y[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            color="darkgoldenrod",
        )

cent_bars_coll = LineCollection([], linewidths=2.5, zorder=10)
ax.add_collection(cent_bars_coll)

cent_scat = ax.scatter(
    [],
    [],
    s=40,
    edgecolor="k",
    linewidth=0.5,
    zorder=11,
    picker=5,
)


def update_plot_data():
    if not centroid_data_list:
        cent_scat.set_offsets(np.zeros((0, 2)))
        cent_bars_coll.set_segments([])
        fig.canvas.draw_idle()
        return

    cx_curr = np.array([c["x"] for c in centroid_data_list])
    cy_curr = np.array([c["y"] for c in centroid_data_list])

    colors_curr = [c["color"] for c in centroid_data_list]
    orients_curr = [c["orient"] for c in centroid_data_list]

    proj_cx, proj_cy = get_projected_coords(cx_curr, cy_curr)

    new_segments = calculate_bar_segments(
        proj_cx,
        proj_cy,
        proj_tx,
        proj_ty,
        orients_curr,
        BAR_LENGTH,
    )

    cent_scat.set_offsets(np.c_[proj_cx, proj_cy])
    cent_scat.set_facecolors(colors_curr)

    cent_bars_coll.set_segments(new_segments)
    cent_bars_coll.set_color(colors_curr)

    fig.canvas.draw_idle()


update_plot_data()


# =======================
# INTERACTION STATE
# =======================

current_mode = "DELETE"
dragging_idx = None
next_insert_id = max([c["id"] for c in centroid_data_list], default=0) + 1


def get_next_id():
    global next_insert_id

    new_id = next_insert_id
    next_insert_id += 1

    return new_id


def on_pick(event):
    global dragging_idx

    if current_mode == "INSERT":
        return

    if event.artist != cent_scat:
        return

    if event.mouseevent.button != 1:
        return

    ind = event.ind[0]
    car_id = centroid_data_list[ind]["id"]

    if current_mode == "DELETE":
        removed = centroid_data_list.pop(ind)
        print(f"DELETE: ID {removed['id']}")
        update_plot_data()

    elif current_mode == "ROTATE":
        current_orientation = centroid_data_list[ind]["orient"]
        new_orientation = "perpendicular" if current_orientation == "parallel" else "parallel"

        centroid_data_list[ind]["orient"] = new_orientation

        print(f"ROTATE: ID {car_id} -> {new_orientation.upper()}")
        update_plot_data()

    elif current_mode == "SWITCH_SIDE":
        current_side = centroid_data_list[ind]["side"]

        if current_side == "left":
            centroid_data_list[ind]["side"] = "right"
            centroid_data_list[ind]["color"] = "hotpink"
        else:
            centroid_data_list[ind]["side"] = "left"
            centroid_data_list[ind]["color"] = "limegreen"

        print(f"SIDE: ID {car_id} -> {centroid_data_list[ind]['side'].upper()}")
        update_plot_data()

    elif current_mode == "MOVE":
        dragging_idx = ind
        print(f"GRABBED: ID {car_id}. Drag to move.")


def on_click(event):
    if current_mode != "INSERT":
        return

    if event.inaxes != ax:
        return

    if event.button != 1:
        return

    if event.xdata is None or event.ydata is None:
        return

    raw_x, raw_y = get_inverse_coords(
        event.xdata,
        event.ydata,
        init_sx,
        init_sy,
        init_yaw,
    )

    new_id = get_next_id()

    new_car = {
        "id": new_id,
        "x": raw_x,
        "y": raw_y,
        "z": 0.0,
        "count": 1,
        "conf": 1.0,
        "orient": "parallel",
        "side": "right",
        "color": "hotpink",
    }

    centroid_data_list.append(new_car)

    print(f"INSERT: New car ID {new_id} at ({raw_x:.2f}, {raw_y:.2f}) - right/parallel")

    update_plot_data()


def on_release(event):
    global dragging_idx

    if dragging_idx is not None and current_mode == "MOVE":
        if event.xdata is not None and event.ydata is not None:
            raw_x, raw_y = get_inverse_coords(
                event.xdata,
                event.ydata,
                init_sx,
                init_sy,
                init_yaw,
            )

            centroid_data_list[dragging_idx]["x"] = raw_x
            centroid_data_list[dragging_idx]["y"] = raw_y

            print(
                f"MOVED: ID {centroid_data_list[dragging_idx]['id']} "
                f"to ({raw_x:.2f}, {raw_y:.2f})"
            )

            update_plot_data()

        dragging_idx = None


fig.canvas.mpl_connect("pick_event", on_pick)
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("button_release_event", on_release)


# =======================
# GUI BUTTONS
# =======================

ax_btn_del = plt.axes([0.03, 0.05, 0.12, 0.05])
ax_btn_rot = plt.axes([0.16, 0.05, 0.12, 0.05])
ax_btn_swi = plt.axes([0.29, 0.05, 0.12, 0.05])
ax_btn_mov = plt.axes([0.42, 0.05, 0.12, 0.05])
ax_btn_ins = plt.axes([0.55, 0.05, 0.12, 0.05])
ax_btn_sav = plt.axes([0.75, 0.05, 0.20, 0.05])

btn_del = Button(ax_btn_del, "DELETE")
btn_rot = Button(ax_btn_rot, "ROTATE")
btn_swi = Button(ax_btn_swi, "SIDE")
btn_mov = Button(ax_btn_mov, "MOVE")
btn_ins = Button(ax_btn_ins, "INSERT")
btn_sav = Button(ax_btn_sav, "Save Filtered")

active_color = "gold"
inactive_color = "0.95"


def set_mode(mode):
    global current_mode, dragging_idx

    current_mode = mode
    dragging_idx = None

    btn_del.color = active_color if mode == "DELETE" else inactive_color
    btn_rot.color = active_color if mode == "ROTATE" else inactive_color
    btn_swi.color = active_color if mode == "SWITCH_SIDE" else inactive_color
    btn_mov.color = active_color if mode == "MOVE" else inactive_color
    btn_ins.color = active_color if mode == "INSERT" else inactive_color

    btn_del.ax.set_facecolor(btn_del.color)
    btn_rot.ax.set_facecolor(btn_rot.color)
    btn_swi.ax.set_facecolor(btn_swi.color)
    btn_mov.ax.set_facecolor(btn_mov.color)
    btn_ins.ax.set_facecolor(btn_ins.color)

    fig.canvas.draw_idle()

    mode_hints = {
        "DELETE": "Click points to delete",
        "ROTATE": "Click points to toggle parallel/perpendicular",
        "SWITCH_SIDE": "Click points to toggle left/right",
        "MOVE": "Click and drag to move points",
        "INSERT": "Click on map to add new car, default right/parallel",
    }

    ax.set_title(f"MODE: {mode} - {mode_hints.get(mode, '')}")


def cb_del(event):
    set_mode("DELETE")


def cb_rot(event):
    set_mode("ROTATE")


def cb_swi(event):
    set_mode("SWITCH_SIDE")


def cb_mov(event):
    set_mode("MOVE")


def cb_ins(event):
    set_mode("INSERT")


def cb_save(event):
    print(f"Saving to {OUTPUT_FILTERED_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILTERED_FILE), exist_ok=True)

    try:
        with open(OUTPUT_FILTERED_FILE, "w") as f:
            f.write("# cluster_id, x, y, z, count, last_conf, orientation, side\n")

            for c in centroid_data_list:
                f.write(
                    f"{c['id']}, "
                    f"{c['x']:.3f}, "
                    f"{c['y']:.3f}, "
                    f"{c['z']:.3f}, "
                    f"{c['count']}, "
                    f"{c['conf']:.3f}, "
                    f"{c['orient']}, "
                    f"{c['side']}\n"
                )

        btn_sav.label.set_text("Saved")
        fig.canvas.draw_idle()

        print(f"Saved {len(centroid_data_list)} cars to {OUTPUT_FILTERED_FILE}")

    except Exception as exc:
        print(f"Error saving filtered file: {exc}")


btn_del.on_clicked(cb_del)
btn_rot.on_clicked(cb_rot)
btn_swi.on_clicked(cb_swi)
btn_mov.on_clicked(cb_mov)
btn_ins.on_clicked(cb_ins)
btn_sav.on_clicked(cb_save)

set_mode("DELETE")

custom_legend = [
    Line2D([0], [0], color="limegreen", lw=3, label="Left"),
    Line2D([0], [0], color="hotpink", lw=3, label="Right"),
    Line2D([0], [0], color="black", lw=1, label="Orientation"),
]

if gt_data_list:
    custom_legend.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gold",
            markersize=10,
            markeredgecolor="black",
            label="Ground Truth",
        )
    )

ax.legend(handles=custom_legend, loc="upper right")

plt.show()