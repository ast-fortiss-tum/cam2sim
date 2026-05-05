import os
import sys

# =======================
# PATH SETUP (workaround per import da root del progetto)
# =======================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
import contextily as cx
from pyproj import Transformer
from utils.coordinates import LAT0, LON0, ODOM0_X, ODOM0_Y, get_projected_coords, load_shift_values

# =========================================================
# ⚙️ ARGUMENT PARSING
# =========================================================
parser = argparse.ArgumentParser(description='Interactive cluster cleaner with optional GT overlay')
parser.add_argument('--gt', '-g', default=None,
                    help='Optional ground truth file for reference (not editable)')
args = parser.parse_args()

# =========================================================
# ⚙️ USER SETTINGS
# =========================================================
trajectory_path = "datasets/2026-02-16-23-07-19/trajectory.txt"
centroid_file   = "datasets/2026-02-16-23-07-19/unified_clusters.txt"

buffer_m = 150
BAR_LENGTH = 4.0

dataset_folder = os.path.dirname(trajectory_path)
shift_file = os.path.join(dataset_folder, "shift.txt")
# Output filename: append "_filtered" before the .txt extension
base, ext = os.path.splitext(centroid_file)
output_filtered_file = f"{base}_filtered{ext}"



# 1. Coordinate Transformers
transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
transformer_wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

# =========================================================
# 1. MATH HELPERS (FORWARD & INVERSE)
# =========================================================

def get_inverse_coords(proj_x, proj_y, shift_x, shift_y, yaw_offset):
    """
    INVERSE: Web Map (EPSG:3857) -> Odom UTM (EPSG:25832)
    Uses proper UTM transformation to match forward pipeline exactly.
    """
    # 1. Web Mercator (EPSG:3857) -> WGS84 Lat/Lon
    lon, lat = transformer_to_4326.transform(proj_x, proj_y)

    # 2. WGS84 -> UTM Zone 32N (EPSG:25832)
    utm_x, utm_y = transformer_wgs84_to_utm.transform(lon, lat)

    # 3. Inverse Yaw Rotation (around ODOM0)
    if abs(yaw_offset) > 1e-9:
        dx = utm_x - ODOM0_X
        dy = utm_y - ODOM0_Y

        c, s = math.cos(-yaw_offset), math.sin(-yaw_offset)  # Negative for inverse
        dx_rot = c * dx - s * dy
        dy_rot = s * dx + c * dy

        utm_x = ODOM0_X + dx_rot
        utm_y = ODOM0_Y + dy_rot

    # 4. Inverse Shift
    raw_x = utm_x - shift_x
    raw_y = utm_y - shift_y

    return raw_x, raw_y

# =========================================================
# 2. DATA LOADING
# =========================================================

def load_trajectory_from_txt(path):
    if not os.path.exists(path): raise FileNotFoundError(f"{path} not found")
    # File format: timestamp, x, y, z, yaw -> read columns 1 and 2 (x, y)
    data = np.loadtxt(path, delimiter=',', comments='#', usecols=(1, 2))
    return data[:, 0], data[:, 1]

def load_calibration(path):
    sx, sy, yaw = 0.0, 0.0, 0.0
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "SHIFT_X=" in line: sx = float(line.split("=")[1])
                if "SHIFT_Y=" in line: sy = float(line.split("=")[1])
                if "YAW_OFFSET=" in line: yaw = float(line.split("=")[1])
    return sx, sy, yaw

def load_ground_truth(path):
    """Load ground truth file for reference display."""
    gt_list = []
    if not os.path.exists(path):
        print(f"⚠️ Ground truth file not found: {path}")
        return gt_list
    
    try:
        raw = np.genfromtxt(path, delimiter=",", comments="#", dtype=None, encoding='utf-8')
        if raw.ndim == 0: 
            raw = np.array([raw])
        for row in raw:
            gt_data = {
                'id': row[0], 
                'x': float(row[1]), 
                'y': float(row[2]), 
                'z': float(row[3])
            }
            gt_list.append(gt_data)
        print(f"✅ Loaded {len(gt_list)} Ground Truth points.")
    except Exception as e:
        print(f"❌ Error loading GT: {e}")
    
    return gt_list

print("--- STARTING INTERACTIVE CLEANER ---")
odom_x, odom_y = load_trajectory_from_txt(trajectory_path)
init_sx, init_sy, init_yaw = load_calibration(shift_file)

# Load Ground Truth if provided
gt_data_list = []
if args.gt:
    print(f"⏳ Loading Ground Truth from {args.gt}...")
    gt_data_list = load_ground_truth(args.gt)

centroid_data_list = []
if os.path.exists(centroid_file):
    print("⏳ Loading Centroids...")
    try:
        raw = np.genfromtxt(centroid_file, delimiter=",", comments="#", dtype=None, encoding='utf-8')
        if raw.ndim == 0: raw = np.array([raw])
        for row in raw:
            # New format (no color column):
            # cluster_id, x, y, z, count, conf, orientation, side
            c_data = {
                'id': row[0], 'x': float(row[1]), 'y': float(row[2]),
                'z': float(row[3]), 'count': int(row[4]), 'conf': float(row[5]),
                'orient': str(row[6]).lower().strip(), 'side': str(row[7]).lower().strip()
            }
            if c_data['side'] == "left": c_data['color'] = 'limegreen'
            elif c_data['side'] == "right": c_data['color'] = 'hotpink'
            else: c_data['color'] = 'gray'
            centroid_data_list.append(c_data)
        print(f"✅ Loaded {len(centroid_data_list)} Centroids.")
    except Exception as e: print(f"❌ Error: {e}")

# =========================================================
# 3. PLOT LOGIC
# =========================================================

proj_tx, proj_ty = get_projected_coords(odom_x, odom_y)
print(f"DEBUG proj_tx range: [{np.min(proj_tx):.2f}, {np.max(proj_tx):.2f}]")
print(f"DEBUG proj_ty range: [{np.min(proj_ty):.2f}, {np.max(proj_ty):.2f}]")

# Project Ground Truth coordinates
proj_gt_x, proj_gt_y = np.array([]), np.array([])
if gt_data_list:
    gt_x = np.array([g['x'] for g in gt_data_list])
    gt_y = np.array([g['y'] for g in gt_data_list])
    proj_gt_x, proj_gt_y = get_projected_coords(gt_x, gt_y)

def calculate_bar_segments(cx, cy, tx, ty, orientations, length):
    segments = []
    for i in range(len(cx)):
        dists = (tx - cx[i])**2 + (ty - cy[i])**2
        idx = np.argmin(dists)
        if idx < len(tx) - 1: angle = np.arctan2(ty[idx+1] - ty[idx], tx[idx+1] - tx[idx])
        else: angle = np.arctan2(ty[idx] - ty[idx-1], tx[idx] - tx[idx-1])
        
        if orientations[i] == "perpendicular": angle += np.pi / 2
        dx, dy = (length/2)*np.cos(angle), (length/2)*np.sin(angle)
        segments.append([(cx[i]-dx, cy[i]-dy), (cx[i]+dx, cy[i]+dy)])
    return segments

def update_plot_data():
    if not centroid_data_list:
        cent_scat.set_offsets(np.zeros((0, 2)))
        cent_bars_coll.set_segments([])
        fig.canvas.draw_idle()
        return

    cx_curr = np.array([c['x'] for c in centroid_data_list])
    cy_curr = np.array([c['y'] for c in centroid_data_list])
    colors_curr = [c['color'] for c in centroid_data_list]
    orients_curr = [c['orient'] for c in centroid_data_list]

    proj_cx, proj_cy = get_projected_coords(cx_curr, cy_curr)
    new_segs = calculate_bar_segments(proj_cx, proj_cy, proj_tx, proj_ty, orients_curr, BAR_LENGTH)
    
    cent_scat.set_offsets(np.c_[proj_cx, proj_cy])
    cent_scat.set_facecolors(colors_curr)
    cent_bars_coll.set_segments(new_segs)
    cent_bars_coll.set_color(colors_curr)
    
    fig.canvas.draw_idle()

# Plot Setup
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(bottom=0.20)
ax.set_xlim(np.min(proj_tx)-buffer_m, np.max(proj_tx)+buffer_m)
ax.set_ylim(np.min(proj_ty)-buffer_m, np.max(proj_ty)+buffer_m)
try:
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron)
except Exception as e:
    print(f"⚠️ Basemap failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

ax.plot(proj_tx, proj_ty, 'r-', alpha=0.5, label='Trajectory')

# Plot Ground Truth (non-editable, yellow squares)
gt_scat = None
if len(proj_gt_x) > 0:
    gt_scat = ax.scatter(proj_gt_x, proj_gt_y, c='gold', s=100, marker='s', 
                         edgecolor='black', linewidth=1, zorder=4, label='Ground Truth')
    # Add GT labels
    for i, gt in enumerate(gt_data_list):
        ax.annotate(f"GT{gt['id']}", (proj_gt_x[i], proj_gt_y[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, color='darkgoldenrod')

cent_bars_coll = LineCollection([], linewidths=2.5, zorder=10)
ax.add_collection(cent_bars_coll)
cent_scat = ax.scatter([], [], s=40, edgecolor='k', linewidth=0.5, zorder=11, picker=5)

update_plot_data()

# =========================================================
# 4. INTERACTION STATE MACHINE
# =========================================================
current_mode = "DELETE"
dragging_idx = None
next_insert_id = max([c['id'] for c in centroid_data_list], default=0) + 1

def get_next_id():
    """Generate next available ID for new cars"""
    global next_insert_id
    new_id = next_insert_id
    next_insert_id += 1
    return new_id

def on_pick(event):
    global dragging_idx
    if current_mode == "INSERT":
        return  # INSERT mode doesn't use pick events
    if event.artist != cent_scat:
        return  # Ignore picks on GT scatter (not editable)
    if event.mouseevent.button != 1:
        return

    ind = event.ind[0]
    c_id = centroid_data_list[ind]['id']

    if current_mode == "DELETE":
        removed = centroid_data_list.pop(ind)
        print(f"❌ DELETE: ID {removed['id']}")
        update_plot_data()

    elif current_mode == "ROTATE":
        curr = centroid_data_list[ind]['orient']
        new_o = 'perpendicular' if curr == 'parallel' else 'parallel'
        centroid_data_list[ind]['orient'] = new_o
        print(f"🔄 ROTATE: ID {c_id} -> {new_o.upper()}")
        update_plot_data()

    elif current_mode == "SWITCH_SIDE":
        curr_s = centroid_data_list[ind]['side']
        if curr_s == 'left':
            centroid_data_list[ind]['side'] = 'right'
            centroid_data_list[ind]['color'] = 'hotpink'
        else:
            centroid_data_list[ind]['side'] = 'left'
            centroid_data_list[ind]['color'] = 'limegreen'
        print(f"↔️ SIDE: ID {c_id} -> {centroid_data_list[ind]['side'].upper()}")
        update_plot_data()

    elif current_mode == "MOVE":
        dragging_idx = ind
        print(f"✊ GRABBED: ID {c_id}. Drag to move...")

def on_click(event):
    """Handle clicks for INSERT mode"""
    if current_mode != "INSERT":
        return
    if event.inaxes != ax:
        return
    if event.button != 1:  # Left click only
        return
    if event.xdata is None or event.ydata is None:
        return

    # Convert click position to UTM coordinates
    raw_x, raw_y = get_inverse_coords(event.xdata, event.ydata, init_sx, init_sy, init_yaw)

    # Create new car with default values
    new_id = get_next_id()
    new_car = {
        'id': new_id,
        'x': raw_x,
        'y': raw_y,
        'z': 0.0,
        'count': 1,
        'conf': 1.0,
        'orient': 'parallel',    # Default: parallel
        'side': 'right',         # Default: right
        'color': 'hotpink'       # Right side color
    }

    centroid_data_list.append(new_car)
    print(f"➕ INSERT: New car ID {new_id} at ({raw_x:.2f}, {raw_y:.2f}) - right/parallel")
    update_plot_data()

def on_release(event):
    global dragging_idx
    if dragging_idx is not None and current_mode == "MOVE":
        if event.xdata is not None and event.ydata is not None:
            # 1. Get new mouse position (Projected Web Mercator)
            new_proj_x, new_proj_y = event.xdata, event.ydata

            # 2. Convert back to Raw Odom Coords using correct Inverse
            raw_x, raw_y = get_inverse_coords(new_proj_x, new_proj_y, init_sx, init_sy, init_yaw)

            # 3. Update Data
            centroid_data_list[dragging_idx]['x'] = raw_x
            centroid_data_list[dragging_idx]['y'] = raw_y

            print(f"📍 MOVED: ID {centroid_data_list[dragging_idx]['id']} to ({raw_x:.2f}, {raw_y:.2f})")
            update_plot_data()

        dragging_idx = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)

# =========================================================
# 5. GUI BUTTONS
# =========================================================

ax_btn_del = plt.axes([0.03, 0.05, 0.12, 0.05])
ax_btn_rot = plt.axes([0.16, 0.05, 0.12, 0.05])
ax_btn_swi = plt.axes([0.29, 0.05, 0.12, 0.05])
ax_btn_mov = plt.axes([0.42, 0.05, 0.12, 0.05])
ax_btn_ins = plt.axes([0.55, 0.05, 0.12, 0.05])
ax_btn_sav = plt.axes([0.75, 0.05, 0.20, 0.05])

btn_del = Button(ax_btn_del, 'DELETE')
btn_rot = Button(ax_btn_rot, 'ROTATE')
btn_swi = Button(ax_btn_swi, 'SIDE')
btn_mov = Button(ax_btn_mov, 'MOVE')
btn_ins = Button(ax_btn_ins, 'INSERT')
btn_sav = Button(ax_btn_sav, 'Save Filtered')

active_color = 'gold'
inactive_color = '0.95'

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
        "MOVE": "Click & Drag to move points",
        "INSERT": "Click on map to add new car (right/parallel)"
    }
    ax.set_title(f"MODE: {mode} - {mode_hints.get(mode, '')}")

def cb_del(event): set_mode("DELETE")
def cb_rot(event): set_mode("ROTATE")
def cb_swi(event): set_mode("SWITCH_SIDE")
def cb_mov(event): set_mode("MOVE")
def cb_ins(event): set_mode("INSERT")

def cb_save(event):
    print(f"💾 Saving to {output_filtered_file}...")
    try:
        with open(output_filtered_file, "w") as f:
            f.write("# cluster_id, x, y, z, count, last_conf, orientation, side\n")
            for c in centroid_data_list:
                f.write(f"{c['id']}, {c['x']:.3f}, {c['y']:.3f}, {c['z']:.3f}, {c['count']}, {c['conf']:.3f}, {c['orient']}, {c['side']}\n")
        btn_sav.label.set_text("✅ Saved!")
        fig.canvas.draw_idle()
        print(f"✅ Saved {len(centroid_data_list)} cars to {output_filtered_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

btn_del.on_clicked(cb_del)
btn_rot.on_clicked(cb_rot)
btn_swi.on_clicked(cb_swi)
btn_mov.on_clicked(cb_mov)
btn_ins.on_clicked(cb_ins)
btn_sav.on_clicked(cb_save)

set_mode("DELETE")

from matplotlib.lines import Line2D
custom_legend = [
    Line2D([0], [0], color='limegreen', lw=3, label='Left'),
    Line2D([0], [0], color='hotpink', lw=3, label='Right'),
    Line2D([0], [0], color='black', lw=1, label='Orientation')
]
# Add GT to legend if loaded
if gt_data_list:
    custom_legend.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', 
                                 markersize=10, markeredgecolor='black', label='Ground Truth'))
ax.legend(handles=custom_legend, loc='upper right')

plt.show()