#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_scenario_segment.py

Computes the scenario segment (common start/end) from successful
real-world trajectories and plots everything on OSM for verification.

Steps:
  1. Load all real-world trajectories, convert UTM → CARLA
  2. Build reference path from longest successful run
  3. Project all successful runs → find scenario start & end
  4. Plot on OSM: all trajectories + scenario segment highlighted

Usage:
    python verify_scenario_segment.py \
        --map_xodr maps/sunny_map/map.xodr \
        --rw_dir /media/davide/New\ Volume/RW_trajectories
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer


# =============================================================================
# XODR PARSING
# =============================================================================

def get_xodr_projection_params(xodr_data):
    geo_match = re.search(r'<geoReference>\s*<!\[CDATA\[(.*?)\]\]>', xodr_data, re.DOTALL)
    geo_ref = geo_match.group(1).strip() if geo_match else "+proj=tmerc"
    offset_match = re.search(r'<offset\s+x="([^"]+)"\s+y="([^"]+)"', xodr_data)
    if offset_match:
        offset = (float(offset_match.group(1)), float(offset_match.group(2)))
    else:
        offset = (0.0, 0.0)
    return {"geo_reference": geo_ref, "offset": offset}


# =============================================================================
# COORDINATE CONVERSIONS
# =============================================================================

def setup_transforms(xodr_path):
    with open(xodr_path, "r") as f:
        xodr_data = f.read()
    params = get_xodr_projection_params(xodr_data)
    xodr_offset = params["offset"]
    proj_string = params["geo_reference"].strip()
    if proj_string == "+proj=tmerc":
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

    tf_utm_to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    tf_wgs_to_proj = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    tf_proj_to_wgs = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
    tf_wgs_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    return tf_utm_to_wgs, tf_wgs_to_proj, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset


def utm_to_carla(utm_x, utm_y, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset):
    lon, lat = tf_utm_to_wgs.transform(utm_x, utm_y)
    proj_x, proj_y = tf_wgs_to_proj.transform(lon, lat)
    carla_x = proj_x + xodr_offset[0]
    carla_y = -(proj_y + xodr_offset[1])
    return carla_x, carla_y


def carla_to_webmercator(carla_x, carla_y, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset):
    local_x = carla_x
    local_y = -carla_y
    proj_x = local_x - xodr_offset[0]
    proj_y = local_y - xodr_offset[1]
    lon, lat = tf_proj_to_wgs.transform(proj_x, proj_y)
    mx, my = tf_wgs_to_merc.transform(lon, lat)
    return mx, my


# =============================================================================
# TRAJECTORY LOADING
# =============================================================================

def load_real_trajectory_utm(path):
    xs, ys = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    xs.append(float(parts[2]))
                    ys.append(float(parts[3]))
                except ValueError:
                    continue
    return np.array(xs), np.array(ys)


# =============================================================================
# REFERENCE PATH & PROJECTION
# =============================================================================

def compute_arc_length(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def build_reference_path(xs, ys, spacing=0.5):
    s = compute_arc_length(xs, ys)
    total_length = s[-1]
    n_points = max(int(total_length / spacing) + 1, 2)
    s_uniform = np.linspace(0, total_length, n_points)
    ref_xs = np.interp(s_uniform, s, xs)
    ref_ys = np.interp(s_uniform, s, ys)
    return ref_xs, ref_ys, s_uniform


def project_onto_reference(traj_xs, traj_ys, ref_xs, ref_ys, ref_s):
    progress = np.zeros(len(traj_xs))
    lateral_dist = np.zeros(len(traj_xs))
    for i in range(len(traj_xs)):
        dists = np.sqrt((ref_xs - traj_xs[i])**2 + (ref_ys - traj_ys[i])**2)
        closest_idx = np.argmin(dists)
        lateral_dist[i] = dists[closest_idx]
        progress[i] = ref_s[closest_idx]
    # Enforce monotonicity
    for i in range(1, len(progress)):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]
    return progress, lateral_dist


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify scenario segment on OSM")
    parser.add_argument("--map_xodr", required=True, help="Path to .xodr file")
    parser.add_argument("--rw_dir", required=True, help="Directory with real-world trajectory files")
    parser.add_argument("--buffer", type=float, default=100, help="Map buffer in meters")
    args = parser.parse_args()

    # --- Setup transforms ---
    tf_utm_to_wgs, tf_wgs_to_proj, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset = \
        setup_transforms(args.map_xodr)

    # --- Load all real-world trajectories ---
    print("=" * 60)
    print("  LOADING REAL-WORLD TRAJECTORIES")
    print("=" * 60)

    SUCCESSFUL_CONDITIONS = {"sunny", "cloudy"}
    rw_trajs = {}  # (condition, run) → (carla_x, carla_y)

    for fname in sorted(os.listdir(args.rw_dir)):
        if not fname.endswith("_trajectory.txt"):
            continue
        match = re.match(r'(\w+?)(\d+)_trajectory\.txt', fname)
        if not match:
            continue
        condition = match.group(1)
        run = int(match.group(2))
        path = os.path.join(args.rw_dir, fname)

        utm_x, utm_y = load_real_trajectory_utm(path)
        carla_xs, carla_ys = [], []
        for ux, uy in zip(utm_x, utm_y):
            cx, cy = utm_to_carla(ux, uy, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset)
            carla_xs.append(cx)
            carla_ys.append(cy)
        carla_xs, carla_ys = np.array(carla_xs), np.array(carla_ys)
        rw_trajs[(condition, run)] = (carla_xs, carla_ys)

        arc = compute_arc_length(carla_xs, carla_ys)
        is_success = condition in SUCCESSFUL_CONDITIONS
        print(f"  {fname}: {len(carla_xs)} pts, {arc[-1]:.1f} m "
              f"{'✅ success' if is_success else '❌ failure'}")

    # --- Build reference path from longest successful run ---
    successful_rw = {k: v for k, v in rw_trajs.items() if k[0] in SUCCESSFUL_CONDITIONS}

    longest_key = None
    longest_length = 0
    for key, (cx, cy) in successful_rw.items():
        length = compute_arc_length(cx, cy)[-1]
        if length > longest_length:
            longest_length = length
            longest_key = key

    print(f"\n  Reference path: {longest_key[0]}{longest_key[1]} ({longest_length:.1f} m)")

    ref_cx, ref_cy = successful_rw[longest_key]
    ref_xs, ref_ys, ref_s = build_reference_path(ref_cx, ref_cy, spacing=0.5)

    # --- Project all successful runs to find scenario segment ---
    print("\n  Projecting successful runs onto reference...")
    start_progresses = []
    end_progresses = []
    run_projections = {}

    for key, (cx, cy) in successful_rw.items():
        progress, lat_dist = project_onto_reference(cx, cy, ref_xs, ref_ys, ref_s)
        start_progresses.append(progress[0])
        end_progresses.append(progress[-1])
        run_projections[key] = progress
        print(f"    {key[0]}{key[1]}: [{progress[0]:.1f} m → {progress[-1]:.1f} m] "
              f"(max lateral: {lat_dist.max():.2f} m)")

    scenario_start = max(start_progresses)
    scenario_end = min(end_progresses)
    scenario_length = scenario_end - scenario_start

    print(f"\n  SCENARIO SEGMENT:")
    print(f"    Start: {scenario_start:.1f} m (latest starting point)")
    print(f"    End:   {scenario_end:.1f} m (earliest finishing point)")
    print(f"    Length: {scenario_length:.1f} m")

    # Find reference path indices for segment start/end
    seg_start_idx = np.argmin(np.abs(ref_s - scenario_start))
    seg_end_idx = np.argmin(np.abs(ref_s - scenario_end))

    # --- Save scenario segment + reference path ---
    import json
    output_json = os.path.join(args.rw_dir, "scenario_segment.json")
    segment_data = {
        "reference_run": f"{longest_key[0]}{longest_key[1]}",
        "reference_length_m": float(longest_length),
        "scenario_start_m": float(scenario_start),
        "scenario_end_m": float(scenario_end),
        "scenario_length_m": float(scenario_length),
        "reference_path_carla_x": ref_xs.tolist(),
        "reference_path_carla_y": ref_ys.tolist(),
        "reference_path_arc_length": ref_s.tolist(),
        "reference_path_spacing_m": 0.5,
    }
    with open(output_json, "w") as f:
        json.dump(segment_data, f, indent=2)
    print(f"\n  Saved: {output_json}")

    # --- Also project failed runs for context ---
    failed_rw = {k: v for k, v in rw_trajs.items() if k[0] not in SUCCESSFUL_CONDITIONS}
    for key, (cx, cy) in failed_rw.items():
        progress, lat_dist = project_onto_reference(cx, cy, ref_xs, ref_ys, ref_s)
        completion = max(0, min(1.0, (progress[-1] - scenario_start) / scenario_length))
        print(f"    {key[0]}{key[1]} (failed): [{progress[0]:.1f} m → {progress[-1]:.1f} m] "
              f"= {completion*100:.1f}% completion")

    # --- Convert everything to Web Mercator for plotting ---
    print("\n  Converting to Web Mercator for plotting...")

    def carla_array_to_merc(cxs, cys):
        mxs, mys = [], []
        for x, y in zip(cxs, cys):
            mx, my = carla_to_webmercator(x, y, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset)
            mxs.append(mx)
            mys.append(my)
        return np.array(mxs), np.array(mys)

    # Reference path in mercator
    ref_mx, ref_my = carla_array_to_merc(ref_xs, ref_ys)

    # Scenario segment in mercator
    seg_mx = ref_mx[seg_start_idx:seg_end_idx + 1]
    seg_my = ref_my[seg_start_idx:seg_end_idx + 1]

    # All trajectories in mercator
    rw_merc = {}
    for key, (cx, cy) in rw_trajs.items():
        mx, my = carla_array_to_merc(cx, cy)
        rw_merc[key] = (mx, my)

    # --- Plot ---
    print("  Plotting...")

    all_mx = np.concatenate([v[0] for v in rw_merc.values()])
    all_my = np.concatenate([v[1] for v in rw_merc.values()])
    buf = args.buffer
    xmin, xmax = all_mx.min() - buf, all_mx.max() + buf
    ymin, ymax = all_my.min() - buf, all_my.max() + buf

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    try:
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"  Warning: Could not load tiles: {e}")

    # Plot scenario segment (thick, behind everything)
    ax.plot(seg_mx, seg_my, color="#4CAF50", linewidth=8, alpha=0.3,
            solid_capstyle="round", zorder=5, label=f"Scenario segment ({scenario_length:.0f} m)")

    # Color scheme
    CONDITION_COLORS = {
        "sunny": "#FF9800",   # orange
        "cloudy": "#607D8B",  # blue-grey
        "snowy": "#90CAF9",   # light blue
    }
    LINE_STYLES = ["-", "--", ":"]

    for (condition, run), (mx, my) in sorted(rw_merc.items()):
        color = CONDITION_COLORS.get(condition, "#888888")
        ls = LINE_STYLES[(run - 1) % len(LINE_STYLES)]
        is_success = condition in SUCCESSFUL_CONDITIONS

        ax.plot(mx, my, color=color, linewidth=2.0, alpha=0.8,
                linestyle=ls, zorder=10, label=f"{condition} run{run}")

        # Start marker
        ax.plot(mx[0], my[0], marker='o', color=color, markersize=7,
                markeredgecolor='black', markeredgewidth=1, zorder=20)

        # End marker
        if is_success:
            ax.plot(mx[-1], my[-1], marker='s', color=color, markersize=7,
                    markeredgecolor='black', markeredgewidth=1, zorder=20)
        else:
            ax.plot(mx[-1], my[-1], marker='x', color=color, markersize=10,
                    markeredgewidth=3, zorder=20)

    # Mark segment start/end with vertical-ish markers
    ax.plot(seg_mx[0], seg_my[0], marker='|', color='#4CAF50', markersize=20,
            markeredgewidth=4, zorder=25, label="Segment start")
    ax.plot(seg_mx[-1], seg_my[-1], marker='|', color='#F44336', markersize=20,
            markeredgewidth=4, zorder=25, label="Segment end")

    ax.set_title("Scenario Segment Definition (from successful real-world runs)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()