#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_steering_vs_progress.py

Plots steering values vs progress (% of scenario segment).
One curve for real-world, one for simulated — overlaid for comparison.

Real-world steering is interpolated (nearest) onto trajectory positions.
Simulated steering is already paired with positions.

All values converted to raw DAVE-2 scale.

Usage:
    python plot_steering_vs_progress.py \
        --map_xodr maps/sunny_map/map.xodr \
        --segment /path/to/scenario_segment.json \
        --rw_traj /path/to/sunny1_trajectory.txt \
        --rw_steering /path/to/steering_sunny1.txt \
        --sim_traj /path/to/sunny_splatfacto_run1_trajectory.json
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer


# =============================================================================
# XODR
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
    return tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset


def utm_to_carla(utm_x, utm_y, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset):
    lon, lat = tf_utm_to_wgs.transform(utm_x, utm_y)
    proj_x, proj_y = tf_wgs_to_proj.transform(lon, lat)
    carla_x = proj_x + xodr_offset[0]
    carla_y = -(proj_y + xodr_offset[1])
    return carla_x, carla_y


# =============================================================================
# LOADERS
# =============================================================================

def load_real_trajectory(path):
    timestamps, xs, ys = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    timestamps.append(float(parts[1]))
                    xs.append(float(parts[2]))
                    ys.append(float(parts[3]))
                except ValueError:
                    continue
    return np.array(timestamps), np.array(xs), np.array(ys)


def load_real_steering(path):
    timestamps, values = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    timestamps.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(timestamps), np.array(values)


def load_sim_trajectory(path):
    with open(path, "r") as f:
        data = json.load(f)
    xs = np.array([p["x"] for p in data])
    ys = np.array([p["y"] for p in data])
    if "steer_raw" in data[0]:
        steering = np.array([p["steer_raw"] for p in data])
    elif "steering" in data[0]:
        steering = np.array([p["steering"] * 3 * np.pi for p in data])
    else:
        steering = np.zeros(len(data))
    return xs, ys, steering


# =============================================================================
# PROJECTION
# =============================================================================

def project_onto_reference(traj_xs, traj_ys, ref_xs, ref_ys, ref_s):
    n = len(traj_xs)
    progress = np.zeros(n)
    for i in range(n):
        dists = np.sqrt((ref_xs - traj_xs[i])**2 + (ref_ys - traj_ys[i])**2)
        progress[i] = ref_s[np.argmin(dists)]
    for i in range(1, n):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]
    return progress


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot steering vs progress (%)")
    parser.add_argument("--map_xodr", required=True, help="Path to .xodr file")
    parser.add_argument("--segment", required=True, help="Path to scenario_segment.json")
    parser.add_argument("--rw_traj", required=True, help="Real-world trajectory .txt (UTM)")
    parser.add_argument("--rw_steering", required=True, help="Real-world steering .txt")
    parser.add_argument("--sim_traj", required=True, help="Simulated trajectory .json (CARLA)")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    # --- Setup ---
    tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset = setup_transforms(args.map_xodr)

    with open(args.segment, "r") as f:
        segment = json.load(f)
    ref_xs = np.array(segment["reference_path_carla_x"])
    ref_ys = np.array(segment["reference_path_carla_y"])
    ref_s = np.array(segment["reference_path_arc_length"])
    seg_start = segment["scenario_start_m"]
    seg_end = segment["scenario_end_m"]
    seg_length = seg_end - seg_start

    # --- Real-world ---
    print("Loading real-world...")
    traj_ts, utm_x, utm_y = load_real_trajectory(args.rw_traj)
    carla_xs, carla_ys = [], []
    for ux, uy in zip(utm_x, utm_y):
        cx, cy = utm_to_carla(ux, uy, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset)
        carla_xs.append(cx)
        carla_ys.append(cy)
    carla_xs, carla_ys = np.array(carla_xs), np.array(carla_ys)

    # Interpolate steering onto trajectory timestamps (nearest match)
    steer_ts, steer_vals = load_real_steering(args.rw_steering)
    # For each trajectory timestamp, find nearest steering timestamp
    rw_steer = np.zeros(len(traj_ts))
    for i, t in enumerate(traj_ts):
        idx = np.argmin(np.abs(steer_ts - t))
        rw_steer[i] = steer_vals[idx]

    rw_progress = project_onto_reference(carla_xs, carla_ys, ref_xs, ref_ys, ref_s)
    rw_mask = (rw_progress >= seg_start) & (rw_progress <= seg_end)
    rw_pct = (rw_progress[rw_mask] - seg_start) / seg_length * 100
    rw_steer_seg = rw_steer[rw_mask]
    print(f"  {np.sum(rw_mask)} pts in segment, steering [{rw_steer_seg.min():.4f}, {rw_steer_seg.max():.4f}]")

    # --- Simulated ---
    print("Loading simulated...")
    sim_x, sim_y, sim_steer = load_sim_trajectory(args.sim_traj)
    sim_progress = project_onto_reference(sim_x, sim_y, ref_xs, ref_ys, ref_s)
    sim_mask = (sim_progress >= seg_start) & (sim_progress <= seg_end)
    sim_pct = (sim_progress[sim_mask] - seg_start) / seg_length * 100
    sim_steer_seg = sim_steer[sim_mask]
    print(f"  {np.sum(sim_mask)} pts in segment, steering [{sim_steer_seg.min():.4f}, {sim_steer_seg.max():.4f}]")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(rw_pct, rw_steer_seg, color="#1E88E5", linewidth=1.2, alpha=0.8, label="Real World")
    ax.plot(sim_pct, sim_steer_seg, color="#E53935", linewidth=1.0, alpha=0.7, label="Simulation")

    ax.axhline(y=0, color='grey', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Progress (%)", fontsize=12)
    ax.set_ylabel("Steering", fontsize=12)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    title = args.title or "Steering vs Progress"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()