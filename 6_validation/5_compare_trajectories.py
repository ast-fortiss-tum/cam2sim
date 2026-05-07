#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trajectories_osm.py

Plots simulated (CARLA coords) and real-world (UTM coords) trajectories
on an OpenStreetMap background.

Both are converted to EPSG:3857 (Web Mercator) for plotting with contextily.

Usage:
    python plot_trajectories_osm.py \
        --sim_trajectory  path/to/trajectory.json \
        --real_trajectory path/to/positions.txt \
        --map_xodr        path/to/map.xodr \
        --output          trajectory_comparison.pdf
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer

# =========================================================
# XODR PARSING (minimal — just what we need)
# =========================================================

def get_xodr_projection_params(xodr_data):
    """Extract geoReference and offset from OpenDRIVE XML."""
    import re
    geo_match = re.search(r'<geoReference>\s*<!\[CDATA\[(.*?)\]\]>', xodr_data, re.DOTALL)
    geo_ref = geo_match.group(1).strip() if geo_match else "+proj=tmerc"

    offset_match = re.search(r'<offset\s+x="([^"]+)"\s+y="([^"]+)"', xodr_data)
    if offset_match:
        offset = (float(offset_match.group(1)), float(offset_match.group(2)))
    else:
        offset = (0.0, 0.0)

    return {"geo_reference": geo_ref, "offset": offset}


# =========================================================
# COORDINATE CONVERSIONS
# =========================================================

def setup_carla_to_wgs84(xodr_path):
    """
    Set up the CARLA → WGS84 (lat/lon) transform chain.

    CARLA → flip Y → subtract xodr offset → inverse TM projection → WGS84
    """
    with open(xodr_path, "r") as f:
        xodr_data = f.read()

    params = get_xodr_projection_params(xodr_data)
    xodr_offset = params["offset"]
    proj_string = params["geo_reference"].strip()
    if proj_string == "+proj=tmerc":
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

    # TM projection coords → WGS84
    transformer = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
    return transformer, xodr_offset


def carla_to_latlon(carla_x, carla_y, transformer, xodr_offset):
    """CARLA (x, y) → WGS84 (lat, lon)."""
    # Reverse CARLA: flip Y back, remove offset
    local_x = carla_x
    local_y = -carla_y  # un-flip Y
    proj_x = local_x - xodr_offset[0]
    proj_y = local_y - xodr_offset[1]
    lon, lat = transformer.transform(proj_x, proj_y)
    return lat, lon


def utm_to_latlon(utm_x, utm_y, utm_crs="EPSG:25832"):
    """UTM (easting, northing) → WGS84 (lat, lon)."""
    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(utm_x, utm_y)
    return lat, lon


def latlon_to_webmercator(lat, lon):
    """WGS84 (lat, lon) → EPSG:3857 (Web Mercator) for contextily plotting."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx, my = transformer.transform(lon, lat)
    return mx, my


# =========================================================
# LOADERS
# =========================================================

def load_sim_trajectory(path):
    """Load simulated trajectory JSON (CARLA coords)."""
    with open(path, "r") as f:
        data = json.load(f)
    xs = np.array([p["x"] for p in data])
    ys = np.array([p["y"] for p in data])
    return xs, ys


def load_real_trajectory(path):
    """Load real-world trajectory TXT (UTM coords)."""
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


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Plot sim + real trajectories on OSM")
    parser.add_argument("--sim_trajectory", required=True,
                        help="Path to simulated trajectory JSON (CARLA coords)")
    parser.add_argument("--real_trajectory", required=True,
                        help="Path to real-world positions TXT (UTM coords)")
    parser.add_argument("--map_xodr", required=True,
                        help="Path to the OpenDRIVE .xodr file for CARLA→WGS84 conversion")
    parser.add_argument("--utm_crs", default="EPSG:25832",
                        help="UTM CRS for real-world data (default: EPSG:25832)")
    parser.add_argument("--buffer", type=float, default=100,
                        help="Map zoom buffer in meters (default: 100)")
    parser.add_argument("--title", default="Trajectory Comparison: Simulation vs Real World",
                        help="Plot title")
    args = parser.parse_args()

    # --- Load trajectories ---
    print(f"Loading simulated trajectory: {args.sim_trajectory}")
    sim_x, sim_y = load_sim_trajectory(args.sim_trajectory)
    print(f"  {len(sim_x)} points, CARLA range: x=[{sim_x.min():.1f}, {sim_x.max():.1f}], "
          f"y=[{sim_y.min():.1f}, {sim_y.max():.1f}]")

    print(f"Loading real-world trajectory: {args.real_trajectory}")
    real_utm_x, real_utm_y = load_real_trajectory(args.real_trajectory)
    print(f"  {len(real_utm_x)} points, UTM range: E=[{real_utm_x.min():.1f}, {real_utm_x.max():.1f}], "
          f"N=[{real_utm_y.min():.1f}, {real_utm_y.max():.1f}]")

    # --- Convert simulated trajectory: CARLA → lat/lon → Web Mercator ---
    print("Converting simulated trajectory: CARLA → WGS84 → EPSG:3857...")
    carla_tf, xodr_offset = setup_carla_to_wgs84(args.map_xodr)

    sim_lats, sim_lons = [], []
    for x, y in zip(sim_x, sim_y):
        lat, lon = carla_to_latlon(x, y, carla_tf, xodr_offset)
        sim_lats.append(lat)
        sim_lons.append(lon)
    sim_lats, sim_lons = np.array(sim_lats), np.array(sim_lons)

    sim_mx, sim_my = latlon_to_webmercator(sim_lats, sim_lons)

    # --- Convert real-world trajectory: UTM → lat/lon → Web Mercator ---
    print(f"Converting real-world trajectory: {args.utm_crs} → WGS84 → EPSG:3857...")
    real_lats, real_lons = [], []
    utm_tf = Transformer.from_crs(args.utm_crs, "EPSG:4326", always_xy=True)
    for ux, uy in zip(real_utm_x, real_utm_y):
        lon, lat = utm_tf.transform(ux, uy)
        real_lats.append(lat)
        real_lons.append(lon)
    real_lats, real_lons = np.array(real_lats), np.array(real_lons)

    real_mx, real_my = latlon_to_webmercator(real_lats, real_lons)

    # --- Sanity check: print geographic overlap ---
    print(f"\n  Sim  lat range: [{sim_lats.min():.6f}, {sim_lats.max():.6f}]")
    print(f"  Real lat range: [{real_lats.min():.6f}, {real_lats.max():.6f}]")
    print(f"  Sim  lon range: [{sim_lons.min():.6f}, {sim_lons.max():.6f}]")
    print(f"  Real lon range: [{real_lons.min():.6f}, {real_lons.max():.6f}]")

    # --- Plot ---
    print("\nPlotting...")
    all_mx = np.concatenate([sim_mx, real_mx])
    all_my = np.concatenate([sim_my, real_my])

    # Buffer in Web Mercator (approximately meters at this latitude)
    buf = args.buffer
    xmin, xmax = all_mx.min() - buf, all_mx.max() + buf
    ymin, ymax = all_my.min() - buf, all_my.max() + buf

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # OSM basemap
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"  Warning: Could not load map tiles: {e}")

    # Plot trajectories
    ax.plot(real_mx, real_my, color="#1E88E5", linewidth=2.5, alpha=0.9,
            label="Real World (Ground Truth)", zorder=10)
    ax.plot(sim_mx, sim_my, color="#E53935", linewidth=2.0, alpha=0.8,
            linestyle="--", label="Simulation (GS)", zorder=11)

    # Start markers
    ax.plot(real_mx[0], real_my[0], marker='o', color="#1E88E5", markersize=10,
            markeredgecolor='black', markeredgewidth=1.5, zorder=20, label="Real start")
    ax.plot(sim_mx[0], sim_my[0], marker='o', color="#E53935", markersize=10,
            markeredgecolor='black', markeredgewidth=1.5, zorder=21, label="Sim start")

    # End markers
    ax.plot(real_mx[-1], real_my[-1], marker='s', color="#1E88E5", markersize=10,
            markeredgecolor='black', markeredgewidth=1.5, zorder=20, label="Real end")
    ax.plot(sim_mx[-1], sim_my[-1], marker='x', color="#E53935", markersize=12,
            markeredgewidth=3, zorder=21, label="Sim end")

    ax.set_title(args.title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()