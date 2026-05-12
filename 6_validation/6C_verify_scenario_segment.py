#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6B_verify_scenario_segment.py

Computes the scenario segment from multiple real-world repetitions of the
same route and plots everything on OpenStreetMap for visual verification.

PATCHED: real-world trajectories are now read from
    data/data_for_validation/real_world_trajectories/<N>_trajectory.csv

instead of data/raw_dataset/<bag_name>/trajectory.csv.

Inputs (all hardcoded under PROJECT_ROOT):

  - data/processed_dataset/<MAP_BAG_NAME>/maps/map.xodr
        OpenDRIVE map used to convert real UTM coordinates into CARLA
        coordinates (and to plot on OSM via WebMercator).

  - data/data_for_validation/real_world_trajectories/<N>_trajectory.csv
        For every real-world repetition (N = 1, 2, 3, ...).
        Produced by extract_trajectories_from_bags.py.
        Columns: timestamp, x, y, z, yaw  (x and y are in UTM EPSG:25832).

Outputs:

  - data/processed_dataset/<MAP_BAG_NAME>/scenario_segment/scenario_segment.json
  - data/processed_dataset/<MAP_BAG_NAME>/scenario_segment/scenario_segment.png

Run:
    python 6B_verify_scenario_segment.py
"""

import os
import re
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer


# =============================================================================
#  HARDCODED CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Bag used for the simulation reconstruction. The map.xodr lives here.
MAP_BAG_NAME = "reference_bag"

# Folder holding the N_trajectory.csv files of the real-world repetitions.
REAL_RUNS_DIR = (
    PROJECT_ROOT / "data" / "data_for_validation" / "real_world_trajectories"
)

# Filename pattern: <index>_trajectory.csv where index is an int.
REAL_RUN_PATTERN = re.compile(r"^(\d+)_trajectory\.csv$")

XODR_PATH = (
    PROJECT_ROOT / "data" / "processed_dataset" / MAP_BAG_NAME / "maps" / "map.xodr"
)

RESULTS_DIR = (
    PROJECT_ROOT / "data" / "processed_dataset" / MAP_BAG_NAME / "scenario_segment"
)

# Plot buffer (in WebMercator meters at this latitude).
PLOT_BUFFER_M = 100.0

# Spacing of the resampled reference path, in meters.
REFERENCE_SPACING_M = 0.5


# =============================================================================
#  XODR PARSING
# =============================================================================

def get_xodr_projection_params(xodr_data):
    geo_match = re.search(
        r"<geoReference>\s*<!\[CDATA\[(.*?)\]\]>",
        xodr_data,
        re.DOTALL,
    )
    geo_ref = geo_match.group(1).strip() if geo_match else "+proj=tmerc"

    offset_match = re.search(
        r'<offset\s+x="([^"]+)"\s+y="([^"]+)"',
        xodr_data,
    )
    if offset_match:
        offset = (float(offset_match.group(1)), float(offset_match.group(2)))
    else:
        offset = (0.0, 0.0)

    return {"geo_reference": geo_ref, "offset": offset}


# =============================================================================
#  COORDINATE TRANSFORMS
# =============================================================================

def setup_transforms(xodr_path):
    with open(xodr_path, "r") as f:
        xodr_data = f.read()

    params = get_xodr_projection_params(xodr_data)
    xodr_offset = params["offset"]
    proj_string = params["geo_reference"].strip()

    if proj_string == "+proj=tmerc":
        proj_string = (
            "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 "
            "+x_0=0 +y_0=0 +datum=WGS84"
        )

    return {
        "utm_to_wgs":  Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True),
        "wgs_to_proj": Transformer.from_crs("EPSG:4326", proj_string, always_xy=True),
        "proj_to_wgs": Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True),
        "wgs_to_merc": Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True),
        "xodr_offset": xodr_offset,
    }


def utm_to_carla(utm_x, utm_y, tfs):
    lon, lat = tfs["utm_to_wgs"].transform(utm_x, utm_y)
    proj_x, proj_y = tfs["wgs_to_proj"].transform(lon, lat)
    carla_x = proj_x + tfs["xodr_offset"][0]
    carla_y = -(proj_y + tfs["xodr_offset"][1])
    return carla_x, carla_y


def carla_to_webmercator(carla_x, carla_y, tfs):
    local_x = carla_x
    local_y = -carla_y
    proj_x = local_x - tfs["xodr_offset"][0]
    proj_y = local_y - tfs["xodr_offset"][1]
    lon, lat = tfs["proj_to_wgs"].transform(proj_x, proj_y)
    mx, my = tfs["wgs_to_merc"].transform(lon, lat)
    return mx, my


def carla_array_to_merc(cxs, cys, tfs):
    mxs, mys = [], []
    for x, y in zip(cxs, cys):
        mx, my = carla_to_webmercator(x, y, tfs)
        mxs.append(mx)
        mys.append(my)
    return np.array(mxs), np.array(mys)


# =============================================================================
#  TRAJECTORY LOADING
# =============================================================================

def load_real_trajectory_csv_to_carla(csv_path, tfs):
    utm_xs, utm_ys = [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "x" not in reader.fieldnames \
                or "y" not in reader.fieldnames:
            raise RuntimeError(
                f"Unexpected CSV header in {csv_path}. "
                f"Got: {reader.fieldnames}. "
                f"Expected columns: timestamp, x, y, z, yaw."
            )

        for row in reader:
            try:
                utm_xs.append(float(row["x"]))
                utm_ys.append(float(row["y"]))
            except (KeyError, ValueError):
                continue

    if not utm_xs:
        raise RuntimeError(f"No data points found in {csv_path}")

    carla_xs, carla_ys = [], []
    for ux, uy in zip(utm_xs, utm_ys):
        cx, cy = utm_to_carla(ux, uy, tfs)
        carla_xs.append(cx)
        carla_ys.append(cy)

    return np.array(carla_xs), np.array(carla_ys)


def discover_real_runs():
    """
    Scan REAL_RUNS_DIR for files matching <N>_trajectory.csv.
    Returns list of (run_label, path_to_csv), sorted by N.
    """
    found = []
    if not REAL_RUNS_DIR.exists():
        print(f"[ERROR] Real runs dir does not exist: {REAL_RUNS_DIR}")
        return found

    for entry in sorted(REAL_RUNS_DIR.iterdir()):
        if not entry.is_file():
            continue
        m = REAL_RUN_PATTERN.match(entry.name)
        if not m:
            continue
        idx = int(m.group(1))
        run_label = f"run{idx}"
        found.append((idx, run_label, entry))

    found.sort(key=lambda t: t[0])
    return [(label, path) for _, label, path in found]


# =============================================================================
#  REFERENCE PATH & PROJECTION
# =============================================================================

def compute_arc_length(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def build_reference_path(xs, ys, spacing=REFERENCE_SPACING_M):
    s = compute_arc_length(xs, ys)
    total_length = s[-1]
    n_points = max(int(total_length / spacing) + 1, 2)
    s_uniform = np.linspace(0, total_length, n_points)
    ref_xs = np.interp(s_uniform, s, xs)
    ref_ys = np.interp(s_uniform, s, ys)
    return ref_xs, ref_ys, s_uniform


def project_onto_reference(traj_xs, traj_ys, ref_xs, ref_ys, ref_s):
    n = len(traj_xs)
    progress = np.zeros(n)
    lateral_dist = np.zeros(n)
    for i in range(n):
        dists = np.sqrt(
            (ref_xs - traj_xs[i]) ** 2 + (ref_ys - traj_ys[i]) ** 2
        )
        idx = int(np.argmin(dists))
        lateral_dist[i] = dists[idx]
        progress[i] = ref_s[idx]
    for i in range(1, n):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]
    return progress, lateral_dist


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  SCENARIO SEGMENT VERIFICATION")
    print("=" * 70)
    print(f"[INFO] Project root:  {PROJECT_ROOT}")
    print(f"[INFO] Map bag:       {MAP_BAG_NAME}")
    print(f"[INFO] Real runs dir: {REAL_RUNS_DIR}")
    print(f"[INFO] XODR:          {XODR_PATH}")
    print(f"[INFO] Results dir:   {RESULTS_DIR}")
    print("=" * 70)

    if not XODR_PATH.exists():
        raise FileNotFoundError(f"XODR file not found: {XODR_PATH}")

    tfs = setup_transforms(XODR_PATH)

    real_runs = discover_real_runs()
    if not real_runs:
        raise RuntimeError(
            f"No <N>_trajectory.csv files found in {REAL_RUNS_DIR}. "
            f"Run extract_trajectories_from_bags.py first."
        )

    print(f"\n[INFO] Loading {len(real_runs)} real-world run(s)...")
    rw_trajs = {}
    for run_label, csv_path in real_runs:
        carla_xs, carla_ys = load_real_trajectory_csv_to_carla(csv_path, tfs)
        rw_trajs[run_label] = (carla_xs, carla_ys)
        arc = compute_arc_length(carla_xs, carla_ys)
        print(f"  {run_label}: {len(carla_xs)} pts, length={arc[-1]:.1f} m  "
              f"<- {csv_path.name}")

    longest_label = None
    longest_length = 0.0
    for label, (cx, cy) in rw_trajs.items():
        length = compute_arc_length(cx, cy)[-1]
        if length > longest_length:
            longest_length = length
            longest_label = label

    print(f"\n[INFO] Reference path: {longest_label} ({longest_length:.1f} m)")

    ref_cx, ref_cy = rw_trajs[longest_label]
    ref_xs, ref_ys, ref_s = build_reference_path(
        ref_cx, ref_cy, spacing=REFERENCE_SPACING_M
    )

    print("\n[INFO] Projecting runs onto reference...")
    start_progresses = []
    end_progresses = []
    for label, (cx, cy) in sorted(rw_trajs.items()):
        progress, lat_dist = project_onto_reference(
            cx, cy, ref_xs, ref_ys, ref_s
        )
        start_progresses.append(progress[0])
        end_progresses.append(progress[-1])
        print(f"  {label}: progress [{progress[0]:.1f} m -> "
              f"{progress[-1]:.1f} m], max lateral={lat_dist.max():.2f} m")

    scenario_start = float(max(start_progresses))
    scenario_end = float(min(end_progresses))
    scenario_length = scenario_end - scenario_start

    print(f"\n[INFO] SCENARIO SEGMENT:")
    print(f"       Start:  {scenario_start:.1f} m  (latest start across runs)")
    print(f"       End:    {scenario_end:.1f} m  (earliest end across runs)")
    print(f"       Length: {scenario_length:.1f} m")

    if scenario_length <= 0:
        print("[WARN] Scenario length is non-positive. "
              "Runs probably do not overlap on the reference path.")

    seg_start_idx = int(np.argmin(np.abs(ref_s - scenario_start)))
    seg_end_idx = int(np.argmin(np.abs(ref_s - scenario_end)))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "scenario_segment.json"

    segment_data = {
        "map_bag":               MAP_BAG_NAME,
        "real_runs":             list(rw_trajs.keys()),
        "reference_run":         longest_label,
        "reference_length_m":    float(longest_length),
        "scenario_start_m":      float(scenario_start),
        "scenario_end_m":        float(scenario_end),
        "scenario_length_m":     float(scenario_length),
        "reference_path_carla_x":  ref_xs.tolist(),
        "reference_path_carla_y":  ref_ys.tolist(),
        "reference_path_arc_length": ref_s.tolist(),
        "reference_path_spacing_m":  float(REFERENCE_SPACING_M),
    }

    with open(out_json, "w") as f:
        json.dump(segment_data, f, indent=2)

    print(f"\n[INFO] Saved: {out_json}")

    # ---- Plot ----
    print("[INFO] Converting to WebMercator for plotting...")

    ref_mx, ref_my = carla_array_to_merc(ref_xs, ref_ys, tfs)
    seg_mx = ref_mx[seg_start_idx:seg_end_idx + 1]
    seg_my = ref_my[seg_start_idx:seg_end_idx + 1]

    rw_merc = {}
    for label, (cx, cy) in rw_trajs.items():
        rw_merc[label] = carla_array_to_merc(cx, cy, tfs)

    all_mx = np.concatenate([v[0] for v in rw_merc.values()])
    all_my = np.concatenate([v[1] for v in rw_merc.values()])
    xmin, xmax = all_mx.min() - PLOT_BUFFER_M, all_mx.max() + PLOT_BUFFER_M
    ymin, ymax = all_my.min() - PLOT_BUFFER_M, all_my.max() + PLOT_BUFFER_M

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.CartoDB.Positron,
        )
    except Exception as e:
        print(f"[WARN] Could not load OSM tiles: {e}")

    ax.plot(
        seg_mx, seg_my,
        color="#4CAF50", linewidth=8, alpha=0.3,
        solid_capstyle="round", zorder=5,
        label=f"Scenario segment ({scenario_length:.0f} m)",
    )

    palette = ["#1E88E5", "#FB8C00", "#8E24AA", "#43A047", "#E53935"]
    line_styles = ["-", "--", ":", "-."]
    for i, (label, (mx, my)) in enumerate(sorted(rw_merc.items())):
        color = palette[i % len(palette)]
        ls = line_styles[i % len(line_styles)]
        ax.plot(
            mx, my,
            color=color, linewidth=2.0, alpha=0.85,
            linestyle=ls, zorder=10, label=label,
        )
        ax.plot(
            mx[0], my[0], marker="o", color=color, markersize=8,
            markeredgecolor="black", markeredgewidth=1, zorder=20,
        )
        ax.plot(
            mx[-1], my[-1], marker="s", color=color, markersize=8,
            markeredgecolor="black", markeredgewidth=1, zorder=20,
        )

    ax.plot(
        seg_mx[0], seg_my[0],
        marker="|", color="#4CAF50", markersize=20,
        markeredgewidth=4, zorder=25, label="Segment start",
    )
    ax.plot(
        seg_mx[-1], seg_my[-1],
        marker="|", color="#F44336", markersize=20,
        markeredgewidth=4, zorder=25, label="Segment end",
    )

    ax.set_title(
        f"Scenario segment from {len(rw_trajs)} real-world repetition(s)",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_axis_off()

    plt.tight_layout()

    out_png = RESULTS_DIR / "scenario_segment.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {out_png}")

    plt.show()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()