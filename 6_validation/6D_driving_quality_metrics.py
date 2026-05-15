#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6D_driving_quality_metrics.py

Drive quality metrics for successful runs:
  1. Min-Frechet distance (sim run vs closest real-world run)
  2. Out-of-corridor: % of sim points outside real-world envelope + mean excess
  3. Steering jitter (std of d_steering / dt)

All runs are trimmed to the scenario segment before comparison.

Expected file layout:
    <rw_dir>/
        trajectory<N>.csv              (UTM x,y + timestamp; one per real-world run)
                                       Header: timestamp,x,y,z,yaw
        steering_cmd_<N>.txt           (optional; DAVE-2 cmds with timestamps)
        scenario_segment.json          (precomputed; or rebuilt via --recompute_segment)

    <sim_dir>/
        <prefix>_run<N>_trajectory.json   (sim runs; <prefix> becomes the "method")

Usage (default: use precomputed segment):
    python 6D_driving_quality_metrics.py \
        --map_xodr data/processed_dataset/reference_bag/maps/map.xodr \
        --segment data/data_for_validation/real_world_trajectories/scenario_segment.json \
        --rw_dir data/data_for_validation/real_world_trajectories \
        --sim_dirs GS=data/data_for_validation/GS_trajectories

Usage (recompute segment from the real-world CSVs in --rw_dir, save it as
<rw_dir>/scenario_segment.json, then run metrics on it):
    python 6D_driving_quality_metrics.py \
        --map_xodr data/processed_dataset/reference_bag/maps/map.xodr \
        --rw_dir data/data_for_validation/real_world_trajectories \
        --sim_dirs GS=data/data_for_validation/GS_trajectories \
        --recompute_segment

When --recompute_segment is passed, --segment is ignored (if provided) and the
resulting scenario_segment.json is written to <rw_dir>/scenario_segment.json,
overwriting any existing file there.
"""

import os
import re
import csv
import json
import argparse
import numpy as np
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

    return tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset


def utm_to_carla(utm_x, utm_y, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset):
    lon, lat = tf_utm_to_wgs.transform(utm_x, utm_y)
    proj_x, proj_y = tf_wgs_to_proj.transform(lon, lat)
    carla_x = proj_x + xodr_offset[0]
    carla_y = -(proj_y + xodr_offset[1])
    return carla_x, carla_y


def utm_array_to_carla(uxs, uys, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset):
    cxs, cys = [], []
    for ux, uy in zip(uxs, uys):
        cx, cy = utm_to_carla(ux, uy, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset)
        cxs.append(cx)
        cys.append(cy)
    return np.array(cxs), np.array(cys)


# =============================================================================
# LOADERS
# =============================================================================

def load_real_trajectory_utm(path):
    """
    Load a real-world trajectory from CSV.

    Expected header:
        timestamp,x,y,z,yaw

    where x,y are UTM coordinates (EPSG:25832) and timestamp is Unix epoch
    in seconds. The z and yaw columns are present but not used here.
    """
    xs, ys, timestamps = [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None \
                or "timestamp" not in reader.fieldnames \
                or "x" not in reader.fieldnames \
                or "y" not in reader.fieldnames:
            raise RuntimeError(
                f"Unexpected CSV header in {path}. "
                f"Got: {reader.fieldnames}. "
                f"Expected at least: timestamp, x, y."
            )
        for row in reader:
            try:
                timestamps.append(float(row["timestamp"]))
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
            except (KeyError, ValueError):
                continue
    return np.array(xs), np.array(ys), np.array(timestamps)


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
        steering = None
    return xs, ys, steering


# =============================================================================
# REFERENCE PATH & PROJECTION
# =============================================================================

def project_onto_reference(traj_xs, traj_ys, ref_xs, ref_ys, ref_s):
    n = len(traj_xs)
    progress = np.zeros(n)
    lateral_dist = np.zeros(n)
    closest_indices = np.zeros(n, dtype=int)

    for i in range(n):
        dists = np.sqrt((ref_xs - traj_xs[i])**2 + (ref_ys - traj_ys[i])**2)
        idx = np.argmin(dists)
        lateral_dist[i] = dists[idx]
        progress[i] = ref_s[idx]
        closest_indices[i] = idx

    for i in range(1, n):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]

    return progress, lateral_dist, closest_indices


def compute_signed_lateral(traj_xs, traj_ys, ref_xs, ref_ys, closest_indices):
    n = len(traj_xs)
    signed_lat = np.zeros(n)

    for i in range(n):
        idx = closest_indices[i]
        if idx < len(ref_xs) - 1:
            tx = ref_xs[idx + 1] - ref_xs[idx]
            ty = ref_ys[idx + 1] - ref_ys[idx]
        else:
            tx = ref_xs[idx] - ref_xs[idx - 1]
            ty = ref_ys[idx] - ref_ys[idx - 1]

        dx = traj_xs[i] - ref_xs[idx]
        dy = traj_ys[i] - ref_ys[idx]

        cross = tx * dy - ty * dx
        norm = np.sqrt(tx**2 + ty**2)
        if norm > 1e-9:
            signed_lat[i] = cross / norm
        else:
            signed_lat[i] = np.sqrt(dx**2 + dy**2)

    return signed_lat


def resample_at_progress(traj_xs, traj_ys, progress, target_progress):
    mask = np.diff(progress, prepend=-1) > 0
    p_clean = progress[mask]
    x_clean = traj_xs[mask]
    y_clean = traj_ys[mask]

    valid = (target_progress >= p_clean[0]) & (target_progress <= p_clean[-1])
    target_clipped = target_progress[valid]

    xs = np.interp(target_clipped, p_clean, x_clean)
    ys = np.interp(target_clipped, p_clean, y_clean)

    return xs, ys, valid


# =============================================================================
# SCENARIO SEGMENT (optional, built from real-world trajectories)
# =============================================================================

def compute_arc_length(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def build_reference_path(xs, ys, spacing=0.5):
    """
    Resample a trajectory uniformly along its arc length so the reference
    path has one point every `spacing` meters.
    """
    s = compute_arc_length(xs, ys)
    total_length = s[-1]
    n_points = max(int(total_length / spacing) + 1, 2)
    s_uniform = np.linspace(0, total_length, n_points)
    ref_xs = np.interp(s_uniform, s, xs)
    ref_ys = np.interp(s_uniform, s, ys)
    return ref_xs, ref_ys, s_uniform


def compute_scenario_segment(rw_carla, rw_dir, spacing=0.5):
    """
    Build a scenario_segment.json from the real-world trajectories already
    loaded into rw_carla = {run_id: (carla_x, carla_y), ...}.

    Logic:
      1. Pick the longest real-world run as the reference path.
      2. Resample it at uniform `spacing` along arc length.
      3. Project every run onto that reference and collect start/end progress.
      4. The scenario segment is the intersection:
            start = max of every run's first projected progress
            end   = min of every run's last  projected progress
      5. Save to <rw_dir>/scenario_segment.json (overwriting if present).

    Returns the segment dict.
    """
    print("\n" + "=" * 70)
    print("  RECOMPUTING SCENARIO SEGMENT")
    print("=" * 70)

    if not rw_carla:
        raise RuntimeError("Cannot recompute scenario segment: no real-world "
                           "trajectories loaded.")

    # 1. Find longest run
    longest_run = None
    longest_length = 0.0
    for run, (cx, cy) in rw_carla.items():
        length = compute_arc_length(cx, cy)[-1]
        if length > longest_length:
            longest_length = length
            longest_run = run

    print(f"  Reference run: trajectory{longest_run}.csv "
          f"(length={longest_length:.1f} m)")

    # 2. Build reference path
    ref_cx, ref_cy = rw_carla[longest_run]
    ref_xs, ref_ys, ref_s = build_reference_path(ref_cx, ref_cy, spacing=spacing)
    print(f"  Reference path: {len(ref_xs)} points, spacing={spacing} m")

    # 3. Project every run onto it
    start_progresses = []
    end_progresses = []
    for run, (cx, cy) in sorted(rw_carla.items()):
        progress, lat_dist, _ = project_onto_reference(
            cx, cy, ref_xs, ref_ys, ref_s
        )
        start_progresses.append(progress[0])
        end_progresses.append(progress[-1])
        print(f"    run{run}: [{progress[0]:.1f} m -> {progress[-1]:.1f} m] "
              f"(max lateral {lat_dist.max():.2f} m)")

    # 4. Segment = intersection
    scenario_start = float(max(start_progresses))
    scenario_end = float(min(end_progresses))
    scenario_length = scenario_end - scenario_start

    if scenario_length <= 0:
        raise RuntimeError(
            f"Computed scenario length is non-positive "
            f"(start={scenario_start:.1f}, end={scenario_end:.1f}). "
            f"The real-world trajectories don't share a common segment."
        )

    print(f"\n  Scenario segment:")
    print(f"    start  = {scenario_start:.1f} m")
    print(f"    end    = {scenario_end:.1f} m")
    print(f"    length = {scenario_length:.1f} m")

    # 5. Save
    segment = {
        "reference_run": f"trajectory{longest_run}",
        "reference_length_m": float(longest_length),
        "scenario_start_m": scenario_start,
        "scenario_end_m": scenario_end,
        "scenario_length_m": float(scenario_length),
        "reference_path_carla_x": ref_xs.tolist(),
        "reference_path_carla_y": ref_ys.tolist(),
        "reference_path_arc_length": ref_s.tolist(),
        "reference_path_spacing_m": float(spacing),
    }

    output_path = os.path.join(rw_dir, "scenario_segment.json")
    with open(output_path, "w") as f:
        json.dump(segment, f, indent=2)
    print(f"  Saved: {output_path}")

    return segment


# =============================================================================
# FRECHET DISTANCE (discrete)
# =============================================================================

def discrete_frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    ca = np.full((n, m), -1.0)

    for i in range(n):
        for j in range(m):
            d = np.sqrt((P[i, 0] - Q[j, 0])**2 + (P[i, 1] - Q[j, 1])**2)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)

    return ca[n - 1, m - 1]


def downsample_trajectory(xs, ys, max_points=500):
    n = len(xs)
    if n <= max_points:
        return xs, ys
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return xs[indices], ys[indices]


# =============================================================================
# CORRIDOR (from real-world runs)
# =============================================================================

def compute_corridor_violations(sim_signed_lateral, corridor_min, corridor_max):
    excess = np.zeros_like(sim_signed_lateral)
    below = sim_signed_lateral < corridor_min
    above = sim_signed_lateral > corridor_max
    excess[below] = corridor_min[below] - sim_signed_lateral[below]
    excess[above] = sim_signed_lateral[above] - corridor_max[above]

    n_out = np.sum(below | above)
    violation_rate = n_out / len(sim_signed_lateral)
    mean_excess = np.mean(excess)
    mean_excess_when_out = np.mean(excess[below | above]) if n_out > 0 else 0.0

    return violation_rate, mean_excess, mean_excess_when_out


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Drive quality metrics")
    parser.add_argument("--map_xodr", required=True, help="Path to .xodr file")
    parser.add_argument("--segment", required=False, default=None,
                        help="Path to scenario_segment.json (precomputed). "
                             "Required unless --recompute_segment is set.")
    parser.add_argument("--recompute_segment", action="store_true",
                        help="Recompute the scenario segment from the real-world "
                             "CSVs in --rw_dir and save it to "
                             "<rw_dir>/scenario_segment.json. When set, --segment "
                             "is ignored. Use this if you replaced or added "
                             "trajectory<N>.csv files and need a fresh segment.")
    parser.add_argument("--rw_dir", required=True,
                        help="Directory with real-world trajectory<N>.csv files")
    parser.add_argument("--sim_dirs", nargs="+", required=True,
                        help="Sim trajectory dirs as label=path "
                             "(e.g. GS=/path/to/sim/dir)")
    parser.add_argument("--completion_threshold", type=float, default=0.95,
                        help="Min completion to be considered successful (default: 0.95)")
    parser.add_argument("--frechet_max_pts", type=int, default=500,
                        help="Max points for Frechet computation (default: 500)")
    args = parser.parse_args()

    if not args.recompute_segment and not args.segment:
        parser.error("Either --segment <path> or --recompute_segment must be given.")

    # --- Setup ---
    tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset = setup_transforms(args.map_xodr)

    # --- Load ALL real-world trajectories ---
    print("=" * 70)
    print("  LOADING REAL-WORLD TRAJECTORIES")
    print("=" * 70)

    # rw_carla: run_id -> (carla_x, carla_y)
    rw_carla = {}
    # rw_steering: run_id -> (steer_vals, steer_ts, steer_cx, steer_cy)
    rw_steering = {}

    # Match: trajectory<N>.csv  (anchored: must start with "trajectory")
    rw_pattern = re.compile(r'^trajectory(\d+)\.csv$')

    for fname in sorted(os.listdir(args.rw_dir)):
        match = rw_pattern.match(fname)
        if not match:
            continue
        run = int(match.group(1))
        path = os.path.join(args.rw_dir, fname)

        utm_x, utm_y, traj_ts = load_real_trajectory_utm(path)
        carla_x, carla_y = utm_array_to_carla(
            utm_x, utm_y, tf_utm_to_wgs, tf_wgs_to_proj, xodr_offset
        )
        rw_carla[run] = (carla_x, carla_y)

        # Load matching steering (prefer cmd, fallback to raw)
        steer_path = os.path.join(args.rw_dir, f"steering_cmd_{run}.txt")
        if not os.path.exists(steer_path):
            steer_path = os.path.join(args.rw_dir, f"steering_{run}.txt")

        if os.path.exists(steer_path):
            steer_ts, steer_vals = load_real_steering(steer_path)
            valid_mask = (steer_ts >= traj_ts[0]) & (steer_ts <= traj_ts[-1])
            steer_ts_valid = steer_ts[valid_mask]
            steer_vals_valid = steer_vals[valid_mask]
            cx_at_steer = np.interp(steer_ts_valid, traj_ts, carla_x)
            cy_at_steer = np.interp(steer_ts_valid, traj_ts, carla_y)
            rw_steering[run] = (steer_vals_valid, steer_ts_valid,
                                 cx_at_steer, cy_at_steer)
            print(f"  {fname}: {len(carla_x)} pts + "
                  f"{len(steer_vals_valid)} steering cmds "
                  f"({os.path.basename(steer_path)})")
        else:
            print(f"  {fname}: {len(carla_x)} pts (no steering file)")

    if not rw_carla:
        raise RuntimeError(
            f"No trajectory<N>.csv files found in {args.rw_dir}. "
            f"Expected files like trajectory1.csv, trajectory2.csv, ..."
        )

    # --- Load or recompute scenario segment ---
    if args.recompute_segment:
        if args.segment:
            print(f"\n[INFO] --recompute_segment set: ignoring "
                  f"--segment {args.segment}")
        segment = compute_scenario_segment(rw_carla, args.rw_dir, spacing=0.5)
    else:
        print(f"\n[INFO] Loading precomputed segment: {args.segment}")
        with open(args.segment, "r") as f:
            segment = json.load(f)

    ref_xs = np.array(segment["reference_path_carla_x"])
    ref_ys = np.array(segment["reference_path_carla_y"])
    ref_s = np.array(segment["reference_path_arc_length"])
    seg_start = segment["scenario_start_m"]
    seg_end = segment["scenario_end_m"]
    seg_length = segment["scenario_length_m"]

    corridor_progress = np.arange(seg_start, seg_end, 0.5)

    # --- Load ALL simulated trajectories ---
    print("\n" + "=" * 70)
    print("  LOADING SIMULATED TRAJECTORIES")
    print("=" * 70)

    # sim_carla: (method, run) -> (carla_x, carla_y)
    sim_carla = {}
    sim_steering = {}

    # Match: <prefix>_run<N>_trajectory.json  (prefix becomes the "method")
    sim_pattern = re.compile(r'^([A-Za-z][A-Za-z0-9_]*)_run(\d+)_trajectory\.json$')

    for entry in args.sim_dirs:
        if "=" not in entry:
            print(f"  [WARN] Ignored --sim_dirs entry without '=': {entry}")
            continue
        dir_label, dir_path = entry.split("=", 1)
        if not os.path.isdir(dir_path):
            print(f"  [WARN] Not a directory: {dir_path}")
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith("_trajectory.json"):
                continue
            match = sim_pattern.match(fname)
            if not match:
                continue
            method = match.group(1)
            run = int(match.group(2))

            path = os.path.join(dir_path, fname)
            carla_x, carla_y, steer = load_sim_trajectory(path)
            sim_carla[(method, run)] = (carla_x, carla_y)
            if steer is not None:
                sim_steering[(method, run)] = steer
            print(f"  {dir_label}/{fname}: {len(carla_x)} pts "
                  f"(method={method}, run={run})")

    # --- Filter to successful runs only ---
    print("\n" + "=" * 70)
    print("  FILTERING TO SUCCESSFUL RUNS (completion >= {:.0f}%)".format(
        args.completion_threshold * 100))
    print("=" * 70)

    def check_completion(carla_x, carla_y):
        progress, _, _ = project_onto_reference(
            carla_x, carla_y, ref_xs, ref_ys, ref_s
        )
        completion = max(0, min(1.0, (progress[-1] - seg_start) / seg_length))
        return completion

    successful_sim = {}
    for key, (cx, cy) in sim_carla.items():
        comp = check_completion(cx, cy)
        if comp >= args.completion_threshold:
            successful_sim[key] = (cx, cy)
            print(f"  OK   {key[0]}_run{key[1]}: {comp*100:.1f}%")
        else:
            print(f"  FAIL {key[0]}_run{key[1]}: {comp*100:.1f}% -- SKIPPED")

    # --- Build corridor (from ALL real-world runs) ---
    print("\n" + "=" * 70)
    print("  BUILDING CORRIDOR & COMPUTING METRICS")
    print("=" * 70)

    rw_signed_lats = []
    rw_trimmed_curves = []  # for Frechet

    for run, (cx, cy) in sorted(rw_carla.items()):
        progress, lat, closest_idx = project_onto_reference(
            cx, cy, ref_xs, ref_ys, ref_s
        )
        signed_lat = compute_signed_lateral(
            cx, cy, ref_xs, ref_ys, closest_idx
        )

        mask = np.diff(progress, prepend=-1) > 0
        p_clean = progress[mask]
        sl_clean = signed_lat[mask]
        valid_cp = ((corridor_progress >= p_clean[0]) &
                    (corridor_progress <= p_clean[-1]))
        sl_resampled = np.interp(corridor_progress[valid_cp], p_clean, sl_clean)

        sl_full = np.full(len(corridor_progress), np.nan)
        sl_full[valid_cp] = sl_resampled
        rw_signed_lats.append(sl_full)

        print(f"  RW run{run}: signed lat range "
              f"[{np.nanmin(sl_full):.3f}, {np.nanmax(sl_full):.3f}] m")

        # Trim for Frechet
        in_seg = (progress >= seg_start) & (progress <= seg_end)
        rw_trimmed_curves.append(
            (run, np.column_stack([cx[in_seg], cy[in_seg]]))
        )

    rw_stack = np.stack(rw_signed_lats, axis=0)
    corridor_min = np.nanmin(rw_stack, axis=0)
    corridor_max = np.nanmax(rw_stack, axis=0)
    corridor_valid = ~np.isnan(corridor_min)

    corridor_width = corridor_max[corridor_valid] - corridor_min[corridor_valid]
    print(f"  Corridor: mean width={np.mean(corridor_width):.3f} m, "
          f"max width={np.max(corridor_width):.3f} m")

    # --- Evaluate each successful sim run ---
    all_results = []

    print()
    for key, (cx, cy) in sorted(successful_sim.items()):
        method, run = key

        progress, lat, closest_idx = project_onto_reference(
            cx, cy, ref_xs, ref_ys, ref_s
        )
        in_seg = (progress >= seg_start) & (progress <= seg_end)
        sim_cx, sim_cy = cx[in_seg], cy[in_seg]

        if len(sim_cx) < 10:
            print(f"  {method}_run{run}: too few points in segment -- skipping")
            continue

        # Frechet vs every real-world run
        sim_curve = np.column_stack([sim_cx, sim_cy])
        frechet_dists = []
        for rw_run, rw_curve in rw_trimmed_curves:
            fd = discrete_frechet_distance(sim_curve, rw_curve)
            frechet_dists.append((rw_run, fd))
        min_frechet = min(frechet_dists, key=lambda x: x[1])

        # Corridor
        signed_lat = compute_signed_lateral(
            cx, cy, ref_xs, ref_ys, closest_idx
        )
        mask = np.diff(progress, prepend=-1) > 0
        p_clean = progress[mask]
        sl_clean = signed_lat[mask]
        valid_cp = ((corridor_progress >= p_clean[0]) &
                    (corridor_progress <= p_clean[-1]))

        if np.sum(valid_cp & corridor_valid) < 10:
            print(f"  {method}_run{run}: insufficient corridor overlap -- skipping")
            continue

        sl_sim = np.interp(corridor_progress[valid_cp], p_clean, sl_clean)
        eval_mask = corridor_valid[valid_cp]
        sl_eval = sl_sim[eval_mask]
        corr_min_eval = corridor_min[valid_cp][eval_mask]
        corr_max_eval = corridor_max[valid_cp][eval_mask]

        viol_rate, mean_excess, mean_excess_out = compute_corridor_violations(
            sl_eval, corr_min_eval, corr_max_eval
        )

        result = {
            "method": method,
            "run": run,
            "min_frechet_m": float(min_frechet[1]),
            "min_frechet_vs": f"run{min_frechet[0]}",
            "corridor_violation_pct": float(viol_rate * 100),
            "mean_excess_m": float(mean_excess),
            "mean_excess_when_out_m": float(mean_excess_out),
        }
        all_results.append(result)

        print(f"  {method}_run{run}: "
              f"Frechet={min_frechet[1]:.3f} m (vs run{min_frechet[0]}), "
              f"out={viol_rate*100:.1f}%, "
              f"mean_excess={mean_excess:.3f} m, "
              f"excess_when_out={mean_excess_out:.3f} m")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  DRIVE QUALITY SUMMARY (successful runs only)")
    print("=" * 70)

    print(f"\n  {'Method':<12} {'Run':>3} "
          f"{'Frechet(m)':>10} {'Out(%)':>8} "
          f"{'MeanExcess(m)':>14} {'ExcessOut(m)':>12}")
    print(f"  {'-'*12} {'-'*3} {'-'*10} {'-'*8} {'-'*14} {'-'*12}")

    for r in all_results:
        print(f"  {r['method']:<12} {r['run']:>3} "
              f"{r['min_frechet_m']:>10.3f} {r['corridor_violation_pct']:>8.1f} "
              f"{r['mean_excess_m']:>14.3f} {r['mean_excess_when_out_m']:>12.3f}")

    # --- Averages per method ---
    print(f"\n  AVERAGES:")
    print(f"  {'Method':<12} {'Runs':>4} "
          f"{'Frechet(m)':>10} {'Out(%)':>8} {'MeanExcess(m)':>14}")
    print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*8} {'-'*14}")

    groups = {}
    for r in all_results:
        groups.setdefault(r["method"], []).append(r)

    for method, runs in sorted(groups.items()):
        n = len(runs)
        avg_f = np.mean([r["min_frechet_m"] for r in runs])
        avg_o = np.mean([r["corridor_violation_pct"] for r in runs])
        avg_e = np.mean([r["mean_excess_m"] for r in runs])
        print(f"  {method:<12} {n:>4} "
              f"{avg_f:>10.3f} {avg_o:>8.1f} {avg_e:>14.3f}")

    # =====================================================================
    #  STEERING JITTER
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEERING JITTER (std of d_steering / dt, within segment)")
    print("=" * 70)

    SIM_DT = 3.0 / 30.0
    RW_MAX_DT = 0.25

    def compute_jitter(steering_arr, progress_arr, seg_start, seg_end,
                       timestamps=None, dt_fixed=None,
                       max_dt_gap=None, subsample=1):
        mask = (progress_arr >= seg_start) & (progress_arr <= seg_end)
        steer_seg = steering_arr[mask]
        if timestamps is not None:
            ts_seg = timestamps[mask]
        if subsample > 1:
            steer_seg = steer_seg[::subsample]
            if timestamps is not None:
                ts_seg = ts_seg[::subsample]
        if len(steer_seg) < 2:
            return None, None
        delta_steer = np.diff(steer_seg)
        if timestamps is not None:
            dt = np.diff(ts_seg)
            if max_dt_gap is not None:
                valid = (dt > 0.05) & (dt <= max_dt_gap)
                delta_steer = delta_steer[valid]
                dt = dt[valid]
            if len(delta_steer) < 2:
                return None, None
            rate = delta_steer / dt
        elif dt_fixed is not None:
            rate = delta_steer / dt_fixed
        else:
            rate = delta_steer
        return float(np.std(rate)), float(np.max(np.abs(rate)))

    jitter_results = []

    # Real-world jitter
    for run, (cx, cy) in sorted(rw_carla.items()):
        if run not in rw_steering:
            continue
        steer_vals, steer_ts, steer_cx, steer_cy = rw_steering[run]
        progress, _, _ = project_onto_reference(
            steer_cx, steer_cy, ref_xs, ref_ys, ref_s
        )
        j, mj = compute_jitter(steer_vals, progress, seg_start, seg_end,
                                timestamps=steer_ts, max_dt_gap=RW_MAX_DT)
        if j is not None:
            jitter_results.append({
                "method": "real_world",
                "run": run,
                "jitter": j,
                "max_jitter": mj,
            })
            print(f"  RW run{run}: jitter={j:.6f} /s, max={mj:.6f} /s")

    # Simulated jitter (successful only)
    for key in sorted(successful_sim.keys()):
        method, run = key
        if key not in sim_steering:
            continue
        cx, cy = successful_sim[key]
        steer = sim_steering[key]
        progress, _, _ = project_onto_reference(cx, cy, ref_xs, ref_ys, ref_s)
        j, mj = compute_jitter(steer, progress, seg_start, seg_end,
                                dt_fixed=SIM_DT, subsample=3)
        if j is not None:
            jitter_results.append({
                "method": method,
                "run": run,
                "jitter": j,
                "max_jitter": mj,
            })
            print(f"  {method} run{run}: jitter={j:.6f} /s, max={mj:.6f} /s")

    # Jitter summary per method
    print(f"\n  JITTER SUMMARY (steering rate: units/second):")
    print(f"  {'Method':<12} {'Runs':>4} {'Avg Jitter':>12} {'Max Jitter':>12}")
    print(f"  {'-'*12} {'-'*4} {'-'*12} {'-'*12}")

    jitter_groups = {}
    for r in jitter_results:
        jitter_groups.setdefault(r["method"], []).append(r)

    for method, runs in sorted(jitter_groups.items()):
        n = len(runs)
        avg_j = np.mean([r["jitter"] for r in runs])
        max_j = np.max([r["max_jitter"] for r in runs])
        print(f"  {method:<12} {n:>4} {avg_j:>12.6f} {max_j:>12.6f}")

    # =====================================================================
    #  SAVE RESULTS TO JSON
    # =====================================================================
    output_path = os.path.join(args.rw_dir, "drive_quality_results.json")
    output_data = {
        "segment": {
            "start_m": seg_start,
            "end_m": seg_end,
            "length_m": seg_length,
        },
        "drive_quality": all_results,
        "drive_quality_averages": [
            {
                "method": method,
                "runs": len(runs),
                "avg_frechet_m": round(float(np.mean(
                    [r["min_frechet_m"] for r in runs])), 3),
                "avg_out_pct": round(float(np.mean(
                    [r["corridor_violation_pct"] for r in runs])), 1),
                "avg_mean_excess_m": round(float(np.mean(
                    [r["mean_excess_m"] for r in runs])), 3),
            }
            for method, runs in sorted(groups.items())
        ],
        "steering_jitter": jitter_results,
        "steering_jitter_averages": [
            {
                "method": method,
                "runs": len(runs),
                "avg_jitter": round(float(np.mean(
                    [r["jitter"] for r in runs])), 6),
                "max_jitter": round(float(np.max(
                    [r["max_jitter"] for r in runs])), 6),
            }
            for method, runs in sorted(jitter_groups.items())
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved: {output_path}")
    print()


if __name__ == "__main__":
    main()