#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_drive_quality.py

Drive quality metrics for successful runs:
  1. Min-Fréchet distance (sim run vs closest real-world run, per scenario)
  2. Out-of-corridor: % of sim points outside real-world envelope + mean excess distance

All runs are trimmed to the scenario segment before comparison.

Usage:
    python metrics_drive_quality.py \
        --map_xodr maps/sunny_map/map.xodr \
        --segment /path/to/scenario_segment.json \
        --rw_dir /path/to/RW_trajectories \
        --sim_dirs \
            SD=/path/to/SD_trajectories \
            GS=/path/to/GS_trajectories
"""

import os
import re
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
    xs, ys, timestamps = [], [], []
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
    # Steering in raw DAVE-2 scale
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
    """Project trajectory onto reference path → (progress, lateral_distance, closest_idx)."""
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

    # Enforce monotonicity
    for i in range(1, n):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]

    return progress, lateral_dist, closest_indices


def compute_signed_lateral(traj_xs, traj_ys, ref_xs, ref_ys, closest_indices):
    """
    Compute signed lateral distance: positive = right of reference, negative = left.
    Uses the reference path tangent at the closest point.
    """
    n = len(traj_xs)
    signed_lat = np.zeros(n)

    for i in range(n):
        idx = closest_indices[i]
        # Tangent direction at closest point
        if idx < len(ref_xs) - 1:
            tx = ref_xs[idx + 1] - ref_xs[idx]
            ty = ref_ys[idx + 1] - ref_ys[idx]
        else:
            tx = ref_xs[idx] - ref_xs[idx - 1]
            ty = ref_ys[idx] - ref_ys[idx - 1]

        # Vector from reference to trajectory point
        dx = traj_xs[i] - ref_xs[idx]
        dy = traj_ys[i] - ref_ys[idx]

        # Cross product gives signed distance (positive = right)
        cross = tx * dy - ty * dx
        norm = np.sqrt(tx**2 + ty**2)
        if norm > 1e-9:
            signed_lat[i] = cross / norm
        else:
            signed_lat[i] = np.sqrt(dx**2 + dy**2)

    return signed_lat


def resample_at_progress(traj_xs, traj_ys, progress, target_progress):
    """Resample a trajectory at uniform progress values via interpolation."""
    # Remove duplicate progress values (keep last)
    mask = np.diff(progress, prepend=-1) > 0
    p_clean = progress[mask]
    x_clean = traj_xs[mask]
    y_clean = traj_ys[mask]

    # Clip target to available range
    valid = (target_progress >= p_clean[0]) & (target_progress <= p_clean[-1])
    target_clipped = target_progress[valid]

    xs = np.interp(target_clipped, p_clean, x_clean)
    ys = np.interp(target_clipped, p_clean, y_clean)

    return xs, ys, valid


# =============================================================================
# FRÉCHET DISTANCE (discrete)
# =============================================================================

def discrete_frechet_distance(P, Q):
    """
    Compute discrete Fréchet distance between two 2D curves.
    P, Q: arrays of shape (n, 2) and (m, 2).
    """
    n = len(P)
    m = len(Q)
    ca = np.full((n, m), -1.0)

    def _dist(i, j):
        return np.sqrt((P[i, 0] - Q[j, 0])**2 + (P[i, 1] - Q[j, 1])**2)

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = _dist(i, j)
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    # Iterative version to avoid recursion limit
    for i in range(n):
        for j in range(m):
            d = _dist(i, j)
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
    """Downsample a trajectory to at most max_points (evenly spaced)."""
    n = len(xs)
    if n <= max_points:
        return xs, ys
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return xs[indices], ys[indices]


# =============================================================================
# CORRIDOR (from real-world runs)
# =============================================================================

def build_corridor(rw_signed_laterals):
    """
    Build corridor bounds from multiple real-world signed lateral arrays.
    All arrays must be sampled at the same progress values.
    Returns (min_lateral, max_lateral) arrays.
    """
    stacked = np.stack(rw_signed_laterals, axis=0)  # (n_runs, n_progress)
    return stacked.min(axis=0), stacked.max(axis=0)


def compute_corridor_violations(sim_signed_lateral, corridor_min, corridor_max):
    """
    Compute how much a sim trajectory violates the corridor.
    Returns:
      violation_rate: fraction of points outside corridor
      mean_excess_m: average distance outside corridor (over all points)
      mean_excess_when_out_m: average distance outside corridor (only over violating points)
    """
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
    parser.add_argument("--segment", required=True, help="Path to scenario_segment.json")
    parser.add_argument("--rw_dir", required=True, help="Directory with real-world trajectories")
    parser.add_argument("--sim_dirs", nargs="+", required=True,
                        help="Sim trajectory dirs as label=path")
    parser.add_argument("--completion_threshold", type=float, default=0.95,
                        help="Min completion to be considered successful (default: 0.95)")
    parser.add_argument("--frechet_max_pts", type=int, default=500,
                        help="Max points for Fréchet computation (default: 500)")
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
    seg_length = segment["scenario_length_m"]

    # Uniform progress values for corridor sampling (every 0.5 m within segment)
    corridor_progress = np.arange(seg_start, seg_end, 0.5)

    # --- Load real-world trajectories → CARLA ---
    print("=" * 70)
    print("  LOADING REAL-WORLD TRAJECTORIES")
    print("=" * 70)

    SUCCESSFUL_CONDITIONS = {"sunny", "cloudy"}
    # rw_carla: (condition, run) → (carla_x, carla_y)
    rw_carla = {}
    # rw_steering: (condition, run) → steering array (matched to positions)
    rw_steering = {}

    for fname in sorted(os.listdir(args.rw_dir)):
        if not fname.endswith("_trajectory.txt"):
            continue
        match = re.match(r'(\w+?)(\d+)_trajectory\.txt', fname)
        if not match:
            continue
        condition = match.group(1)
        run = int(match.group(2))
        path = os.path.join(args.rw_dir, fname)

        utm_x, utm_y, traj_ts = load_real_trajectory_utm(path)
        carla_x, carla_y = utm_array_to_carla(utm_x, utm_y, tf_utm_to_wgs,
                                                tf_wgs_to_proj, xodr_offset)
        rw_carla[(condition, run)] = (carla_x, carla_y)

        # Try to load matching steering file
        # Try steering_cmd_ first (DAVE-2 command), fall back to steering_ (wheel response)
        steer_path = os.path.join(args.rw_dir, f"steering_cmd_{condition}_{run}.txt")
        if not os.path.exists(steer_path):
            steer_path = os.path.join(args.rw_dir, f"steering_{condition}_{run}.txt")
        if os.path.exists(steer_path):
            steer_ts, steer_vals = load_real_steering(steer_path)
            # Interpolate POSITIONS at steering command timestamps (not the reverse)
            # This keeps one point per actual DAVE-2 decision with real Δt
            valid_mask = (steer_ts >= traj_ts[0]) & (steer_ts <= traj_ts[-1])
            steer_ts_valid = steer_ts[valid_mask]
            steer_vals_valid = steer_vals[valid_mask]
            cx_at_steer = np.interp(steer_ts_valid, traj_ts, carla_x)
            cy_at_steer = np.interp(steer_ts_valid, traj_ts, carla_y)
            # Store: positions at steering times, steering values, steering timestamps
            rw_steering[(condition, run)] = (steer_vals_valid, steer_ts_valid,
                                              cx_at_steer, cy_at_steer)
            print(f"  {fname}: {len(carla_x)} pts + {len(steer_vals_valid)} steering cmds ({os.path.basename(steer_path)})")
        else:
            print(f"  {fname}: {len(carla_x)} pts (no steering file)")

    # --- Load simulated trajectories ---
    print("\n" + "=" * 70)
    print("  LOADING SIMULATED TRAJECTORIES")
    print("=" * 70)

    # sim_carla: (method, condition, run) → (carla_x, carla_y)
    sim_carla = {}
    # sim_steering: (method, condition, run) → steering array
    sim_steering = {}

    for entry in args.sim_dirs:
        if "=" not in entry:
            continue
        dir_label, dir_path = entry.split("=", 1)
        if not os.path.isdir(dir_path):
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith("_trajectory.json"):
                continue
            # condition_method_runN or condition_runN
            match2 = re.match(r'([a-z]+)_([a-z]+)_run(\d+)_trajectory\.json', fname)
            match1 = re.match(r'([a-z]+)_run(\d+)_trajectory\.json', fname)
            if match2:
                condition = match2.group(1)
                method = match2.group(2)
                run = int(match2.group(3))
            elif match1:
                condition = match1.group(1)
                method = dir_label
                run = int(match1.group(2))
            else:
                continue

            path = os.path.join(dir_path, fname)
            carla_x, carla_y, steer = load_sim_trajectory(path)
            sim_carla[(method, condition, run)] = (carla_x, carla_y)
            if steer is not None:
                sim_steering[(method, condition, run)] = steer
            print(f"  {method}/{fname}: {len(carla_x)} pts")

    # --- Filter to successful runs only ---
    print("\n" + "=" * 70)
    print("  FILTERING TO SUCCESSFUL RUNS (completion >= {:.0f}%)".format(
        args.completion_threshold * 100))
    print("=" * 70)

    def check_completion(carla_x, carla_y):
        progress, _, _ = project_onto_reference(carla_x, carla_y, ref_xs, ref_ys, ref_s)
        completion = max(0, min(1.0, (progress[-1] - seg_start) / seg_length))
        return completion

    successful_sim = {}
    for key, (cx, cy) in sim_carla.items():
        comp = check_completion(cx, cy)
        if comp >= args.completion_threshold:
            successful_sim[key] = (cx, cy)
            print(f"  ✅ {key[0]}_{key[1]}_run{key[2]}: {comp*100:.1f}%")
        else:
            print(f"  ❌ {key[0]}_{key[1]}_run{key[2]}: {comp*100:.1f}% — SKIPPED")

    # --- Build corridors per scenario (from successful real-world runs) ---
    print("\n" + "=" * 70)
    print("  BUILDING CORRIDORS & COMPUTING METRICS")
    print("=" * 70)

    conditions_with_success = sorted(set(k[0] for k in rw_carla.keys()
                                         if k[0] in SUCCESSFUL_CONDITIONS))

    all_results = []

    for condition in conditions_with_success:
        print(f"\n  --- {condition.upper()} ---")

        # Get real-world runs for this condition
        rw_runs = {k: v for k, v in rw_carla.items() if k[0] == condition}

        # Project real-world runs and resample at corridor_progress
        rw_resampled = []  # list of (xs, ys) resampled at corridor_progress
        rw_signed_lats = []  # signed lateral at corridor_progress

        for key, (cx, cy) in sorted(rw_runs.items()):
            progress, lat, closest_idx = project_onto_reference(cx, cy, ref_xs, ref_ys, ref_s)
            signed_lat = compute_signed_lateral(cx, cy, ref_xs, ref_ys, closest_idx)

            # Resample at corridor progress
            rx, ry, valid = resample_at_progress(cx, cy, progress, corridor_progress)
            # Also resample signed lateral
            mask = np.diff(progress, prepend=-1) > 0
            p_clean = progress[mask]
            sl_clean = signed_lat[mask]
            valid_cp = (corridor_progress >= p_clean[0]) & (corridor_progress <= p_clean[-1])
            sl_resampled = np.interp(corridor_progress[valid_cp], p_clean, sl_clean)

            # Pad to full length if needed (for runs that don't cover full segment)
            sl_full = np.full(len(corridor_progress), np.nan)
            sl_full[valid_cp] = sl_resampled

            rw_resampled.append((rx, ry))
            rw_signed_lats.append(sl_full)
            print(f"    RW {key[0]}{key[1]}: signed lat range "
                  f"[{np.nanmin(sl_full):.3f}, {np.nanmax(sl_full):.3f}] m")

        # Build corridor (ignoring NaN — use only points covered by all runs)
        rw_stack = np.stack(rw_signed_lats, axis=0)  # (n_runs, n_progress)
        corridor_min = np.nanmin(rw_stack, axis=0)
        corridor_max = np.nanmax(rw_stack, axis=0)
        corridor_valid = ~np.isnan(corridor_min)  # progress points with valid corridor

        corridor_width = corridor_max[corridor_valid] - corridor_min[corridor_valid]
        print(f"    Corridor: mean width={np.mean(corridor_width):.3f} m, "
              f"max width={np.max(corridor_width):.3f} m")

        # --- Compute Fréchet distances: real-world runs trimmed to segment ---
        # First, prepare trimmed real-world curves for Fréchet
        rw_trimmed_curves = []
        for key, (cx, cy) in sorted(rw_runs.items()):
            progress, _, _ = project_onto_reference(cx, cy, ref_xs, ref_ys, ref_s)
            in_seg = (progress >= seg_start) & (progress <= seg_end)
            tx, ty = cx[in_seg], cy[in_seg]
            rw_trimmed_curves.append((key, np.column_stack([tx, ty])))

        # --- Evaluate each successful sim run ---
        sim_for_condition = {k: v for k, v in successful_sim.items() if k[1] == condition}

        for key, (cx, cy) in sorted(sim_for_condition.items()):
            method, cond, run = key

            # Trim sim to segment
            progress, lat, closest_idx = project_onto_reference(cx, cy, ref_xs, ref_ys, ref_s)
            in_seg = (progress >= seg_start) & (progress <= seg_end)
            sim_cx, sim_cy = cx[in_seg], cy[in_seg]

            if len(sim_cx) < 10:
                print(f"    {method}_run{run}: too few points in segment — skipping")
                continue

            # --- Min-Fréchet ---
            sim_curve = np.column_stack([sim_cx, sim_cy])

            frechet_dists = []
            for rw_key, rw_curve in rw_trimmed_curves:
                fd = discrete_frechet_distance(sim_curve, rw_curve)
                frechet_dists.append((rw_key, fd))

            min_frechet = min(frechet_dists, key=lambda x: x[1])

            # --- Corridor violations ---
            signed_lat = compute_signed_lateral(cx, cy, ref_xs, ref_ys, closest_idx)
            # Resample sim signed lateral at corridor progress
            mask = np.diff(progress, prepend=-1) > 0
            p_clean = progress[mask]
            sl_clean = signed_lat[mask]
            valid_cp = (corridor_progress >= p_clean[0]) & (corridor_progress <= p_clean[-1])

            if np.sum(valid_cp & corridor_valid) < 10:
                print(f"    {method}_run{run}: insufficient corridor overlap — skipping")
                continue

            sl_sim = np.interp(corridor_progress[valid_cp], p_clean, sl_clean)

            # Only evaluate where corridor is valid
            eval_mask = corridor_valid[valid_cp]
            sl_eval = sl_sim[eval_mask]
            corr_min_eval = corridor_min[valid_cp][eval_mask]
            corr_max_eval = corridor_max[valid_cp][eval_mask]

            viol_rate, mean_excess, mean_excess_out = compute_corridor_violations(
                sl_eval, corr_min_eval, corr_max_eval)

            result = {
                "method": method,
                "condition": condition,
                "run": run,
                "min_frechet_m": min_frechet[1],
                "min_frechet_vs": f"{min_frechet[0][0]}{min_frechet[0][1]}",
                "corridor_violation_pct": viol_rate * 100,
                "mean_excess_m": mean_excess,
                "mean_excess_when_out_m": mean_excess_out,
            }
            all_results.append(result)

            print(f"    {method}_run{run}: "
                  f"Fréchet={min_frechet[1]:.3f} m (vs {min_frechet[0][0]}{min_frechet[0][1]}), "
                  f"out={viol_rate*100:.1f}%, "
                  f"mean_excess={mean_excess:.3f} m, "
                  f"excess_when_out={mean_excess_out:.3f} m")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  DRIVE QUALITY SUMMARY (successful runs only)")
    print("=" * 70)

    print(f"\n  {'Method':<12} {'Condition':<10} {'Run':>3} "
          f"{'Fréchet(m)':>10} {'Out(%)':>8} {'MeanExcess(m)':>14} {'ExcessOut(m)':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*3} {'-'*10} {'-'*8} {'-'*14} {'-'*12}")

    for r in all_results:
        print(f"  {r['method']:<12} {r['condition']:<10} {r['run']:>3} "
              f"{r['min_frechet_m']:>10.3f} {r['corridor_violation_pct']:>8.1f} "
              f"{r['mean_excess_m']:>14.3f} {r['mean_excess_when_out_m']:>12.3f}")

    # --- Averages per method+condition ---
    print(f"\n  AVERAGES:")
    print(f"  {'Method':<12} {'Condition':<10} {'Runs':>4} "
          f"{'Fréchet(m)':>10} {'Out(%)':>8} {'MeanExcess(m)':>14}")
    print(f"  {'-'*12} {'-'*10} {'-'*4} {'-'*10} {'-'*8} {'-'*14}")

    groups = {}
    for r in all_results:
        key = (r["method"], r["condition"])
        groups.setdefault(key, []).append(r)

    for (method, condition), runs in sorted(groups.items()):
        n = len(runs)
        avg_f = np.mean([r["min_frechet_m"] for r in runs])
        avg_o = np.mean([r["corridor_violation_pct"] for r in runs])
        avg_e = np.mean([r["mean_excess_m"] for r in runs])
        print(f"  {method:<12} {condition:<10} {n:>4} "
              f"{avg_f:>10.3f} {avg_o:>8.1f} {avg_e:>14.3f}")

    # =====================================================================
    #  STEERING JITTER (successful runs only, within segment)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEERING JITTER (std of Δsteering, within segment)")
    print("=" * 70)

    SIM_DT = 3.0 / 30.0  # 0.1s — every 3rd frame at 30 FPS
    RW_MAX_DT = 0.25     # skip real-world gaps larger than this (dropped msgs)

    def compute_jitter(steering_arr, progress_arr, seg_start, seg_end,
                       timestamps=None, dt_fixed=None, max_dt_gap=None, subsample=1):
        """
        Compute jitter = std(Δsteering/Δt) and max jitter = max(|Δsteering/Δt|).
        Units: steering change per second.

        timestamps: actual timestamps (real-world). Used to compute Δt and skip gaps.
        dt_fixed: fixed Δt for sim runs (after subsampling).
        max_dt_gap: skip pairs where Δt exceeds this (real-world dropped messages).
        subsample: take every Nth sample (e.g. 3 for SD to match GS rate).
        """
        mask = (progress_arr >= seg_start) & (progress_arr <= seg_end)
        steer_seg = steering_arr[mask]

        if timestamps is not None:
            ts_seg = timestamps[mask]

        # Subsample
        if subsample > 1:
            steer_seg = steer_seg[::subsample]
            if timestamps is not None:
                ts_seg = ts_seg[::subsample]

        if len(steer_seg) < 2:
            return None, None

        delta_steer = np.diff(steer_seg)

        if timestamps is not None:
            dt = np.diff(ts_seg)
            # Skip gaps (dropped messages)
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
            rate = delta_steer  # fallback: no normalization

        return float(np.std(rate)), float(np.max(np.abs(rate)))

    # Real-world jitter (using cmd/steering_target timestamps)
    jitter_results = []

    for condition in conditions_with_success:
        rw_runs = {k: v for k, v in rw_carla.items() if k[0] == condition}

        for (cond, run), (cx, cy) in sorted(rw_runs.items()):
            if (cond, run) not in rw_steering:
                continue
            steer_vals, steer_ts, steer_cx, steer_cy = rw_steering[(cond, run)]
            # Project steering positions onto reference
            progress, _, _ = project_onto_reference(steer_cx, steer_cy, ref_xs, ref_ys, ref_s)
            j, mj = compute_jitter(steer_vals, progress, seg_start, seg_end,
                                    timestamps=steer_ts, max_dt_gap=RW_MAX_DT)
            if j is not None:
                jitter_results.append({
                    "method": "real_world",
                    "condition": cond,
                    "run": run,
                    "jitter": j,
                    "max_jitter": mj,
                })
                print(f"  RW {cond}{run}: jitter={j:.6f} /s, max={mj:.6f} /s")

    # Simulated jitter (successful only)
    # SD: subsample every 3rd frame to match GS rate (~10 Hz)
    # GS: already PREDICT_EVERY=3, but logged every frame (repeated values) → subsample 3
    for key in sorted(successful_sim.keys()):
        method, condition, run = key
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
                "condition": condition,
                "run": run,
                "jitter": j,
                "max_jitter": mj,
            })
            print(f"  {method} {condition} run{run}: jitter={j:.6f} /s, max={mj:.6f} /s")

    # Jitter summary per method+condition
    print(f"\n  JITTER SUMMARY (steering rate: units/second):")
    print(f"  {'Method':<12} {'Condition':<10} {'Runs':>4} {'Avg Jitter':>12} {'Max Jitter':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*4} {'-'*12} {'-'*12}")

    jitter_groups = {}
    for r in jitter_results:
        key = (r["method"], r["condition"])
        jitter_groups.setdefault(key, []).append(r)

    for (method, condition), runs in sorted(jitter_groups.items()):
        n = len(runs)
        avg_j = np.mean([r["jitter"] for r in runs])
        max_j = np.max([r["max_jitter"] for r in runs])
        print(f"  {method:<12} {condition:<10} {n:>4} {avg_j:>12.6f} {max_j:>12.6f}")

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
                "condition": condition,
                "runs": len(runs),
                "avg_frechet_m": round(float(np.mean([r["min_frechet_m"] for r in runs])), 3),
                "avg_out_pct": round(float(np.mean([r["corridor_violation_pct"] for r in runs])), 1),
                "avg_mean_excess_m": round(float(np.mean([r["mean_excess_m"] for r in runs])), 3),
            }
            for (method, condition), runs in sorted(groups.items())
        ],
        "steering_jitter": jitter_results,
        "steering_jitter_averages": [
            {
                "method": method,
                "condition": condition,
                "runs": len(runs),
                "avg_jitter": round(float(np.mean([r["jitter"] for r in runs])), 6),
                "max_jitter": round(float(np.max([r["max_jitter"] for r in runs])), 6),
            }
            for (method, condition), runs in sorted(jitter_groups.items())
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved: {output_path}")

    print()


if __name__ == "__main__":
    main()