#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6C_metrics_drive_quality.py

PATCHED: real-world trajectories and steering predictions are now read from
    data/data_for_validation/real_world_trajectories/<N>_trajectory.csv
    data/data_for_validation/real_world_trajectories/<N>_steering_predictions.txt

The steering files are optional (jitter reale is computed only when present).
"""

import os
import re
import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer


# =============================================================================
#  HARDCODED CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MAP_BAG_NAME = "reference_bag"

REAL_RUNS_DIR = (
    PROJECT_ROOT / "data" / "data_for_validation" / "real_world_trajectories"
)
REAL_RUN_PATTERN = re.compile(r"^(\d+)_trajectory\.csv$")

XODR_PATH = (
    PROJECT_ROOT / "data" / "processed_dataset" / MAP_BAG_NAME / "maps" / "map.xodr"
)

SEGMENT_PATH = (
    PROJECT_ROOT / "data" / "processed_dataset" / MAP_BAG_NAME
    / "scenario_segment" / "scenario_segment.json"
)

SIM_RUNS_ROOT = (
    PROJECT_ROOT / "data" / "data_for_carla" / MAP_BAG_NAME / "drive_results"
)

RESULTS_DIR = (
    PROJECT_ROOT / "data" / "processed_dataset" / MAP_BAG_NAME / "drive_evaluation"
)

OUTPUT_JSON = RESULTS_DIR / "drive_quality_results.json"

COMPLETION_THRESHOLD = 0.95
CORRIDOR_SPACING_M = 0.5
FRECHET_MAX_PTS = 500

SIM_DT_S = 3.0 / 30.0
SIM_SUBSAMPLE = 3

RW_MAX_DT_S = 0.25


# =============================================================================
#  XODR + TRANSFORMS
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
        "xodr_offset": xodr_offset,
    }


def utm_to_carla(utm_x, utm_y, tfs):
    lon, lat = tfs["utm_to_wgs"].transform(utm_x, utm_y)
    proj_x, proj_y = tfs["wgs_to_proj"].transform(lon, lat)
    carla_x = proj_x + tfs["xodr_offset"][0]
    carla_y = -(proj_y + tfs["xodr_offset"][1])
    return carla_x, carla_y


def utm_array_to_carla(uxs, uys, tfs):
    cxs, cys = [], []
    for ux, uy in zip(uxs, uys):
        cx, cy = utm_to_carla(ux, uy, tfs)
        cxs.append(cx)
        cys.append(cy)
    return np.array(cxs), np.array(cys)


# =============================================================================
#  LOADERS
# =============================================================================

def load_real_trajectory_csv(csv_path):
    utm_xs, utm_ys, timestamps = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "x" not in reader.fieldnames \
                or "y" not in reader.fieldnames \
                or "timestamp" not in reader.fieldnames:
            raise RuntimeError(
                f"Unexpected CSV header in {csv_path}. "
                f"Got: {reader.fieldnames}. "
                f"Expected: timestamp, x, y, z, yaw."
            )

        for row in reader:
            try:
                timestamps.append(float(row["timestamp"]))
                utm_xs.append(float(row["x"]))
                utm_ys.append(float(row["y"]))
            except (KeyError, ValueError):
                continue

    if not utm_xs:
        raise RuntimeError(f"No data rows in {csv_path}")

    return np.array(utm_xs), np.array(utm_ys), np.array(timestamps)


def load_real_steering(txt_path):
    timestamps, values = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                timestamps.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(timestamps), np.array(values)


def load_sim_trajectory(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        raise RuntimeError(f"Empty sim trajectory: {json_path}")

    xs = np.array([float(p["x"]) for p in data])
    ys = np.array([float(p["y"]) for p in data])

    if "steer_raw" in data[0]:
        steering = np.array([float(p["steer_raw"]) for p in data])
    elif "steering_raw_rad" in data[0]:
        steering = np.array([float(p["steering_raw_rad"]) for p in data])
    elif "steer_norm" in data[0]:
        steering = np.array([float(p["steer_norm"]) * 3.0 * np.pi for p in data])
    else:
        steering = None

    return xs, ys, steering


def discover_real_runs(tfs):
    """
    Scan REAL_RUNS_DIR for <N>_trajectory.csv files. For each, optionally
    load the matching <N>_steering_predictions.txt if present.
    """
    real_runs = {}

    if not REAL_RUNS_DIR.exists():
        print(f"[ERROR] Real runs dir not found: {REAL_RUNS_DIR}")
        return real_runs

    entries = []
    for entry in sorted(REAL_RUNS_DIR.iterdir()):
        if not entry.is_file():
            continue
        m = REAL_RUN_PATTERN.match(entry.name)
        if not m:
            continue
        idx = int(m.group(1))
        entries.append((idx, entry))

    entries.sort(key=lambda t: t[0])

    for idx, csv_path in entries:
        run_label = f"run{idx}"

        utm_x, utm_y, ts = load_real_trajectory_csv(csv_path)
        cx, cy = utm_array_to_carla(utm_x, utm_y, tfs)

        steering_path = REAL_RUNS_DIR / f"{idx}_steering_predictions.txt"

        if steering_path.exists():
            st_ts, st_vals = load_real_steering(steering_path)
            valid = (st_ts >= ts[0]) & (st_ts <= ts[-1])
            st_ts_v = st_ts[valid]
            st_vals_v = st_vals[valid]
            cx_at_steer = np.interp(st_ts_v, ts, cx)
            cy_at_steer = np.interp(st_ts_v, ts, cy)
            steering = {
                "timestamps": st_ts_v,
                "values": st_vals_v,
                "carla_x": cx_at_steer,
                "carla_y": cy_at_steer,
            }
            print(f"  {run_label}: {len(cx)} pts + "
                  f"{len(st_vals_v)} steering cmds  <- {csv_path.name}")
        else:
            steering = None
            print(f"  {run_label}: {len(cx)} pts (no steering)  "
                  f"<- {csv_path.name}")

        real_runs[run_label] = {
            "carla_x": cx,
            "carla_y": cy,
            "timestamps": ts,
            "steering": steering,
        }

    return real_runs


def discover_sim_runs():
    runs = []

    if not SIM_RUNS_ROOT.exists():
        print(f"[WARN] Sim runs root does not exist: {SIM_RUNS_ROOT}")
        return runs

    pattern = re.compile(r"^(.*)_run(\d+)$")

    for child in sorted(SIM_RUNS_ROOT.iterdir()):
        if not child.is_dir():
            continue

        traj = child / "trajectory.json"
        if not traj.exists():
            traj = child / "data" / "trajectory.json"
        if not traj.exists():
            print(f"[WARN] No trajectory.json in {child.name}")
            continue

        match = pattern.match(child.name)
        if match:
            method = match.group(1)
            run_idx = int(match.group(2))
        else:
            method = child.name
            run_idx = 0

        runs.append({
            "run_name":        child.name,
            "method":          method,
            "run_idx":         run_idx,
            "trajectory_path": traj,
        })

    return runs


# =============================================================================
#  PROJECTION / LATERAL
# =============================================================================

def project_onto_reference(traj_xs, traj_ys, ref_xs, ref_ys, ref_s):
    n = len(traj_xs)
    progress = np.zeros(n)
    lateral = np.zeros(n)
    closest = np.zeros(n, dtype=int)

    for i in range(n):
        dists = np.sqrt((ref_xs - traj_xs[i]) ** 2 + (ref_ys - traj_ys[i]) ** 2)
        idx = int(np.argmin(dists))
        closest[i] = idx
        lateral[i] = dists[idx]
        progress[i] = ref_s[idx]

    for i in range(1, n):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]

    return progress, lateral, closest


def compute_signed_lateral(traj_xs, traj_ys, ref_xs, ref_ys, closest_idx):
    n = len(traj_xs)
    signed = np.zeros(n)

    for i in range(n):
        k = closest_idx[i]
        if k < len(ref_xs) - 1:
            tx = ref_xs[k + 1] - ref_xs[k]
            ty = ref_ys[k + 1] - ref_ys[k]
        else:
            tx = ref_xs[k] - ref_xs[k - 1]
            ty = ref_ys[k] - ref_ys[k - 1]

        dx = traj_xs[i] - ref_xs[k]
        dy = traj_ys[i] - ref_ys[k]

        cross = tx * dy - ty * dx
        norm = np.sqrt(tx * tx + ty * ty)
        if norm > 1e-9:
            signed[i] = cross / norm
        else:
            signed[i] = np.sqrt(dx * dx + dy * dy)

    return signed


def completion_fraction(progress, seg_start, seg_length):
    return max(0.0, min(1.0, (progress[-1] - seg_start) / seg_length))


# =============================================================================
#  FRECHET
# =============================================================================

def downsample_curve(xs, ys, max_points):
    n = len(xs)
    if n <= max_points:
        return xs, ys
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return xs[idx], ys[idx]


def discrete_frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    if n == 0 or m == 0:
        return float("inf")

    ca = np.empty((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            d = float(np.hypot(P[i, 0] - Q[j, 0], P[i, 1] - Q[j, 1]))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)

    return float(ca[n - 1, m - 1])


# =============================================================================
#  CORRIDOR
# =============================================================================

def corridor_violation(sim_signed, corr_min, corr_max):
    excess = np.zeros_like(sim_signed)
    below = sim_signed < corr_min
    above = sim_signed > corr_max
    excess[below] = corr_min[below] - sim_signed[below]
    excess[above] = sim_signed[above] - corr_max[above]

    n_out = int(np.sum(below | above))
    n = len(sim_signed)
    if n == 0:
        return 0.0, 0.0, 0.0

    rate = n_out / n
    mean_exc = float(np.mean(excess))
    mean_exc_out = float(np.mean(excess[below | above])) if n_out > 0 else 0.0
    return rate, mean_exc, mean_exc_out


# =============================================================================
#  JITTER
# =============================================================================

def compute_jitter(values, progress, seg_start, seg_end,
                   timestamps=None, dt_fixed=None,
                   max_dt_gap=None, subsample=1):
    mask = (progress >= seg_start) & (progress <= seg_end)
    v_seg = values[mask]
    if timestamps is not None:
        ts_seg = timestamps[mask]

    if subsample > 1:
        v_seg = v_seg[::subsample]
        if timestamps is not None:
            ts_seg = ts_seg[::subsample]

    if len(v_seg) < 2:
        return None, None

    dv = np.diff(v_seg)

    if timestamps is not None:
        dt = np.diff(ts_seg)
        if max_dt_gap is not None:
            valid = (dt > 0.0) & (dt <= max_dt_gap)
            dv = dv[valid]
            dt = dt[valid]
        if len(dv) < 2:
            return None, None
        rate = dv / dt
    elif dt_fixed is not None:
        rate = dv / float(dt_fixed)
    else:
        rate = dv

    return float(np.std(rate)), float(np.max(np.abs(rate)))


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  DRIVE QUALITY METRICS")
    print("=" * 70)
    print(f"[INFO] Project root:  {PROJECT_ROOT}")
    print(f"[INFO] Map bag:       {MAP_BAG_NAME}")
    print(f"[INFO] Real runs dir: {REAL_RUNS_DIR}")
    print(f"[INFO] XODR:          {XODR_PATH}")
    print(f"[INFO] Segment JSON:  {SEGMENT_PATH}")
    print(f"[INFO] Sim runs root: {SIM_RUNS_ROOT}")
    print(f"[INFO] Results dir:   {RESULTS_DIR}")
    print("=" * 70)

    if not XODR_PATH.exists():
        raise FileNotFoundError(f"XODR not found: {XODR_PATH}")
    if not SEGMENT_PATH.exists():
        raise FileNotFoundError(
            f"Scenario segment not found: {SEGMENT_PATH}. "
            f"Run 6B_verify_scenario_segment.py first."
        )

    tfs = setup_transforms(XODR_PATH)

    with open(SEGMENT_PATH, "r") as f:
        segment = json.load(f)

    ref_xs = np.array(segment["reference_path_carla_x"])
    ref_ys = np.array(segment["reference_path_carla_y"])
    ref_s = np.array(segment["reference_path_arc_length"])
    seg_start = float(segment["scenario_start_m"])
    seg_end = float(segment["scenario_end_m"])
    seg_length = float(segment["scenario_length_m"])

    if seg_length <= 0:
        raise RuntimeError("Scenario length is non-positive.")

    corridor_progress = np.arange(seg_start, seg_end, CORRIDOR_SPACING_M)
    print(f"[INFO] Segment: start={seg_start:.1f} m, end={seg_end:.1f} m, "
          f"length={seg_length:.1f} m, "
          f"corridor grid={len(corridor_progress)} pts")

    print("\n[INFO] Loading real-world runs...")
    real_runs = discover_real_runs(tfs)
    if not real_runs:
        raise RuntimeError(
            f"No <N>_trajectory.csv files in {REAL_RUNS_DIR}."
        )

    print("\n[INFO] Loading sim runs...")
    sim_run_specs = discover_sim_runs()
    if not sim_run_specs:
        raise RuntimeError(
            f"No sim trajectory.json files under {SIM_RUNS_ROOT}."
        )

    sim_runs = {}
    for spec in sim_run_specs:
        cx, cy, steer = load_sim_trajectory(spec["trajectory_path"])
        sim_runs[spec["run_name"]] = {
            "method":   spec["method"],
            "run_idx":  spec["run_idx"],
            "carla_x":  cx,
            "carla_y":  cy,
            "steering": steer,
        }
        print(f"  {spec['run_name']}: {len(cx)} pts "
              f"(method={spec['method']}, run={spec['run_idx']})")

    print(f"\n[INFO] Filtering sim runs (completion >= "
          f"{COMPLETION_THRESHOLD * 100:.0f}%)...")

    successful_sim = {}
    sim_completion = {}
    for run_name, info in sim_runs.items():
        progress, _, _ = project_onto_reference(
            info["carla_x"], info["carla_y"], ref_xs, ref_ys, ref_s
        )
        comp = completion_fraction(progress, seg_start, seg_length)
        sim_completion[run_name] = comp
        ok = comp >= COMPLETION_THRESHOLD
        mark = "OK   " if ok else "FAIL "
        print(f"  [{mark}] {run_name}: completion={comp * 100:.1f}%")
        if ok:
            successful_sim[run_name] = info

    print("\n[INFO] Projecting real-world runs and building corridor...")

    rw_proj = {}
    rw_signed_resampled = []
    rw_segment_curves = {}

    for label, run in sorted(real_runs.items()):
        cx, cy = run["carla_x"], run["carla_y"]
        progress, lateral, closest = project_onto_reference(
            cx, cy, ref_xs, ref_ys, ref_s
        )
        signed_lat = compute_signed_lateral(cx, cy, ref_xs, ref_ys, closest)
        rw_proj[label] = {
            "progress":   progress,
            "signed_lat": signed_lat,
            "carla_x":    cx,
            "carla_y":    cy,
        }

        keep = np.diff(progress, prepend=-1) > 0
        p_clean = progress[keep]
        s_clean = signed_lat[keep]
        in_range = (corridor_progress >= p_clean[0]) & \
                   (corridor_progress <= p_clean[-1])
        s_full = np.full(len(corridor_progress), np.nan)
        if np.any(in_range):
            s_full[in_range] = np.interp(
                corridor_progress[in_range], p_clean, s_clean
            )
        rw_signed_resampled.append(s_full)

        in_seg = (progress >= seg_start) & (progress <= seg_end)
        rw_segment_curves[label] = (cx[in_seg], cy[in_seg])

        print(f"  {label}: signed lat range "
              f"[{np.nanmin(s_full):.3f}, {np.nanmax(s_full):.3f}] m")

    rw_stack = np.stack(rw_signed_resampled, axis=0)
    corridor_min = np.nanmin(rw_stack, axis=0)
    corridor_max = np.nanmax(rw_stack, axis=0)
    corridor_valid = ~np.isnan(corridor_min)
    width = corridor_max[corridor_valid] - corridor_min[corridor_valid]
    if width.size > 0:
        print(f"  Corridor: mean width={np.mean(width):.3f} m, "
              f"max width={np.max(width):.3f} m, "
              f"valid points={int(corridor_valid.sum())}/{len(corridor_valid)}")

    print("\n[INFO] Computing metrics for successful sim runs...")
    per_run_results = []

    for run_name, info in sorted(successful_sim.items()):
        cx, cy = info["carla_x"], info["carla_y"]
        progress, _, closest = project_onto_reference(
            cx, cy, ref_xs, ref_ys, ref_s
        )
        in_seg = (progress >= seg_start) & (progress <= seg_end)
        sim_cx, sim_cy = cx[in_seg], cy[in_seg]

        if len(sim_cx) < 10:
            print(f"  [SKIP] {run_name}: only {len(sim_cx)} pts in segment")
            continue

        sim_ds_x, sim_ds_y = downsample_curve(sim_cx, sim_cy, FRECHET_MAX_PTS)
        sim_curve = np.column_stack([sim_ds_x, sim_ds_y])

        frechet_per_real = []
        for label, (rcx, rcy) in rw_segment_curves.items():
            rcx_ds, rcy_ds = downsample_curve(rcx, rcy, FRECHET_MAX_PTS)
            rw_curve = np.column_stack([rcx_ds, rcy_ds])
            fd = discrete_frechet_distance(sim_curve, rw_curve)
            frechet_per_real.append((label, fd))

        best_label, best_fd = min(frechet_per_real, key=lambda kv: kv[1])

        signed_lat = compute_signed_lateral(cx, cy, ref_xs, ref_ys, closest)
        keep = np.diff(progress, prepend=-1) > 0
        p_clean = progress[keep]
        s_clean = signed_lat[keep]
        in_range = (corridor_progress >= p_clean[0]) & \
                   (corridor_progress <= p_clean[-1])

        if int(np.sum(in_range & corridor_valid)) < 10:
            viol_rate = float("nan")
            mean_exc = float("nan")
            mean_exc_out = float("nan")
            print(f"  [WARN] {run_name}: insufficient corridor overlap")
        else:
            s_sim = np.interp(corridor_progress[in_range], p_clean, s_clean)
            eval_mask = corridor_valid[in_range]
            s_eval = s_sim[eval_mask]
            cmin_eval = corridor_min[in_range][eval_mask]
            cmax_eval = corridor_max[in_range][eval_mask]
            viol_rate, mean_exc, mean_exc_out = corridor_violation(
                s_eval, cmin_eval, cmax_eval
            )

        if info["steering"] is not None:
            jitter_std, jitter_max = compute_jitter(
                info["steering"], progress, seg_start, seg_end,
                dt_fixed=SIM_DT_S, subsample=SIM_SUBSAMPLE,
            )
        else:
            jitter_std = jitter_max = None

        result = {
            "run_name":               run_name,
            "method":                 info["method"],
            "run_idx":                info["run_idx"],
            "completion_pct":         round(sim_completion[run_name] * 100.0, 2),
            "min_frechet_m":          round(best_fd, 4),
            "min_frechet_vs":         best_label,
            "corridor_violation_pct": round(viol_rate * 100.0, 2)
                                       if not np.isnan(viol_rate) else None,
            "mean_excess_m":          round(mean_exc, 4)
                                       if not np.isnan(mean_exc) else None,
            "mean_excess_when_out_m": round(mean_exc_out, 4)
                                       if not np.isnan(mean_exc_out) else None,
            "steering_jitter_std":    round(jitter_std, 6)
                                       if jitter_std is not None else None,
            "steering_jitter_max":    round(jitter_max, 6)
                                       if jitter_max is not None else None,
        }
        per_run_results.append(result)

        print(f"  {run_name}: "
              f"Frechet={best_fd:.3f} m (vs {best_label}), "
              f"out={(viol_rate * 100.0 if not np.isnan(viol_rate) else float('nan')):.1f}%, "
              f"jitter_std="
              f"{(jitter_std if jitter_std is not None else float('nan')):.4f}")

    print("\n[INFO] Computing real-world steering jitter...")
    rw_jitter = []
    for label, run in sorted(real_runs.items()):
        steering = run["steering"]
        if steering is None:
            continue
        progress, _, _ = project_onto_reference(
            steering["carla_x"], steering["carla_y"],
            ref_xs, ref_ys, ref_s,
        )
        j_std, j_max = compute_jitter(
            steering["values"], progress, seg_start, seg_end,
            timestamps=steering["timestamps"], max_dt_gap=RW_MAX_DT_S,
        )
        if j_std is not None:
            rw_jitter.append({
                "run_label":           label,
                "steering_jitter_std": round(j_std, 6),
                "steering_jitter_max": round(j_max, 6),
            })
            print(f"  {label}: jitter_std={j_std:.4f}, max={j_max:.4f}")

    by_method = {}
    for r in per_run_results:
        by_method.setdefault(r["method"], []).append(r)

    def _safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    def _safe_std(xs):
        xs = [x for x in xs if x is not None]
        return float(np.std(xs)) if xs else None

    method_summary = []
    for method, rs in sorted(by_method.items()):
        method_summary.append({
            "method":                 method,
            "n_successful_runs":      len(rs),
            "mean_frechet_m":         _safe_mean([r["min_frechet_m"] for r in rs]),
            "std_frechet_m":          _safe_std([r["min_frechet_m"] for r in rs]),
            "mean_corridor_violation_pct":
                _safe_mean([r["corridor_violation_pct"] for r in rs]),
            "mean_excess_m":          _safe_mean([r["mean_excess_m"] for r in rs]),
            "mean_jitter_std":        _safe_mean([r["steering_jitter_std"] for r in rs]),
        })

    print("\n" + "=" * 70)
    print("  SUMMARY (successful runs only, mean +/- std per method)")
    print("=" * 70)
    print(f"  {'Method':<14} {'N':>3} "
          f"{'Frechet(m)':>14} {'Out(%)':>10} "
          f"{'Excess(m)':>10} {'Jitter':>10}")
    print(f"  {'-' * 14} {'-' * 3} {'-' * 14} {'-' * 10} "
          f"{'-' * 10} {'-' * 10}")
    for s in method_summary:
        f_mean = s["mean_frechet_m"]
        f_std = s["std_frechet_m"]
        out_pct = s["mean_corridor_violation_pct"]
        excess = s["mean_excess_m"]
        jit = s["mean_jitter_std"]

        f_str = (f"{f_mean:.3f}+/-{f_std:.3f}"
                 if f_mean is not None and f_std is not None else "--")
        o_str = f"{out_pct:.1f}" if out_pct is not None else "--"
        e_str = f"{excess:.3f}" if excess is not None else "--"
        j_str = f"{jit:.4f}" if jit is not None else "--"

        print(f"  {s['method']:<14} {s['n_successful_runs']:>3} "
              f"{f_str:>14} {o_str:>10} {e_str:>10} {j_str:>10}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "map_bag":              MAP_BAG_NAME,
        "real_runs":            list(real_runs.keys()),
        "completion_threshold": COMPLETION_THRESHOLD,
        "segment": {
            "start_m":  seg_start,
            "end_m":    seg_end,
            "length_m": seg_length,
        },
        "sim_completion":         {k: round(v * 100.0, 2)
                                    for k, v in sim_completion.items()},
        "per_run":                per_run_results,
        "per_method_summary":     method_summary,
        "real_world_jitter":      rw_jitter,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Saved: {OUTPUT_JSON}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()