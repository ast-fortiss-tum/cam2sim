#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all_trajectories.py

Plots ALL trajectories (real-world + simulated) against the scenario segment
on an OSM background. Groups by condition (sunny/cloudy/snowy) in subplots.
Generates LaTeX table with metrics at the end.

Usage:
    python plot_all_trajectories.py \
        --map_xodr maps/sunny_map/map.xodr \
        --segment /media/davide/New\ Volume/RW_trajectories/scenario_segment.json \
        --rw_dir /media/davide/New\ Volume/RW_trajectories \
        --sim_dirs \
            SD=/media/davide/New\ Volume/SD_trajectories \
            splatfacto=/path/to/splatfacto_trajectories \
            nerfacto=/path/to/nerfacto_trajectories \
            only_carla=/path/to/Only_Carla_trajectories
"""

import os
import re
import json
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

    tf_proj_to_wgs = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
    tf_wgs_to_proj = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    tf_utm_to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    tf_wgs_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    return tf_proj_to_wgs, tf_wgs_to_proj, tf_utm_to_wgs, tf_wgs_to_merc, xodr_offset


def carla_to_merc(carla_x, carla_y, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset):
    proj_x = carla_x - xodr_offset[0]
    proj_y = (-carla_y) - xodr_offset[1]
    lon, lat = tf_proj_to_wgs.transform(proj_x, proj_y)
    mx, my = tf_wgs_to_merc.transform(lon, lat)
    return mx, my


def carla_array_to_merc(cxs, cys, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset):
    mxs, mys = [], []
    for x, y in zip(cxs, cys):
        mx, my = carla_to_merc(x, y, tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset)
        mxs.append(mx)
        mys.append(my)
    return np.array(mxs), np.array(mys)


def utm_to_merc(utm_x, utm_y, tf_utm_to_wgs, tf_wgs_to_merc):
    lon, lat = tf_utm_to_wgs.transform(utm_x, utm_y)
    mx, my = tf_wgs_to_merc.transform(lon, lat)
    return mx, my


def utm_array_to_merc(uxs, uys, tf_utm_to_wgs, tf_wgs_to_merc):
    mxs, mys = [], []
    for ux, uy in zip(uxs, uys):
        mx, my = utm_to_merc(ux, uy, tf_utm_to_wgs, tf_wgs_to_merc)
        mxs.append(mx)
        mys.append(my)
    return np.array(mxs), np.array(mys)


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
# PROJECTION ONTO REFERENCE PATH
# =============================================================================

def project_onto_reference(traj_xs, traj_ys, ref_xs, ref_ys, ref_s):
    """Project trajectory onto reference path → 1D progress (arc length)."""
    progress = np.zeros(len(traj_xs))
    for i in range(len(traj_xs)):
        dists = np.sqrt((ref_xs - traj_xs[i])**2 + (ref_ys - traj_ys[i])**2)
        closest_idx = np.argmin(dists)
        progress[i] = ref_s[closest_idx]
    # Enforce monotonicity
    for i in range(1, len(progress)):
        if progress[i] < progress[i - 1]:
            progress[i] = progress[i - 1]
    return progress


# =============================================================================
# LOADERS
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


def load_sim_trajectory(path):
    with open(path, "r") as f:
        data = json.load(f)
    xs = np.array([p["x"] for p in data])
    ys = np.array([p["y"] for p in data])
    return xs, ys


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(completions_rw, completions_sim, conditions, fail_threshold=95.0):
    """
    Generate LaTeX table with metrics.

    completions_rw: dict (condition, run) → completion%
    completions_sim: dict (method, condition, run) → completion%
    conditions: list of condition names
    """

    # Map internal method names to display names
    DOMAIN_MAP = [
        ("real",        "Real"),
        ("splatfacto",  "3DGS"),
        ("nerfacto",    "NeRF"),
        ("SD",          "SD"),
        ("only_carla",  "Sim"),
    ]

    def get_runs(method_key, condition):
        """Get completion values for a method/condition combo."""
        if method_key == "real":
            return {k: v for k, v in completions_rw.items() if k[0] == condition}
        else:
            return {k: v for k, v in completions_sim.items()
                    if k[0] == method_key and k[1] == condition}

    def get_fail_rate(method_key, condition):
        """Returns (n_failed, n_total) string like '1/3'."""
        runs = get_runs(method_key, condition)
        if not runs:
            return "---"
        n_total = len(runs)
        n_failed = sum(1 for v in runs.values() if v < fail_threshold)
        return f"{n_failed}/{n_total}"

    def get_completion_rate(method_key, condition):
        """Returns completion as 'min - avg - max'."""
        runs = get_runs(method_key, condition)
        if not runs:
            return "---"
        vals = list(runs.values())
        vmin = np.min(vals)
        vavg = np.mean(vals)
        vmax = np.max(vals)
        return f"{vavg:.0f} - {vmax:.0f} - {vmin:.0f}"

    # Build LaTeX
    n_cond = len(conditions)
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{System-level evaluation results across weather conditions.}")
    lines.append(r"\label{tab:system_eval}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "c" * n_cond + "}")
    lines.append(r"\hline")

    # Header
    cond_headers = " & ".join([f"\\textbf{{{c.capitalize()}}}" for c in conditions])
    lines.append(r"\textbf{Metric} & \textbf{Domain} & " + cond_headers + r" \\")
    lines.append(r"\hline")

    # Helper: empty cells for section header rows
    empty_cond = " & " * n_cond

    # ===================== SUCCESS / FAIL =====================
    lines.append(r"\small{\textbf{Success / Fail}}" + " &" + empty_cond + r" \\")

    # --- FAIL RATE ---
    for i, (method_key, display_name) in enumerate(DOMAIN_MAP):
        cells = []
        for condition in conditions:
            cells.append(get_fail_rate(method_key, condition))
        metric_col = "Fail Rate" if i == 0 else ""
        row = f"{metric_col} & {display_name} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    # --- COMPLETION RATE (%) — avg - max - min ---
    for i, (method_key, display_name) in enumerate(DOMAIN_MAP):
        cells = []
        for condition in conditions:
            cells.append(get_completion_rate(method_key, condition))
        if i == 0:
            metric_col = r"\makecell[l]{Completion Rate (\%)\\\scriptsize{avg--max--min}}"
        else:
            metric_col = ""
        row = f"{metric_col} & {display_name} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    # --- FAILURE TYPE --- (OR--CC--OS--US)
    FAILURE_TYPES = {
        ("real", "sunny"):       "--",
        ("real", "cloudy"):      "--",
        ("real", "snowy"):       "3 - 0 - 0 - 3",
        ("splatfacto", "sunny"): "--",
        ("splatfacto", "cloudy"): "--",
        ("splatfacto", "snowy"): "2 - 1 - 0 - 3",
        ("nerfacto", "sunny"):   "3 - 0 - 3 - 0",
        ("nerfacto", "cloudy"):  "--",
        ("nerfacto", "snowy"):   "3 - 0 - 0 - 3",
        ("SD", "sunny"):         "3 - 0 - 0 - 3",
        ("SD", "cloudy"):        "0 - 1 - 1 - 0",
        ("SD", "snowy"):         "2 - 1 - 3 - 0",
        ("only_carla", "sunny"): "3 - 0 - 1 - 2",
        ("only_carla", "cloudy"): "2 - 1 - 3 - 0",
        ("only_carla", "snowy"): "2 - 1 - 2 - 1",
    }

    for i, (method_key, display_name) in enumerate(DOMAIN_MAP):
        cells = []
        for condition in conditions:
            cells.append(FAILURE_TYPES.get((method_key, condition), "--"))
        if i == 0:
            metric_col = r"\makecell[l]{Failure Type\textsuperscript{*}\\\scriptsize{OR--CC--OS--US}}"
        else:
            metric_col = ""
        row = f"{metric_col} & {display_name} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\noindent\scriptsize{\textsuperscript{*}OR = Out of Road, CC = Car Crash, OS = Oversteer, US = Understeer.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot all trajectories vs scenario segment on OSM")
    parser.add_argument("--map_xodr", required=True, help="Path to .xodr file")
    parser.add_argument("--segment", required=True, help="Path to scenario_segment.json")
    parser.add_argument("--rw_dir", required=True, help="Directory with real-world trajectory files")
    parser.add_argument("--sim_dirs", nargs="+", required=True,
                        help="Simulated trajectory dirs as label=path (e.g. SD=/path/to/dir)")
    parser.add_argument("--buffer", type=float, default=100, help="Map buffer in meters")
    parser.add_argument("--fail_threshold", type=float, default=95.0,
                        help="Completion %% below which a run is considered failed (default: 95)")
    args = parser.parse_args()

    # --- Setup transforms ---
    tf_proj_to_wgs, tf_wgs_to_proj, tf_utm_to_wgs, tf_wgs_to_merc, xodr_offset = \
        setup_transforms(args.map_xodr)

    # --- Load scenario segment (reference path in CARLA coords) ---
    with open(args.segment, "r") as f:
        segment = json.load(f)

    ref_carla_x = np.array(segment["reference_path_carla_x"])
    ref_carla_y = np.array(segment["reference_path_carla_y"])
    ref_s = np.array(segment["reference_path_arc_length"])
    seg_start = segment["scenario_start_m"]
    seg_end = segment["scenario_end_m"]
    seg_length = segment["scenario_length_m"]

    # Segment portion of reference path
    seg_mask = (ref_s >= seg_start) & (ref_s <= seg_end)
    seg_cx = ref_carla_x[seg_mask]
    seg_cy = ref_carla_y[seg_mask]

    # Convert to mercator
    seg_mx, seg_my = carla_array_to_merc(seg_cx, seg_cy,
                                          tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset)
    ref_mx, ref_my = carla_array_to_merc(ref_carla_x, ref_carla_y,
                                          tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset)

    print(f"Scenario segment: {seg_length:.1f} m [{seg_start:.1f} → {seg_end:.1f}]")

    # --- Completion tracking ---
    # (condition, run) → completion%
    completions_rw = {}
    # (method, condition, run) → completion%
    completions_sim = {}

    # --- Load real-world trajectories ---
    print("\nLoading real-world trajectories...")
    rw_trajs = {}
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
        mx, my = utm_array_to_merc(utm_x, utm_y, tf_utm_to_wgs, tf_wgs_to_merc)
        carla_x, carla_y = utm_array_to_carla(utm_x, utm_y, tf_utm_to_wgs,
                                               tf_wgs_to_proj, xodr_offset)
        progress = project_onto_reference(carla_x, carla_y, ref_carla_x, ref_carla_y, ref_s)
        rw_trajs[(condition, run)] = (mx, my)

        completion = max(0, min(1.0, (progress[-1] - seg_start) / seg_length)) * 100
        completions_rw[(condition, run)] = completion

        status = ""
        if progress[0] > seg_start + 5.0:
            status = f" ⚠️  STARTS LATE (progress={progress[0]:.1f} m > segment start={seg_start:.1f} m)"
        print(f"  {fname}: {len(mx)} pts, progress [{progress[0]:.1f} → {progress[-1]:.1f}] m, "
              f"completion={completion:.1f}%{status}")

    # --- Load simulated trajectories ---
    print("\nLoading simulated trajectories...")
    sim_trajs = {}
    for entry in args.sim_dirs:
        if "=" not in entry:
            print(f"  WARNING: Skipping '{entry}' — expected label=path")
            continue
        method, dir_path = entry.split("=", 1)
        if not os.path.isdir(dir_path):
            print(f"  WARNING: Not a directory: {dir_path}")
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith("_trajectory.json"):
                continue

            condition = None
            run = None
            effective_method = None

            # Pattern 1: only_carla_{condition}_map_run{N}_trajectory.json
            match = re.match(r'only_carla_([a-z]+)_map_run(\d+)_trajectory\.json', fname)
            if match:
                condition = match.group(1)
                run = int(match.group(2))
                effective_method = method

            # Pattern 2: condition_method_run{N}_trajectory.json
            if condition is None:
                match = re.match(r'([a-z]+)_([a-z]+)_run(\d+)_trajectory\.json', fname)
                if match:
                    condition = match.group(1)
                    effective_method = match.group(2)
                    run = int(match.group(3))

            # Pattern 3: condition_run{N}_trajectory.json
            if condition is None:
                match = re.match(r'([a-z]+)_run(\d+)_trajectory\.json', fname)
                if match:
                    condition = match.group(1)
                    run = int(match.group(2))
                    effective_method = method

            if condition is None:
                continue

            path = os.path.join(dir_path, fname)
            sim_x, sim_y = load_sim_trajectory(path)
            mx, my = carla_array_to_merc(sim_x, sim_y,
                                          tf_proj_to_wgs, tf_wgs_to_merc, xodr_offset)
            progress = project_onto_reference(sim_x, sim_y, ref_carla_x, ref_carla_y, ref_s)
            sim_trajs[(effective_method, condition, run)] = (mx, my)

            completion = max(0, min(1.0, (progress[-1] - seg_start) / seg_length)) * 100
            completions_sim[(effective_method, condition, run)] = completion

            status = ""
            if progress[0] > seg_start + 5.0:
                status = f" ⚠️  STARTS LATE (progress={progress[0]:.1f} m > segment start={seg_start:.1f} m)"
            print(f"  {effective_method}/{fname}: {len(mx)} pts, progress [{progress[0]:.1f} → {progress[-1]:.1f}] m, "
                  f"completion={completion:.1f}%{status}")

    # --- Determine conditions and methods ---
    CONDITION_ORDER = ["sunny", "cloudy", "snowy"]
    all_conditions = set(k[0] for k in rw_trajs.keys())
    conditions = [c for c in CONDITION_ORDER if c in all_conditions]
    methods = sorted(set(k[0] for k in sim_trajs.keys()))
    n_cols = len(conditions)

    print(f"\nConditions: {conditions}")
    print(f"Methods: {methods}")

    # --- Compute global bounds from all trajectories ---
    all_mx_list = [ref_mx]
    all_my_list = [ref_my]
    for mx, my in rw_trajs.values():
        all_mx_list.append(mx)
        all_my_list.append(my)
    for mx, my in sim_trajs.values():
        all_mx_list.append(mx)
        all_my_list.append(my)
    all_mx_cat = np.concatenate(all_mx_list)
    all_my_cat = np.concatenate(all_my_list)
    buf = args.buffer
    xmin, xmax = all_mx_cat.min() - buf, all_mx_cat.max() + buf
    ymin, ymax = all_my_cat.min() - buf, all_my_cat.max() + buf

    # --- Colors ---
    RW_COLOR = "#1E88E5"
    METHOD_COLORS = {
        "SD": "#E53935",
        "splatfacto": "#FF9800",
        "nerfacto": "#9C27B0",
        "only_carla": "#43A047",
    }
    LINE_STYLES = ["-", "--", ":"]

    # --- Plot: one column per condition ---
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 10))
    if n_cols == 1:
        axes = [axes]

    for col, condition in enumerate(conditions):
        ax = axes[col]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            print(f"  Warning: Could not load tiles: {e}")

        ax.plot(seg_mx, seg_my, color="#4CAF50", linewidth=8, alpha=0.25,
                solid_capstyle="round", zorder=5)
        ax.plot(seg_mx[0], seg_my[0], marker='|', color='#4CAF50', markersize=20,
                markeredgewidth=4, zorder=25)
        ax.plot(seg_mx[-1], seg_my[-1], marker='|', color='#F44336', markersize=20,
                markeredgewidth=4, zorder=25)

        rw_runs = {k: v for k, v in rw_trajs.items() if k[0] == condition}
        for (cond, run), (mx, my) in sorted(rw_runs.items()):
            ls = LINE_STYLES[(run - 1) % len(LINE_STYLES)]
            label = f"RW run{run}"
            ax.plot(mx, my, color=RW_COLOR, linewidth=2.0, alpha=0.8,
                    linestyle=ls, zorder=10, label=label)
            ax.plot(mx[0], my[0], marker='o', color=RW_COLOR, markersize=6,
                    markeredgecolor='black', markeredgewidth=1, zorder=20)
            ax.plot(mx[-1], my[-1], marker='s', color=RW_COLOR, markersize=6,
                    markeredgecolor='black', markeredgewidth=1, zorder=20)

        for method in methods:
            color = METHOD_COLORS.get(method, "#888888")
            method_runs = {k: v for k, v in sim_trajs.items()
                           if k[0] == method and k[1] == condition}
            for (m, cond, run), (mx, my) in sorted(method_runs.items()):
                ls = LINE_STYLES[(run - 1) % len(LINE_STYLES)]
                label = f"{method} run{run}"
                ax.plot(mx, my, color=color, linewidth=1.8, alpha=0.8,
                        linestyle=ls, zorder=12, label=label)
                ax.plot(mx[0], my[0], marker='o', color=color, markersize=6,
                        markeredgecolor='black', markeredgewidth=1, zorder=20)
                ax.plot(mx[-1], my[-1], marker='x', color=color, markersize=10,
                        markeredgewidth=2.5, zorder=20)

        ax.set_title(f"{condition.capitalize()}", fontsize=14)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_axis_off()

    fig.suptitle(f"All Trajectories vs Scenario Segment ({seg_length:.0f} m)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # =================================================================
    #  LATEX TABLE
    # =================================================================
    print("\n" + "=" * 70)
    print("  LATEX TABLE")
    print("=" * 70)

    latex = generate_latex_table(
        completions_rw, completions_sim, conditions,
        fail_threshold=args.fail_threshold
    )
    print(latex)

    print("\nDone.")


if __name__ == "__main__":
    main()