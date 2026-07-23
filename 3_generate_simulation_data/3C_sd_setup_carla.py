#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3C_sd_setup_carla.py

Launch the CARLA simulator (CarlaUE4.sh) at low quality, with a custom
Glass.uasset swap so that car windows render as opaque/uniform materials
instead of CARLA's default transparent/reflective glass.

This is required for the Stable Diffusion branch: ControlNet needs vehicles
to appear as solid blobs in semantic/instance maps, otherwise transparent
glass breaks the conditioning.

Workflow:
  1. Backup the current Glass.uasset of CARLA into assets/Carla/Glass.uasset.original
     (if not already backed up).
  2. Overwrite CARLA's Glass.uasset with assets/Carla/Glass.uasset (custom).
  3. Launch CARLA.
  4. On exit (clean or crash), restore the original Glass.uasset.

Reads from (project root):
    utils/config.py
        CARLA_INSTALLATION_PATH   - root of the CARLA installation
        CARLA_GLASS_PATH          - relative path of Glass.uasset inside CARLA
        ASSET_PATH                - folder containing the custom assets

Writes to:
    The configured CARLA Glass.uasset is temporarily replaced while CARLA runs.
    The original asset is restored when this script exits.

Parameters:
    None.

Usage:
    python 3_generate_simulation_data/3C_sd_setup_carla.py
"""

import os
import sys
import subprocess


# =======================
# PATH SETUP
# =======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )

if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


# =======================
# IMPORTS
# =======================

from utils.config import (
    CARLA_INSTALLATION_PATH,
    CARLA_GLASS_PATH,
    ASSET_PATH,
)


# =======================
# HELPERS
# =======================

def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# =======================
# MAIN
# =======================

def main():
    # ---------- Validate config ----------
    if not CARLA_INSTALLATION_PATH:
        raise ValueError(
            "CARLA_INSTALLATION_PATH is not set. Please set it in utils/config.py."
        )
    if not os.path.exists(CARLA_INSTALLATION_PATH):
        raise ValueError(
            f"CARLA installation path does not exist: {CARLA_INSTALLATION_PATH}"
        )

    carla_script_path = os.path.join(CARLA_INSTALLATION_PATH, "CarlaUE4.sh")
    if not os.path.exists(carla_script_path):
        raise ValueError(
            f"CarlaUE4.sh does not exist at: {carla_script_path}"
        )

    if not CARLA_GLASS_PATH:
        raise ValueError(
            "CARLA_GLASS_PATH is not set. Please set it in utils/config.py."
        )

    glass_file_path = os.path.join(CARLA_INSTALLATION_PATH, CARLA_GLASS_PATH)
    if not os.path.exists(glass_file_path):
        raise ValueError(
            f"CARLA Glass file does not exist: {glass_file_path}\n"
            f"Check CARLA_GLASS_PATH in utils/config.py."
        )

    glass_file_basename = os.path.basename(CARLA_GLASS_PATH)
    override_glass_file_path = os.path.join(
        ASSET_PATH, "Carla", glass_file_basename
    )
    backup_path = os.path.join(
        ASSET_PATH, "Carla", f"{glass_file_basename}.original"
    )

    if not os.path.exists(override_glass_file_path):
        raise ValueError(
            f"Custom Glass override not found: {override_glass_file_path}\n"
            f"Make sure assets/Carla/{glass_file_basename} exists."
        )

    # ---------- Snapshot the current CARLA glass file ----------
    current_glass_data = read_bytes(glass_file_path)

    if os.path.exists(backup_path):
        # If we already have a backup, keep using IT as the source of truth
        # for the "original" — this protects us against accidentally backing
        # up an already-modified glass file from a previous interrupted run.
        print(f"Existing backup found, using it as original: {backup_path}")
        original_glass_data = read_bytes(backup_path)
    else:
        # First run: snapshot current glass as the canonical original.
        print(f"No backup found. Saving current glass as original to: {backup_path}")
        write_bytes(backup_path, current_glass_data)
        original_glass_data = current_glass_data

    # ---------- Overwrite CARLA glass with the custom one ----------
    print(f"Overriding CARLA glass with custom version from: {override_glass_file_path}")
    override_glass_data = read_bytes(override_glass_file_path)
    write_bytes(glass_file_path, override_glass_data)

    # ---------- Launch CARLA (Low quality) ----------
    try:
        print("Starting CARLA with Low Quality...")
        result = subprocess.run(
            [carla_script_path, "-quality-level=Low"],
            check=True,
        )
        print("CARLA finished with return code:", result.returncode)
    except subprocess.CalledProcessError as e:
        print("CARLA script failed with error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)
    finally:
        # Always restore the original glass file, even on crash or Ctrl+C.
        print("Restoring original CARLA glass file...")
        write_bytes(glass_file_path, original_glass_data)
        print("Glass file restored.")


if __name__ == "__main__":
    main()