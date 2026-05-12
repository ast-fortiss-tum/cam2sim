#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess


# =======================
# PATH SETUP
# =======================

# Folder where this script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local utils folder expected next to this script:
# <script_folder>/utils/
LOCAL_UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")

if not os.path.isdir(LOCAL_UTILS_DIR):
    raise FileNotFoundError(
        f"Expected utils folder next to this script, but not found: {LOCAL_UTILS_DIR}"
    )

# Force Python to import from this script folder first.
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)


# =======================
# IMPORTS
# =======================

from utils.config import CARLA_INSTALLATION_PATH


# =======================
# MAIN
# =======================

def main():
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

    try:
        print("Starting CARLA...")

        # Normal start
        # result = subprocess.run([carla_script_path], check=True)

        # Low quality start, if needed:
        result = subprocess.run([carla_script_path, "-quality-level=Low"], check=True)

        print("CARLA finished with return code:", result.returncode)

    except subprocess.CalledProcessError as e:
        print("CARLA script failed with error:", e)

    except Exception as e:
        print("An unexpected error occurred:", e)


if __name__ == "__main__":
    main()