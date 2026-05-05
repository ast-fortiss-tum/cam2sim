import json
import os
import sys

# =======================
# PATH SETUP
# =======================
# This makes imports work even if the script is launched from another directory.
# The utils/ folder is expected to be in the same folder as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# =======================
# IMPORTS
# =======================
from utils.map_data import get_street_data, fetch_osm_data

from utils.save_data import (
    create_map_folders,
    save_vehicle_data,
    save_map_data,
    save_osm_data,
    get_existing_osm_data,
)

from utils.other import ensure_carla_functionality
from utils.plotting import create_plot, show_plot, get_output


# =======================
# CONFIGURATION
# =======================

# Address used to fetch the OSM map.
ADDRESS = "Guerickestraße, Alte Heide, Munich"

# Required map name.
# This name defines the output folder:
# data/maps/<MAP_NAME>/
MAP_NAME = "reference_bag"

# Output root.
# This is relative to the directory where you run the script.
MAP_OUTPUT_ROOT = os.path.join("data", "maps")

# Radius around the address used for OSM data extraction.
DIST = 200

# Map creation mode.
# Use:
# - "manual" to select hero car and parking areas manually
# - "auto" to select only the hero car and inject parking areas later from clusters
MODE = "auto"

# If True, CARLA functionality is not checked.
# If False, the script checks that CARLA functionality is available.
NO_CARLA = False

# If True, load previously saved OSM data from disk.
# If False, fetch fresh OSM data from ADDRESS.
SKIP_FETCH = False


# =======================
# HELPERS
# =======================

def load_existing_vehicle_data(folder_name):
    """
    Load existing vehicle data from the selected map folder.

    This is used only to preserve previously saved offset values when re-running
    the script with the same distance.
    """
    vehicle_data_path = os.path.join(folder_name, "vehicle_data.json")

    if not os.path.exists(vehicle_data_path):
        return None

    with open(vehicle_data_path, "r") as f:
        return json.load(f)


# =======================
# MAIN SCRIPT
# =======================

def main():
    # 1. Validate config
    if not MAP_NAME or not isinstance(MAP_NAME, str):
        raise ValueError("MAP_NAME must be a non-empty string.")

    if MODE not in {"manual", "auto"}:
        raise ValueError("MODE must be either 'manual' or 'auto'.")

    # 2. Set up output paths
    maps_root = os.path.abspath(MAP_OUTPUT_ROOT)
    folder_name = os.path.join(maps_root, MAP_NAME)

    print("=" * 70)
    print("MAP DATA GENERATION")
    print("=" * 70)
    print(f"Address: {ADDRESS}")
    print(f"Map name: {MAP_NAME}")
    print(f"Distance: {DIST}")
    print(f"Mode: {MODE}")
    print(f"No CARLA: {NO_CARLA}")
    print(f"Skip OSM fetch: {SKIP_FETCH}")
    print(f"Output folder: {folder_name}")

    # 3. Load existing vehicle data to preserve offsets when re-running
    existing_vehicle_data = load_existing_vehicle_data(folder_name)

    # 4. Check CARLA functionality if requested
    if not NO_CARLA:
        ensure_carla_functionality()

    # 5. Fetch or load OSM data
    if not SKIP_FETCH:
        print(f"\nFetching OSM data for: {ADDRESS}")
        osm_data = get_street_data(ADDRESS, dist=DIST)
    else:
        print(f"\nLoading existing OSM data from: {folder_name}")
        osm_data = get_existing_osm_data(folder_name)

    # 6. Create map folders
    create_map_folders(folder_name)

    # 7. Save OSM data if freshly fetched
    if not SKIP_FETCH:
        save_osm_data(folder_name, osm_data)

    # 8. Process map geometry
    print("\nProcessing map geometry: nodes, edges, and buildings")
    graph, edges, buildings = fetch_osm_data(folder_name)

    # 9. Open GUI
    print(f"\nMode: {MODE.upper()}. Opening GUI.")

    if MODE == "manual":
        print("Instructions: Select the hero car position and parking areas.")
    else:
        print("Instructions: Select only the hero car position.")
        print("Parking areas will be injected later from cluster data.")
        print("Click the start position and then close the window.")

    create_plot(buildings, edges, ADDRESS)
    show_plot()

    # 10. Retrieve GUI output
    output_json = get_output(DIST)

    # 11. Preserve previous offset values if possible
    if existing_vehicle_data is not None:
        if (
            existing_vehicle_data.get("dist") == DIST
            and existing_vehicle_data.get("offset") is not None
            and output_json is not None
        ):
            output_json["offset"] = existing_vehicle_data["offset"]
            print("Copied offset values from existing vehicle data.")

    # 12. Save map and vehicle data
    if output_json is not None:
        save_map_data(folder_name, osm_data, NO_CARLA)
        save_vehicle_data(folder_name, output_json)

        print(f"\n[{MODE.upper()}] Map and vehicle data saved to:")
        print(folder_name)

        if MODE != "manual":
            print(
                "JSON created with hero car data. "
                "Run create_vehicle_data_from_centroids.py to add clusters."
            )
    else:
        print("\nPlot closed without selection. Nothing saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()