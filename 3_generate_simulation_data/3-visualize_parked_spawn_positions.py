import carla
import argparse
import json
import os
import time
import random
import numpy as np
from dotenv import load_dotenv

# Import your existing config/utils
from config import CARLA_IP, CARLA_PORT, MAPS_FOLDER_NAME
from utils.save_data import get_map_data
# LAT0, LON0 no longer needed - using OSM center as origin
# We need these specific functions to calculate the offset correctly
from utils.carla_simulator import (
    generate_world_map,
    get_filtered_vehicle_blueprints,
    get_osm_center,
    get_xodr_geo_reference,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Spawn Positions from JSON")
    parser.add_argument("--map", type=str, required=True, help="Name of the map folder (inside maps/)")
    parser.add_argument("--json", type=str, required=True, help="Path to the generated vehicle_data.json")
    parser.add_argument("--offset-x", type=float, default=0.0, help="Manual X offset adjustment (meters)")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Manual Y offset adjustment (meters)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Connect to CARLA
    print(f"🔌 Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(20.0)

    # 2. Load Map Data
    print(f"🗺️  Loading Map: {args.map}...")
    map_data = get_map_data(args.map, (100, 100))

    if map_data is None:
        print(f"❌ CRITICAL ERROR: Could not load map '{args.map}'. Check spelling.")
        return

    # 3. Get OSM center and check XODR projection
    osm_file = os.path.join(MAPS_FOLDER_NAME, args.map, "map.osm")
    osm_center_lat, osm_center_lon = get_osm_center(osm_file)
    print(f"🔄 OSM center: ({osm_center_lat:.6f}, {osm_center_lon:.6f})")

    # Check what projection the XODR uses
    xodr_proj = get_xodr_geo_reference(map_data["xodr_data"])
    if xodr_proj:
        print(f"📍 XODR projection: {xodr_proj}")

    # 4. Get offset from vehicle_data.json (calculated by create_vehicle_data_from_centroids.py)
    # This offset compensates for the difference between OSM center and XODR center
    # and cancels out the size/2 translation in get_translation_vector
    vehicle_data = map_data.get("vehicle_data", {})
    json_offset = vehicle_data.get("offset", {"x": 0.0, "y": 0.0, "heading": 0.0})

    # Allow command-line args to OVERRIDE or ADD to the JSON offset
    offset_data = {
        "x": json_offset.get("x", 0.0) + args.offset_x,
        "y": json_offset.get("y", 0.0) + args.offset_y,
        "heading": json_offset.get("heading", 0.0)
    }

    print(f"📐 Offset from JSON: x={json_offset.get('x', 0):.2f}, y={json_offset.get('y', 0):.2f}")
    if args.offset_x != 0 or args.offset_y != 0:
        print(f"📐 Manual adjustment: x={args.offset_x}, y={args.offset_y}")
    print(f"📐 Final offset: x={offset_data['x']:.2f}, y={offset_data['y']:.2f}")



    # Load the OpenDRIVE map
    world = generate_world_map(client, map_data["xodr_data"])
    blueprint_library = world.get_blueprint_library()

    # 3. Load Generated Spawn Positions
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON file not found: {args.json}")

    print(f"📂 Loading Spawn Data from: {args.json}")
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    spawn_list = data.get("spawn_positions", [])
    print(f"found {len(spawn_list)} positions to spawn.")

    # 4. Filter Vehicles
    vehicle_bp_list = get_filtered_vehicle_blueprints(world)

    spawned_actors = []

    try:
        print("🚗 Spawning vehicles...")
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        for item in spawn_list:
            # Get the Raw Point
            raw_loc = item.get("start") 
            heading = item.get("heading", 0.0)

            if raw_loc is None: continue

            # --- APPLY OFFSET TRANSFORMATION ---
            # This moves the point from "Raw Map Coordinates" to "CARLA World Coordinates"
            # using the offset we loaded from the map data.
            final_pos = raw_loc

            # CARLA Transform (x, y, z)
            # z + 0.5 to prevent clipping
            carla_loc = carla.Location(x=final_pos[0], y=final_pos[1], z=final_pos[2] + 0.5)
            
            # Use heading directly - no rotation correction needed
            # (coordinates and headings are in same projection space as XODR roads)
            carla_rot = carla.Rotation(pitch=0.0, yaw=heading, roll=0.0)
            
            transform = carla.Transform(carla_loc, carla_rot)

            bp = random.choice(vehicle_bp_list)
            vehicle = world.try_spawn_actor(bp, transform)
            
            if vehicle:
                vehicle.set_simulate_physics(False) 
                spawned_actors.append(vehicle)

        print(f"✅ Successfully spawned {len(spawned_actors)} vehicles.")

        if spawned_actors:
            spectator = world.get_spectator()
            first_car_trans = spawned_actors[0].get_transform()
            spectator.set_transform(carla.Transform(
                first_car_trans.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            ))

        print("Press Ctrl+C to exit and destroy actors.")
        
        while True:
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\nCleaning up...")
    finally:
        print(f"🗑️ Destroying {len(spawned_actors)} actors...")
        client.apply_batch([carla.command.DestroyActor(x) for x in spawned_actors])
        print("Done.")

if __name__ == "__main__":
    main()