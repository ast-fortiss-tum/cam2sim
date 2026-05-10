import random
import carla
from queue import Queue
import os
import json
import ast
import xml.etree.ElementTree as ET
import math
import numpy as np
from PIL import Image
import math
from utils.save_data import get_dataset_data, get_map_data
# from create_vehicle_data_from_centroids import odom_xy_to_latlon  # se è in un file diverso, dimmelo e te lo sistemo
from utils.config import VERTEX_DISTANCE, MAX_ROAD_LENGTH, WALL_HEIGHT, EXTRA_WIDTH, ROTATION_DEGREES, CAR_SPACING, \
    FORWARDS_PARKING_PROBABILITY

def load_instance_color_map(filepath):
    mapped_colors = {}
    if not os.path.exists(filepath):
        print(f" Warning: {filepath} not found. No color remapping will occur.")
        return mapped_colors
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        for key_str, val_str in raw_data.items():
            #  REPLACE NULL VALUES WITH 128,128,128
            if val_str is None:
                rgb_val = [128, 128, 128]
            else:
                rgb_val = [int(x) for x in val_str.split(',')]
            
            rgb_key = ast.literal_eval(key_str)
            mapped_colors[rgb_key] = rgb_val
        print(f" Loaded {len(mapped_colors)} color mappings.")
    except Exception as e:
        print(f" Error loading color map: {e}")
    return mapped_colors

def process_instance_map_fixed(inst_raw_data, global_color_map):
    bgra = np.frombuffer(inst_raw_data.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((inst_raw_data.height, inst_raw_data.width, 4))
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] 
    
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    output = np.zeros((inst_raw_data.height, inst_raw_data.width, 3), dtype=np.uint8)
    
    out_r = (g_id * 37 + b_id * 13) % 200 + 55
    out_g = (g_id * 17 + b_id * 43) % 200 + 55
    out_b = (g_id * 29 + b_id * 53) % 200 + 55
    
    output[vehicle_mask, 0] = out_r[vehicle_mask]
    output[vehicle_mask, 1] = out_g[vehicle_mask]
    output[vehicle_mask, 2] = out_b[vehicle_mask]
    
    if global_color_map:
        for old_rgb, new_rgb in global_color_map.items():
            source = np.array(old_rgb, dtype=np.uint8)
            target = np.array(new_rgb, dtype=np.uint8)
            mask = np.all(output == source, axis=-1)
            if np.any(mask):
                output[mask] = target

    return Image.fromarray(output)

def cleanup_old_sensors(hero_vehicle):
    print(" Cleaning up old sensors on Hero...")
    world = hero_vehicle.get_world()
    attached = [x for x in world.get_actors() if x.parent and x.parent.id == hero_vehicle.id]
    count = 0
    for sensor in attached:
        if 'sensor' in sensor.type_id:
            sensor.destroy()
            count += 1
    print(f" Removed {count} old sensors.")
    
def get_map_size(xodr_data): #search for header in xodr
    root = ET.fromstring(xodr_data)
    header = root.find("header")

    north = float(header.attrib.get("north", "0"))
    south = float(header.attrib.get("south", "0"))
    east = float(header.attrib.get("east", "0"))
    west = float(header.attrib.get("west", "0"))

    width = east - west
    height = north - south

    return width, height

def get_xodr_center(xodr_data):
    """
    Get the center coordinates of the XODR map.
    Returns (center_x, center_y) in XODR local coordinates.
    """
    root = ET.fromstring(xodr_data)
    header = root.find("header")

    north = float(header.attrib.get("north", "0"))
    south = float(header.attrib.get("south", "0"))
    east = float(header.attrib.get("east", "0"))
    west = float(header.attrib.get("west", "0"))

    center_x = (east + west) / 2
    center_y = (north + south) / 2

    return center_x, center_y

def get_xodr_geo_offset(xodr_data):
    """
    Extract the geographic offset from XODR header.
    This tells us how XODR local coords relate to the projection coords.

    Returns: (offset_x, offset_y, hdg) or None if not found
    """
    root = ET.fromstring(xodr_data)
    header = root.find("header")
    offset_elem = header.find("offset")

    if offset_elem is not None:
        return (
            float(offset_elem.attrib.get("x", "0")),
            float(offset_elem.attrib.get("y", "0")),
            float(offset_elem.attrib.get("hdg", "0"))
        )
    return None

def get_xodr_geo_reference(xodr_data):
    """
    Extract the geoReference projection string from XODR header.
    This tells us what projection CARLA's Osm2Odr used.
    """
    root = ET.fromstring(xodr_data)
    header = root.find("header")
    geo_ref = header.find("geoReference")

    if geo_ref is not None and geo_ref.text:
        proj_string = geo_ref.text.strip()
        print(f"[DEBUG] XODR geoReference: {proj_string}")
        return proj_string
    return None


def get_xodr_projection_params(xodr_data):
    """
    Extract both geoReference and offset from XODR header.
    Returns a dict with all parameters needed for coordinate conversion.

    Returns:
        dict with keys:
            - geo_reference: projection string (e.g., "+proj=tmerc")
            - offset: tuple (x, y) offset values
            - bounds: dict with north, south, east, west
            - center_local: tuple (x, y) center in XODR local coords
    """
    root = ET.fromstring(xodr_data)
    header = root.find("header")

    # Get geoReference
    geo_ref = header.find("geoReference")
    proj_string = "+proj=tmerc"  # default
    if geo_ref is not None and geo_ref.text:
        proj_string = geo_ref.text.strip()

    # Get offset
    offset_elem = header.find("offset")
    offset_x, offset_y = 0.0, 0.0
    if offset_elem is not None:
        offset_x = float(offset_elem.attrib.get("x", "0"))
        offset_y = float(offset_elem.attrib.get("y", "0"))

    # Get bounds
    north = float(header.attrib.get("north", "0"))
    south = float(header.attrib.get("south", "0"))
    east = float(header.attrib.get("east", "0"))
    west = float(header.attrib.get("west", "0"))

    # Calculate XODR center in local coordinates
    center_x = (east + west) / 2
    center_y = (north + south) / 2

    print(f"[DEBUG] XODR projection params:")
    print(f"        geoReference: {proj_string}")
    print(f"        offset: ({offset_x:.2f}, {offset_y:.2f})")
    print(f"        bounds: N={north:.2f} S={south:.2f} E={east:.2f} W={west:.2f}")
    print(f"        center (local): ({center_x:.2f}, {center_y:.2f})")

    return {
        "geo_reference": proj_string,
        "offset": (offset_x, offset_y),
        "bounds": {
            "north": north,
            "south": south,
            "east": east,
            "west": west
        },
        "center_local": (center_x, center_y)
    }


def calculate_osm_to_xodr_offset(osm_center_lat, osm_center_lon, xodr_params):
    """
    Calculate the offset between OSM center and XODR center.

    This offset is needed because:
    - OSM contains all nodes (buildings, POIs, etc.)
    - XODR only contains roads
    - Their bounding boxes (and thus centers) are different!

    Returns:
        tuple (offset_x, offset_y) in meters to add to coordinates
    """
    from pyproj import Transformer

    # Get XODR projection (default is +proj=tmerc with lon_0=0)
    proj_string = xodr_params["geo_reference"].strip()
    if proj_string == "+proj=tmerc":
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    # Project OSM center to XODR projection space
    osm_proj_x, osm_proj_y = transformer.transform(osm_center_lon, osm_center_lat)

    # Convert to XODR local coordinates
    xodr_offset = xodr_params["offset"]
    osm_local_x = osm_proj_x + xodr_offset[0]
    osm_local_y = osm_proj_y + xodr_offset[1]

    # Get XODR center
    xodr_center = xodr_params["center_local"]

    # Calculate difference
    diff_x = xodr_center[0] - osm_local_x
    diff_y = xodr_center[1] - osm_local_y

    print(f"[DEBUG] OSM center in XODR local: ({osm_local_x:.2f}, {osm_local_y:.2f})")
    print(f"[DEBUG] XODR center: ({xodr_center[0]:.2f}, {xodr_center[1]:.2f})")
    print(f"[DEBUG] OSM-to-XODR offset: ({diff_x:.2f}m, {diff_y:.2f}m)")

    return (diff_x, diff_y)

def calculate_grid_convergence(lat, lon, central_meridian=0):
    """
    Calculate grid convergence for transverse Mercator projection.
    Grid convergence is the angle between grid north and true north.

    For CARLA's +proj=tmerc with lon_0=0, this causes a rotation
    that increases with distance from the central meridian.

    Returns: convergence in degrees (positive = grid north is east of true north)
    """
    import math
    delta_lon = math.radians(lon - central_meridian)
    lat_rad = math.radians(lat)
    convergence_rad = math.atan(math.tan(delta_lon) * math.sin(lat_rad))
    return math.degrees(convergence_rad)

def get_osm_center(osm_file_path):
    """
    Parse OSM file to find the bounding box center.
    Returns (center_lat, center_lon)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(osm_file_path)
    root = tree.getroot()

    # Find bounds element
    bounds = root.find("bounds")
    if bounds is not None:
        minlat = float(bounds.attrib["minlat"])
        maxlat = float(bounds.attrib["maxlat"])
        minlon = float(bounds.attrib["minlon"])
        maxlon = float(bounds.attrib["maxlon"])

        center_lat = (minlat + maxlat) / 2
        center_lon = (minlon + maxlon) / 2
        return center_lat, center_lon

    # Fallback: compute from all nodes
    lats, lons = [], []
    for node in root.findall(".//node"):
        lats.append(float(node.attrib["lat"]))
        lons.append(float(node.attrib["lon"]))

    if lats and lons:
        return (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2

    return None, None

def calculate_auto_offset_from_osm(osm_file_path, hero_lat, hero_lon):
    """
    Calculate the CARLA offset based on where the hero car (LAT0/LON0) is
    relative to the OSM map center.

    The OSM center becomes approximately (size_x/2, size_y/2) in XODR coords.
    The offset from hero to OSM center gives us the CARLA offset needed.

    Parameters:
        osm_file_path: Path to the map.osm file
        hero_lat, hero_lon: The hero car's first position (LAT0, LON0)

    Returns: dict with x, y offset values
    """
    from pyproj import Geod

    # Get OSM center
    osm_center_lat, osm_center_lon = get_osm_center(osm_file_path)

    if osm_center_lat is None:
        print("Could not find OSM center, using zero offset")
        return {"x": 0.0, "y": 0.0, "heading": 0.0}

    print(f"[AUTO-OFFSET] OSM center: ({osm_center_lat:.6f}, {osm_center_lon:.6f})")
    print(f"[AUTO-OFFSET] Hero car:   ({hero_lat:.6f}, {hero_lon:.6f})")

    # Calculate distance and azimuth from OSM center to hero car
    geod = Geod(ellps="WGS84")
    azimuth, _, distance = geod.inv(osm_center_lon, osm_center_lat, hero_lon, hero_lat)

    # Convert to local XY offset
    # Note: Y is negated to match CARLA's coordinate system (Y-axis flipped)
    azimuth_rad = math.radians(azimuth)
    offset_x = distance * math.sin(azimuth_rad)   # East component
    offset_y = -distance * math.cos(azimuth_rad)  # North component NEGATED for CARLA

    print(f"[AUTO-OFFSET] Hero is {distance:.2f}m from OSM center")
    print(f"[AUTO-OFFSET] Offset: x={offset_x:.2f}m, y={offset_y:.2f}m (Y negated for CARLA)")

    # The hero car is at (offset_x, offset_y) relative to OSM center
    # In CARLA/XODR, the OSM center is at approximately (0, 0) or the map center
    # So we need to translate by this offset

    return {
        "x": offset_x,
        "y": offset_y,
        "heading": 0.0  # Will be set based on ROTATION_DEGREES
    }



def update_synchronous_mode(world, tm, bool, fps = 1):
    settings = world.get_settings()
    if bool:
        settings.fixed_delta_seconds = 1 / fps
    settings.synchronous_mode = bool
    world.apply_settings(settings)

    tm.set_synchronous_mode(bool)

def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    # 0,0,0 → 128,64,128
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    # 70,130,180 → 0,0,0
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)

def carla_image_to_pil(image) -> Image.Image:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
    return Image.fromarray(rgb_array)

def remove_sensor(sensor):
    if sensor is not None:
        sensor.stop()
        sensor.destroy()

def delete_all_vehicles(world):
    actors = world.get_actors().filter('vehicle.*')
    for actor in actors:
        actor.destroy()

def get_spectator_transform(world):
    spectator = world.get_spectator()
    transform = spectator.get_transform()
    return transform

def set_spectator_transform(world, transform):
    spectator = world.get_spectator()
    spectator.set_transform(transform)

def generate_world_map(client, xodr_data):
    return client.generate_opendrive_world(
        xodr_data,
        carla.OpendriveGenerationParameters(
            vertex_distance=VERTEX_DISTANCE,
            max_road_length=MAX_ROAD_LENGTH,
            wall_height=WALL_HEIGHT,
            additional_width=EXTRA_WIDTH,
            smooth_junctions=True,
            enable_mesh_visibility=True
        )
    )

def spawn_parked_cars_front_of_hero(world, vehicle_library, entry, hero_car_data):
    """
    Spawn a car temporarily in front of the hero to detect its instance color.
    The car will be teleported to its real position later.

    Parameters:
        hero_car_data: Either a dict with 'position' and 'heading' keys,
                       or a carla.Vehicle actor to get position from
    """
    spawned_actors = []

    # 1. Extract color from JSON
    target_color = None
    if "color" in entry and entry["color"]:
        try:
            target_color = [int(x) for x in entry["color"].split(',')]
        except:
            pass
    if target_color is None:
        target_color = [128, 128, 128]

    # 2. Get hero position and calculate spawn point in front
    if hasattr(hero_car_data, 'get_transform'):
        # It's a CARLA actor - get position directly
        hero_tf = hero_car_data.get_transform()
        hero_x = hero_tf.location.x
        hero_y = hero_tf.location.y
        hero_yaw = hero_tf.rotation.yaw
    else:
        # It's a dict from vehicle_data.json - need to transform
        hero_raw = hero_car_data.get("position", [0, 0, 0])
        hero_pos = hero_raw
        hero_x = hero_pos[0]
        hero_y = hero_pos[1]
        hero_yaw = hero_car_data.get("heading", 0)

    # Spawn 15m in front of hero (in the direction hero is facing)
    spawn_distance = 15.0
    yaw_rad = math.radians(hero_yaw)
    spawn_x = hero_x + spawn_distance * math.cos(yaw_rad)
    spawn_y = hero_y + spawn_distance * math.sin(yaw_rad)
    spawn_z = 0.5

    heading = entry["heading"]

    # 3. Choose Blueprint and Set Color
    blueprint = random.choice(vehicle_library)
    if blueprint.has_attribute('color'):
        color_str = f"{target_color[0]},{target_color[1]},{target_color[2]}"
        blueprint.set_attribute('color', color_str)

    veh_heading = heading
    if entry["mode"].strip() == "perpendicular" and random.random() < 0.5:
        veh_heading = (veh_heading + 180) % 360

    transform = carla.Transform(
        carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
        carla.Rotation(yaw=veh_heading)
    )

    try:
        actor = world.spawn_actor(blueprint, transform)
        actor.set_simulate_physics(False)
        spawned_actors.append(actor)
    except RuntimeError:
        pass

    return spawned_actors

def get_unique_colors_from_sensor(inst_raw_data):
    """Extracts unique RGB colors from the instance sensor data."""
    bgra = np.frombuffer(inst_raw_data.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((inst_raw_data.height, inst_raw_data.width, 4))
    
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] 
    
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    
    valid_b = b_id[vehicle_mask]
    valid_g = g_id[vehicle_mask]
    
    if len(valid_b) == 0:
        return set()

    out_r = (valid_g * 37 + valid_b * 13) % 200 + 55
    out_g = (valid_g * 17 + valid_b * 43) % 200 + 55
    out_b = (valid_g * 29 + valid_b * 53) % 200 + 55
    
    colors = np.stack((out_r, out_g, out_b), axis=-1)
    unique_colors = np.unique(colors, axis=0)
    
    return set(map(tuple, unique_colors))



def flush_all_queues(seg_q, rgb_q, inst_q):
    """Helper to aggressively empty queues to prevent memory buildup"""
    while not seg_q.empty(): seg_q.get()
    while not rgb_q.empty(): rgb_q.get()
    while not inst_q.empty(): inst_q.get()


def get_inverse_transform(carla_transform):
    # 1. Extract CARLA data
    c_loc = carla_transform.location
    c_rot = carla_transform.rotation

    # 2. Revert the manual Z offset
    adjusted_z = c_loc.z - 0.5

    # 3. Return directly (no transformation needed)
    return {
        "location": {
            "x": c_loc.x,
            "y": c_loc.y,
            "z": adjusted_z
        },
        "rotation": {
            "yaw": c_rot.yaw % 360,
            "pitch": c_rot.pitch,
            "roll": c_rot.roll
        }
    }

def get_filtered_vehicle_blueprints(world):
    """
    Returns a list of vehicle blueprints filtering out:
    - 2-wheeled vehicles (bikes, motorcycles)
    - Heavy machinery (trucks, buses)
    - Emergency vehicles
    - Specific excluded models (mustang, etc)
    """
    blueprint_library = world.get_blueprint_library()
    all_vehicles = blueprint_library.filter('vehicle.*')
    
    # The comprehensive list from Script 1
    forbidden_keywords = [
        "mustang", "police", "impala", "carlacola", "cybertruck", "t2", "sprinter", 
        "firetruck", "ambulance", "bus", "truck", "van", "bingle", "microlino", 
        "vespa", "yamaha", "kawasaki", "harley", "bh", "gazelle", "diamondback", 
        "crossbike", "century", "omafiets", "low_rider", "ninja", "zx125", "yzf", 
        "fuso", "rosa", "isetta", "tesla", "audi", "rubicon", "lincoln"
    ]
    
    vehicle_library = []
    
    for bp in all_vehicles:
        # Filter 1: Must have 4 wheels
        if bp.has_attribute('number_of_wheels'):
            if int(bp.get_attribute('number_of_wheels')) != 4: continue
            
        # Filter 2: Must be Generation 1 (From Script 1 logic)
        if bp.has_attribute('generation'):
            if int(bp.get_attribute('generation')) != 1: continue
            
        # Filter 3: Forbidden Keywords
        if any(k in bp.id.lower() for k in forbidden_keywords): continue
        
        vehicle_library.append(bp)
        
    return vehicle_library

def sensor_callback(sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)

def spawn_sensor(bp_lib,type,w,h,fov,world,sensor_tf,hero_vehicle):
    sensor_queue = Queue()
    sensor_bp = bp_lib.find(type)
    sensor_bp.set_attribute('image_size_x', str(w))
    sensor_bp.set_attribute('image_size_y', str(h))
    sensor_bp.set_attribute('fov', fov)
    sensor_sensor = world.spawn_actor(sensor_bp, sensor_tf, attach_to=hero_vehicle)
    sensor_sensor.listen(lambda d: sensor_callback(d, sensor_queue))
    return sensor_sensor, sensor_queue