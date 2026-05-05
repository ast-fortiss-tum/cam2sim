import os

import math
import osmnx as ox
import requests
from pyproj import Geod
from shapely.geometry import LineString, Point
import geopandas as gpd

from utils.debug import debug_loading_osm_data

geod = Geod(ellps="WGS84")

def get_origin_lat_lon(edges, address):
    """Convert a Point to (lat, lon) tuple."""
    center_point = edges.union_all().centroid
    origin_lat = center_point.y
    origin_lon = center_point.x
    return origin_lat, origin_lon

def get_street_data(address, dist=500):
    debug_loading_osm_data(address)
    point = ox.geocode(address)

    west, south, east, north = ox.utils_geo.bbox_from_point(point, dist=dist)

    query = f"""
    [out:xml][timeout:25];
    (
      way["building"]({south},{west},{north},{east});
      way["highway"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
      relation["highway"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """

    headers = {
        "User-Agent": "cam2sim-replication/0.1",
        "Accept": "*/*",
    }
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={'data': query},
        headers=headers,
        timeout=60,
    )

    if not response.ok:
        raise ValueError(
            f"Overpass-Request failed (status {response.status_code}): "
            f"{response.text[:500]}"
        )

    return response.content

def generate_spawn_gdf(edges, offset=4.0, offset_left=6.0, offset_right=5.2, override = False):
    lines = []
    sides = []
    for idx, row in edges.iterrows():
        geom = row.geometry

        if not isinstance(geom, LineString):
            continue

        # Parking tags (can be 'yes', 'parallel', 'diagonal', etc.)
        park_tags = {
            "parking:left": str(row.get("parking:left", "")).lower(),
            "parking:right": str(row.get("parking:right", "")).lower(),
            "parking:both": str(row.get("parking:both", "")).lower(),
            "parking:lane:left": str(row.get("parking:lane:left", "")).lower(),
            "parking:lane:right": str(row.get("parking:lane:right", "")).lower(),
            "parking:lane:both": str(row.get("parking:lane:both", "")).lower(),
        }

        # set park_accept to accept lane parking
        park_accept = {"yes", "parallel", "diagonal", "lane", "parallel:both", "diagonal:both", "lane:both", "on_street", "on_kerb"}

        left_allowed = (
                park_tags["parking:left"] in park_accept or
                park_tags["parking:lane:left"] in park_accept or
                park_tags["parking:both"] in park_accept or
                park_tags["parking:lane:both"] in park_accept or
                override
        )

        # Controllare se è consentito parcheggiare sulla destra
        right_allowed = (
                park_tags["parking:right"] in park_accept or
                park_tags["parking:lane:right"] in park_accept or
                park_tags["parking:both"] in park_accept or
                park_tags["parking:lane:both"] in park_accept or
                override
        )

        # Create parallel offsets conditionally
        if left_allowed:
            left = geom.parallel_offset(offset_left / 111.32e3, 'left', join_style=2)
            if isinstance(left, LineString):
                lines.append(left)
                sides.append("left")

        if right_allowed:
            right = geom.parallel_offset(offset_right / 111.32e3, 'right', join_style=2)
            if isinstance(right, LineString):
                lines.append(right)
                sides.append("right")

    return gpd.GeoDataFrame({'geometry': lines, 'side': sides}, crs="EPSG:4326")

def get_heading(start_lat, start_lon, end_lat, end_lon):
    """
    Calculate heading (yaw) for CARLA.
    Uses local projection centered on start point.

    CARLA yaw convention:
    - 0° = positive X (East)
    - 90° = positive Y (South in CARLA, since Y is flipped)
    - 180° = negative X (West)
    - 270° = negative Y (North in CARLA)
    """
    from pyproj import Transformer
    import math

    # Project centered on start point
    proj_string = f"+proj=tmerc +lat_0={start_lat} +lon_0={start_lon} +k=1 +x_0=0 +y_0=0 +datum=WGS84"
    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    # Start is at origin, get end position in projected coords
    # In projection: +X = East, +Y = North
    end_x, end_y = transformer.transform(end_lon, end_lat)

    # Convert to CARLA coords: CARLA +Y = South = -proj_Y
    # Direction in CARLA: (end_x, -end_y)
    # CARLA yaw = atan2(carla_dy, carla_dx) = atan2(-end_y, end_x)
    heading_rad = math.atan2(-end_y, end_x)
    heading_deg = math.degrees(heading_rad)

    return (heading_deg + 360) % 360

def latlon_to_carla(origin_lat, origin_lon, lat, lon, rotation_deg=0.0):
    """
    Convert lat/lon to local Cartesian coordinates for CARLA.
    Uses transverse Mercator projection centered on the origin point,
    matching CARLA's Osm2Odr behavior.

    Parameters:
        origin_lat, origin_lon: The reference point (typically OSM center)
        lat, lon: The point to convert
        rotation_deg: Grid convergence correction in degrees.
                      This accounts for the difference between the projection
                      grid north and the XODR grid north.
                      For Munich with XODR using +proj=tmerc (lon_0=0),
                      this is approximately -8.69 degrees.

    Returns:
        (x, y, z) tuple in CARLA world coordinates
    """
    from pyproj import Transformer
    import math

    # Use origin (OSM center) as projection center
    proj_string = f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +k=1 +x_0=0 +y_0=0 +datum=WGS84"
    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    # With projection centered on origin, origin transforms to (0, 0)
    target_x, target_y = transformer.transform(lon, lat)

    # Apply grid convergence rotation if specified
    # This rotates the coordinates to match the XODR grid orientation
    if abs(rotation_deg) > 1e-6:
        theta = math.radians(rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated_x = cos_t * target_x - sin_t * target_y
        rotated_y = sin_t * target_x + cos_t * target_y
        target_x, target_y = rotated_x, rotated_y

    # Negate Y for CARLA's coordinate system (Y points south)
    return (target_x, -target_y, 0.0)


def latlon_to_xodr_local(lat, lon, xodr_geo_reference, xodr_offset):
    """
    Convert lat/lon to XODR local coordinates using the EXACT same projection
    and offset as the XODR file. This ensures coordinates match CARLA's world.

    Parameters:
        lat, lon: WGS84 coordinates
        xodr_geo_reference: The projection string from XODR (e.g., "+proj=tmerc")
        xodr_offset: Tuple (offset_x, offset_y) from XODR header

    Returns:
        (x, y, z) in XODR local coordinates with Y negated for CARLA convention
    """
    from pyproj import Transformer

    # Use the exact projection from XODR
    # If it's just "+proj=tmerc", add default parameters
    proj_string = xodr_geo_reference.strip()
    if proj_string == "+proj=tmerc":
        # Default TM projection with central meridian at 0
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

    transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    # Project lat/lon to the TM projection
    proj_x, proj_y = transformer.transform(lon, lat)

    # Apply XODR offset to get local coordinates
    # XODR local = projected + offset (since offset is negative, this subtracts)
    local_x = proj_x + xodr_offset[0]
    local_y = proj_y + xodr_offset[1]

    # Negate Y for CARLA convention (CARLA Y-axis is flipped: south is positive)
    return (local_x, -local_y, 0.0)

#def save_graph_to_osm(filepath, G):
#    ox.io.save_graph_xml(G, filepath=filepath)

def fetch_osm_data(map_folder):
    map_path = os.path.join(map_folder, "map.osm")
    G = ox.graph.graph_from_xml(map_path, simplify=False, retain_all=True)

    all_tags = [
        "parking:left", "parking:right", "parking:both",
        "parking:lane:left", "parking:lane:right", "parking:lane:both"
    ]
    tag_dict = {tag: True for tag in all_tags}
    parking_data = ox.features_from_xml(map_path, tags=tag_dict).reset_index()

    available_columns = [col for col in all_tags if col in parking_data.columns]
    parking_info = parking_data[["id"] + available_columns]
    parking_info["id"] = parking_info["id"].astype(str)

    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    tags = {"building": True}
    buildings = ox.features_from_xml(map_path, tags=tags)

    allowed_highways = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "unclassified", "service"
    }

    edges = edges[edges["highway"].isin(allowed_highways)]

    edges = edges[edges["highway"].isin(allowed_highways)]

    edges["osmid_str"] = edges["osmid"].astype(str)
    edges = edges.merge(parking_info, left_on="osmid_str", right_on="id", how="left")

    print(edges.columns)

    return G, edges, buildings