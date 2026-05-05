import os
import xml.etree.ElementTree as ET
import shapely.geometry
from math import cos, radians
import trimesh
import numpy as np
import pyrender
from PIL import Image

def building_to_trimesh(coords, height, center_lat, center_lon):
    local_points = [geo_to_local(lat, lon, center_lat, center_lon) for lat, lon in coords]
    scale = 25
    scaled_points = [(x * scale, y * scale) for x, y in local_points]

    if scaled_points[0] != scaled_points[-1]:
        scaled_points.append(scaled_points[0])

    if len(scaled_points) < 4:
        return None

    polygon = shapely.geometry.Polygon(scaled_points)
    if not polygon.is_valid:
        return None

    mesh = trimesh.creation.extrude_polygon(polygon, height=height * scale)
    return mesh

def get_osm_building_nodes(root):
    nodes = {}
    for node in root.findall('node'):
        if node.attrib.get('visible', 'true') != 'true':
            continue
        node_id = node.attrib['id']
        lat = float(node.attrib['lat'])
        lon = float(node.attrib['lon'])
        nodes[node_id] = (lat, lon)
    return nodes

def geo_to_local(lat, lon, center_lat, center_lon):
    dx = latlon_to_m(center_lat, center_lon, center_lat, lon)[1]
    dy = latlon_to_m(center_lat, center_lon, lat, center_lon)[0]
    return dx, dy

def latlon_to_m(lat1, lon1, lat2, lon2):
    lat_m = (lat2 - lat1) * 111320
    lon_m = (lon2 - lon1) * 40075000 * cos(radians((lat1 + lat2) / 2)) / 360
    return lat_m, lon_m

def get_buildings_object(osm_data):
    LEVEL_HEIGHT = 3.0

    root = ET.fromstring(osm_data)

    nodes = get_osm_building_nodes(root)

    buildings = []
    building_tags = []
    for way in root.findall('way'):
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
        if 'building' in tags:
            coords = [nodes[nd.attrib['ref']] for nd in way.findall('nd') if nd.attrib['ref'] in nodes]
            if coords:
                buildings.append(coords)
                building_tags.append(tags)

    # Mittelpunkt berechnen
    all_lats = [lat for geb in buildings for lat, _ in geb]
    all_lons = [lon for geb in buildings for _, lon in geb]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    all_meshes = []

    for coords, tags in zip(buildings, building_tags):
        levels = float(tags.get('building:levels', 1))
        height = levels * LEVEL_HEIGHT
        offset_x = sum(geo_to_local(lat, lon, center_lat, center_lon)[0] for lat, lon in coords) / len(coords)
        offset_z = sum(geo_to_local(lat, lon, center_lat, center_lon)[1] for lat, lon in coords) / len(coords)
        offset = (offset_x, 0, offset_z)

        mesh = building_to_trimesh(coords, height, center_lat, center_lon)
        mesh.apply_translation(offset)
        all_meshes.append(mesh)

    # Export all meshes to a single .obj file
    combined = trimesh.util.concatenate(all_meshes)
    #combined.export('buildings.obj')
    return combined

def get_building_mesh(map_path):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    mesh = trimesh.load_mesh(os.path.join(map_path, "buildings.obj"))
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_pyrender)
    return scene

def get_building_rendering(scene, camera_pos=[0, -50, 20], look_dir=[0, 1, -0.4], resolution=(512, 512), fov_deg=55):

    if hasattr(scene, "camera_node"):
        scene.remove_node(scene.camera_node)

    camera_pos = np.array(camera_pos)
    look_dir = np.array(look_dir)
    target = camera_pos + look_dir
    pose = look_at(camera_pos, target)

    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg))
    #camera = pyrender.IntrinsicsCamera(fx=491.6,fy=491.6,cx=256,cy=256,znear=0.1,zfar=1000)
    cam_node = scene.add(camera, pose=pose)
    scene.camera_node = cam_node  # merken

    renderer = pyrender.OffscreenRenderer(*resolution)
    color, _ = renderer.render(scene)
    renderer.delete()

    image = Image.fromarray(color).convert("L")
    return image

def look_at(camera_pos, target, up=np.array([0, 0, 1])):
    """Erzeugt eine LookAt-Transformationsmatrix."""
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)
    rot = np.column_stack((right, true_up, forward))
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = camera_pos
    return transform

def render_bw_image(obj_path, camera_pos, look_dir, image_path="output_bw.png", resolution=(800, 600), fov_deg=60):
    """
    Rendert ein Schwarz-Weiß-Bild einer .obj-Datei aus gegebener Kameraposition und Richtung.
    """
    # 1. Mesh laden
    scene = get_building_mesh(obj_path)
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg))
    #camera = pyrender.IntrinsicsCamera(fx=491.6,fy=491.6,cx=256,cy=256,znear=0.1,zfar=100000000)

    # 2. Kamera definieren
    camera_pos = np.array(camera_pos)
    look_dir = np.array(look_dir)
    target = camera_pos + look_dir
    up = np.array([0, 0, 1])

    # 3. View-Transform berechnen
    def look_at(cam_pos, target, up):
        forward = (target - cam_pos)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        mat = np.eye(4)
        mat[:3, :3] = np.vstack([right, true_up, -forward])
        mat[:3, 3] = cam_pos
        return np.linalg.inv(mat)

    cam_pose = look_at(camera_pos, target, up)
    scene.add(camera, pose=cam_pose)

    # 4. Renderer
    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    r.delete()

    # 5. Graustufen speichern
    img = Image.fromarray(color).convert("L")
    img.save(image_path)
    print(f"✅ Schwarz-Weiß-Bild gespeichert unter: {image_path}")


#render_bw_image(
#    obj_path="../maps/guerickestrae_alte_heide_munich_25_08_01",
#    camera_pos=[00, -5000, -1800],
#    look_dir=[1, 0.0, -2.5],
#    image_path="export_bw.png"
#)

#scene = get_building_mesh("../maps/guerickestrae_alte_heide_munich_25_08_01")
#image = get_building_rendering(scene, camera_pos=[50, -100, 20], look_dir=[0, 1, -0.4])
#image.save("preview_image.png")