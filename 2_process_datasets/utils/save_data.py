import json
import os
import shutil

import unicodedata
import re
import datetime
from utils.config import MAPS_FOLDER_NAME, DATASETS_FOLDER_NAME, MODEL_FOLDER_NAME
from utils.buildings import get_buildings_object
from PIL import Image
#from utils.map_data import save_graph_to_osm


def get_map_folder_name(address) -> str:
    return get_folder_name(address)

def get_dataset_folder_name(video_filename) -> str:
    return get_folder_name(video_filename)

def get_folder_name(name) -> str:
    name_ascii = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name_ascii = name_ascii.lower()
    name_clean = re.sub(r'[^\w\s]', '', name_ascii)  # Sonderzeichen entfernen
    name_clean = re.sub(r'\s+', '_', name_clean)  # Leerzeichen durch _ ersetzen
    date_str = datetime.datetime.now().strftime('%y_%m_%d')
    folder_name = f"{name_clean}_{date_str}"
    return folder_name


def create_map_folders(output_folder):
    os.makedirs(MAPS_FOLDER_NAME, exist_ok=True)  # Create the main maps folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

def create_dataset_folders(output_folder):
    os.makedirs(DATASETS_FOLDER_NAME, exist_ok=True)  # Create the main datasets folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# def create_output_folders(output_folder, delete = True):
#     os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
#     if os.path.exists(output_folder) and delete:
#         shutil.rmtree(output_folder)
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(os.path.join(output_folder, "carla"), exist_ok=True)
#     os.makedirs(os.path.join(output_folder, "output"), exist_ok=True)
#     os.makedirs(os.path.join(output_folder, "depth"), exist_ok=True)
#     os.makedirs(os.path.join(output_folder, "seg"), exist_ok=True)
#     os.makedirs(os.path.join(output_folder, "canny"), exist_ok=True)

def save_vehicle_data(output_folder, output_json):
    with open(os.path.join(output_folder,"vehicle_data.json"), "w") as f:
        json.dump(output_json, f, indent=2)

def save_osm_data(output_folder, osm_data):
    with open(os.path.join(output_folder,"map.osm"), "wb") as f:
        f.write(osm_data)

def get_existing_osm_data(output_folder):
    with open(os.path.join(output_folder,"map.osm"), "rb") as f:
        osm_data = f.read()
    return osm_data

def save_map_data(output_folder, osm_data, no_carla=False):
    osm_filename = os.path.join(output_folder, "map.osm")
    #save_graph_to_osm(osm_filename, G)

    #with open(osm_filename, "rb") as f:
    #    osm_data = f.read()
    try:
        buildings_obj = get_buildings_object(osm_data)
        buildings_obj.export(os.path.join(output_folder, "buildings.obj"))
    except Exception as e:
        print(f"Error exporting buildings: {e}")

    if not no_carla:

        save_xodr_data(output_folder, osm_data)
        #os.remove(osm_filename)

def save_xodr_data(output_folder, osm_data):
    import carla

    settings = carla.Osm2OdrSettings()
    settings.set_osm_way_types(
        ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
         "tertiary", "tertiary_link", "unclassified", "residential"])

    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    with open(os.path.join(output_folder, "map.xodr"), 'w') as f:
        f.write(xodr_data)

    return xodr_data

def save_dataset(folder_path, hf_dataset):
    hf_dataset.save_to_disk(folder_path)

def create_dotenv():
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("HF_TOKEN=\n")

def delete_image_files(folder_path):
    images_folder = os.path.join(folder_path, "images")
    canny_folder = os.path.join(folder_path, "canny")
    segmentation_folder = os.path.join(folder_path, "segmentation")

    if os.path.exists(images_folder):
        shutil.rmtree(images_folder)
    if os.path.exists(canny_folder):
        shutil.rmtree(canny_folder)
    if os.path.exists(segmentation_folder):
        shutil.rmtree(segmentation_folder)

def get_map_data(map_name, starting_image_size=None, no_carla=False):
    map_folder = os.path.join(MAPS_FOLDER_NAME, map_name)
    if not os.path.exists(map_folder):
        return None

    xodr_file = os.path.join(map_folder, "map.xodr")
    osm_file = os.path.join(map_folder, "map.osm")
    vehicle_data_file = os.path.join(map_folder, "vehicle_data.json")
    trajectory_data_file = os.path.join(map_folder, "trajectory_positions.json")

    # 1. Load Map Geometry
    xodr_data = None
    if os.path.exists(xodr_file):
        with open(xodr_file, 'r') as f:
            xodr_data = f.read()
    elif os.path.exists(osm_file) and not no_carla:
        with open(osm_file, 'r') as f:
            osm_data = f.read()
        xodr_data = save_xodr_data(map_folder, osm_data)
    elif not no_carla:
        return None

    # 2. Load Vehicle Data (Optional safe return)
    vehicle_data = None
    if os.path.exists(vehicle_data_file):
        with open(vehicle_data_file, 'r') as f:
            vehicle_data = json.load(f)
    else:
        # If vehicle_data is missing, return None or a minimal dict to avoid crashes later
        return None 

    # 3. Load Trajectory Data (CRASH FIX: Check existence first)
    trajectory_data = None
    if os.path.exists(trajectory_data_file):
        with open(trajectory_data_file, 'r') as f:
            trajectory_data = json.load(f)

    # 4. Load Starting Image
    starting_image_path = os.path.join(map_folder,"starting_image.jpg")
    starting_image = None
    if os.path.exists(starting_image_path) and starting_image_size is not None:
        starting_image = Image.open(starting_image_path).resize(starting_image_size)

    return {
        "xodr_data": xodr_data,
        "vehicle_data": vehicle_data,
        "starting_image": starting_image,
        "trajectory_data": trajectory_data
    }

def get_dataset_data(dataset_name):
    dataset_folder = os.path.join(DATASETS_FOLDER_NAME, dataset_name)
    if not os.path.exists(dataset_folder):
        return None

    config_file = os.path.join(dataset_folder, "camera.json")
    if not os.path.exists(config_file):
        return None

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    print(config_data)
    return {
        "size": config_data.get("image_size", {"x": 512, "y": 512}),
        "camera": config_data.get("camera"),
        "pitch": config_data.get("pitch"),
    }

def get_model_data(model_name):
    model_folder = os.path.join(MODEL_FOLDER_NAME, model_name)
    if not os.path.exists(model_folder):
        return None

    config_file = os.path.join(model_folder, "config.json")
    if not os.path.exists(config_file):
        return None

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    return {
        "size": config_data.get("image_size", {"x": 512, "y": 512}),
        "camera": config_data.get("camera"),
        "pitch": config_data.get("pitch"),
        "stable_diffusion_model": config_data.get("stable_diffusion_model"), #
        "lora_weights": config_data.get("lora_weights"),
        "controlnet_segmentation": config_data.get("controlnet_segmentation"),
        "controlnet_tempconsistency": config_data.get("controlnet_tempconsistency"),
        "controlnet_depth": config_data.get("controlnet_depth"),
        "controlnet_canny": config_data.get("controlnet_canny"),
    }

def save_camera_calibration(model_name, camera_calibration):
    model_folder = os.path.join(MODEL_FOLDER_NAME, model_name)
    if not os.path.exists(model_folder):
        return False

    config_file = os.path.join(model_folder, "config.json")
    if not os.path.exists(config_file):
        return False
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    config_data["camera"]["calibration"] = camera_calibration
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    return True

def save_arguments(args, output_folder):
    args_dict = vars(args)  # Convert Namespace to dictionary
    with open(os.path.join(output_folder, "arguments.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

def get_saved_arguments(output_folder):
    args_file = os.path.join(output_folder, "arguments.json")
    if not os.path.exists(args_file):
        return None
    with open(args_file, "r") as f:
        args_dict = json.load(f)
    return args_dict