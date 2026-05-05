import argparse

from utils.config import CARLA_IP, CARLA_PORT


def parse_dataset_args():
    parser = argparse.ArgumentParser(description="Arguments for creating and processing the Dataset")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder of images to be processed")
    parser.add_argument("--name", type=str, help="Name of the dataset to be created")
    parser.add_argument("--upload", action='store_true', default=False, help="Uploads the dataset to Huggingface Hub")
    args = parser.parse_args()
    return args

def parse_map_args():
    parser = argparse.ArgumentParser(description="Arguments for creating a map from an address")
    parser.add_argument("--address", type=str, required=True, help="Name of the address, a street, or a location (be specific, add the city)")
    parser.add_argument("--name", type=str, help="Name of the map to be created (if not provided, it will be derived from the address)")
    parser.add_argument("--dist", type=int, default=250, help="Distance in meters around the address to fetch the map data")
    parser.add_argument("--no_carla", action='store_true', default=False, help="Disables conversion to .xodr file for Carla (which carla is required for)")
    parser.add_argument("--skip_fetch", action='store_true', default=False, help="Skips the osm File fetch (only works if it already exists)")
    parser.add_argument("--mode", choices=["manual", "clusters"], default="manual", help="manual = usa GUI, all = genera tutti i segmenti automaticamente")
    args = parser.parse_args()
    return args

def parse_finetuning_args():
    parser = argparse.ArgumentParser(description="Arguments for finetuning models on the Dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to be used for finetuning")

    args = parser.parse_args()
    return args

def parse_carla_scenario_args():
    parser = argparse.ArgumentParser(description="Arguments for generationg scenario in Carla")
    parser.add_argument("--map", type=str, required=True, help="Name of the map to be used for generationg scenario in Carla")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset which we want to simulate in Carla")
    parser.add_argument("--carla_ip", type=str, default=CARLA_IP, help=f"IP address of the Carla server (default: {CARLA_IP})")
    parser.add_argument("--carla_port", type=int, default=CARLA_PORT, help=f"Port of the Carla server (default: {CARLA_PORT})")

    args = parser.parse_args()
    return args

def parse_testing_args():
    parser = argparse.ArgumentParser(description="Arguments for testing the model on a map")
    parser.add_argument("--map", type=str, required=True, help="Name of the map to be used for testing")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset which we want to simulate in Carla")
    parser.add_argument("--model", type=str, required=False, help="Name of the model to be used for testing")
    parser.add_argument("--only_carla", action='store_true', default=False, help="Skips the Stable Diffusion part and only runs the Carla simulation, saves the results as a .png file")
    parser.add_argument("--yolo", action='store_true', default=False, help="Runs Yolo Vehicle-Detection only for the pygame window output")
    parser.add_argument("--carla_ip", type=str, default=CARLA_IP, help=f"IP address of the Carla server (default: {CARLA_IP})")
    parser.add_argument("--carla_port", type=int, default=CARLA_PORT, help=f"Port of the Carla server (default: {CARLA_PORT})")
    parser.add_argument("--other_vehicles", type=int, default=0, help="The amount of other vehicles driving around (default: 0)")
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save the results of the testing")
    parser.add_argument("--no_save", action='store_true', default=False, help="Disables saving the results of the testing")
    parser.add_argument("--seconds", type=int, default=60, help="Amount of seconds the final video should be.")
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process (default: all)')
    # --- NUOVI ARGOMENTI AGGIUNTI QUI ---
    parser.add_argument("--sched_start", type=str, default='[0.11, 0.91, 0.0]', help="Start Schedule list as string, e.g. '[0.31, 0.0, 0.0]'")
    parser.add_argument("--sched_end", type=str, default='[0.90, 1.00, 0.1]', help="End Schedule list as string, e.g. '[1.0, 0.3, 0.3]'")
    parser.add_argument('--model_dir', type=str, help='Path to split model directory')
    parser.add_argument('--run_number', type=int, default=1, help='Run iteration number')
    parser.add_argument('--num_parts', type=int, default=3, help='Number of model parts')
    args = parser.parse_args()
    return args

def parse_export_args():
    parser = argparse.ArgumentParser(description="Arguments for exporting a video from a test")
    parser.add_argument("--name", type=str, required=True, help="Name the output")
    parser.add_argument("--dir", type=str, default="output", help="Override the output folder name")
    parser.add_argument("--only_generated", action='store_true', default=False, help="Only saves the generated images to a video")

    args = parser.parse_args()
    return args

def parse_generation_args():
    parser = argparse.ArgumentParser(description="Redoing the generation of images from saved carla images")
    parser.add_argument("--output_dir", type=str, required=True, help="Name the output folder")
    parser.add_argument("--split", default=20, type=int, help="The Split in percent, where the handoff between consistency model and segmentation should happen")
    parser.add_argument("--set_seed", action='store_true', default=False, help="Activates a set seed")
    parser.add_argument("--guidance", type=float, default=4.5, help="Sets the guidance scale")
    parser.add_argument("--rotate", action='store_true', default=False, help="Rotates the controlnets")
    parser.add_argument("--max_images", type=int, help="Maximum amount of Images")
    

    args = parser.parse_args()
    return args

def parse_validation_args():
    parser = argparse.ArgumentParser(description="Validating the generation of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Name the output folder")
    parser.add_argument("--segmentation", action='store_true', default=False, help="Does SSS validation")
    parser.add_argument("--cpl", action='store_true', default=False, help="Does CPL validation in addition to the selected one")
    args = parser.parse_args()
    return args

def parse_distribution_args():
    parser = argparse.ArgumentParser(description="Validating the Distribution")
    parser.add_argument("--output_dir", type=str, required=True, help="Name the output folder")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the Dataset for the BaseLine")
    args = parser.parse_args()
    return args

def parse_calibration_args():
    parser = argparse.ArgumentParser(description="Calibrating the Carla Camera using the checkerboard video")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to be processed")
    parser.add_argument("--model", type=str, help="Name of the model to save the settings to. If empty will only output the results")
    parser.add_argument("--plot", action='store_true', default=False, help="Plots each frames results if a checkerboard was found")
    args = parser.parse_args()
    return args

def parse_calibration2_args():
    parser = argparse.ArgumentParser(description="Calibrating the Carla Camera using the checkerboard video")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to be processed")
    parser.add_argument("--frame", type=int, default = 50, help="FrameId of the video")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to save the settings to & get from.")
    args = parser.parse_args()
    return args

def parse_figure_args():
    parser = argparse.ArgumentParser(description="Get the figure of a csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Name the output folder")
    parser.add_argument("--value", type=str, required=True, help="Name the value")
    parser.add_argument("--pgf", type=str, help="Name the pfg file without the .pfg ending")
    parser.add_argument("--file", type=str, required=True, help="Name the csv file")
    parser.add_argument("--name", type=str, help="Name of the value, --value by default")
    parser.add_argument("--no_seed", action='store_true', default=False,
                        help="Activates random seed")
    parser.add_argument("--rotate", action='store_true', default=False,
                        help="Activates rotate")

    args = parser.parse_args()
    return args