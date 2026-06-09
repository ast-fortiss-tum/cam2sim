"""
Resolve where the cam2sim Stable Diffusion models live.

The chosen path is persisted in `data/.sd_root` (project root).

Resolution order (per call):
  1. If `override` is given (e.g. --output-root from 4A), use it.
     The value is written to `data/.sd_root` so later runs find it
     without the flag.
  2. Else, if `data/.sd_root` exists, read and use the path from it.
  3. Else, fall back to <PROJECT_ROOT>/data/stable_diff_models and
     persist it as the chosen path. This makes the project work
     out of the box right after `git clone`, without forcing the
     user to configure anything.

The returned path is always absolute. The directory is created if
it does not exist.
"""

import os

SD_ROOT_FILE = "data/.sd_root"
DEFAULT_SD_ROOT_REL = "data/stable_diff_models"


def resolve_sd_root(project_root, override=None):
    """
    Returns (path, action_label):
      - path:         absolute path to the SD root
      - action_label: short string for logging ("--output-root",
                      "data/.sd_root", "default")
    """
    path_file = os.path.join(project_root, SD_ROOT_FILE)

    if override is not None:
        chosen = os.path.abspath(override)
        _persist(path_file, chosen)
        action = "--output-root"
    elif os.path.exists(path_file):
        with open(path_file, "r") as f:
            chosen = f.read().strip()
        if not chosen:
            raise ValueError(f"SD root file is empty: {path_file}")
        action = SD_ROOT_FILE
    else:
        chosen = os.path.abspath(os.path.join(project_root, DEFAULT_SD_ROOT_REL))
        _persist(path_file, chosen)
        action = f"default ({DEFAULT_SD_ROOT_REL})"

    os.makedirs(chosen, exist_ok=True)
    return chosen, action


def _persist(path_file, path):
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, "w") as f:
        f.write(path + "\n")


def detect_num_trained_parts(models_base_dir):
    """
    Count contiguous fully-trained parts starting from part_0.

    A part is fully trained if it contains all of:
      - config.json
      - stable_diffusion/pytorch_lora_weights.safetensors
      - controlnet_segmentation/diffusion_pytorch_model.safetensors
      - controlnet_instance/diffusion_pytorch_model.safetensors
      - controlnet_tempconsistency/diffusion_pytorch_model.safetensors

    Stops at the first part that is missing or incomplete: this is
    intentional, partial training (part_0 OK, part_2 OK but part_1
    broken) would cause wrong trajectory chunking and silently bad
    inference. Better to surface the gap.
    """
    if not os.path.isdir(models_base_dir):
        return 0

    required = [
        "config.json",
        "stable_diffusion/pytorch_lora_weights.safetensors",
        "controlnet_segmentation/diffusion_pytorch_model.safetensors",
        "controlnet_instance/diffusion_pytorch_model.safetensors",
        "controlnet_tempconsistency/diffusion_pytorch_model.safetensors",
    ]

    n = 0
    while True:
        part_dir = os.path.join(models_base_dir, f"part_{n}")
        if not os.path.isdir(part_dir):
            break
        if not all(os.path.exists(os.path.join(part_dir, f)) for f in required):
            break
        n += 1
    return n 

def require_trained_parts(models_base_dir, bag_name):
    """
    Verify models_base_dir exists and contains at least one fully-trained
    part. Returns num_parts. Raises FileNotFoundError with an actionable
    message otherwise.
    """
    if not os.path.isdir(models_base_dir):
        raise FileNotFoundError(
            f"SD models directory not found: {models_base_dir}\n"
            f"Run 4A_train_stable_diff.py --bag-name {bag_name} first."
        )
    num_parts = detect_num_trained_parts(models_base_dir)
    if num_parts == 0:
        raise FileNotFoundError(
            f"No fully-trained model part found under {models_base_dir}.\n"
            f"Re-run 4A_train_stable_diff.py --bag-name {bag_name} to complete training."
        )
    return num_parts

from dataclasses import dataclass

@dataclass
class SDBagPaths:
    replay_dataset: str
    semantic: str
    instance: str
    metadata: str
    trajectory: str
    bag_sd_dir: str
    models_base: str

def build_sd_bag_paths(project_root, sd_root, bag_stem):
    replay = os.path.join(project_root, "data", "processed_dataset", bag_stem, "carla_replay_dataset_sd")
    bag_sd_dir = os.path.join(sd_root, bag_stem)
    return SDBagPaths(
        replay_dataset=replay,
        semantic=os.path.join(replay, "semantic"),
        instance=os.path.join(replay, "instance"),
        metadata=os.path.join(replay, "data", "all_frame_data.json"),
        trajectory=os.path.join(project_root, "data", "data_for_carla", bag_stem,
                                "trajectory_positions_rear_odom_yaw.json"),
        bag_sd_dir=bag_sd_dir,
        models_base=os.path.join(bag_sd_dir, "SD_Training_Outputs_Split"),
    )