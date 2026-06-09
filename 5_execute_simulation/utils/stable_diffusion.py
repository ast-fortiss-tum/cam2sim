import os
import json
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from huggingface_hub import login, snapshot_download
from utils.config import (STABLE_DIFF_PROMPT, 
                    STABLE_DIFF_STEPS, 
                    SEGMENTATION_COND_SCALE, 
                    MODEL_FOLDER_NAME, 
                    STATIC_PROMPT, NEGATIVE_PROMPT, 
                    CONTROL_START, CONTROL_END)
from safetensors import safe_open
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

GUIDANCE_SCALE = 3.0

def load_pipeline_models(model_root, device):
    config_path = os.path.join(model_root, "config.json")
    with open(config_path, "r") as f:
        model_data = json.load(f)

    print("\ Loading ControlNet Models...")
    # Load ControlNets
    cnet_seg = ControlNetModel.from_pretrained(os.path.join(model_root, model_data["controlnet_segmentation"]), torch_dtype=torch.float16)
    cnet_temp = ControlNetModel.from_pretrained(os.path.join(model_root, model_data["controlnet_tempconsistency"]), torch_dtype=torch.float16)
    cnet_inst = ControlNetModel.from_pretrained(os.path.join(model_root, model_data["controlnet_instance"]), torch_dtype=torch.float16)

    # Load Pipeline [Seg, Inst, Temp]
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_data["stable_diffusion_model"],
        controlnet=[cnet_seg, cnet_inst, cnet_temp], 
        torch_dtype=torch.float16,
        safety_checker=None,                # <-- skip safety classifier load
        requires_safety_checker=False,      # <-- tell diffusers it's intentional
    ).to(device)

    # Load LoRA
    lora_path = os.path.join(model_root, model_data["lora_weights"])
    print(f" Loading LoRA from: {lora_path}")
    if lora_path.endswith(".safetensors"):
        lora_state_dict = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys(): lora_state_dict[key] = f.get_tensor(key)
    else:
        lora_state_dict = torch.load(lora_path, map_location="cpu")
    
    pipe.load_lora_weights(lora_state_dict)
    return pipe, model_data


def generate_image_realtime(
    pipe, seg_image, inst_image, model_data, prev_image, prompt,
    guidance=GUIDANCE_SCALE,
    control_start=None, control_end=None,
    guess_mode=True,            # <-- nuovo, era hardcoded
    seed=None,                  # <-- nuovo, None = random
    num_inference_steps=50,
):
    """
    Generates one frame using specific ControlNet parameters, DYNAMIC PROMPT, 
    and DYNAMIC SCHEDULES.
    """
    # Default fallbacks if None passed (optional, safety net)
    if control_start is None: control_start = [0.41, 0.0, 0.0]
    if control_end is None:   control_end   = [1.0, 0.4, 0.4]

    # 1. Prepare Control Images
    ctrl_temp = prev_image if prev_image is not None else seg_image
    
    # ControlNet Input Order: [Seg, Inst, Temp]
    control_images = [seg_image, inst_image, ctrl_temp]
    
    # 2. Parameter Configuration
    current_temp_scale = 1.1 if prev_image is not None else 0.0
    controlnet_scales = [0.7, 0.7, current_temp_scale]

    generator = torch.Generator(device=pipe.device).manual_seed(50) 

    # 3. Call Pipeline
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=control_images,
            negative_prompt= NEGATIVE_PROMPT,
            controlnet_conditioning_scale=controlnet_scales,
            height=model_data["size"]["y"],
            width=model_data["size"]["x"],
            num_inference_steps=50, #STABLE_DIFF_STEPS, # Ensure this matches your config
            
            # --- DYNAMIC APPLIED SCHEDULES ---
            control_guidance_start=control_start, 
            control_guidance_end=control_end,     
            
            guidance_scale=guidance,
            guess_mode=True, 
            output_type="pil",
            generator=generator
        )
    return result.images[0]

# =======================
# TRAJECTORY-BASED MODEL SELECTION
# =======================
# Same chunking logic the training uses to split the dataset into per-part
# shards. Inference scripts (5E offline, 5F closed-loop, 5G grid search)
# rebuild the same chunks and dispatch each frame to the matching SD model
# part based on the hero (x, y) position along the planned trajectory.

def split_trajectory_into_parts(trajectory_points, num_parts):
    """Split the trajectory into num_parts equal chunks (same as training)."""
    total = len(trajectory_points)
    chunk_size = total // num_parts
    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = total if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(trajectory_points[start:end])
    return chunks


def find_closest_trajectory_point(xy, trajectory_chunk):
    """
    Index of the trajectory point closest to (x, y).

    Args:
        xy: 2-element tuple/list/array (x, y). Callers pass either
            (frame_location["x"], frame_location["y"]) for offline scripts
            or (cur_loc.x, cur_loc.y) for the closed-loop script.
        trajectory_chunk: list of {"transform": {"location": {"x":..,"y":..}}}.

    Returns:
        (closest_idx, min_distance)
    """
    x, y = float(xy[0]), float(xy[1])
    min_dist = float("inf")
    closest_idx = 0
    for idx, point in enumerate(trajectory_chunk):
        traj_x = point["transform"]["location"]["x"]
        traj_y = point["transform"]["location"]["y"]
        dist = np.sqrt((x - traj_x) ** 2 + (y - traj_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    return closest_idx, min_dist


def select_model_part(xy, trajectory_chunks):
    """
    Find which chunk (i.e. which SD model part) the (x, y) belongs to.
    Returns (best_part_index, distance_to_that_chunk_in_meters).
    """
    best_part = 0
    best_distance = float("inf")
    for part_idx, chunk in enumerate(trajectory_chunks):
        _, dist = find_closest_trajectory_point(xy, chunk)
        if dist < best_distance:
            best_distance = dist
            best_part = part_idx
    return best_part, best_distance


# =======================
# REPLAY DATASET LOADING (offline scripts only: 5E, 5G)
# =======================

def load_replay_data(sem_folder, inst_folder, metadata_path, max_frames=None):
    """
    Load semantic + instance maps and metadata from the replay dataset
    produced by 5A_sd_trajectory_only_carla.py.

    Returns:
        seg_list:       list of PIL.Image (RGB)
        inst_list:      list of PIL.Image (RGB)
        frame_data:     list of metadata dicts (location, rotation, caption)
        frame_indices:  list of frame_id ints

    Frames whose semantic or instance file is missing on disk are silently
    skipped and counted in a warning at the end.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    print(f"\n[INFO] Loading replay dataset from: {os.path.dirname(metadata_path)}")

    with open(metadata_path, "r") as f:
        all_frame_data = json.load(f)

    if max_frames is not None and max_frames > 0:
        all_frame_data = all_frame_data[:max_frames]

    seg_list, inst_list, frame_data, frame_indices = [], [], [], []
    n_missing = 0

    for item in tqdm(all_frame_data, desc="Reading frames"):
        frame_id = item["frame"]
        filename = f"{frame_id:06d}.png"
        seg_path = os.path.join(sem_folder, filename)
        inst_path = os.path.join(inst_folder, filename)
        if not os.path.exists(seg_path) or not os.path.exists(inst_path):
            n_missing += 1
            continue
        seg_list.append(Image.open(seg_path).convert("RGB"))
        inst_list.append(Image.open(inst_path).convert("RGB"))
        frame_data.append(item)
        frame_indices.append(frame_id)

    if n_missing > 0:
        print(f"[WARN] Missing files for {n_missing} frames (skipped).")
    print(f"[INFO] Loaded {len(seg_list)} frames.")
    return seg_list, inst_list, frame_data, frame_indices