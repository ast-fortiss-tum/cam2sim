import os
import json
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from huggingface_hub import login, snapshot_download
from utils.config import (NEGATIVE_PROMPT)
from safetensors import safe_open
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# =======================
# DEFAULT INFERENCE CONFIG
# =======================

# ControlNet conditioning schedule: [seg, inst, temp]
# Controls when each ControlNet starts/stops influencing the diffusion process.
CONTROL_START = [0.0, 0.0, 0.35]
CONTROL_END = [1.0, 0.6, 0.55]

# Per-ControlNet conditioning scales [seg, inst, temp].
# When prev_image is None (first frame) the temporal scale is set to 0
# so the temporal ControlNet has no effect on that frame.
CONTROLNET_SCALES_SEG = 0.7
CONTROLNET_SCALES_INST = 0.7
CONTROLNET_SCALES_TEMP = 1.1

# Classifier-free guidance strength.
GUIDANCE_SCALE = 3.0

DEFAULT_SEED = 50

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
    guess_mode=True,
    seed=DEFAULT_SEED,
    num_inference_steps=50,
    seg_scale=CONTROLNET_SCALES_SEG,
    inst_scale=CONTROLNET_SCALES_INST,
    temp_scale=CONTROLNET_SCALES_TEMP,
):
    """
    Generate a single frame with the SD + 3-ControlNet pipeline.

    Args:
        pipe, model_data:   pipeline objects from load_pipeline_models().
        seg_image, inst_image, prev_image: PIL conditioning inputs for
                            the segmentation, instance and temporal
                            ControlNets respectively. prev_image may be
                            None on the first frame, in which case the
                            temporal ControlNet is disabled
                            (its conditioning scale is forced to 0).
        prompt:             text prompt for the diffusion step.
        guidance:           classifier-free guidance scale.
        control_start, control_end: per-ControlNet [seg, inst, temp]
                            schedules in [0, 1]. If None, falls back to
                            the module-level CONTROL_START / CONTROL_END.
        guess_mode:         enables ControlNet guess mode.
        seed:               integer seed for reproducible generation,
                            or None to draw a fresh seed every call.
        num_inference_steps: number of denoising steps.
        seg_scale, inst_scale, temp_scale: conditioning scales applied
                            to each ControlNet. temp_scale is ignored
                            on the first frame.

    Returns:
        PIL.Image of the generated frame.
    """
    if control_start is None:
        control_start = CONTROL_START
    if control_end is None:
        control_end = CONTROL_END

    ctrl_temp = prev_image if prev_image is not None else seg_image
    control_images = [seg_image, inst_image, ctrl_temp]

    effective_temp_scale = temp_scale if prev_image is not None else 0.0
    controlnet_scales = [seg_scale, inst_scale, effective_temp_scale]

    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=control_images,
            negative_prompt=NEGATIVE_PROMPT,
            controlnet_conditioning_scale=controlnet_scales,
            height=model_data["size"]["y"],
            width=model_data["size"]["x"],
            num_inference_steps=num_inference_steps,
            control_guidance_start=control_start,
            control_guidance_end=control_end,
            guidance_scale=guidance,
            guess_mode=guess_mode,
            output_type="pil",
            generator=generator,
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