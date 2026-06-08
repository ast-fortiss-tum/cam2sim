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
        use_safetensors=True,
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
    pipe, 
    seg_image, 
    inst_image, 
    model_data, 
    prev_image, 
    prompt, 
    guidance=3.0,
    control_start=None, 
    control_end=None
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