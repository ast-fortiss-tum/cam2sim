import os
import json
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from huggingface_hub import login, snapshot_download
from config import (STABLE_DIFF_PROMPT, 
                    STABLE_DIFF_STEPS, 
                    SEGMENTATION_COND_SCALE, 
                    MODEL_FOLDER_NAME, 
                    STATIC_PROMPT, NEGATIVE_PROMPT, 
                    CONTROL_START, CONTROL_END)
from safetensors import safe_open

def load_stable_diffusion_pipeline(model_name, model_data):
    controlnet_segmentation = ControlNetModel.from_pretrained("./"+os.path.join(MODEL_FOLDER_NAME,model_name,model_data["controlnet_segmentation"]), torch_dtype=torch.float16)
    controlnet_tempconsistency = ControlNetModel.from_pretrained("./"+os.path.join(MODEL_FOLDER_NAME,model_name,model_data["controlnet_tempconsistency"]), torch_dtype=torch.float16)
    #controlnet_depth = ControlNetModel.from_pretrained("./"+os.path.join(MODEL_FOLDER_NAME,model_name,model_data["controlnet_depth"]), torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_data["stable_diffusion_model"],
        controlnet=[controlnet_segmentation, controlnet_tempconsistency],
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(get_device())

    pipe.load_lora_weights(os.path.join(MODEL_FOLDER_NAME,model_name,model_data["lora_weights"]), dtype=torch.float16)
    return pipe

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def generate_image(pipe, seg_image, model_data, prev_image = None, split = 20, guidance = 4.5, set_seed = False, rotate = False):
    #from PIL import Image
    #seg_image = Image.open("test.jpg").resize((512,512))

    if set_seed:
        generator = torch.manual_seed(50)

    split_value = split / 100

    split_1_1 = split_value if split_value < 1.0 else 0.99
    split_1_2 = 0.0

    split_2_1 = 1.0
    split_2_2 = split_value if split_value > 0.0 else 0.01

    if rotate:
        split_1_1_save = split_1_1
        split_2_1_save = split_2_1

        split_1_1 = split_1_2
        split_1_2 = split_1_1_save

        split_2_1 = split_2_2
        split_2_2 = split_2_1_save

    return pipe(
        STABLE_DIFF_PROMPT,
        image=[seg_image, prev_image if prev_image is not None else seg_image],
        negative_prompt="blurry, distorted, street without street lines",
        controlnet_conditioning_scale=[SEGMENTATION_COND_SCALE,0.6 if prev_image is not None else 0.0],
        height=model_data["size"]["y"],
        width=model_data["size"]["x"],
        control_image=prev_image if prev_image is not None else seg_image,
        num_inference_steps=STABLE_DIFF_STEPS,
        control_guidance_start=[split_1_1, split_1_2],
        control_guidance_end=[split_2_1, split_2_2],
        guidance_scale=guidance,
        guess_mode=True,
        output_type="pil"
    ).images[0]
#generator=generator,

def check_essential_files(model_dir: str) -> bool:
    if not os.path.exists(model_dir): return False
    required = ["config.json", "stable_diffusion/pytorch_lora_weights.safetensors", "controlnet_instance/diffusion_pytorch_model.safetensors"]
    for r in required:
        if not os.path.exists(os.path.join(model_dir, r)): return False
    return True
############# HAS TO BE MOVED IN STABLE DIFF UTILS ####################
def download_models_and_config(repo_id, local_dir, token):
    if check_essential_files(local_dir):
        print(f"✅ Models found locally in {local_dir}.")
        return local_dir
    print(f"⏳ Downloading models from {repo_id}...")
    login(token=token, add_to_git_credential=False)
    # Using snapshot_download logic from your script
    return snapshot_download(
        repo_id=repo_id, 
        repo_type="model", 
        local_dir=local_dir, 
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json", "lora_weights/*", "stable_diffusion/*", 
            "controlnet_segmentation/*", "controlnet_instance/*", "controlnet_tempconsistency/*"
        ]
    )
############# HAS TO BE MOVED IN STABLE DIFF UTILS ####################
def load_pipeline_models(model_root, device):
    config_path = os.path.join(model_root, "config.json")
    with open(config_path, "r") as f:
        model_data = json.load(f)

    print("\n⏳ Loading ControlNet Models...")
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
    print(f"⏳ Loading LoRA from: {lora_path}")
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
    # --- NEW ARGUMENTS ---
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