#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Placeholder Imports/Constants (Replace with your actual values/imports) ---
# NOTE: In a real script, 'load_model' would be imported from 'keras.models' or 'tensorflow.keras.models'
try:
    from keras.models import load_model
except ImportError:
    # Define a dummy for demonstration if Keras is not installed
    def load_model(path, compile=False, safe_mode=False):
        print(f"Warning: Dummy load_model called for {path}. Install Keras/TensorFlow to run inference.")
        class DummyModel:
            def compile(self, loss, metrics): pass
            def predict(self, image_array, verbose=0):
                # Returns dummy output shape (1, 2)
                return np.array([[0.0, 0.5]])
        return DummyModel()

FIXED_THROTTLE = False
MAX_STEERING = 9.0
STEERING = 0  # Index for steering in model output
THROTTLE = 1  # Index for throttle in model output
# -------------------------------------------------------------------------------


# --- DAVE-2 Model Preprocessing Parameters ---
crop_top = 204
crop_bottom = 35
target_h, target_w = 66, 200 # Target DAVE-2 dimensions
# ---------------------------------------------

# --- GLOBAL CONSTANT FOR SAVING FOLDER ---
DAVE2_OUTPUT_FOLDER = "dave2_input_images"


def preprocess_image_for_dave2(pil_img, save_path=None):
    """
    Applies the full DAVE-2 preprocessing.
    
    Input:  pil_img (PIL Image): The input image.
            save_path (str or None): Path to save the final 200x66 image.
    Output: float32 array (1, 66, 200, 3)
    """
    # 1. Resize back to the original training size (800x503)
    original_w, original_h = 800, 503
    pil_img = pil_img.resize((original_w, original_h), Image.BILINEAR)

    # 2. Apply the cropping
    width, height = pil_img.size
    # Crop from (0, crop_top) to (width, height - crop_bottom)
    pil_img = pil_img.crop((0, crop_top, width, height - crop_bottom)) 

    # 3. Resize to the target DAVE-2 dimensions (200x66)
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
    
    # --- SAVE THE FINAL PROCESSED IMAGE ---
    if save_path:
        try:
            # Check if directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pil_img.save(save_path)
            # print(f"Saved DAVE-2 input image (200x66) to: {save_path}")
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")
    # ------------------------------------------------
    
    # --- Convert to numpy array float32 and add batch dimension ---
    x = np.asarray(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)

    return x


class Dave2Model:

    model = None
    frame_counter = 0 # New counter to track processed frames
    
    # --- SMOOTHING VARIABLES ---
    prev_steering = 0.0
    alpha = 0.85
    is_first_frame = True

    def parse_model_outputs(self, outputs):
        """Extracts steering and throttle from the model's output array."""
        res = []
        for i in range(outputs.shape[1]):
            res.append(outputs[0][i])
        return res
    
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False, safe_mode=False)
        self.model.compile(loss="sgd", metrics=["mse"])
        
        # Initialize smoothing state
        self.prev_steering = 0.0
        self.is_first_frame = True

    def calculate_dave2_image(self, image):
        """
        Processes a PIL Image, saves the final input image to disk using a 
        progressive counter, and returns the predicted steering and throttle.
        
        Input: image (PIL Image): The input image (assumed to be RGB).
        Output: (steering: float, throttle: float)
        """
        # Increment the counter for this frame
        self.frame_counter += 1
        
        # Construct the unique filename (e.g., 'dave2_input_images/frame_000001.png')
        filename = f"frame_{self.frame_counter:06d}.png"
        save_path = os.path.join(DAVE2_OUTPUT_FOLDER, filename)
        
        # Step 1: Preprocess the image and save it
        image_array = preprocess_image_for_dave2(image, save_path=save_path)

        # Step 2: Model prediction
        outputs = self.model.predict(image_array, verbose=0)
        parsed_outputs = self.parse_model_outputs(outputs)

        # Step 3: Extract and constrain steering/throttle
        steering_prediction = 0.
        throttle = 0.
        if len(parsed_outputs) > 0:        
            steering_prediction = parsed_outputs[STEERING]
            throttle = parsed_outputs[THROTTLE]

        # --- APPLY SMOOTHING ---
        if self.is_first_frame:
            # If it's the first frame, we don't have a previous value.
            # Set previous to current to avoid starting from 0.0 arbitrarily.
            self.prev_steering = steering_prediction
            self.is_first_frame = False
            steering = steering_prediction
        else:
            # Apply the smoothing formula
            # steering_cmd = alpha * previous + (1 - alpha) * current
            steering = (self.alpha * self.prev_steering) + ((1.0 - self.alpha) * steering_prediction)
            
            # Update history for next frame
            self.prev_steering = steering
        # -----------------------

        if FIXED_THROTTLE:
            throttle = 1.

        # Apply maximum steering limits
        steering = min(steering, MAX_STEERING)
        steering = max(steering, MAX_STEERING * -1.0)

        return steering, throttle