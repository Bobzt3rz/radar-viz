import cv2
import numpy as np
import os
import glob
import sys
from typing import List, Optional, Any

from modules.optical_flow import OpticalFlow

def calculate_and_save_flow_custom(
    input_dir: str, 
    output_dir: str, 
    use_cv_inference: bool = False
) -> None:
    """
    Reads sequential RGB images, calculates optical flow using the specified
    method (inference or inference_cv) on the custom calculator object,
    and saves the resulting flow fields as .npy files.
    """
    
    print(f"--- Starting Optical Flow Processing ---")
    print(f"Input Directory: {input_dir}")

    flow_calculator_object = OpticalFlow()
    
    # --- Determine the method and print status ---
    if use_cv_inference:
        inference_method = flow_calculator_object.inference_cv
        mode_str = "Custom CV (inference_cv)"
    else:
        inference_method = flow_calculator_object.inference
        mode_str = "Custom NN (inference)"
    print(f"Mode: Using {mode_str}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find and sort all image files (crucial for correct sequencing)
    image_files: List[str] = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_files:
        print(f"Error: No images found in {input_dir}. Exiting.")
        return

    # 1. Initialize the calculator with the first frame (Frame 0)
    try:
        # Read the first image in RGB (assuming your inference methods expect RGB)
        frame_0_rgb = cv2.cvtColor(cv2.imread(image_files[0]), cv2.COLOR_BGR2RGB)
        
        # The first call to inference (or inference_cv) often primes the calculator
        # by storing the initial frame (Frame A).
        inference_method(frame_0_rgb)
            
    except Exception as e:
        print(f"Error during initial frame setup/inference: {e}. Exiting.")
        return
    
    # 2. Process remaining frames (starting from index 1)
    for i in range(1, len(image_files)):
        current_file = image_files[i]
        
        try:
            current_frame_rgb = cv2.cvtColor(cv2.imread(current_file), cv2.COLOR_BGR2RGB)
            frame_id: str = os.path.basename(current_file).split('.')[0]
        except Exception as e:
            print(f"Error reading image {current_file}: {e}. Skipping.")
            continue
            
        # 3. Calculate Flow using the chosen method
        try:
            # The flow calculator's method is called with the current frame (Frame B).
            # It should internally calculate flow from (A) to (B) and store (B) for the next iteration.
            flow_map: np.ndarray = inference_method(current_frame_rgb)
        except Exception as e:
            print(f"Error calculating flow for frame {frame_id} using {mode_str}: {e}. Skipping.")
            continue
        
        # 4. Save the Flow Map
        output_filepath: str = os.path.join(output_dir, f"{frame_id}.npy")
        
        # Ensure the output is a valid 2-channel flow map
        if flow_map.ndim == 3 and flow_map.shape[2] == 2:
            np.save(output_filepath, flow_map)
            print(f"Processed frame {frame_id}: Saved flow map to {os.path.basename(output_filepath)}")
        else:
            print(f"Error: Inference for frame {frame_id} returned invalid shape {flow_map.shape}. Skipping save.")


    print("--- Optical Flow Processing Complete ---")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python generate_flow.py <input_img_dir> <output_flow_dir> [cv_mode]")
        print("  [cv_mode]: Optional. Pass 'cv' to use the inference_cv method.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    use_cv = len(sys.argv) == 4 and sys.argv[3].lower() == 'cv'
            
    calculate_and_save_flow_custom(
        input_dir, 
        output_dir, 
        use_cv_inference=use_cv
    )