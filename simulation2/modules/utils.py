from PIL import Image
import os
import numpy as np
from typing import Optional # Import Optional if you haven't already

def save_image(img_array: np.ndarray, file_path: str) -> bool:
    """
    Saves a NumPy array representing an image to a specified file path.

    Args:
        img_array: The NumPy array containing image data (e.g., HxWx3 for RGB).
                   Assumes data type is compatible with Pillow (like uint8).
        file_path: The full path (including filename and extension, e.g.,
                   'output/images/frame_001.png') where the image should be saved.

    Returns:
        True if the image was saved successfully, False otherwise.
    """
    if not isinstance(img_array, np.ndarray) or img_array.size == 0:
        print(f"Error: Invalid image data provided for saving to {file_path}.")
        return False
    if not isinstance(file_path, str) or not file_path:
        print("Error: Invalid file path provided for saving image.")
        return False

    try:
        # --- 1. Ensure Directory Exists ---
        directory = os.path.dirname(file_path)
        if directory: # Check if directory part exists (not just filename)
            os.makedirs(directory, exist_ok=True) # Create directories if they don't exist

        # --- 2. Convert NumPy array to Pillow Image ---
        # Assumes img_array is in a format Pillow understands (like HxWx3 RGB uint8)
        img = Image.fromarray(img_array)

        # --- 3. Save the Image ---
        # Pillow infers format from the file extension in file_path
        img.save(file_path)
        # print(f"Image saved successfully to: {file_path}") # Optional confirmation
        return True

    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")
        return False