import numpy as np
from PIL import Image
import os

def create_random_noise_texture(width=256, height=256, filename="random_noise_texture.png"):
    """
    Generates a random noise texture image and saves it to a file.

    This type of texture is generally good for optical flow algorithms
    because it provides unique local gradients and features for tracking.

    Args:
        width (int): The width of the texture image in pixels.
        height (int): The height of the texture image in pixels.
        filename (str): The name of the file to save the texture as (e.g., 'texture.png').
    """
    try:
        # 1. Generate random pixel data
        # Creates a (height, width, 3) array with random floats between 0.0 and 1.0
        random_data = np.random.rand(height, width, 3)

        # 2. Scale to 0-255 and convert to unsigned 8-bit integer format
        image_data = (random_data * 255).astype(np.uint8)

        # 3. Create PIL Image object from the numpy array
        img = Image.fromarray(image_data, 'RGB')

        # 4. Save the image to the specified file
        filepath = os.path.abspath(filename) # Get absolute path for clarity
        img.save(filepath)

        print(f"Successfully created random noise texture: {filepath}")
        print(f"Image size: {width}x{height}")

    except Exception as e:
        print(f"Error creating texture file '{filename}': {e}")

# --- Configuration ---
TEXTURE_WIDTH = 256
TEXTURE_HEIGHT = 256
OUTPUT_FILENAME = "optical_flow_texture.png" # You can change this name

# --- Generate the texture ---
if __name__ == "__main__":
    # Ensure the script generates the texture in the current directory
    # or specify a full path in OUTPUT_FILENAME if needed.
    create_random_noise_texture(TEXTURE_WIDTH, TEXTURE_HEIGHT, OUTPUT_FILENAME)

    print("\nHow to use:")
    print(f"1. A file named '{OUTPUT_FILENAME}' has been created in the current directory.")
    print( "2. Update the TEXTURE_FILE variable in your 'main.py' to point to this file:")
    print(f"   TEXTURE_FILE = \"/path/to/your/directory/{OUTPUT_FILENAME}\"")
    print( "3. Rerun 'main.py'.")