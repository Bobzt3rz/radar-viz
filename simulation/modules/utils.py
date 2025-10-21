# In main.py or a utils.py
import numpy as np

def save_as_ply(points: np.ndarray, filename: str):
    """Saves an (N, 3) point cloud as a .ply file."""
    # Create the PLY header
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    
    # Use np.savetxt to write the points, prepending the header
    with open(filename, 'w') as f:
        f.write(header)
        # Use a simple format for the data
        np.savetxt(f, points, fmt='%f %f %f')
    
    print(f"Saved {len(points)} points to {filename}")