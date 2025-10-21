import numpy as np
import matplotlib.pyplot as plt
from OpenGL.GLU import gluProject
from typing import Tuple

def build_velocity_map(id_map: np.ndarray, id_to_velocity: dict) -> np.ndarray:
    """
    Converts a (H, W, 3) ID map into a (H, W, 3) velocity map
    using a lookup dictionary.
    """
    H, W, _ = id_map.shape
    # Create an output array for velocities
    velocity_map = np.zeros((H, W, 3), dtype=np.float32)

    # This is the fast, vectorized way:
    for id_color_tuple, velocity in id_to_velocity.items():
        # Create a boolean mask where the ID map matches this color
        mask = np.all(id_map == id_color_tuple, axis=-1)
        # Apply the velocity to all pixels in the mask
        velocity_map[mask] = velocity

    return velocity_map

def project_to_2d(world_points: np.ndarray, modelview: np.ndarray, projection: np.ndarray, viewport: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Projects a (H, W, 3) map of 3D world points into
    a (H, W, 2) map of 2D pixel coordinates.
    """
    H, W, _ = world_points.shape
    pixel_coords = np.empty((H, W, 2), dtype=float)

    for v_idx in range(H):
        for u_idx in range(W):
            wx, wy, wz = world_points[v_idx, u_idx]
            # Handle background (0,0,0) points
            if wx == 0 and wy == 0 and wz == 0:
                pixel_coords[v_idx, u_idx] = [u_idx, v_idx] # No motion
                continue
                
            u_gl, v_gl, _ = gluProject(wx, wy, wz, modelview, projection, viewport)
            
            # Flip the 'v' coordinate from OpenGL (bottom-left)
            # to NumPy (top-left)
            v_numpy = (H - 1) - v_gl
      
            pixel_coords[v_idx, u_idx] = [u_gl, v_numpy]
            
    return pixel_coords

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

def save_flow_map(flow_map: np.ndarray, filename: str):
    """Saves an (H, W, 2) flow map as a .npy file."""
    np.save(filename, flow_map)
    print(f"Saved flow map to {filename}")

def save_flow_histogram(flow_map: np.ndarray, filename: str):
    """
    Calculates and saves a histogram of the flow vector magnitudes.
    """
    # 1. Calculate the magnitude (speed) for every pixel
    magnitudes = np.linalg.norm(flow_map, axis=2)
    
    # 2. Flatten the 2D array of magnitudes into a 1D list
    magnitudes_flat = magnitudes.flatten()
    
    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(magnitudes_flat, bins=100, range=(0, np.max(magnitudes_flat)))
    
    # 4. Use a log scale for the Y-axis (Count)
    # This is critical, as it will prevent the "zero-flow" pixels
    # from completely dwarfing all other values.
    plt.yscale('log') 
    
    plt.title('Flow Vector Magnitude Distribution')
    plt.xlabel('Flow Magnitude (pixels/frame)')
    plt.ylabel('Count (log scale)')
    
    # 5. Save the figure
    plt.savefig(filename)
    plt.close() # Close the plot to free memory
    print(f"Saved flow histogram to {filename}")

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    # rad_max = 3.0  # Set a fixed max flow of 10 pixels. Tune this value!
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)