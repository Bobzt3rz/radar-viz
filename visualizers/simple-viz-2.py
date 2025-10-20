import numpy as np
from PIL import Image, ImageDraw
import os
import glob
from pypcd4 import PointCloud
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # Needed for colorbar placement

# --- Configuration ---
# Set the directory paths
IMAGE_DIR = '/home/bobberman/Downloads/URBAN_F0-20251009T193110Z-1-002/URBAN_F0/1_IMAGE/RIGHT' # Path to your 00xxxx.png files
RADAR_DIR = '/home/bobberman/Downloads/URBAN_F0-20251009T193110Z-1-002/URBAN_F0/3_RADAR/PCD'   # Path to your 00xxxx.pcd files
OUTPUT_DIR = '/home/bobberman/programming/radar/radar-viz/outputs/projection_results' # Directory to save images with projected points

MIN_DEPTH = 0.05 
MAX_DEPTH = 100.0
MIN_HEIGHT = -100.0  # e.g., filter out points 2 meters below the camera center
MAX_HEIGHT = 100.0   # e.g., filter out points 5 meters above the camera center

POINT_SIZE = 10 # Point size for visualization
SNR_THRESHOLD = 10.0

# --- Camera and Radar Calibration Data ---

# 1. Rotation Matrix (R_cr: Radar to Camera)
R_cr = np.array([
    [0.999925123881632, -0.0117293489522970, 0.00348840987535661],
    [0.0116809544074019, 0.999839501963135, 0.0135840206949982],
    [-0.00364718171132664, -0.0135422556195484, 0.999901647852577]
], dtype=np.float64)

# 2. Translation Vector (t_cr: Radar to Camera)
# TODO: HM seems like i needed to negate x to fit the points better... check this in more detail
t_cr = np.array([-0.239261664237513, 0.9462445453737781, 1.307386642291325], dtype=np.float64).reshape(3, 1)

# 3. Camera Intrinsic Matrix (K) - Camera2 (RIGHT)
K = np.array([
    [647.665206888116, 0, 367.691476534482],
    [0, 647.665543907575, 285.201609563427],
    [0, 0, 1]
], dtype=np.float64)

# 4. Radial Distortion Coefficients (D) - k1, k2
D = np.array([-0.231756400305989, 0.129011020676044], dtype=np.float64) 


def read_radar_pcd(filepath):
    """
    Reads a PCD file, extracts x, y, z, and power, and filters points 
    based on a global SNR_THRESHOLD.
    
    Returns:
        np.ndarray: A 3xN numpy array of [x, y, z] points for points above the threshold.
    """
    try:
        pc: PointCloud = PointCloud.from_path(filepath)
        
        # 1. Extract necessary fields (x, y, z, power)
        # Note: The 'power' field is being used as the SNR proxy based on your PCD description.
        # Result is an N x 4 array
        points_and_power_Nx4 = pc.numpy(("x", "y", "z", "power"))
        
        # 2. Filter by power/SNR threshold
        power_values = points_and_power_Nx4[:, 3]
        snr_mask = power_values >= SNR_THRESHOLD
        
        # 3. Apply mask and extract only x, y, z coordinates
        points_filtered_Nx3 = points_and_power_Nx4[snr_mask, :3]
        
        # 4. Transpose to 3xN for matrix multiplication
        radar_points = points_filtered_Nx3.T
        
        print(f"  -> Original points: {points_and_power_Nx4.shape[0]}, Filtered points: {radar_points.shape[1]}")

        return radar_points
        
    except Exception as e:
        print(f"Error reading {filepath} with pypcd4: {e}")
        return None


def project_radar_to_image(radar_points_3xN, K, R_cr, t_cr, D, image_shape):
    """
    Projects 3D radar points onto the 2D image plane and returns pixel coords, 
    depths, and the mask of points that were visible/valid.
    """
    H, W = image_shape
    N = radar_points_3xN.shape[1]
    
    # 1. Extrinsic Transformation (Radar -> Camera Coordinates)
    P_c = R_cr @ radar_points_3xN + t_cr
    
    # Extract ALL depths and heights for masking
    all_depths = P_c[2, :] # z_c
    all_heights = P_c[1, :] # y_c
    
    # --- CRITICAL FILTERING STEP ---
    # Combine depth, and new height filters
    depth_mask = (all_depths > MIN_DEPTH) & (all_depths < MAX_DEPTH)
    height_mask = (all_heights > MIN_HEIGHT) & (all_heights < MAX_HEIGHT) # New Height Filter
    
    # The final set of valid 3D points
    valid_3d_mask = depth_mask & height_mask

    P_c_valid = P_c[:, valid_3d_mask]
    
    if P_c_valid.shape[1] == 0:
        return np.empty((N, 2)), np.empty((N,)), np.full(N, False) 

    # 2. Project to Normalized Coordinates
    depths = P_c_valid[2, :]
    u_norm = P_c_valid[0, :] / depths
    v_norm = P_c_valid[1, :] / depths
    
    # 3a. Apply Radial Distortion
    r2 = u_norm**2 + v_norm**2
    D_factor = 1 + D[0] * r2 + D[1] * r2**2
    
    u_dist = u_norm * D_factor
    v_dist = v_norm * D_factor
    
    # 3b. Final Pixel Mapping (Intrinsic)
    P_dist_homo = np.stack([u_dist, v_dist, np.ones_like(u_dist)])
    P_pix = K @ P_dist_homo
    
    u_pix = P_pix[0, :]
    v_pix = P_pix[1, :]
    
    # Step 2: Filter points outside the image boundaries
    in_bounds_mask = (u_pix >= 0) & (u_pix < W) & \
                     (v_pix >= 0) & (v_pix < H)
    
    # 4. Construct the full index mask (combines steps 1 and 2)
    # We need to map the in_bounds_mask back to the original index N
    
    # Create the final pixel array (N x 2) and initialize with NaNs for non-visible points
    all_pixels = np.full((N, 2), np.nan)
    
    # Find the indices in the original N points that were valid after filtering
    valid_indices = np.where(valid_3d_mask)[0]
    
    # Map the in_bounds_mask back to the original N indices
    final_visible_indices = valid_indices[in_bounds_mask]
    
    # Fill in the coordinates for the visible points
    all_pixels[final_visible_indices, 0] = u_pix[in_bounds_mask]
    all_pixels[final_visible_indices, 1] = v_pix[in_bounds_mask]
    
    # Create the final boolean mask for the main function
    final_mask = np.full(N, False)
    final_mask[final_visible_indices] = True

    return all_pixels, all_depths, final_mask


def visualize_radar_projection(image_path, radar_points, uvs, point_depths, filtered_idx, output_path, point_size=POINT_SIZE):
    """
    Creates the image visualization with a fixed output size and includes a depth colorbar 
    on the right by manually reserving space within the fixed padding.
    """
    
    # --- FIXED PARAMETERS ---
    DPI = 100 
    PAD_HORIZONTAL = 300 # Total horizontal padding (150px left, 150px right)
    PAD_VERTICAL = 200   # Total vertical padding (100px top, 100px bottom)
    
    # Colorbar specific spacing (in normalized units relative to the final figure)
    CBAR_WIDTH_NORM = 0.03 # 3% of total width for the colorbar
    CBAR_PAD_NORM = 0.015  # 1.5% padding between the image and colorbar
    CBAR_MARGIN_NORM = 0.05 # Margin for colorbar within the overall vertical padding
    
    try:
        # Load image once to get original pixel dimensions
        image = plt.imread(image_path)
        # H_IMG, W_IMG are the actual pixel dimensions of the image being plotted
        H_IMG, W_IMG = image.shape[0], image.shape[1] 
    except Exception as e:
        print(f"Could not read image for visualization: {e}")
        return

    # --- CALCULATE DYNAMIC OUTPUT SIZE ---
    W_OUT = W_IMG + PAD_HORIZONTAL
    H_OUT = H_IMG + PAD_VERTICAL
    
    # Calculate Centered Padding
    PAD_LEFT_PIX = PAD_HORIZONTAL / 2
    PAD_BOTTOM_PIX = PAD_VERTICAL / 2
    
    # --- CALCULATE NORMALIZED MARGINS (Critical for Fixed Size) ---
    
    # 1. Calculate the normalized left/bottom padding based on the total output size
    left_norm = PAD_LEFT_PIX / W_OUT
    bottom_norm = PAD_BOTTOM_PIX / H_OUT
    
    # 2. Adjust the right margin for the main image to make space for the colorbar
    # The space available for the image: (W_OUT - PAD_LEFT_PIX - PAD_RIGHT_PIX) / W_OUT 
    # The right side of the image plot is pushed inward by the CBAR_PAD_NORM and CBAR_WIDTH_NORM
    image_right_space = 1 - CBAR_PAD_NORM - CBAR_WIDTH_NORM - (PAD_HORIZONTAL - PAD_LEFT_PIX) / W_OUT
    
    # Note: top and bottom are used for the main axes and the colorbar height
    top_norm = (H_OUT - PAD_BOTTOM_PIX) / H_OUT

    # --- FIGURE SETUP ---
    fig_width_in = W_OUT / DPI
    fig_height_in = H_OUT / DPI
    
    fig, ax1 = plt.subplots(1, 1, figsize=(fig_width_in, fig_height_in))
    
    # 1. Set main plot axes position (Shrinking the right side for the colorbar)
    plt.subplots_adjust(
        left=left_norm, 
        right=image_right_space, 
        top=top_norm, 
        bottom=bottom_norm
    ) 
    
    # Filter visible points
    visible_points = uvs[filtered_idx]
    visible_depths = point_depths[filtered_idx]
    
    # Display image
    ax1.imshow(image)
    ax1.set_xlim(0, W_IMG)
    ax1.set_ylim(H_IMG, 0)
    
    # 2. Scatter plot
    scatter = ax1.scatter( # Scatter is saved to a variable for use with colorbar
        visible_points[:, 0], 
        visible_points[:, 1], 
        c=visible_depths,      
        cmap='jet',            
        vmin=MIN_DEPTH,        
        vmax=MAX_DEPTH,        
        s=point_size,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8
    )
    
    ax1.axis('off')

    # --- COLORBAR CREATION (Manual Axes) ---
    
    # Calculate the position for the new colorbar axes (cax)
    # The cax should sit between image_right_space and the total right margin (1 - PAD_RIGHT_PIX / W_OUT)
    cax_left = image_right_space + CBAR_PAD_NORM
    cax_bottom = bottom_norm + CBAR_MARGIN_NORM
    cax_width = CBAR_WIDTH_NORM
    cax_height = top_norm - bottom_norm - 2 * CBAR_MARGIN_NORM # Height should match main plot minus margin
    
    # Create the new axes in the margin
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    
    # Plot the colorbar into the manually created axes
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Depth (m)', fontsize=12, weight='bold')

    # --- SAVE THE FIGURE ---
    # Save with explicit DPI and zero padding to guarantee W_OUT x H_OUT pixel output.
    fig.savefig(
        output_path, 
        dpi=DPI, 
        pad_inches=0 
    )
    
    plt.close(fig)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '00????.png')))
    
    if not image_files:
        print(f"No PNG files found in {IMAGE_DIR}. Please check path and file format.")
        return

    # Process files
    for image_path in image_files:
        frame_id = os.path.splitext(os.path.basename(image_path))[0]
        pcd_path = os.path.join(RADAR_DIR, f'{frame_id}.pcd')
        output_path = os.path.join(OUTPUT_DIR, f'{frame_id}_depth_colored.png')

        if not os.path.exists(pcd_path):
            print(f"Skipping {frame_id}: Corresponding PCD file not found.")
            continue
            
        print(f"Processing frame {frame_id}...")
        
        # 1. Load Image
        try:
            img = Image.open(image_path).convert("RGB")
            W, H = img.size 
            draw = ImageDraw.Draw(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

       # 2. Load Radar Points (3xN)
        radar_points_3xN = read_radar_pcd(pcd_path)
        if radar_points_3xN is None:
            continue
        
        # Transpose to (N, 3) for easier indexing in the main loop/visualization
        radar_points_Nx3 = radar_points_3xN.T

        all_uvs, all_depths, final_mask = project_radar_to_image(
            radar_points_3xN, K, R_cr, t_cr, D, (H, W)
        )
            
        # 4. Visualize and Save
        if np.sum(final_mask) == 0:
            print(f"Frame {frame_id}: No points projected within image bounds.")
            continue
            
        visualize_radar_projection(
            image_path,
            radar_points_Nx3, 
            all_uvs, 
            all_depths, 
            final_mask,
            output_path
        )
        print(f"Saved depth-colored projection for {frame_id} to {output_path}")

if __name__ == '__main__':
    main()