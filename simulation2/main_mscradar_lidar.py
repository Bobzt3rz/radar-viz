#!/usr/bin/env python3

"""
main_mscradar_lidar_SYNCED_GREEDY.py

Sanity check script to project LiDAR points onto the **LEFT** camera image.

This script implements a **greedy 1-to-1 synchronization**:
1.  For each image, it finds the (un-used) LiDAR frame with the
    closest timestamp.
2.  It only accepts the match if `abs(ts_img - ts_lidar) < MAX_TIME_DIFFERENCE`.
3.  Once a LiDAR frame is matched, it is "removed" and cannot be
    matched to another image.
4.  Uses LEFT camera intrinsics and ORIGINAL LiDAR extrinsics.
"""

import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
from pypcd4 import PointCloud
import open3d as o3d

import numpy as np
import cv2
from scipy.spatial.transform import Rotation


# --- 1. Dataset Configuration & Constants ---

# !!! IMPORTANT: Update this path to your local copy of the dataset !!!
DATASET_ROOT = Path.home() / "Downloads" / "URBAN_D0-20251112T180635Z-1-002" / "URBAN_D0"

if not DATASET_ROOT.is_dir():
    print(f"Error: Dataset root not found at {DATASET_ROOT}", file=sys.stderr)
    print("Please update the DATASET_ROOT variable in this script.", file=sys.stderr)
    sys.exit(1)

# Lidar points with intensity < this value will be filtered out.
INTENSITY_THRESHOLD = 0.0 
FRAME_LIMIT = 9999 # Process this many frames

# ---
# Synchronization bounds
# ---
# The maximum allowed time difference (in seconds) between an image
# and its matched LiDAR frame. 0.1 = 100ms.
MAX_TIME_DIFFERENCE = 0.50

# --- 2. Data Loading Functions ---

def load_timestamps(file_path: Path) -> Dict[str, float]:
    """
    Loads a timestamp file.
    Format: `frame_id sec nsec`
    
    Returns a dictionary mapping frame_id (str) to timestamp (float).
    """
    timestamps: Dict[str, float] = {}
    if not file_path.exists():
        print(f"Error: Timestamp file not found: {file_path}")
        return timestamps
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3: # frame_id sec nsec
                frame_id = parts[0]
                ts = float(parts[1]) + float(parts[2]) * 1e-9
                timestamps[frame_id] = ts
    return timestamps

def load_lidar_pcd(file_path: Path) -> PointCloud:
    """
    Parses a .pcd file (including binary_compressed)
    using the pypcd4 library.
    
    Returns a pypcd4.PointCloud object or None on failure.
    """
    if not file_path.exists():
        return None

    try:
        pc = PointCloud.from_path(str(file_path))
        return pc
    except Exception as e:
        print(f"Error loading PCD file {file_path}: {e}", file=sys.stderr)
        return None


# --- 3. Preprocessing Functions ---

def preprocess_lidar_points(pc: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 3D points from LiDAR PCD and filters by INTENSITY_THRESHOLD.
    Assumes points are ALREADY in the "Paper" frame.
    
    Returns:
    - p_paper_lidar (N, 3): Points in "Paper" lidar frame (Z-fwd, X-right, Y-down).
    - ranges_meters (N,): Range data for color-coding.
    """
    if pc is None:
        return np.array([]), np.array([])
        
    pc_data = pc.pc_data
    
    # 1. Extract necessary fields
    try:
        data = np.vstack([
            pc_data['x'], pc_data['y'], pc_data['z'],
            pc_data['intensity'], pc_data['range']
        ]).T.astype(np.float64)
    except (ValueError, KeyError) as e:
        print(f"  Error: PCD missing required fields (x, y, z, intensity, range). {e}")
        return np.array([]), np.array([])

    # 2. Filter by intensity threshold
    intensity_values = data[:, 3]
    mask = intensity_values >= INTENSITY_THRESHOLD
    
    # 3. Apply mask and extract points and ranges
    # We assume these are already in the "Paper-Lidar" frame.
    p_paper_lidar = data[mask, :3]
    
    # Range is in uint32 millimeters, convert to float meters
    ranges_meters = data[mask, 4]
    
    return p_paper_lidar, ranges_meters


# --- 4. Sanity Check Functions ---
def project_points_to_image(
    points_3d_paper_lidar: np.ndarray, 
    K: np.ndarray, 
    T_paper_cam_from_lidar: np.ndarray,
    original_color_data: np.ndarray  ### NEW PARAMETER
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ### NEW RETURN
    """
    Projects 3D points from the "Paper-Lidar" frame onto the 2D image plane.
    Assumes all inputs are ALREADY in the "Paper" coordinate system.
    
    Returns filtered 2D points, camera-Z-depths, and filtered color data.
    """
    if points_3d_paper_lidar.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]) ### NEW RETURN

    num_points = points_3d_paper_lidar.shape[0]
    points_3d_hom = np.hstack((points_3d_paper_lidar, np.ones((num_points, 1))))
    
    points_cam_hom = (T_paper_cam_from_lidar @ points_3d_hom.T).T
    points_cam = points_cam_hom[:, 0:3]
    
    depths = points_cam[:, 2] # Camera Z-depths
    
    valid_mask = depths > 0.1  # 10cm threshold
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([]) ### NEW RETURN
        
    points_cam_valid = points_cam[valid_mask]
    depths_valid = depths[valid_mask]
    
    ### NEW: Filter the color_data with the SAME MASK
    color_data_valid = original_color_data[valid_mask]
    
    u_norm = points_cam_valid[:, 0] / depths_valid
    v_norm = points_cam_valid[:, 1] / depths_valid
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u_pix = fx * u_norm + cx
    v_pix = fy * v_norm + cy
    
    points_2d = np.vstack((u_pix, v_pix)).T
    
    return points_2d, depths_valid, color_data_valid ### NEW RETURN

def draw_points_on_image(
    image: np.ndarray, 
    points_2d: np.ndarray, 
    depth_data: np.ndarray,  # I renamed 'depths' to 'depth_data' for clarity
    percentile_min: float = 5.0,
    percentile_max: float = 95.0
) -> np.ndarray:
    """
    Draws 2D points on an image, color-coded by depth_data (e.g., range).
    
    This version uses dynamic percentile clipping for better color contrast
    and a perceptually uniform colormap (VIRIDIS).
    """
    if points_2d.shape[0] == 0 or depth_data.shape[0] == 0:
        return image

    # 1. Calculate dynamic min/max clips using percentiles
    # This auto-adjusts the colormap to the relevant data range
    # in this specific frame, dramatically improving contrast.
    min_clip = np.percentile(depth_data, percentile_min)
    max_clip = np.percentile(depth_data, percentile_max)
    
    # Add a small epsilon to prevent division by zero if all points are the same
    epsilon = 1e-6 

    # 2. Clip data to this dynamic range
    depths_clipped = np.clip(depth_data, min_clip, max_clip)
    
    # 3. Normalize the clipped data
    depths_norm = (depths_clipped - min_clip) / (max_clip - min_clip + epsilon)
    
    # 4. Scale to 8-bit for the colormap
    depths_8bit = (depths_norm * 255).astype(np.uint8)
    
    # 5. Apply a perceptually uniform colormap (better than JET)
    # Good alternatives: cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA
    colors_mapped = cv2.applyColorMap(depths_8bit, cv2.COLORMAP_VIRIDIS)
    
    # --- Drawing logic (same as before) ---
    img_with_points = image.copy()
    h, w = img_with_points.shape[:2]
    
    # Ensure we don't have a mismatch (shouldn't happen, but safe)
    num_points = min(points_2d.shape[0], colors_mapped.shape[0])
    
    for i in range(num_points):
        u, v = points_2d[i]
        
        if 0 <= u < w and 0 <= v < h:
            color = tuple(int(c) for c in colors_mapped[i, 0, :])
            cv2.circle(img_with_points, (int(u), int(v)), 1, color, -1)
            
    return img_with_points

# --- 5. Main Execution ---
def main():
    """
    Main function to load, preprocess, and project.
    """
    print(f"Starting MSC-RAD4R **LiDAR** Projection Sanity Check...")
    print(f"Targeting **LEFT** Camera (per CALIBRATION_README.txt)...")
    print(f"Using **Greedy 1-to-1 Sync (Bound: {MAX_TIME_DIFFERENCE*1000:.0f}ms)**...")
    print(f"Dataset Root: {DATASET_ROOT}\n")

    # --- Define sub-paths ---
    lidar_dir_base = DATASET_ROOT / "2_LIDAR"
    lidar_pcd_dir = lidar_dir_base / "PCD"
    img_dir = DATASET_ROOT / "1_IMAGE" / "LEFT"
    
    output_dir = Path("output") / "lidar_projections"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving projection images to: {output_dir}")

    # --- 1. Load Calibration Data ---
    print("--- 1. Loading Calibration Data (for **LEFT** Camera) ---")
    
    K = np.array([
        [647.665206888116, 0, 367.691476534482],
        [0, 647.665543907575, 285.201609563427],
        [0, 0, 1]
    ], dtype=np.float64)

    D = np.array([-0.231756400305989, 0.129011020676044, 0.0, 0.0, 0.0], dtype=np.float64) 

    r_cl = np.array([1.226053849071323, -1.196572101163224, 1.192295102346085])
    t_cl = np.array([0.1278447154599633, -0.314979325909709, -0.9064288085364153])

    R, _ = cv2.Rodrigues(r_cl)

    T_paper_cam_from_lidar = np.eye(4, dtype=np.float64)
    T_paper_cam_from_lidar[0:3, 0:3] = R
    T_paper_cam_from_lidar[0:3, 3] = t_cl
    
    print(f"Loaded K_paper (Left Cam):\n{K}")
    print(f"Loaded T_paper_cam_from_lidar (Paper Frame, ORIGINAL):\n{T_paper_cam_from_lidar}\n")

    # --- 2. Load All Timestamps ---
    print("--- 2. Loading timestamps for synchronization ---")
    image_ts_map = load_timestamps(DATASET_ROOT / "1_IMAGE" / "timestamp_image_left.txt")
    lidar_ts_pool = load_timestamps(lidar_dir_base / "timestamp_lidar.txt")
    
    if not image_ts_map or not lidar_ts_pool:
        print("Error: Could not load timestamp files. Exiting.")
        return

    print(f"Loaded {len(image_ts_map)} image timestamps.")
    print(f"Loaded {len(lidar_ts_pool)} LiDAR timestamps into pool.")

    # --- 3. STAGE 1: Find all 1-to-1 matches ---
    print(f"--- 3. Finding 1-to-1 matches within {MAX_TIME_DIFFERENCE*1000:.0f}ms ---")
    
    sorted_image_items = sorted(image_ts_map.items(), key=lambda item: item[1])
    matches: List[Tuple[str, str]] = []
    
    for img_frame_id, img_ts in sorted_image_items:
        if not lidar_ts_pool:
            print("  LiDAR pool is empty. Stopping match search.")
            break
            
        best_lidar_id = min(
            lidar_ts_pool, 
            key=lambda k: abs(img_ts - lidar_ts_pool[k])
        )
        
        best_diff = abs(img_ts - lidar_ts_pool[best_lidar_id])
        
        if best_diff <= MAX_TIME_DIFFERENCE:
            matches.append((img_frame_id, best_lidar_id))
            del lidar_ts_pool[best_lidar_id]
        else:
            pass
            
    print(f"Found {len(matches)} 1-to-1 matches.")

    # --- 4. STAGE 2: Process and Project the Matched Frames ---
    print(f"--- 4. Starting SYNCED Projection Loop (up to {FRAME_LIMIT} frames) ---")

    frames_processed = 0
    for img_frame_id, lidar_frame_id in matches:
        if frames_processed >= FRAME_LIMIT:
            print(f"Reached {FRAME_LIMIT} frame limit. Exiting loop.")
            break
            
        img_path = img_dir / f"{img_frame_id}.png"
        lidar_path = lidar_pcd_dir / f"{lidar_frame_id}.pcd"
        
        # 1. Load and Undistort Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image {img_path.name}. Skipping.")
            continue
        img_undistorted = cv2.undistort(img, K, D)
        
        # 2. Load and Preprocess Lidar Data
        pc_lidar = load_lidar_pcd(lidar_path)
        
        if pc_lidar is None:
            print(f"Warning: File missing: {lidar_path.name}. Skipping.")
            continue
        
        p_paper_lidar, color_data = preprocess_lidar_points(pc_lidar)
        
        if p_paper_lidar.shape[0] == 0:
            print(f"Warning: No points passed intensity threshold in {lidar_frame_id}.pcd. Skipping.")
            continue

        # --- 3. Transform Points to Camera Frame ---
        num_points = p_paper_lidar.shape[0]
        points_lidar_hom = np.hstack((
            p_paper_lidar, 
            np.ones((num_points, 1))
        ))
        
        points_cam_hom = (T_paper_cam_from_lidar @ points_lidar_hom.T).T
        p_paper_cam = points_cam_hom[:, 0:3]
            
        # 6. Efficient Projection (re-using p_paper_cam)
        depths = p_paper_cam[:, 2] # Camera Z-depths
        
        valid_mask = depths > 0.1 # Filter points behind camera
        if not np.any(valid_mask):
            points_2d = np.array([])
            color_data_filtered = np.array([])
        else:
            points_cam_valid = p_paper_cam[valid_mask]
            depths_valid = depths[valid_mask]
            color_data_filtered = color_data[valid_mask]
            
            # Project to 2D
            u_norm = points_cam_valid[:, 0] / depths_valid
            v_norm = points_cam_valid[:, 1] / depths_valid
            
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            u_pix = fx * u_norm + cx
            v_pix = fy * v_norm + cy
            
            points_2d = np.vstack((u_pix, v_pix)).T

        # 7. Draw Points
        img_with_points = draw_points_on_image(img_undistorted, points_2d, color_data_filtered)
        
        # 8. Save Image
        output_path = output_dir / f"{img_frame_id}_sync_with_{lidar_frame_id}.png"
        cv2.imwrite(str(output_path), img_with_points)
        print(f"  Saved: {output_path.name} ({points_2d.shape[0]} points)")
        frames_processed += 1

    print(f"\n\nSanity check complete. Processed {frames_processed} matched frames.")
    print(f"Check the projection results in: {output_dir}")


if __name__ == "__main__":
    if not DATASET_ROOT.is_dir():
        print("Please update the DATASET_ROOT variable at the top of this script.")
    else:
        main()