#!/usr/bin/env python3

"""
main_mscradar.py

Main executable for running the radar-camera velocity estimation pipeline
on the MSC-RAD4R dataset.

(Fix 6: Reverted to color-by-depth and removed SNR filtering
         per user request.)
"""

import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
from pypcd4 import PointCloud

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

# ---
# Coordinate System Definitions
# ---
# "Paper" Frame: +Z Forward, +X left, +Y Down
# ---

# === CHANGE 1: Set threshold to -infinity to plot ALL points ===
SNR_THRESHOLD = -np.inf # User request: Plot all points, no filtering
FRAME_LIMIT = 9999 # Process this many frames

# ---
# Synchronization bounds
# ---
MAX_TIME_DIFFERENCE = 0.05 # 50ms


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

def load_radar_pcd(file_path: Path) -> PointCloud:
    """
    Parses a .pcd file (including binary_compressed)
    using the pypcd4 library.
    """
    if not file_path.exists():
        return None

    try:
        pc = PointCloud.from_path(str(file_path))
        return pc
    except Exception as e:
        print(f"Error loading PCD file {file_path}: {e}", file=sys.stderr)
        return None


# --- 3. Preprocessing Functions (Following coordinate_systems.md) ---

def preprocess_radar_points(pc: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 3D points from PCD and filters by SNR_THRESHOLD.
    
    Returns:
    - p_paper_radar (N, 3): Points in "Paper" radar frame (Z-fwd, X-left, Y-down).
    - snr_values (N,): SNR data (for potential use, even if not for coloring).
    """
    if pc is None:
        return np.array([]), np.array([])
        
    pc_data = pc.pc_data
    
    try:
        points_and_power = np.vstack([
            pc_data['x'], pc_data['y'], pc_data['z'], pc_data['power']
        ]).T.astype(np.float64)
    except (ValueError, KeyError) as e:
        print(f"  Error: PCD missing required fields (x, y, z, power). {e}")
        return np.array([]), np.array([])

    # Filter by power/SNR threshold
    power_values = points_and_power[:, 3]
    snr_mask = power_values >= SNR_THRESHOLD # This mask will now be all True
    
    if not np.any(snr_mask):
        return np.array([]), np.array([])
        
    # Apply mask and extract coordinates and SNR
    p_paper_radar = points_and_power[snr_mask, :3]
    snr_values_filtered = points_and_power[snr_mask, 3]
    
    return p_paper_radar, snr_values_filtered


# --- 4. Sanity Check Functions ---

def project_points_to_image(
    points_3d_paper_radar: np.ndarray, 
    K: np.ndarray, 
    T_paper_cam_from_radar: np.ndarray,
    original_color_data: np.ndarray  # (This is SNR, but we won't use it for coloring)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects 3D points from the "Paper-Radar" frame onto the 2D image plane.
    Assumes all inputs are ALREADY in the "Paper" coordinate system.
    
    Returns filtered 2D points, camera-Z-depths, and filtered color data (SNR).
    """
    if points_3d_paper_radar.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    num_points = points_3d_paper_radar.shape[0]
    points_3d_hom = np.hstack((points_3d_paper_radar, np.ones((num_points, 1))))
    
    points_cam_hom = (T_paper_cam_from_radar @ points_3d_hom.T).T
    points_cam = points_cam_hom[:, 0:3]
    
    depths = points_cam[:, 2] # Camera Z-depths
    
    valid_mask = depths > 0.1  # 10cm threshold
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
        
    points_cam_valid = points_cam[valid_mask]
    depths_valid = depths[valid_mask]
    
    # Filter the color_data (SNR) with the SAME MASK
    color_data_valid = original_color_data[valid_mask]
    
    u_norm = points_cam_valid[:, 0] / depths_valid
    v_norm = points_cam_valid[:, 1] / depths_valid
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u_pix = fx * u_norm + cx
    v_pix = fy * v_norm + cy
    
    points_2d = np.vstack((u_pix, v_pix)).T
    
    return points_2d, depths_valid, color_data_valid # (Returns depths_valid and snr_data_valid)

def draw_points_on_image(
    image: np.ndarray, 
    points_2d: np.ndarray, 
    color_data: np.ndarray,  # This will now be the DEPTH data
    percentile_min: float = 5.0,
    percentile_max: float = 95.0
) -> np.ndarray:
    """
    Draws 2D points on an image, color-coded by color_data (e.g., DEPTH).
    
    Uses dynamic percentile clipping for better color contrast
    and a perceptually uniform colormap (VIRIDIS).
    """
    if points_2d.shape[0] == 0 or color_data.shape[0] == 0:
        return image

    min_clip = np.percentile(color_data, percentile_min)
    max_clip = np.percentile(color_data, percentile_max)
    
    epsilon = 1e-6 
    data_clipped = np.clip(color_data, min_clip, max_clip)
    data_norm = (data_clipped - min_clip) / (max_clip - min_clip + epsilon)
    data_8bit = (data_norm * 255).astype(np.uint8)
    
    colors_mapped = cv2.applyColorMap(data_8bit, cv2.COLORMAP_VIRIDIS)
    
    img_with_points = image.copy()
    h, w = img_with_points.shape[:2]
    
    num_points = min(points_2d.shape[0], colors_mapped.shape[0])
    
    for i in range(num_points):
        u, v = points_2d[i]
        
        if 0 <= u < w and 0 <= v < h:
            color = tuple(int(c) for c in colors_mapped[i, 0, :])
            cv2.circle(img_with_points, (int(u), int(v)), 2, color, -1)
            
    return img_with_points

# --- 5. Main Execution ---

def main():
    """
    Main function to load, preprocess (to "Paper" frame), and project.
    """
    print(f"Starting MSC-RAD4R **Radar** Projection Sanity Check...")
    print(f"Targeting **left** Camera...")
    print(f"Using **Greedy 1-to-1 Sync (Bound: {MAX_TIME_DIFFERENCE*1000:.0f}ms)**...")
    print(f"SNR Filter: **DISABLED** (Plotting all points)")
    print(f"Coloring by: **DEPTH**")
    print(f"Dataset Root: {DATASET_ROOT}\n")

    # --- Define sub-paths ---
    radar_pcd_dir = DATASET_ROOT / "3_RADAR" / "PCD"
    radar_ts_file = DATASET_ROOT / "3_RADAR" / "timestamp_radar.txt"
    
    img_dir = DATASET_ROOT / "1_IMAGE" / "LEFT"
    img_ts_file = DATASET_ROOT / "1_IMAGE" / "timestamp_image_left.txt"
    
    output_dir = Path("output") / "radar"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving projection images to: {output_dir}\n")

    # --- 1. Load Calibration Data ---
    print("--- 1. Loading Calibration Data (for **LEFT** Camera) ---")
    
    T_paper_cam_from_radar = np.eye(4, dtype=np.float64)
    T_paper_cam_from_radar[0:3, 0:3] = np.array([
        [0.999925123881632, -0.0117293489522970,  0.00348840987535661],
        [0.0116809544074019,  0.999839501963135,  0.0135840206949982 ],
        [-0.00364718171132664, -0.0135422556195484, 0.999901647852577 ]
    ])
    T_paper_cam_from_radar[0:3, 3] = np.array([
        0.239261664237513, 0.9462445453737781, 1.307386642291325
    ])
    
    K = np.array([
        [647.665206888116, 0, 367.691476534482],
        [0, 647.665543907575, 285.201609563427],
        [0, 0, 1]
    ], dtype=np.float64)

    D = np.array([-0.231756400305989, 0.129011020676044, 0.0, 0.0, 0.0], dtype=np.float64) 
    
    print(f"Loaded K_paper (left Cam):\n{K}")
    print(f"Loaded D_paper (left Cam):\n{D}")
    print(f"Loaded T_paper_cam_from_radar (Paper Frame):\n{T_paper_cam_from_radar}\n")


    # --- 2. Load All Timestamps ---
    print("--- 2. Loading timestamps for synchronization ---")
    image_ts_map = load_timestamps(img_ts_file)
    radar_ts_pool = load_timestamps(radar_ts_file)
    
    if not image_ts_map or not radar_ts_pool:
        print("Error: Could not load timestamp files. Exiting.")
        return

    print(f"Loaded {len(image_ts_map)} image timestamps.")
    print(f"Loaded {len(radar_ts_pool)} Radar timestamps into pool.")

    # --- 3. STAGE 1: Find all 1-to-1 matches ---
    print(f"--- 3. Finding 1-to-1 matches within {MAX_TIME_DIFFERENCE*1000:.0f}ms ---")
    
    sorted_image_items = sorted(image_ts_map.items(), key=lambda item: item[1])
    matches: List[Tuple[str, str]] = []
    
    for img_frame_id, img_ts in sorted_image_items:
        if not radar_ts_pool:
            print("  Radar pool is empty. Stopping match search.")
            break
            
        best_radar_id = min(
            radar_ts_pool, 
            key=lambda k: abs(img_ts - radar_ts_pool[k])
        )
        
        best_diff = abs(img_ts - radar_ts_pool[best_radar_id])
        
        if best_diff <= MAX_TIME_DIFFERENCE:
            matches.append((img_frame_id, best_radar_id))
            del radar_ts_pool[best_radar_id]
        else:
            pass
            
    print(f"Found {len(matches)} 1-to-1 matches.")

    # --- 4. STAGE 2: Process and Project the Matched Frames ---
    print(f"--- 4. Starting SYNCED Projection Loop (up to {FRAME_LIMIT} frames) ---")
    
    frames_processed = 0
    for img_frame_id, radar_frame_id in matches:
        if frames_processed >= FRAME_LIMIT:
            print(f"Reached {FRAME_LIMIT} frame limit. Exiting loop.")
            break
            
        img_path = img_dir / f"{img_frame_id}.png"
        radar_path = radar_pcd_dir / f"{radar_frame_id}.pcd"
        
        # 1. Load and Undistort Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image {img_path.name}. Skipping.")
            continue
        img_undistorted = cv2.undistort(img, K, D)
        
        # 2. Load and Preprocess Radar Data
        pc_radar = load_radar_pcd(radar_path)
        if pc_radar is None:
            print(f"Warning: File missing: {radar_path.name}. Skipping.")
            continue
        
        # p_paper_radar is (N, 3)
        # snr_data is (N,)
        p_paper_radar, snr_data = preprocess_radar_points(pc_radar)
        
        if p_paper_radar.shape[0] == 0:
            # This should only happen now if the file is empty
            print(f"Warning: No points found in {radar_frame_id}.pcd. Skipping.")
            continue

        # 3. Project ALL Points (using "Paper" data)
        # We still pass snr_data, but we'll ignore the filtered result
        points_2d, depths, _ = project_points_to_image(
            p_paper_radar, K, T_paper_cam_from_radar, snr_data
        )
        
        # === CHANGE 2: Pass 'depths' to 'draw_points_on_image' ===
        # 4. Draw Points, color-coded by DEPTH
        img_with_points = draw_points_on_image(img_undistorted, points_2d, depths)
        
        # 5. Save Image
        output_path = output_dir / f"{img_frame_id}_sync_with_{radar_frame_id}.png"
        cv2.imwrite(str(output_path), img_with_points)
        print(f"  Saved: {output_path.name} ({points_2d.shape[0]} points)")
        frames_processed += 1

    print(f"\n\nSanity check complete. Processed {frames_processed} matched frames.")
    print(f"Check the results in: {output_dir}")


if __name__ == "__main__":
    if not DATASET_ROOT.is_dir():
        print("Please update the DATASET_ROOT variable at the top of this script.")
    else:
        main()