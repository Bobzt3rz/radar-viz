#!/usr/bin/env python3

"""
stereo_depth_map_generator.py

Generates dense depth maps from stereo camera pairs (LEFT and RIGHT).
Overlays the depth map on the image for visual calibration quality checking.

Uses the stereo calibration parameters from URBAN_D0 dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2


# --- Configuration ---
DATASET_ROOT = Path.home() / "Downloads" / "URBAN_D0-20251112T180635Z-1-002" / "URBAN_D0"

# Output configuration
FRAME_LIMIT = 100  # Process this many frames
OUTPUT_BLEND_ALPHA = 0.6  # Transparency for depth overlay (0=only image, 1=only depth)

# Stereo matching parameters - TUNED for urban driving scenes
STEREO_BLOCK_SIZE = 5  # Smaller = more detail, but noisier
STEREO_NUM_DISPARITIES = 96  # Reduced from 128 for better performance
STEREO_MIN_DISPARITY = 0

# Post-processing
USE_WLS_FILTER = True  # Weighted Least Squares filter for smoothing
WLS_LAMBDA = 8000.0
WLS_SIGMA = 1.5

# --- Stereo Calibration Parameters from URBAN_CALIBRATION_CAMERA ---

# Camera1 (LEFT) Intrinsic Matrix
K_LEFT = np.array([
    [646.955997072896, 0.0, 374.835206750483],
    [0.0, 647.313486650587, 274.110353312482],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Camera1 (LEFT) Distortion
D_LEFT = np.array([-0.233159251613014, 0.142542553330781, 0.0, 0.0], dtype=np.float64)

# Camera2 (RIGHT) Intrinsic Matrix
K_RIGHT = np.array([
    [647.665206888116, 0.0, 367.691476534482],
    [0.0, 647.665543907575, 285.201609563427],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Camera2 (RIGHT) Distortion
D_RIGHT = np.array([-0.231756400305989, 0.129011020676044, 0.0, 0.0], dtype=np.float64)

# Rotation matrix (from LEFT to RIGHT camera)
R = np.array([
    [0.999995421151889, -0.00180011943011057, -0.00243253885758453],
    [0.00180115338716526, 0.999998288487383, 0.000422928813994127],
    [0.00243177337188796, -0.000427308253070042, 0.999996951938317]
], dtype=np.float64)

# Translation vector (from LEFT to RIGHT camera, in millimeters)
T_mm = np.array([-497.793948992697, -2.18455982352058, 3.78473995227301])
T = T_mm / 1000.0  # Convert to meters

# Baseline in meters (important for depth calculation)
BASELINE = np.linalg.norm(T)  # ~0.498 meters


# --- Helper Functions ---

def load_timestamps(file_path: Path) -> Dict[str, float]:
    """Load timestamp file mapping frame_id to timestamp."""
    timestamps: Dict[str, float] = {}
    if not file_path.exists():
        print(f"Error: Timestamp file not found: {file_path}")
        return timestamps
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                frame_id = parts[0]
                ts = float(parts[1]) + float(parts[2]) * 1e-9
                timestamps[frame_id] = ts
    return timestamps


def rectify_stereo_pair(img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rectify stereo image pair using the calibration parameters.
    
    Returns:
        - img_left_rect: Rectified left image
        - img_right_rect: Rectified right image
        - Q: Disparity-to-depth reprojection matrix
        - roi_left: Valid region of interest in left image
    """
    img_size = (img_left.shape[1], img_left.shape[0])  # (width, height)
    
    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_LEFT, D_LEFT,
        K_RIGHT, D_RIGHT,
        img_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 0 = crop to valid pixels only, 1 = keep all pixels
    )
    
    # Compute rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_LEFT, D_LEFT, R1, P1, img_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_RIGHT, D_RIGHT, R2, P2, img_size, cv2.CV_32FC1
    )
    
    # Apply rectification
    img_left_rect = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    return img_left_rect, img_right_rect, Q, roi_left


def compute_stereo_depth(img_left_rect: np.ndarray, img_right_rect: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute depth map from rectified stereo pair with improved matching and filtering.
    
    Returns:
        depth_map: Depth in meters (H, W)
    """
    # Convert to grayscale for stereo matching
    gray_left = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)
    
    # Create LEFT matcher with tuned parameters for urban scenes
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=STEREO_MIN_DISPARITY,
        numDisparities=STEREO_NUM_DISPARITIES,
        blockSize=STEREO_BLOCK_SIZE,
        P1=8 * 3 * STEREO_BLOCK_SIZE**2,
        P2=32 * 3 * STEREO_BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,  # Increased from 10 for more confidence
        speckleWindowSize=50,  # Reduced for less aggressive filtering
        speckleRange=2,  # Reduced from 32 for stricter filtering
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute left disparity
    disparity_left = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    if USE_WLS_FILTER:
        # Create RIGHT matcher for WLS filtering
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        
        # Compute right disparity
        disparity_right = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0
        
        # Create WLS filter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(WLS_LAMBDA)
        wls_filter.setSigmaColor(WLS_SIGMA)
        
        # Apply filter
        disparity = wls_filter.filter(disparity_left, gray_left, None, disparity_right)
    else:
        disparity = disparity_left
    
    # Extract focal length and baseline from Q matrix
    focal_length = Q[2, 3]
    baseline = 1.0 / Q[3, 2]
    
    # Compute depth (avoid division by zero)
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    valid_mask = disparity > 0.5  # Minimum disparity threshold
    depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
    
    # Filter out unrealistic depths
    depth_map[depth_map < 0.5] = 0  # Too close
    depth_map[depth_map > 100.0] = 0  # Too far
    
    # Additional post-processing: median filter to remove salt-and-pepper noise
    depth_map = cv2.medianBlur(depth_map, 5)
    
    return depth_map


def create_depth_visualization(
    img: np.ndarray,
    depth_map: np.ndarray,
    alpha: float = 0.6,
    min_depth: float = 1.0,  # Start at 1m instead of 0.5m
    max_depth: float = 80.0  # Cap at 80m for urban scenes
) -> np.ndarray:
    """
    Create a visualization by overlaying colored depth map on the image.
    
    Args:
        img: Original image (H, W, 3)
        depth_map: Depth in meters (H, W)
        alpha: Blending factor (0=only image, 1=only depth)
        min_depth: Minimum depth for color mapping
        max_depth: Maximum depth for color mapping
    
    Returns:
        Blended image with depth overlay
    """
    # Create a mask for valid depth values
    valid_mask = (depth_map > 0) & (depth_map < max_depth)
    
    # Normalize depth to 0-255 range for colormap
    depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
    depth_clipped = np.clip(depth_map, min_depth, max_depth)
    depth_normalized[valid_mask] = (
        255 * (max_depth - depth_clipped[valid_mask]) / (max_depth - min_depth)
    ).astype(np.uint8)
    
    # Apply colormap (closer = red/yellow, farther = blue/purple)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Set invalid regions to black
    depth_colored[~valid_mask] = [0, 0, 0]
    
    # Blend with original image
    blended = cv2.addWeighted(img, 1 - alpha, depth_colored, alpha, 0)
    
    # Add depth scale legend
    blended = add_depth_scale(blended, min_depth, max_depth)
    
    return blended


def add_depth_scale(img: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Add a color scale legend to the image."""
    h, w = img.shape[:2]
    
    # Create color bar
    bar_width = 30
    bar_height = 200
    bar_x = w - bar_width - 20
    bar_y = 50
    
    # Generate gradient
    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8)
    gradient = np.tile(gradient.reshape(-1, 1), (1, bar_width))
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    
    # Place gradient on image
    img_copy = img.copy()
    img_copy[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = gradient_colored
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255)
    
    # Top label (min depth - closer)
    cv2.putText(img_copy, f"{min_depth:.1f}m", 
                (bar_x - 45, bar_y + 10), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Bottom label (max depth - farther)
    cv2.putText(img_copy, f"{max_depth:.1f}m", 
                (bar_x - 45, bar_y + bar_height), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img_copy


def process_stereo_pair(
    img_left_path: Path,
    img_right_path: Path,
    output_dir: Path,
    frame_id: str
) -> bool:
    """
    Process a stereo pair and generate depth map visualization.
    
    Returns True on success, False on failure.
    """
    # Load images
    img_left = cv2.imread(str(img_left_path))
    img_right = cv2.imread(str(img_right_path))
    
    if img_left is None or img_right is None:
        print(f"Error: Could not load stereo pair for frame {frame_id}")
        return False
    
    # Ensure images have the same size
    if img_left.shape != img_right.shape:
        print(f"Error: Image size mismatch for frame {frame_id}")
        return False
    
    # Rectify stereo pair
    img_left_rect, img_right_rect, Q, roi = rectify_stereo_pair(img_left, img_right)
    
    # Compute depth map
    depth_map = compute_stereo_depth(img_left_rect, img_right_rect, Q)
    
    # Create visualization
    depth_vis = create_depth_visualization(img_left_rect, depth_map, alpha=OUTPUT_BLEND_ALPHA)
    
    # Save outputs
    output_path_vis = output_dir / f"{frame_id}_depth_overlay.png"
    output_path_depth = output_dir / f"{frame_id}_depth_raw.png"
    
    # Save visualization
    cv2.imwrite(str(output_path_vis), depth_vis)
    
    # Save raw depth map as 16-bit PNG (depth * 256 for millimeter precision)
    depth_mm = (depth_map * 1000).astype(np.uint16)  # Convert to millimeters
    cv2.imwrite(str(output_path_depth), depth_mm)
    
    # Calculate statistics
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) > 0:
        print(f"  Frame {frame_id}: "
              f"Mean depth={np.mean(valid_depths):.2f}m, "
              f"Min={np.min(valid_depths):.2f}m, "
              f"Max={np.max(valid_depths):.2f}m, "
              f"Valid pixels={len(valid_depths)}/{depth_map.size} "
              f"({100*len(valid_depths)/depth_map.size:.1f}%)")
    
    return True


def main():
    """Main processing function."""
    print("Starting Stereo Depth Map Generator...")
    print(f"Dataset Root: {DATASET_ROOT}\n")
    
    if not DATASET_ROOT.is_dir():
        print(f"Error: Dataset root not found at {DATASET_ROOT}", file=sys.stderr)
        sys.exit(1)
    
    # Define paths
    left_img_dir = DATASET_ROOT / "1_IMAGE" / "LEFT"
    right_img_dir = DATASET_ROOT / "1_IMAGE" / "RIGHT"
    
    output_dir = Path("output") / "stereo_depth_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Print calibration info
    print("=== Stereo Calibration Info ===")
    print(f"Baseline: {BASELINE:.4f} meters ({BASELINE*1000:.2f} mm)")
    print(f"Left Camera Focal Length: {K_LEFT[0,0]:.2f} pixels")
    print(f"Right Camera Focal Length: {K_RIGHT[0,0]:.2f} pixels")
    print(f"Stereo Parameters:")
    print(f"  Block Size: {STEREO_BLOCK_SIZE}")
    print(f"  Num Disparities: {STEREO_NUM_DISPARITIES}")
    print(f"  Blend Alpha: {OUTPUT_BLEND_ALPHA}\n")
    
    # Load timestamps (we'll just process frames in order)
    left_ts_map = load_timestamps(DATASET_ROOT / "1_IMAGE" / "timestamp_image_left.txt")
    right_ts_map = load_timestamps(DATASET_ROOT / "1_IMAGE" / "timestamp_image_right.txt")
    
    if not left_ts_map or not right_ts_map:
        print("Error: Could not load timestamp files.")
        sys.exit(1)
    
    # Find common frame IDs (should be synchronized)
    common_frames = sorted(set(left_ts_map.keys()) & set(right_ts_map.keys()))
    
    print(f"Found {len(common_frames)} common stereo pairs")
    print(f"Processing up to {FRAME_LIMIT} frames...\n")
    
    # Process frames
    frames_processed = 0
    frames_success = 0
    
    for frame_id in common_frames[:FRAME_LIMIT]:
        left_path = left_img_dir / f"{frame_id}.png"
        right_path = right_img_dir / f"{frame_id}.png"
        
        if process_stereo_pair(left_path, right_path, output_dir, frame_id):
            frames_success += 1
        
        frames_processed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Processed: {frames_processed} frames")
    print(f"Successful: {frames_success} frames")
    print(f"Failed: {frames_processed - frames_success} frames")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - *_depth_overlay.png: Visualization with depth overlay")
    print(f"  - *_depth_raw.png: Raw depth in millimeters (16-bit)")


if __name__ == "__main__":
    main()