#!/usr/bin/env python3

"""
interactive_calibration_radar.py

Interactive GUI tool for fine-tuning RADAR-Camera calibration.
Allows real-time adjustment of:
- Camera intrinsics (fx, fy, cx, cy)
- Rotation (roll, pitch, yaw) 
- Translation (tx, ty, tz)

Usage:
    python interactive_calibration_radar.py
    
Then adjust sliders and press 's' to save the current parameters.
Use 'a' and 'd' keys to navigate between frames.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
from pypcd4 import PointCloud

# --- Configuration ---
DATASET_ROOT = Path.home() / "Downloads" / "URBAN_D0-20251112T180635Z-1-002" / "URBAN_D0"

# Maximum time difference for frame synchronization (in seconds)
MAX_TIME_DIFFERENCE = 0.01

POWER_THRESHOLD = 0.0  # Minimum power threshold for radar points

# Initial calibration values (from your radar calibration)
INITIAL_FX = 646.955997072896
INITIAL_FY = 647.313486650587
INITIAL_CX = 374.835206750483
INITIAL_CY = 274.110353312482

# Original Rodrigues vector [rx, ry, rz]
INITIAL_RX = -0.01356389260362083
INITIAL_RY = 0.003567994251137598
INITIAL_RZ = 0.01170580277613928

# Original translation [tx, ty, tz]
INITIAL_TX = -0.239261664237513
INITIAL_TY = 0.9462445453737781
INITIAL_TZ = 1.307386642291325

# Distortion coefficients (we won't adjust these interactively)
D_PAPER = np.array([-0.233159251613014, 0.142542553330781, 0.0, 0.0], dtype=np.float64)


# --- Global state for sliders ---
class CalibrationState:
    def __init__(self):
        # Intrinsics (stored as integers for sliders, divide by scale factor)
        self.fx = int(INITIAL_FX * 10)  # Scale by 10 for finer control
        self.fy = int(INITIAL_FY * 10)
        self.cx = int(INITIAL_CX * 10)
        self.cy = int(INITIAL_CY * 10)
        
        # Rotation (stored as milliradians for slider precision)
        self.rx = int(INITIAL_RX * 1000)
        self.ry = int(INITIAL_RY * 1000)
        self.rz = int(INITIAL_RZ * 1000)
        
        # Translation (stored as millimeters)
        self.tx = int(INITIAL_TX * 1000)
        self.ty = int(INITIAL_TY * 1000)
        self.tz = int(INITIAL_TZ * 1000)
        
        # Frame management
        self.frame_pairs: List[Tuple[str, str]] = []  # List of (image_id, radar_id) tuples
        self.current_frame_idx = 0
        
        # Cached data for current frame
        self.img_undistorted = None
        self.points_3d_radar = None
        self.color_data = None


state = CalibrationState()


# --- Helper Functions ---

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
            if len(parts) == 3:  # frame_id sec nsec
                frame_id = parts[0]
                ts = float(parts[1]) + float(parts[2]) * 1e-9
                timestamps[frame_id] = ts
    return timestamps


def find_synchronized_frames() -> List[Tuple[str, str]]:
    """
    Find all synchronized image-radar frame pairs using greedy matching.
    Returns list of (image_frame_id, radar_frame_id) tuples.
    """
    print("Loading timestamps for frame synchronization...")
    
    image_ts_map = load_timestamps(DATASET_ROOT / "1_IMAGE" / "timestamp_image_left.txt")
    radar_ts_pool = load_timestamps(DATASET_ROOT / "3_RADAR" / "timestamp_radar.txt")
    
    if not image_ts_map or not radar_ts_pool:
        print("Error: Could not load timestamp files.")
        return []
    
    print(f"Found {len(image_ts_map)} image timestamps")
    print(f"Found {len(radar_ts_pool)} radar timestamps")
    
    # Sort images by timestamp
    sorted_image_items = sorted(image_ts_map.items(), key=lambda item: item[1])
    
    matches: List[Tuple[str, str]] = []
    
    for img_frame_id, img_ts in sorted_image_items:
        if not radar_ts_pool:
            break
        
        # Find best remaining radar frame
        best_radar_id = min(
            radar_ts_pool,
            key=lambda k: abs(img_ts - radar_ts_pool[k])
        )
        
        best_diff = abs(img_ts - radar_ts_pool[best_radar_id])
        
        if best_diff <= MAX_TIME_DIFFERENCE:
            matches.append((img_frame_id, best_radar_id))
            del radar_ts_pool[best_radar_id]
    
    print(f"Found {len(matches)} synchronized frame pairs\n")
    return matches


def load_radar_pcd(file_path: Path) -> PointCloud:
    """Load a .pcd file using pypcd4."""
    if not file_path.exists():
        return None
    try:
        pc = PointCloud.from_path(str(file_path))
        return pc
    except Exception as e:
        print(f"Error loading PCD file {file_path}: {e}", file=sys.stderr)
        return None


def preprocess_radar_points(pc: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and filter radar points using 'power' field."""
    if pc is None:
        return np.array([]), np.array([])
    
    pc_data = pc.pc_data
    
    try:
        # Radar has 'power' instead of 'intensity'
        data = np.vstack([
            pc_data['x'], pc_data['y'], pc_data['z'],
            pc_data['power'], pc_data['range']
        ]).T.astype(np.float64)
    except (ValueError, KeyError) as e:
        print(f"Error: PCD missing required fields. {e}")
        print(f"Available fields: {list(pc_data.dtype.names)}")
        return np.array([]), np.array([])
    
    power_values = data[:, 3]
    mask = power_values >= POWER_THRESHOLD
    
    p_paper_radar = data[mask, :3]
    ranges_meters = data[mask, 4]
    
    return p_paper_radar, ranges_meters


def project_points_to_image(
    points_3d_paper_radar: np.ndarray,
    K: np.ndarray,
    T_paper_cam_from_radar: np.ndarray,
    original_color_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D radar points to 2D image plane."""
    if points_3d_paper_radar.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    
    num_points = points_3d_paper_radar.shape[0]
    points_3d_hom = np.hstack((points_3d_paper_radar, np.ones((num_points, 1))))
    
    points_cam_hom = (T_paper_cam_from_radar @ points_3d_hom.T).T
    points_cam = points_cam_hom[:, 0:3]
    
    depths = points_cam[:, 2]
    
    valid_mask = depths > 0.1
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    points_cam_valid = points_cam[valid_mask]
    depths_valid = depths[valid_mask]
    color_data_valid = original_color_data[valid_mask]
    
    u_norm = points_cam_valid[:, 0] / depths_valid
    v_norm = points_cam_valid[:, 1] / depths_valid
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u_pix = fx * u_norm + cx
    v_pix = fy * v_norm + cy
    
    points_2d = np.vstack((u_pix, v_pix)).T
    
    return points_2d, depths_valid, color_data_valid


def draw_points_on_image(
    image: np.ndarray,
    points_2d: np.ndarray,
    depth_data: np.ndarray,
    percentile_min: float = 5.0,
    percentile_max: float = 95.0
) -> np.ndarray:
    """Draw colored points on image."""
    if points_2d.shape[0] == 0 or depth_data.shape[0] == 0:
        return image
    
    min_clip = np.percentile(depth_data, percentile_min)
    max_clip = np.percentile(depth_data, percentile_max)
    epsilon = 1e-6
    
    depths_clipped = np.clip(depth_data, min_clip, max_clip)
    depths_norm = (depths_clipped - min_clip) / (max_clip - min_clip + epsilon)
    depths_8bit = (depths_norm * 255).astype(np.uint8)
    
    colors_mapped = cv2.applyColorMap(depths_8bit, cv2.COLORMAP_VIRIDIS)
    
    img_with_points = image.copy()
    h, w = img_with_points.shape[:2]
    
    num_points = min(points_2d.shape[0], colors_mapped.shape[0])
    
    for i in range(num_points):
        u, v = points_2d[i]
        if 0 <= u < w and 0 <= v < h:
            color = tuple(int(c) for c in colors_mapped[i, 0, :])
            cv2.circle(img_with_points, (int(u), int(v)), 2, color, -1)  # Larger circles for radar
    
    return img_with_points


def load_frame_data(image_frame_id: str, radar_frame_id: str) -> bool:
    """
    Load image and radar data for a specific frame pair.
    Updates state.img_undistorted, state.points_3d_radar, and state.color_data.
    Returns True on success, False on failure.
    """
    # Load image
    img_path = DATASET_ROOT / "1_IMAGE" / "LEFT" / f"{image_frame_id}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return False
    
    # Build K for undistortion (using current intrinsics from sliders)
    K = np.array([
        [state.fx / 10.0, 0.0, state.cx / 10.0],
        [0.0, state.fy / 10.0, state.cy / 10.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    state.img_undistorted = cv2.undistort(img, K, D_PAPER)
    
    # Load radar - Note: The path is 3_RADAR, not 2_radar
    radar_path = DATASET_ROOT / "3_RADAR" / "PCD" / f"{radar_frame_id}.pcd"
    pc = load_radar_pcd(radar_path)
    if pc is None:
        print(f"Error: Could not load radar file {radar_path}")
        return False
    
    state.points_3d_radar, state.color_data = preprocess_radar_points(pc)
    
    print(f"Loaded frame {state.current_frame_idx + 1}/{len(state.frame_pairs)}: "
          f"Image={image_frame_id}, Radar={radar_frame_id} "
          f"({state.points_3d_radar.shape[0]} points)")
    
    return True


def next_frame():
    """Load the next frame pair."""
    if state.current_frame_idx < len(state.frame_pairs) - 1:
        state.current_frame_idx += 1
        img_id, radar_id = state.frame_pairs[state.current_frame_idx]
        if load_frame_data(img_id, radar_id):
            update_display()
        else:
            # Revert on failure
            state.current_frame_idx -= 1


def prev_frame():
    """Load the previous frame pair."""
    if state.current_frame_idx > 0:
        state.current_frame_idx -= 1
        img_id, radar_id = state.frame_pairs[state.current_frame_idx]
        if load_frame_data(img_id, radar_id):
            update_display()
        else:
            # Revert on failure
            state.current_frame_idx += 1


# --- Rendering Function ---

def render_projection():
    """Render the current projection based on slider values."""
    # Build K matrix from current slider values
    K = np.array([
        [state.fx / 10.0, 0.0, state.cx / 10.0],
        [0.0, state.fy / 10.0, state.cy / 10.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Build rotation vector from current slider values
    r_cl = np.array([
        state.rx / 1000.0,
        state.ry / 1000.0,
        state.rz / 1000.0
    ])
    
    # Build translation vector from current slider values
    t_cl = np.array([
        state.tx / 1000.0,
        state.ty / 1000.0,
        state.tz / 1000.0
    ])
    
    # Convert Rodrigues to rotation matrix
    R, _ = cv2.Rodrigues(r_cl)
    
    # Build transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = t_cl
    
    # Project points
    points_2d, depths, color_filtered = project_points_to_image(
        state.points_3d_radar,
        K,
        T,
        state.color_data
    )
    
    # Draw on image
    result = draw_points_on_image(state.img_undistorted, points_2d, color_filtered)
    
    # Add parameter display overlay
    overlay = result.copy()
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 0)
    
    # Get current frame IDs
    img_id, radar_id = state.frame_pairs[state.current_frame_idx]
    
    params_text = [
        f"Frame {state.current_frame_idx + 1}/{len(state.frame_pairs)}: Img={img_id} Radar={radar_id}",
        f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}",
        f"rx={r_cl[0]:.4f} ry={r_cl[1]:.4f} rz={r_cl[2]:.4f}",
        f"tx={t_cl[0]:.4f} ty={t_cl[1]:.4f} tz={t_cl[2]:.4f}",
        f"Points: {points_2d.shape[0]}",
        "A/D: Navigate frames | 's': Save | 'r': Reset | 'q': Quit"
    ]
    
    for i, text in enumerate(params_text):
        cv2.putText(overlay, text, (10, y_offset + i*20), 
                    font, font_scale, color, thickness, cv2.LINE_AA)
    
    return overlay


# --- Slider Callbacks ---

def on_fx_change(val):
    state.fx = val
    update_display()

def on_fy_change(val):
    state.fy = val
    update_display()

def on_cx_change(val):
    state.cx = val
    update_display()

def on_cy_change(val):
    state.cy = val
    update_display()

def on_rx_change(val):
    state.rx = val
    update_display()

def on_ry_change(val):
    state.ry = val
    update_display()

def on_rz_change(val):
    state.rz = val
    update_display()

def on_tx_change(val):
    state.tx = val
    update_display()

def on_ty_change(val):
    state.ty = val
    update_display()

def on_tz_change(val):
    state.tz = val
    update_display()


def update_display():
    """Re-render and display the projection."""
    img = render_projection()
    cv2.imshow('Calibration Tool', img)


def save_parameters():
    """Save current parameters to a file."""
    output_file = Path("calibration_params_radar.txt")
    
    with open(output_file, 'w') as f:
        f.write("# Camera Intrinsics (K matrix)\n")
        f.write(f"fx = {state.fx / 10.0}\n")
        f.write(f"fy = {state.fy / 10.0}\n")
        f.write(f"cx = {state.cx / 10.0}\n")
        f.write(f"cy = {state.cy / 10.0}\n")
        f.write("\n# Rotation (Rodrigues vector)\n")
        f.write(f"rx = {state.rx / 1000.0}\n")
        f.write(f"ry = {state.ry / 1000.0}\n")
        f.write(f"rz = {state.rz / 1000.0}\n")
        f.write("\n# Translation (meters)\n")
        f.write(f"tx = {state.tx / 1000.0}\n")
        f.write(f"ty = {state.ty / 1000.0}\n")
        f.write(f"tz = {state.tz / 1000.0}\n")
        f.write("\n# Python code snippet:\n")
        f.write(f"K = np.array([\n")
        f.write(f"    [{state.fx / 10.0}, 0.0, {state.cx / 10.0}],\n")
        f.write(f"    [0.0, {state.fy / 10.0}, {state.cy / 10.0}],\n")
        f.write(f"    [0.0, 0.0, 1.0]\n")
        f.write(f"], dtype=np.float64)\n\n")
        f.write(f"r_cl = np.array([{state.rx / 1000.0}, {state.ry / 1000.0}, {state.rz / 1000.0}])\n")
        f.write(f"t_cl = np.array([{state.tx / 1000.0}, {state.ty / 1000.0}, {state.tz / 1000.0}])\n")
    
    print(f"\nParameters saved to {output_file}")
    print(f"fx={state.fx/10.0:.2f}, fy={state.fy/10.0:.2f}, cx={state.cx/10.0:.2f}, cy={state.cy/10.0:.2f}")
    print(f"rx={state.rx/1000.0:.4f}, ry={state.ry/1000.0:.4f}, rz={state.rz/1000.0:.4f}")
    print(f"tx={state.tx/1000.0:.4f}, ty={state.ty/1000.0:.4f}, tz={state.tz/1000.0:.4f}")


def reset_parameters():
    """Reset to initial values."""
    state.fx = int(INITIAL_FX * 10)
    state.fy = int(INITIAL_FY * 10)
    state.cx = int(INITIAL_CX * 10)
    state.cy = int(INITIAL_CY * 10)
    state.rx = int(INITIAL_RX * 1000)
    state.ry = int(INITIAL_RY * 1000)
    state.rz = int(INITIAL_RZ * 1000)
    state.tx = int(INITIAL_TX * 1000)
    state.ty = int(INITIAL_TY * 1000)
    state.tz = int(INITIAL_TZ * 1000)
    
    # Update all sliders (using new offset values)
    cv2.setTrackbarPos('fx (*0.1)', 'Calibration Tool', state.fx)
    cv2.setTrackbarPos('fy (*0.1)', 'Calibration Tool', state.fy)
    cv2.setTrackbarPos('cx (*0.1)', 'Calibration Tool', state.cx)
    cv2.setTrackbarPos('cy (*0.1)', 'Calibration Tool', state.cy)
    cv2.setTrackbarPos('rx (mrad)', 'Calibration Tool', state.rx + 10000)
    cv2.setTrackbarPos('ry (mrad)', 'Calibration Tool', state.ry + 10000)
    cv2.setTrackbarPos('rz (mrad)', 'Calibration Tool', state.rz + 10000)
    cv2.setTrackbarPos('tx (mm)', 'Calibration Tool', state.tx + 5000)
    cv2.setTrackbarPos('ty (mm)', 'Calibration Tool', state.ty + 5000)
    cv2.setTrackbarPos('tz (mm)', 'Calibration Tool', state.tz + 5000)
    
    update_display()
    print("Reset to initial parameters")


# --- Main Function ---

def main():
    """Load data and start interactive calibration."""
    print("Starting Interactive RADAR Calibration Tool...")
    print(f"Dataset Root: {DATASET_ROOT}\n")
    
    if not DATASET_ROOT.is_dir():
        print(f"Error: Dataset root not found at {DATASET_ROOT}", file=sys.stderr)
        sys.exit(1)
    
    # Find all synchronized frame pairs
    state.frame_pairs = find_synchronized_frames()
    
    if len(state.frame_pairs) == 0:
        print("Error: No synchronized frame pairs found!")
        sys.exit(1)
    
    # Start with first frame
    state.current_frame_idx = 0
    print(f"Starting with first frame: {state.frame_pairs[0][0]}/{state.frame_pairs[0][1]}")
    
    # Load initial frame
    img_id, radar_id = state.frame_pairs[state.current_frame_idx]
    if not load_frame_data(img_id, radar_id):
        print("Error: Could not load initial frame data")
        sys.exit(1)
    
    # Create window and sliders
    cv2.namedWindow('Calibration Tool', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration Tool', 1280, 720)
    
    # Intrinsics sliders (scaled by 10)
    # Range: 0-15000 allows fx/fy from 0-1500, cx/cy from 0-1500
    cv2.createTrackbar('fx (*0.1)', 'Calibration Tool', state.fx, 15000, on_fx_change)
    cv2.createTrackbar('fy (*0.1)', 'Calibration Tool', state.fy, 15000, on_fy_change)
    cv2.createTrackbar('cx (*0.1)', 'Calibration Tool', state.cx, 15000, on_cx_change)
    cv2.createTrackbar('cy (*0.1)', 'Calibration Tool', state.cy, 15000, on_cy_change)
    
    # Rotation sliders (milliradians, offset by 10000 to allow range -10 to +10 radians)
    # Range: 0-20000 allows rx/ry/rz from -10 to +10 radians
    cv2.createTrackbar('rx (mrad)', 'Calibration Tool', state.rx + 10000, 20000, 
                       lambda v: on_rx_change(v - 10000))
    cv2.createTrackbar('ry (mrad)', 'Calibration Tool', state.ry + 10000, 20000,
                       lambda v: on_ry_change(v - 10000))
    cv2.createTrackbar('rz (mrad)', 'Calibration Tool', state.rz + 10000, 20000,
                       lambda v: on_rz_change(v - 10000))
    
    # Translation sliders (millimeters, offset by 5000 to allow range -5m to +5m)
    # Range: 0-10000 allows tx/ty/tz from -5 to +5 meters
    cv2.createTrackbar('tx (mm)', 'Calibration Tool', state.tx + 5000, 10000,
                       lambda v: on_tx_change(v - 5000))
    cv2.createTrackbar('ty (mm)', 'Calibration Tool', state.ty + 5000, 10000,
                       lambda v: on_ty_change(v - 5000))
    cv2.createTrackbar('tz (mm)', 'Calibration Tool', state.tz + 5000, 10000,
                       lambda v: on_tz_change(v - 5000))
    
    print("\nControls:")
    print("  - 'a'/'d' keys: Navigate between frames")
    print("  - Adjust sliders to change calibration")
    print("  - Press 's' to save current parameters")
    print("  - Press 'r' to reset to initial values")
    print("  - Press 'q' to quit\n")
    
    # Initial render
    update_display()
    
    # Event loop
    while True:
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_parameters()
        elif key == ord('r'):
            reset_parameters()
        elif key == ord('a'):
            prev_frame()
        elif key == ord('d'):
            next_frame()
    
    cv2.destroyAllWindows()
    print("\nCalibration tool closed.")


if __name__ == "__main__":
    main()