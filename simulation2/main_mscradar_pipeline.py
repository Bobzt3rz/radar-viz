#!/usr/bin/env python3

"""
main_mscradar_pipeline.py

This script adapts the CARLA-based velocity estimation and clustering
pipeline to run on the real MSC-RAD4R dataset.

It now includes a validation step that uses a synchronized LiDAR point
cloud as ground truth. The 'clusters' (kept points) and 'noise_points'
(removed points) are scored against the LiDAR data using a KD-Tree
and a fixed-radius search to generate TP, FP, TN, and FN metrics.
"""

import sys
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pypcd4 import PointCloud
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree # <-- NEW: For LiDAR validation

# --- Import all the original logic modules ---
# (Assuming they are in a 'modules' folder)
try:
    from modules.optical_flow import OpticalFlow
    from modules.utils import save_image, save_frame_histogram, save_clustering_analysis_plot
    from modules.clustering import cluster_detections_6d
    from modules.types import NoiseType, DetectionTuple, Vector3, Matrix4x4, FlowField
    from modules.velocity_solver import solve_full_velocity, calculate_reprojection_error
except ImportError:
    print("Error: Could not import from 'modules' directory.", file=sys.stderr)
    print("Please ensure 'modules' (with optical_flow.py, clustering.py, etc.)", file=sys.stderr)
    print("is in the same directory as this script.", file=sys.stderr)
    sys.exit(1)


# --- 1. Dataset Configuration & Constants ---

# !!! IMPORTANT: Update this path to your local copy of the dataset !!!
DATASET_ROOT = Path.home() / "Downloads" / "URBAN_D0-20251112T180635Z-1-002" / "URBAN_D0"

if not DATASET_ROOT.is_dir():
    print(f"Error: Dataset root not found at {DATASET_ROOT}", file=sys.stderr)
    print("Please update the DATASET_ROOT variable in this script.", file=sys.stderr)
    sys.exit(1)

# --- Path to your predicted ego velocity file ---
# !!! IMPORTANT: Update this path if needed !!!
EGO_VELOCITY_FILE = Path.home() / "programming/radar/radar-viz/data/predicted_ego_vel.txt"

if not EGO_VELOCITY_FILE.is_file():
    print(f"Error: Ego velocity file not found at {EGO_VELOCITY_FILE}", file=sys.stderr)
    sys.exit(1)

# ---
# Coordinate System Definitions
# ---
# We assume the "Paper" Frame (+Z Fwd, +X Right, +Y Down) for all
# loaded PCD data and calibration files, matching coordinate_systems.md.
# ---

FRAME_LIMIT = 9999 # Process this many frames
MAX_TIME_DIFFERENCE = 0.50 # 50ms sync bound
REPROJ_PERCENTILE_THRESHOLD = 90.0
KDTREE_SEARCH_RADIUS = 1 # Meters

# --- 2. NEW/MODIFIED Data Types ---

@dataclass
class RadarDetection:
    """Holds the data we load from the real MSC-RAD4R PCD file."""
    position_paper: np.ndarray  # 3D (x,y,z) in "Paper" coords
    radial_velocity: float    # 'doppler' field (m/s)
    power: float              # 'power' field (SNR in dB)

class MockCamera:
    """Mocks the Camera object with data loaded from files."""
    def __init__(self, K: np.ndarray, D: np.ndarray, image_shape: Tuple[int, int]):
        self._K = K
        self._D = D
        self.fx: float = K[0, 0]
        self.fy: float = K[1, 1]
        self.cx: float = K[0, 2]
        self.cy: float = K[1, 2]
        self.image_height: int = image_shape[0]
        self.image_width: int = image_shape[1]

    def get_intrinsics_matrix(self) -> np.ndarray:
        return self._K

# --- 3. Data Loading Functions ---

def load_timestamps(file_path: Path) -> Dict[str, float]:
    """Loads a timestamp file. (From main_mscradar.py)"""
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

def load_radar_pcd_to_detections(file_path: Path) -> List[RadarDetection]:
    """
    MODIFIED: Loads the MSC-RAD4R PCD file and returns List[RadarDetection].
    """
    detections = []
    try:
        pc = PointCloud.from_path(str(file_path))
    except Exception as e:
        print(f"Error loading PCD file {file_path}: {e}", file=sys.stderr)
        return detections

    pc_data = pc.pc_data
    
    try:
        data_stack = [
            pc_data['x'], pc_data['y'], pc_data['z'],
            pc_data['doppler'], pc_data['power']
        ]
        data_array = np.vstack(data_stack).T.astype(np.float64)
    except (ValueError, KeyError) as e:
        print(f"  Error: PCD missing required fields (x, y, z, doppler, power). {e}")
        return detections

    for row in data_array:
        pos_paper = row[0:3]
        v_rad = row[3]  # Using 'doppler' field
        power = row[4]
        
        detections.append(RadarDetection(
            position_paper=pos_paper,
            radial_velocity=v_rad,
            power=power
        ))
    return detections

def load_ego_velocity(file_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    NEW: Loads the predicted_ego_vel.txt file.
    """
    ego_vel_map: Dict[str, Tuple[float, float]] = {}
    if not file_path.is_file():
        print(f"Error: Ego velocity file not found: {file_path}")
        return ego_vel_map
        
    with open(file_path, 'r') as f:
        next(f) # Skip header line
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                index_str = parts[0]
                speed = float(parts[1])
                alpha = float(parts[2]) # Assuming this is yaw rate in rad/s
                ego_vel_map[index_str] = (speed, alpha)
    return ego_vel_map

def load_lidar_pcd(file_path: Path) -> np.ndarray:
    """
    NEW: Loads a LiDAR PCD file and returns just the (x,y,z) points.
    Assumes points are ALREADY in the "Paper" frame (LiDAR origin).
    """
    if not file_path.exists():
        return np.array([])

    try:
        pc = PointCloud.from_path(str(file_path))
    except Exception as e:
        print(f"Error loading PCD file {file_path}: {e}", file=sys.stderr)
        return np.array([])
    
    pc_data = pc.pc_data
    
    try:
        # We only need x, y, z for the ground truth
        data = np.vstack([
            pc_data['x'], pc_data['y'], pc_data['z']
        ]).T.astype(np.float64)
        return data
    except (ValueError, KeyError) as e:
        print(f"  Error: LiDAR PCD missing required fields (x, y, z). {e}")
        return np.array([])

def transform_points_3d(points: np.ndarray, T_B_from_A: np.ndarray) -> np.ndarray:
    """
    NEW: Helper function to transform (N, 3) points using a 4x4 matrix.
    """
    if points.shape[0] == 0:
        return np.array([])
    
    num_points = points.shape[0]
    points_A_hom = np.hstack((points, np.ones((num_points, 1))))
    points_B_hom = (T_B_from_A @ points_A_hom.T).T
    points_B = points_B_hom[:, 0:3]
    return points_B

# --- 4. NEW EGO-MOTION & PLOTTING ---

@dataclass
class ValidationStats:
    """NEW: Holds the validation results."""
    tp: int = 0  # True Positives (Kept, Match)
    fp: int = 0  # False Positives (Kept, No Match)
    tn: int = 0  # True Negatives (Removed, No Match)
    fn: int = 0  # False Negatives (Removed, Match)
    kept_oor: int = 0    # Kept, Out-of-Range (Unscorable)
    removed_oor: int = 0 # Removed, Out-of-Range (Unscorable)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

def validate_detections_with_lidar(
    kept_points: List[DetectionTuple],
    removed_points: List[DetectionTuple],
    lidar_gt_points: np.ndarray,
    search_radius: float = 0.5
) -> Optional[ValidationStats]:
    """
    MODIFIED: Validates points using a DYNAMIC 3D Bounding Box.
    - The box is defined by the 1st and 99th percentiles of the
      LiDAR data itself along each axis (X, Y, Z).
    - Points outside this box are 'Out-of-Range'.
    - Points inside are validated against the KD-Tree.
    """
    if lidar_gt_points.shape[0] < 100: # Need enough points for percentiles
        print("  Warning: Not enough LiDAR points (<100) for percentile ROI. Skipping.")
        return None

    if not kept_points and not removed_points:
        print("  Warning: No radar points to validate.")
        return None
        
    percentile = 99.0
    # --- NEW: 1. Define Dynamic ROI Bounding Box ---
    # Get the 1st and 99th percentiles for each axis
    x_min, x_max = np.percentile(lidar_gt_points[:, 0], [100.0 - percentile, percentile])
    y_min, y_max = np.percentile(lidar_gt_points[:, 1], [100.0 - percentile, percentile])
    z_min, z_max = np.percentile(lidar_gt_points[:, 2], [100.0 - percentile, percentile])
    
    # Add a small buffer to the Z-min to avoid clipping points
    # (e.g., if 1st percentile is 5m, we still want to score points at 4m)
    z_min = min(z_min, 1.0) # Ensure we score points close to the sensor
    
    print(f"  LiDAR Dynamic ROI (X): {x_min:.1f}m to {x_max:.1f}m")
    print(f"  LiDAR Dynamic ROI (Y): {y_min:.1f}m to {y_max:.1f}m")
    print(f"  LiDAR Dynamic ROI (Z): {z_min:.1f}m to {z_max:.1f}m")

    # 2. Build the KD-Tree from the LiDAR ground truth points
    try:
        lidar_kdtree = KDTree(lidar_gt_points)
    except Exception as e:
        print(f"  Error building KDTree: {e}")
        return None
        
    stats = ValidationStats()

    # --- NEW: Helper function to check scorable points ---
    def check_points_in_roi(points_pos: np.ndarray):
        """
        Takes (N, 3) points, returns a boolean (N,) scorable_mask
        and the (M, 3) subset of points that are inside the ROI.
        """
        if points_pos.shape[0] == 0:
            return np.array([]), np.array([])
            
        mask_x = (points_pos[:, 0] >= x_min) & (points_pos[:, 0] <= x_max)
        mask_y = (points_pos[:, 1] >= y_min) & (points_pos[:, 1] <= y_max)
        mask_z = (points_pos[:, 2] >= z_min) & (points_pos[:, 2] <= z_max)
        
        scorable_mask = mask_x & mask_y & mask_z
        points_to_validate = points_pos[scorable_mask]
        
        return scorable_mask, points_to_validate

    # 3. Validate the "Kept" points (Clusters)
    kept_pos_list = [det[4] for det in kept_points]
    if kept_pos_list:
        kept_pos = np.array(kept_pos_list)
        scorable_mask, points_to_validate = check_points_in_roi(kept_pos)
        
        stats.kept_oor = (~scorable_mask).sum()
        
        if points_to_validate.shape[0] > 0:
            results = lidar_kdtree.query_ball_point(points_to_validate, r=search_radius)
            for r in results:
                if len(r) > 0:
                    stats.tp += 1
                else:
                    stats.fp += 1

    # 4. Validate the "Removed" points (Noise)
    removed_pos_list = [det[4] for det in removed_points]
    if removed_pos_list:
        removed_pos = np.array(removed_pos_list)
        scorable_mask, points_to_validate = check_points_in_roi(removed_pos)

        stats.removed_oor = (~scorable_mask).sum()

        if points_to_validate.shape[0] > 0:
            results = lidar_kdtree.query_ball_point(points_to_validate, r=search_radius)
            for r in results:
                if len(r) > 0:
                    stats.fn += 1
                else:
                    stats.tn += 1
    
    # 5. Calculate Precision, Recall, and F1-Score
    precision_denom = stats.tp + stats.fp
    if precision_denom > 0:
        stats.precision = stats.tp / precision_denom
    
    recall_denom = stats.tp + stats.fn
    if recall_denom > 0:
        stats.recall = stats.tp / recall_denom
        
    f1_denom = stats.precision + stats.recall
    if f1_denom > 0:
        stats.f1_score = 2 * (stats.precision * stats.recall) / f1_denom

    return stats


def calculate_relative_pose(speed: float, yaw_rate: float, delta_t: float) -> Matrix4x4:
    """
    NEW: Calculates the relative camera pose T_A_to_B (T_camB_from_camA)
    from a unicycle model (speed, yaw_rate).
    """
    d_theta = yaw_rate * delta_t
    
    if abs(yaw_rate) < 1e-6:
        d_z = speed * delta_t
        d_x = 0.0
    else:
        radius = speed / yaw_rate
        d_z = radius * np.sin(d_theta)
        d_x = radius * (1.0 - np.cos(d_theta))
    
    T_B_from_A = np.eye(4)
    R_y = Rotation.from_euler('y', d_theta).as_matrix()
    T_B_from_A[0:3, 0:3] = R_y
    T_B_from_A[0:3, 3] = np.array([d_x, 0.0, d_z])
    
    try:
        T_A_to_B = np.linalg.inv(T_B_from_A)
        return T_A_to_B
    except np.linalg.LinAlgError:
        print("Error inverting relative pose, returning identity")
        return np.eye(4)


def save_noise_analysis_plot(
    frame_id: str,
    all_solved_points: List[DetectionTuple],
    clusters: List[List[DetectionTuple]],
    noise_points: List[List[DetectionTuple]],
    output_dir: Path,
    val_stats_reproj: Optional[ValidationStats] = None,
    val_stats_cluster: Optional[ValidationStats] = None,
    val_stats_combined: Optional[ValidationStats] = None
):
    """
    MODIFIED: Creates and saves a 2x2 plot visualizing filter results.
    - Title now includes F1-Score and all validation stats.
    - Reprojection plot uses the global REPROJ_PERCENTILE_THRESHOLD.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Extract data for Reprojection Error plot (Request 1) ---
    all_reproj_errors = [det[2] for det in all_solved_points]
    if all_reproj_errors:
        # Calculate the threshold *just for this plot's histogram*
        reproj_threshold_px = np.percentile(all_reproj_errors, REPROJ_PERCENTILE_THRESHOLD)
    else:
        reproj_threshold_px = 0.0

    points_kept = [err for err in all_reproj_errors if err < reproj_threshold_px]
    points_filtered = [err for err in all_reproj_errors if err >= reproj_threshold_px]
    num_kept = len(points_kept)
    num_filtered = len(points_filtered)

    # --- Extract data for Clustering plots (Request 2) ---
    clustered_pos = [det[4] for c in clusters for det in c]
    clustered_vel = [det[5] for c in clusters for det in c]
    clustered_vel_mags = [np.linalg.norm(v) for v in clustered_vel]
    
    noisy_pos = [det[4] for det in noise_points]
    noisy_vel = [det[5] for det in noise_points]
    noisy_vel_mags = [np.linalg.norm(v) for v in noisy_vel]
    
    num_clustered = len(clustered_pos)
    num_noisy = len(noisy_pos)

    # --- Create Figure ---
    fig = plt.figure(figsize=(16, 12), dpi=100)

    # --- NEW: Multi-line Dynamic Title with F1-Score ---
    title_str = f"Filter Analysis - Frame {frame_id}\n"
    total_solved = len(all_solved_points)
    
    if val_stats_cluster:
        s = val_stats_cluster
        total_scorable = s.tp + s.fp + s.tn + s.fn
        total_oor = s.kept_oor + s.removed_oor
        title_str += f"Total Solved: {total_solved} | Scorable (in range): {total_scorable} | Out-of-Range: {total_oor}\n"
    else:
        title_str += f"Total Solved: {total_solved} | Clustered: {num_clustered} | Noise: {num_noisy}\n"

    # Define a helper lambda to format the stats
    def format_stats(s: ValidationStats) -> str:
        oor = s.kept_oor + s.removed_oor
        return f"F1: {s.f1_score:.4f} (P:{s.precision:.3f} R:{s.recall:.3f}) | TP:{s.tp} FP:{s.fp} TN:{s.tn} FN:{s.fn} | OOR:{oor}"

    if val_stats_reproj:
        title_str += f"\nReproj Stats:   {format_stats(val_stats_reproj)}"
    if val_stats_cluster:
        title_str += f"\nClustering Stats: {format_stats(val_stats_cluster)}"
    if val_stats_combined:
        title_str += f"\nCombined Stats:   {format_stats(val_stats_combined)}"
    
    fig.suptitle(title_str, fontsize=12) # Reduced font size for more text

    # --- 1. Reprojection Error (Kept vs. Filtered) ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(f"Reprojection Error ({REPROJ_PERCENTILE_THRESHOLD}th Percentile Threshold)")
    
    if all_reproj_errors:
        all_errs = all_reproj_errors
        max_err = max(np.percentile(all_errs, 99) if all_errs else 1.0, 1.0)
        max_err = min(max_err, 5.0) # Clip at 5px
        bins = np.linspace(0, max_err, 50)
        ax1.hist(points_kept, bins=bins, alpha=0.7, label=f"Kept ({num_kept})", color='blue')
        ax1.hist(points_filtered, bins=bins, alpha=0.7, label=f"Filtered ({num_filtered})", color='red')
        ax1.set_xlabel("Reprojection Error (pixels)")
        ax1.set_ylabel("Count")
        ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 2. Velocity Magnitude (Clustered vs. Noise) ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Solved 3D Velocity Magnitude (Clustered vs. Noise)")
    if clustered_vel_mags or noisy_vel_mags:
        max_vel = max(np.percentile(clustered_vel_mags, 98) if clustered_vel_mags else 30, 30)
        bins = np.linspace(0, max_vel, 50)
        ax2.hist(clustered_vel_mags, bins=bins, alpha=0.7, label=f"Clustered ({num_clustered})", color='blue')
        ax2.hist(noisy_vel_mags, bins=bins, alpha=0.7, label=f"Noise ({num_noisy})", color='red')
        ax2.set_xlabel("Velocity (m/s)")
        ax2.set_ylabel("Count")
        ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- 3. Position Scatter 3D "Cube" (Clustered vs. Noise) ---
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title("3D Position (Clustered vs. Noise)")
    if clustered_pos:
        c_pos_arr = np.array(clustered_pos)
        ax3.scatter(c_pos_arr[:, 0], c_pos_arr[:, 2], c_pos_arr[:, 1], s=2, alpha=0.8, label="Clustered", color='blue')
    if noisy_pos:
        n_pos_arr = np.array(noisy_pos)
        ax3.scatter(n_pos_arr[:, 0], n_pos_arr[:, 2], n_pos_arr[:, 1], s=2, alpha=0.5, label="Noise", color='red')
    ax3.set_xlabel("X-Paper (m)")
    ax3.set_ylabel("Z-Paper (m)")
    ax3.set_zlabel("Y-Paper (m)")
    ax3.invert_zaxis()
    ax3.legend()

    # --- 4. Velocity Scatter 3D "Cube" (Clustered vs. Noise) ---
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title("Solved 3D Velocity (Clustered vs. Noise)")
    if clustered_vel:
        c_vel_arr = np.array(clustered_vel)
        ax4.scatter(c_vel_arr[:, 0], c_vel_arr[:, 2], c_vel_arr[:, 1], s=2, alpha=0.8, label="Clustered", color='blue')
    if noisy_vel:
        n_vel_arr = np.array(noisy_vel)
        ax4.scatter(n_vel_arr[:, 0], n_vel_arr[:, 2], n_vel_arr[:, 1], s=2, alpha=0.5, label="Noise", color='red')
    ax4.set_xlabel("Vel-X (m/s)")
    ax4.set_ylabel("Vel-Z (m/s)")
    ax4.set_zlabel("Vel-Y (m/s)")
    ax4.invert_zaxis()
    ax4.legend()

    # --- Save Figure ---
    # Adjusted top margin to make room for the larger title
    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    output_path = output_dir / f"{frame_id}_clustering_analysis.png"
    plt.savefig(str(output_path))
    plt.close(fig)

def project_and_get_depths(
    detections: List[DetectionTuple],
    T_Cam_from_Radar: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts all 3D points from the list, projects them,
    and returns 2D points and their corresponding depths.
    """
    # 1. Extract all 3D points (det_tuple[4] is pos_3d)
    points_3d_radar = [det[4] for det in detections]
    
    if not points_3d_radar:
        return np.array([]), np.array([])
        
    points_3d_radar = np.array(points_3d_radar) # (N, 3)
    
    # 2. Project them (standard projection logic)
    points_cam = transform_points_3d(points_3d_radar, T_Cam_from_Radar)
    if points_cam.shape[0] == 0:
        return np.array([]), np.array([])
    
    depths = points_cam[:, 2] # Camera Z-depths
    
    valid_mask = depths > 0.1  # 10cm threshold
    if not np.any(valid_mask):
        return np.array([]), np.array([])
        
    points_cam_valid = points_cam[valid_mask]
    depths_valid = depths[valid_mask]
    
    u_norm = points_cam_valid[:, 0] / depths_valid
    v_norm = points_cam_valid[:, 1] / depths_valid
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u_pix = fx * u_norm + cx
    v_pix = fy * v_norm + cy
    
    points_2d = np.vstack((u_pix, v_pix)).T
    
    return points_2d, depths_valid

def draw_points_on_image(
    image: np.ndarray, 
    points_2d: np.ndarray, 
    depth_data: np.ndarray,
    percentile_min: float = 5.0,
    percentile_max: float = 95.0
) -> np.ndarray:
    """
    Draws 2D points on an image, color-coded by depth_data (e.g., depth).
    """
    if points_2d.shape[0] == 0 or depth_data.shape[0] == 0:
        return image

    min_clip = np.percentile(depth_data, percentile_min)
    max_clip = np.percentile(depth_data, percentile_max)
    
    epsilon = 1e-6 
    data_clipped = np.clip(depth_data, min_clip, max_clip)
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


# --- 5. REFACTORED Solver Function ---

def estimate_velocities_from_data(
    radar_detections: List[RadarDetection],
    flow: FlowField,
    camera: MockCamera,
    T_A_to_B: Matrix4x4, # Relative pose (T_camB_from_camA)
    T_A_to_R: Matrix4x4, # Static extrinsic (T_radar_from_camA)
    world_delta_t: float
) -> List[DetectionTuple]:
    """
    REFACTORED: Calculates full velocity for loaded MSC-RAD4R detections.
    """
    frame_results: List[DetectionTuple] = []

    try:
        T_Cam_from_Radar_static = np.linalg.inv(T_A_to_R)
    except np.linalg.LinAlgError:
        print("  Error inverting T_A_to_R to find static extrinsic.")
        return frame_results

    for detection in radar_detections:
        point_radar_coords = detection.position_paper
        speed_radial = detection.radial_velocity
        
        point_cam_B = transform_points_3d(point_radar_coords.reshape(1,3), T_Cam_from_Radar_static)[0]
        depth_B = point_cam_B[2]

        if depth_B <= 1e-3: continue

        uq = point_cam_B[0] / depth_B
        vq = point_cam_B[1] / depth_B

        xq_pix_f = camera.fx * uq + camera.cx
        yq_pix_f = camera.fy * vq + camera.cy
        xq_pix = int(round(xq_pix_f))
        yq_pix = int(round(yq_pix_f))

        if not (0 <= xq_pix < camera.image_width and 0 <= yq_pix < camera.image_height):
            continue

        dx, dy = flow[yq_pix, xq_pix]
        xp_pix_f = xq_pix_f - dx
        yp_pix_f = yq_pix_f - dy

        up = (xp_pix_f - camera.cx) / camera.fx
        vp = (yp_pix_f - camera.cy) / camera.fy

        full_vel_vector_radar = solve_full_velocity(
            up=up, vp=vp, uq=uq, vq=vq, d=depth_B, delta_t=world_delta_t,
            T_A_to_B=T_A_to_B, T_A_to_R=T_A_to_R,
            speed_radial=speed_radial, point_radar_coords=point_radar_coords,
            return_in_radar_coords=True
        )
        
        if full_vel_vector_radar is None:
            continue
            
        frame_displacement_error = calculate_reprojection_error(
            full_vel_radar_A=full_vel_vector_radar,
            point_radar_B=point_radar_coords,
            T_Cam_from_Radar=T_Cam_from_Radar_static,
            T_CamB_from_CamA=T_A_to_B,
            flow=flow, 
            camera=camera, 
            xq_pix_f=xq_pix_f, 
            yq_pix_f=yq_pix_f, 
            delta_t=world_delta_t
        )
         
        if frame_displacement_error is not None:
            full_vel_magnitude = float(np.linalg.norm(full_vel_vector_radar))
            
            result_tuple: DetectionTuple = (
                full_vel_magnitude,
                0.0, # velocity_error_magnitude (dummy)
                frame_displacement_error,
                NoiseType.REAL, # noiseType (dummy, 1=REAL)
                point_radar_coords,
                full_vel_vector_radar,
                np.array([0.0, 0.0, 0.0]) # ground_truth_vel_radar (dummy)
            )
            frame_results.append(result_tuple)
            
    return frame_results


# --- 6. Main Execution ---

def main():
    """
    Main pipeline function.
    """
    print(f"Starting MSC-RAD4R Noise Filtering Pipeline...")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Ego Velocity: {EGO_VELOCITY_FILE}\n")
    
    # --- Define sub-paths ---
    radar_pcd_dir = DATASET_ROOT / "3_RADAR" / "PCD"
    radar_ts_file = DATASET_ROOT / "3_RADAR" / "timestamp_radar.txt"
    
    img_dir = DATASET_ROOT / "1_IMAGE" / "LEFT"
    img_ts_file = DATASET_ROOT / "1_IMAGE" / "timestamp_image_left.txt"

    # --- NEW: LiDAR Paths ---
    lidar_pcd_dir = DATASET_ROOT / "2_LIDAR" / "PCD"
    lidar_ts_file = DATASET_ROOT / "2_LIDAR" / "timestamp_lidar.txt"
    
    output_dir_analysis = Path("output") / "mscradar_pipeline_analysis"
    output_dir_clustered = Path("output") / "projections_clustered_only"
    output_dir_reproj = Path("output") / "projections_reproj_error_only"
    output_dir_combined = Path("output") / "projections_combined_filter"
    
    output_dir_analysis.mkdir(parents=True, exist_ok=True)
    output_dir_clustered.mkdir(parents=True, exist_ok=True)
    output_dir_reproj.mkdir(parents=True, exist_ok=True)
    output_dir_combined.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving analysis plots to: {output_dir_analysis}")
    print(f"Saving Clustered projections to: {output_dir_clustered}")
    print(f"Saving Reproj Error projections to: {output_dir_reproj}")
    print(f"Saving Combined Filter projections to: {output_dir_combined}\n")

    # --- 1. Load Static Calibration (LEFT Camera) ---
    print("--- 1. Loading Calibration Data (LEFT Camera) ---")
    
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
    
    try:
        T_A_to_R_static = np.linalg.inv(T_paper_cam_from_radar)
        T_Cam_from_Radar_static = T_paper_cam_from_radar 
        print("Loaded and inverted T_paper_cam_from_radar successfully.")
    except np.linalg.LinAlgError:
        print("FATAL: Could not invert T_paper_cam_from_radar. Exiting.")
        return
        
    print(f"Loaded K_paper (Right Cam):\n{K}\n")

    # --- LiDAR to Radar Transform ---
    r_cl = np.array([1.226053849071323, -1.196572101163224, 1.192295102346085])
    t_cl = np.array([0.1278447154599633, -0.314979325909709, -0.9064288085364153])

    R_cl, _ = cv2.Rodrigues(r_cl)

    T_paper_cam_from_lidar = np.eye(4, dtype=np.float64)
    T_paper_cam_from_lidar[0:3, 0:3] = R_cl
    T_paper_cam_from_lidar[0:3, 3] = t_cl

    T_paper_radar_from_lidar = np.linalg.inv(T_paper_cam_from_radar) @ T_paper_cam_from_lidar
    
    
    # --- 2. Load Timestamps & Ego Velocity ---
    print("--- 2. Loading Timestamps and Ego Velocity ---")
    image_ts_map = load_timestamps(img_ts_file)
    radar_ts_pool = load_timestamps(radar_ts_file)
    lidar_ts_pool = load_timestamps(lidar_ts_file) # <-- NEW
    ego_vel_map = load_ego_velocity(EGO_VELOCITY_FILE)
    
    if not image_ts_map or not radar_ts_pool or not ego_vel_map or not lidar_ts_pool: # <-- MODIFIED
        print("Error: Could not load timestamp or ego velocity files. Exiting.")
        return

    print(f"Loaded {len(image_ts_map)} image timestamps.")
    print(f"Loaded {len(radar_ts_pool)} Radar timestamps into pool.")
    print(f"Loaded {len(lidar_ts_pool)} LiDAR timestamps into pool.") # <-- NEW
    print(f"Loaded {len(ego_vel_map)} ego velocity entries.")

    # --- 3. STAGE 1: Find 1-to-1-to-1 matches ---
    print(f"--- 3. Finding 3-way matches within {MAX_TIME_DIFFERENCE*1000:.0f}ms ---")
    
    sorted_image_items = sorted(image_ts_map.items(), key=lambda item: item[1])
    # --- MODIFIED: matches list now holds (img_id, radar_id, lidar_id) ---
    matches: List[Tuple[str, str, str]] = []
    
    for img_frame_id, img_ts in sorted_image_items:
        if not radar_ts_pool or not lidar_ts_pool: break # Need both
            
        # Find best radar frame
        best_radar_id = min(radar_ts_pool, key=lambda k: abs(img_ts - radar_ts_pool[k]))
        best_radar_diff = abs(img_ts - radar_ts_pool[best_radar_id])
        
        # Find best lidar frame
        best_lidar_id = min(lidar_ts_pool, key=lambda k: abs(img_ts - lidar_ts_pool[k]))
        best_lidar_diff = abs(img_ts - lidar_ts_pool[best_lidar_id])
        
        # Check if *both* are within the time bound
        if best_radar_diff <= MAX_TIME_DIFFERENCE and best_lidar_diff <= MAX_TIME_DIFFERENCE:
            # Found a 3-way match!
            matches.append((img_frame_id, best_radar_id, best_lidar_id))
            
            # Remove both from their pools
            del radar_ts_pool[best_radar_id]
            del lidar_ts_pool[best_lidar_id]
            
    print(f"Found {len(matches)} 3-way matches.") # <-- MODIFIED
    if len(matches) < 2:
        print("Error: Not enough matches to run optical flow pipeline. Exiting.")
        return

    # --- 4. STAGE 2: Process Matched Frames ---
    print(f"--- 4. Starting Pipeline Loop (up to {FRAME_LIMIT} frames) ---")
    
    optical_flow_calculator = OpticalFlow()
    
    try:
        img_frame_id_A, _, _ = matches[0] # <-- MODIFIED
        img_path_A = img_dir / f"{img_frame_id_A}.png"
        image_A = cv2.imread(str(img_path_A))
        if image_A is None: raise FileNotFoundError(f"Missing {img_path_A}")
        
        img_undistorted_A = cv2.undistort(image_A, K, D)
        frame_A_rgb = cv2.cvtColor(img_undistorted_A, cv2.COLOR_BGR2RGB)
        
        optical_flow_calculator.inference(frame_A_rgb) 
        mock_camera = MockCamera(K, D, frame_A_rgb.shape)
        
    except Exception as e:
        print(f"Error loading initial frame {matches[0][0]}: {e}")
        return

    # --- Main Processing Loop ---
    frames_processed = 0
    for i in range(1, min(len(matches), FRAME_LIMIT + 1)):
        
        # --- MODIFIED: Unpack 3-way match ---
        img_frame_id_A, radar_frame_id_A, _ = matches[i-1]
        img_frame_id_B, radar_frame_id_B, lidar_frame_id_B = matches[i]
        
        img_path_B = img_dir / f"{img_frame_id_B}.png"
        radar_path_B = radar_pcd_dir / f"{radar_frame_id_B}.pcd"
        lidar_path_B = lidar_pcd_dir / f"{lidar_frame_id_B}.pcd" # <-- NEW
        
        validation_stats: Optional[ValidationStats] = None # <-- NEW

        try:
            image_B = cv2.imread(str(img_path_B))
            if image_B is None: raise FileNotFoundError(f"Missing {img_path_B}")
            
            radar_detections: List[RadarDetection] = load_radar_pcd_to_detections(radar_path_B)
            
            # --- NEW: Load LiDAR Ground Truth ---
            lidar_points_paper = load_lidar_pcd(lidar_path_B)
            
            if not radar_detections:
                print(f"Warning: No radar points in {radar_path_B.name}. Skipping.")
                continue
            
            img_undistorted_B = cv2.undistort(image_B, K, D)
            frame_B_rgb = cv2.cvtColor(img_undistorted_B, cv2.COLOR_BGR2RGB)
        
        except Exception as e:
            print(f"Error loading data for frame {img_frame_id_B}: {e}. Skipping.")
            continue
            
        print(f"--- Processing Frame {i} (Img: {img_frame_id_B}) ---")

        flow = optical_flow_calculator.inference(frame_B_rgb)
        
        delta_t = image_ts_map[img_frame_id_B] - image_ts_map[img_frame_id_A]
        
        index_str = f"{i:04d}"
        speed, alpha = ego_vel_map.get(index_str, (0.0, 0.0))
        if speed == 0.0 and alpha == 0.0:
            print(f"  Warning: No ego-velocity for index {index_str}. Assuming static.")
            
        T_A_to_B = calculate_relative_pose(speed, alpha, delta_t)

        current_frame_results: List[DetectionTuple] = estimate_velocities_from_data(
            radar_detections, 
            flow, 
            mock_camera,
            T_A_to_B,
            T_A_to_R_static,
            delta_t
        )
        print(f"  Solved velocity for {len(current_frame_results)} of {len(radar_detections)} points.")

        if not current_frame_results:
            print("  No points to cluster. Skipping.")
            continue

        all_reproj_errors = [det[2] for det in current_frame_results]
        
        dynamic_reproj_threshold = np.percentile(all_reproj_errors, REPROJ_PERCENTILE_THRESHOLD)
        
        print(f"  Dynamic Reproj Threshold ({REPROJ_PERCENTILE_THRESHOLD}th percentile): {dynamic_reproj_threshold:.4f} px")
            
        clusters, noise_points = cluster_detections_6d(
            detections=current_frame_results,
            eps=1.0, min_samples=3, velocity_weight=4.0   
        )
        
        # This is the list of "kept" points
        clustered_results = [det for c in clusters for det in c]
        
        num_clustered = len(clustered_results)
        num_noisy = len(noise_points)
        
        avg_clustered_err = np.mean([det[2] for det in clustered_results]) if num_clustered > 0 else 0.0
        avg_noisy_err = np.mean([det[2] for det in noise_points]) if num_noisy > 0 else 0.0
        
        print(f"  Clustering: {num_clustered} clustered ({len(clusters)} clusters) | "
              f"{num_noisy} noise (filtered)")
        print(f"  Avg Reproj Err (Clustered): {avg_clustered_err:.4f} px")
        print(f"  Avg Reproj Err (Noise):     {avg_noisy_err:.4f} px")

        # --- NEW: Run LiDAR Validation for ALL 3 Filters ---
        validation_stats_reproj: Optional[ValidationStats] = None
        validation_stats_cluster: Optional[ValidationStats] = None
        validation_stats_combined: Optional[ValidationStats] = None
        
        reproj_kept = []
        combined_kept = []
        if lidar_points_paper.shape[0] > 0:
            print(f"  Validating {len(current_frame_results)} points against {lidar_points_paper.shape[0]} LiDAR points...")
            
            # 1. Transform LiDAR points to Radar (Paper) frame (once)
            lidar_points_radar_frame = transform_points_3d(
                lidar_points_paper,
                T_paper_radar_from_lidar
            )
            
            # 2. Define the 3 filter groups
            
            # --- Filter 1: Reprojection Only ---
            reproj_kept = [det for det in current_frame_results if det[2] < dynamic_reproj_threshold]
            reproj_removed = [det for det in current_frame_results if det[2] >= dynamic_reproj_threshold]
            
            # --- Filter 2: Clustering Only ---
            cluster_kept = clustered_results # (Already defined)
            cluster_removed = noise_points    # (Already defined)

            # --- Filter 3: Combined (Clustering AND Reprojection) ---
            combined_kept = [det for det in cluster_kept if det[2] < dynamic_reproj_threshold]
            # "Removed" is any point not in the "combined_kept" list
            combined_kept_ids = set(id(det) for det in combined_kept)
            combined_removed = [det for det in current_frame_results if id(det) not in combined_kept_ids]

            # 3. Run Validation for each
            print("    Running validation for Reprojection Filter...")
            validation_stats_reproj = validate_detections_with_lidar(
                reproj_kept, reproj_removed, lidar_points_radar_frame, search_radius=KDTREE_SEARCH_RADIUS
            )
            
            print("    Running validation for Clustering Filter...")
            validation_stats_cluster = validate_detections_with_lidar(
                cluster_kept, cluster_removed, lidar_points_radar_frame, search_radius=KDTREE_SEARCH_RADIUS
            )
            
            print("    Running validation for Combined Filter...")
            validation_stats_combined = validate_detections_with_lidar(
                combined_kept, combined_removed, lidar_points_radar_frame, search_radius=KDTREE_SEARCH_RADIUS
            )

            # 4. Print all results
            def print_stats(name: str, s: ValidationStats):
                scorable = s.tp + s.fp + s.tn + s.fn
                oor = s.kept_oor + s.removed_oor
                print(f"  {name:16} F1: {s.f1_score:.4f} (P:{s.precision:.3f}, R:{s.recall:.3f}) | "
                      f"TP:{s.tp} FP:{s.fp} TN:{s.tn} FN:{s.fn} | Scorable: {scorable} | OOR: {oor}")

            if validation_stats_reproj:
                print_stats("Reproj Stats:", validation_stats_reproj)
            if validation_stats_cluster:
                print_stats("Clustering Stats:", validation_stats_cluster)
            if validation_stats_combined:
                print_stats("Combined Stats:", validation_stats_combined)
        else:
            print("  Warning: No LiDAR points loaded for validation.")

        save_noise_analysis_plot(
            img_frame_id_B,
            current_frame_results,
            clusters,
            noise_points,
            output_dir_analysis,
            validation_stats_reproj,
            validation_stats_cluster,
            validation_stats_combined
        )
        
        # --- Create Filtered Lists ---
        
        # 1. Clustered only
        # (already have 'clustered_results')
        
        # 2. Reprojection error only
        reproj_results = reproj_kept

        # 3. Combined (Clustered AND Reprojection)
        combined_results = combined_kept
        # --- Save Filtered Projection Images ---
        
        filter_outputs = {
            "clustered": (clustered_results, output_dir_clustered),
            "reproj_error": (reproj_results, output_dir_reproj),
            "combined": (combined_results, output_dir_combined),
        }

        for filter_name, (results_list, out_dir) in filter_outputs.items():
            if not results_list:
                continue
                
            points_2d, depths = project_and_get_depths(
                results_list,
                T_Cam_from_Radar_static,
                K
            )
            
            if points_2d.shape[0] == 0:
                continue

            image_with_points = draw_points_on_image(
                img_undistorted_B, # BGR image
                points_2d,
                depths # Color by depth
            )
            
            output_path = out_dir / f"{img_frame_id_B}_sync_{radar_frame_id_B}_{filter_name}.png"
            cv2.imwrite(str(output_path), image_with_points)
            
        print(f"  Saved filtered projections (Clustered: {len(clustered_results)}, Reproj: {len(reproj_results)}, Combined: {len(combined_results)})")
        
        frames_processed += 1

    print(f"\n\Pipeline complete. Processed {frames_processed} frames.")
    print(f"Check the analysis in: {output_dir_analysis}")
    print(f"Check the projections in: {output_dir_clustered}, {output_dir_reproj}, {output_dir_combined}")


if __name__ == "__main__":
    if not DATASET_ROOT.is_dir():
        print("Please update the DATASET_ROOT variable at the top of this script.")
    elif not EGO_VELOCITY_FILE.is_file():
        print("Please update the EGO_VELOCITY_FILE variable at the top of this script.")
    else:
        main()