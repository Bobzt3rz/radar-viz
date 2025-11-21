import sys
import os
import glob
import numpy as np
import cv2
import itertools
from typing import List, Dict, Tuple
from tqdm import tqdm  # You may need to run: pip install tqdm

# --- Import your specific modules ---
# Ensure 'modules' folder is in the same directory
from modules.clustering import cluster_detections_6d
from modules.types import NoiseType, DetectionTuple
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera, RadarDetection

# --- CONFIGURATION: SEARCH RANGES ---
# Adjust these ranges based on your data scale
PARAM_GRID = {
    'eps': np.arange(2.0, 6.0, 0.5),          # Test wider connections
    'min_samples': [3, 4, 5, 6],              # Test lower density requirements
    'velocity_weight': np.arange(1.0, 6.0, 1.0), 
}

# --- 1. Data Loader (Runs Once) ---
def cache_simulation_data(max_frames_to_load=None):
    """
    Loads flow, poses, and radar data, solves for velocity, 
    and returns a list of ALL frame detections (unclustered).
    """
    print("--- Phase 1: Caching Simulation Data (Running Solver) ---")
    
    # Paths (Copied from your main_carla.py)
    CARLA_OUTPUT_DIR = "../carla/output"
    DELTA_T = 0.05
    CAM_DIR = os.path.join(CARLA_OUTPUT_DIR, "camera_rgb")
    PLY_DIR = os.path.join(CARLA_OUTPUT_DIR, "radar_ply")
    POSES_DIR = os.path.join(CARLA_OUTPUT_DIR, "poses")
    CALIB_DIR = os.path.join(CARLA_OUTPUT_DIR, "calib")
    FLOW_DIR = os.path.join(CARLA_OUTPUT_DIR, "flow")

    # Load Calibration
    try:
        K_cam = np.loadtxt(os.path.join(CALIB_DIR, "intrinsics.txt"))
        T_A_to_R_static = np.loadtxt(os.path.join(CALIB_DIR, "extrinsics_radar_from_camera.txt"))
        mock_camera = MockCamera(K_cam)
    except Exception as e:
        print(f"Critical Error: Could not load calibration. {e}")
        sys.exit(1)

    all_image_files = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    if not all_image_files:
        print("Error: No images found.")
        sys.exit(1)

    cached_frames = []
    
    # Limit frames if just testing the script
    limit = len(all_image_files) if max_frames_to_load is None else min(len(all_image_files), max_frames_to_load)

    for frame_count in tqdm(range(1, limit), desc="Solving Velocities"):
        image_path_B = all_image_files[frame_count]
        frame_id_B = os.path.basename(image_path_B).split('.')[0]
        
        pose_file = os.path.join(POSES_DIR, f"{frame_id_B}_relative_pose.txt")
        ply_file = os.path.join(PLY_DIR, f"{frame_id_B}.ply")
        flow_file = os.path.join(FLOW_DIR, f"{frame_id_B}.npy")

        if not (os.path.exists(pose_file) and os.path.exists(ply_file) and os.path.exists(flow_file)):
            continue

        # Load Data
        T_A_to_B = np.loadtxt(pose_file)
        radar_detections = load_radar_ply(ply_file)
        flow = np.load(flow_file)

        if not radar_detections:
            continue

        # Run Solver
        frame_detections = estimate_velocities_from_data(
            radar_detections, flow, mock_camera, 
            T_A_to_B, T_A_to_R_static, DELTA_T
        )
        
        if frame_detections:
            cached_frames.append(frame_detections)

    print(f"Successfully cached {len(cached_frames)} frames with valid data.")
    return cached_frames

# --- 2. Evaluation Logic ---
def evaluate_parameters(all_frames_data, eps, min_samples, velocity_weight):
    """
    Runs clustering on cached data and calculates Global F1 Score.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for frame_detections in all_frames_data:
        clusters, noise_points = cluster_detections_6d(
            detections=frame_detections,
            eps=eps,
            min_samples=min_samples,
            velocity_weight=velocity_weight
        )

        # --- Calculate Metrics (Simplified from main_carla.py) ---
        
        # 1. Analyze Clusters (TP vs FP)
        for cluster in clusters:
            for det in cluster:
                noise_type = det[3]
                if noise_type == NoiseType.REAL:
                    total_tp += 1
                else:
                    # Multipath or Random inside a cluster = False Positive
                    total_fp += 1
        
        # 2. Analyze Noise (FN) - We don't care about TN for F1 score
        for det in noise_points:
            noise_type = det[3]
            if noise_type == NoiseType.REAL:
                # Real point filtered out = False Negative
                total_fn += 1
                
    # Calculate F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # # F-Beta Score Formula: (1 + beta^2) * (P * R) / ((beta^2 * P) + R)
    # # F2 Score (Beta = 2) -> Recall is 2x as important as Precision
    # beta = 2.0
    # if (precision + recall) > 0:
    #     f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    # else:
    #     f2_score = 0.0
    
    # # RETURN F2 INSTEAD OF F1
    return f1, precision, recall

# --- 3. Main Execution ---
if __name__ == "__main__":
    
    # A. Load Data
    # Set max_frames_to_load to a small number (e.g. 50) for debugging, or None for full run
    cached_data = cache_simulation_data(max_frames_to_load=None) 
    
    if not cached_data:
        print("No data to optimize.")
        sys.exit(1)

    # B. Generate Grid
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Phase 2: Grid Search ({len(param_combinations)} combinations) ---")

    best_score = -1.0
    best_params = None
    best_metrics = (0, 0) # Prec, Rec

    # C. Run Optimization
    # We use tqdm to show progress bar
    for params in tqdm(param_combinations, desc="Optimizing"):
        
        f1, prec, rec = evaluate_parameters(
            cached_data, 
            params['eps'], 
            params['min_samples'], 
            params['velocity_weight']
        )
        
        if f1 > best_score:
            best_score = f1
            best_params = params
            best_metrics = (prec, rec)

    # D. Report Results
    print("\n" + "="*40)
    print("OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Best F1 Score: {best_score*100:.2f}%")
    print(f"  - Precision: {best_metrics[0]*100:.2f}%")
    print(f"  - Recall:    {best_metrics[1]*100:.2f}%")
    print("-" * 20)
    print("Optimal Parameters:")
    print(f"  eps:             {best_params['eps']}")
    print(f"  min_samples:     {best_params['min_samples']}")
    print(f"  velocity_weight: {best_params['velocity_weight']}")
    print("="*40)
    
    print("\nRecommendation: Update these values in main_carla.py")