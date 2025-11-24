import sys
import os
import glob
import numpy as np
import itertools
from typing import List, Tuple
from tqdm import tqdm

# --- Imports ---
# Ensure these match your project structure
from modules.clustering import cluster_detections_anisotropic, filter_clusters_median
from modules.types import NoiseType
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

# --- CONFIGURATION ---

# 1. The Fixed "Optimal" Clustering Parameters (From your previous run)
FIXED_CLUSTERING_PARAMS = {
    'eps': 1.5,
    'min_samples': 8,
    'weight_vz': 6,
    'weight_vxy': 3
}

# 2. The Search Grid for Filtering
PARAM_GRID = {
    # How far (m/s) a point can be from the median before being deleted.
    # Testing strict (2.0) to loose (15.0) thresholds.
    'purge_threshold': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
}

# --- Data Loader (Same as before) ---
def cache_simulation_data(max_frames_to_load=None):
    print("--- Phase 1: Caching Simulation Data ---")
    # ... (Identical logic to your previous script, omitted for brevity) ...
    # You can copy-paste the cache_simulation_data function from your previous file here.
    # For now, I will assume it's the exact same function.
    
    CARLA_OUTPUT_DIR = "../carla/output"
    DELTA_T = 0.05
    CAM_DIR = os.path.join(CARLA_OUTPUT_DIR, "camera_rgb")
    PLY_DIR = os.path.join(CARLA_OUTPUT_DIR, "radar_ply")
    POSES_DIR = os.path.join(CARLA_OUTPUT_DIR, "poses")
    CALIB_DIR = os.path.join(CARLA_OUTPUT_DIR, "calib")
    FLOW_DIR = os.path.join(CARLA_OUTPUT_DIR, "flow")

    try:
        K_cam = np.loadtxt(os.path.join(CALIB_DIR, "intrinsics.txt"))
        T_A_to_R_static = np.loadtxt(os.path.join(CALIB_DIR, "extrinsics_radar_from_camera.txt"))
        mock_camera = MockCamera(K_cam)
    except Exception as e:
        print(f"Critical Error: Could not load calibration. {e}")
        sys.exit(1)

    all_image_files = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    cached_frames = []
    limit = len(all_image_files) if max_frames_to_load is None else min(len(all_image_files), max_frames_to_load)

    for frame_count in tqdm(range(1, limit), desc="Solving Velocities"):
        image_path_B = all_image_files[frame_count]
        frame_id_B = os.path.basename(image_path_B).split('.')[0]
        pose_file = os.path.join(POSES_DIR, f"{frame_id_B}_relative_pose.txt")
        ply_file = os.path.join(PLY_DIR, f"{frame_id_B}.ply")
        flow_file = os.path.join(FLOW_DIR, f"{frame_id_B}.npy")

        if not (os.path.exists(pose_file) and os.path.exists(ply_file) and os.path.exists(flow_file)):
            continue

        T_A_to_B = np.loadtxt(pose_file)
        radar_detections = load_radar_ply(ply_file)
        flow = np.load(flow_file)

        if not radar_detections: continue

        frame_detections = estimate_velocities_from_data(
            radar_detections, flow, mock_camera, 
            T_A_to_B, T_A_to_R_static, DELTA_T
        )
        if frame_detections: cached_frames.append(frame_detections)

    return cached_frames

# --- Evaluation Logic ---
def evaluate_filter_performance(all_frames_data, purge_threshold):
    """
    Evaluates how well the filter cleans up the Fixed Clusters.
    Metric: F1 Score of keeping REAL points vs removing GHOST points.
    """
    tp = 0 # Real kept
    fp = 0 # Ghost kept (Bad)
    tn = 0 # Ghost removed (Good)
    fn = 0 # Real removed (Bad)

    for frame_detections in all_frames_data:
        
        # 1. Run Fixed Clustering (Step 1)
        raw_clusters, _ = cluster_detections_anisotropic(
            frame_detections,
            eps=FIXED_CLUSTERING_PARAMS['eps'],
            min_samples=FIXED_CLUSTERING_PARAMS['min_samples'],
            weight_vz=FIXED_CLUSTERING_PARAMS['weight_vz'],
            weight_vxy=FIXED_CLUSTERING_PARAMS['weight_vxy']
        )
        
        # 2. Run Filtering (Step 2 - The variable being optimized)
        refined_clusters, purge_noise = filter_clusters_median(
            raw_clusters,
            purge_threshold=purge_threshold
        )
        
        # 3. Calculate Stats
        
        # Check what was KEPT (Inside refined_clusters)
        for cluster in refined_clusters:
            for det in cluster:
                noise_type = det[3] # Index 3 is noise_type
                if noise_type == NoiseType.REAL:
                    tp += 1
                else:
                    # Kept a Multipath/Noise point -> False Positive
                    fp += 1
                    
        # Check what was REMOVED (Inside purge_noise)
        for det in purge_noise:
            noise_type = det[3]
            if noise_type == NoiseType.REAL:
                # Removed a Real point -> False Negative (Over-filtering)
                fn += 1
            else:
                # Removed a Noise point -> True Negative (Successful Purge)
                tn += 1

    # --- Calculate F1 ---
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return f1, precision, recall, tn

# --- Main Execution ---
if __name__ == "__main__":
    
    # Load Data
    cached_data = cache_simulation_data(max_frames_to_load=None)
    if not cached_data:
        print("No data.")
        sys.exit(1)
        
    print(f"\n--- Phase 2: Optimizing Filter (Threshold Search) ---")
    print(f"Fixed Clustering Params: {FIXED_CLUSTERING_PARAMS}")
    
    best_score = -1.0
    best_thresh = None
    best_stats = (0, 0, 0) # Prec, Rec, GhostsKilled

    for thresh in tqdm(PARAM_GRID['purge_threshold'], desc="Sweeping Thresholds"):
        
        f1, prec, rec, ghosts_killed = evaluate_filter_performance(cached_data, thresh)
        
        # Verbose print to see trade-offs
        # print(f"Thresh {thresh}: F1={f1:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | Ghosts Killed={ghosts_killed}")
        
        if f1 > best_score:
            best_score = f1
            best_thresh = thresh
            best_stats = (prec, rec, ghosts_killed)

    print("\n" + "="*40)
    print("FILTER OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Best F1 Score: {best_score*100:.2f}%")
    print(f"  - Precision: {best_stats[0]*100:.2f}% (Purity)")
    print(f"  - Recall:    {best_stats[1]*100:.2f}% (Retention)")
    print(f"  - Total Ghosts Removed: {best_stats[2]}")
    print("-" * 20)
    print(f"Optimal Purge Threshold: {best_thresh} m/s")
    print("="*40)