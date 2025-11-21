import sys
import os
import glob
import numpy as np
import itertools
from tqdm import tqdm

# --- Import your specific modules ---
from modules.clustering import cluster_detections_6d
from modules.tracking import ClusterTracker
from modules.types import NoiseType
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

# ==========================================
# 1. LOCK IN YOUR STAGE 1 RESULTS
# ==========================================
FIXED_CLUSTERING_PARAMS = {
    'eps': 4.5,              
    'min_samples': 3,        
    'velocity_weight': 5.0,  
}

# ==========================================
# 2. DEFINE TRACKER SEARCH GRID
# ==========================================
TRACKER_GRID = {
    # Relax hits/history to allow tracks to survive jitter
    'hit_threshold': [3, 4],        
    'dist_threshold': [3.0, 4.0],   # Allow centroids to jump 3-4m (fast cars)
    'max_history': [2, 3],          # Must be > 1 to handle flickering

    # PHYSICS PARAMETERS
    # Relax angle significantly
    'ghost_angle_thresh': [80.0, 100.0], 
    
    # FORCE higher velocity tolerance. 
    # 2.0 is too tight for DBSCAN centroids. Start at 5.0.
    'ghost_vel_thresh': [5.0, 8.0, 10.0]
}

# --- Helper: Data Cacher ---
def cache_clustered_sequence(max_frames_to_load=None):
    print("--- Phase 1: Caching Clustered Data (Pre-computation) ---")
    
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
        print(f"Calibration Error: {e}")
        return []

    all_image_files = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    limit = len(all_image_files) if max_frames_to_load is None else min(len(all_image_files), max_frames_to_load)
    
    if limit == 0: return []

    cached_sequence = []

    for frame_count in tqdm(range(1, limit), desc="Clustering Frames"):
        image_path_B = all_image_files[frame_count]
        frame_id_B = os.path.basename(image_path_B).split('.')[0]
        
        pose_file = os.path.join(POSES_DIR, f"{frame_id_B}_relative_pose.txt")
        ply_file = os.path.join(PLY_DIR, f"{frame_id_B}.ply")
        flow_file = os.path.join(FLOW_DIR, f"{frame_id_B}.npy")

        if not (os.path.exists(pose_file) and os.path.exists(ply_file) and os.path.exists(flow_file)):
            continue

        # Load
        T_A_to_B = np.loadtxt(pose_file) # EGO MOTION MATRIX
        radar_detections = load_radar_ply(ply_file)
        flow = np.load(flow_file)

        if not radar_detections: continue

        # Solve
        frame_detections = estimate_velocities_from_data(
            radar_detections, flow, mock_camera, 
            T_A_to_B, T_A_to_R_static, DELTA_T
        )
        
        if not frame_detections: continue

        # Cluster (Using Locked Parameters)
        clusters, _ = cluster_detections_6d(
            detections=frame_detections,
            **FIXED_CLUSTERING_PARAMS
        )
        
        cached_sequence.append({
            'clusters': clusters,
            'all_detections': frame_detections,
            'T_A_to_B': T_A_to_B # <--- STORE EGO MOTION FOR TRACKER
        })

    return cached_sequence

# --- Evaluation Logic ---
def evaluate_tracker(cached_sequence, params):
    # Initialize with ALL params from the grid
    tracker = ClusterTracker(
        dist_threshold=params['dist_threshold'],
        max_history=params['max_history'],
        hit_threshold=params['hit_threshold'],
        ghost_angle_thresh=params['ghost_angle_thresh'],
        ghost_vel_thresh=params['ghost_vel_thresh']
    )
    
    total_tp, total_fp, total_fn = 0, 0, 0
    DELTA_T = 0.05

    for frame_data in cached_sequence:
        # 1. Run Tracker with Ego-Motion Compensation
        confirmed_clusters = tracker.update(
            frame_data['clusters'], 
            DELTA_T, 
            T_parent_to_current=frame_data['T_A_to_B'] # <--- Pass the matrix here
        )
        
        # 2. Flatten Results
        confirmed_points = [det for cluster in confirmed_clusters for det in cluster]
        all_gt_points = frame_data['all_detections']
        
        # 3. Calculate Stats
        current_tp = sum(1 for det in confirmed_points if det[3] == NoiseType.REAL)
        current_fp = sum(1 for det in confirmed_points if det[3] != NoiseType.REAL)
        
        total_real_in_frame = sum(1 for det in all_gt_points if det[3] == NoiseType.REAL)
        current_fn = total_real_in_frame - current_tp
        
        total_tp += current_tp
        total_fp += current_fp
        total_fn += current_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall

# --- Main ---
if __name__ == "__main__":
    
    # 1. Cache Data
    cached_data = cache_clustered_sequence(max_frames_to_load=None) 
    if not cached_data: sys.exit(1)

    # 2. Grid Search
    keys, values = zip(*TRACKER_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Phase 2: Optimizing Tracker ({len(param_combinations)} combinations) ---")

    results = []
    
    for params in tqdm(param_combinations, desc="Testing Trackers"):
        f1, prec, rec = evaluate_tracker(cached_data, params)
        results.append((f1, prec, rec, params))

    # 3. Sort and Display
    # Filter for viable candidates (Recall > 80%)
    viable = [r for r in results if r[2] > 0.80]
    
    if not viable:
        print("Warning: No candidates achieved >80% Recall. Showing best F1 instead.")
        viable = results
        
    # Sort by F1 Score instead of Precision
    viable.sort(key=lambda x: x[0], reverse=True) 
    
    print("\n" + "="*60)
    print("TOP 3 CONFIGURATIONS (Optimized for BALANCE/F1)")
    
    print("="*60)
    print(f"{'Precision':<10} {'Recall':<10} {'F1':<10} | Parameters")
    print("-" * 60)
    
    for i in range(min(3, len(viable))):
        res = viable[i]
        p_str = f"{res[1]*100:.2f}%"
        r_str = f"{res[2]*100:.2f}%"
        f_str = f"{res[0]*100:.2f}%"
        print(f"{p_str:<10} {r_str:<10} {f_str:<10} | {res[3]}")
    
    print("="*60)
    print("Recommendation: Choose the top one. High Precision = No Ghosts.")