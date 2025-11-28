import sys
import os
import glob
import numpy as np
import itertools
import concurrent.futures
from scipy import stats
from typing import List, Tuple, Dict
from tqdm import tqdm
from collections import namedtuple

# --- Imports ---
from modules.clustering import cluster_detections_6d, filter_clusters_quantile, filter_clusters_mad, filter_static_points
from modules.types import NoiseType, DetectionTuple
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

# --- CONFIGURATION ---

# 1. Fixed DBSCAN Params (The Baseline)
FIXED_CLUSTERING_PARAMS = {
    'eps': 8.7205,
    'min_samples': 11,
    'velocity_weight': 2.5279
}

# 2. Search Grid for 3D Box Filter
# We test different deep ratios for Lateral (X), Longitudinal (Y), and Vertical (Z)
PARAM_GRID = {
    'x': [1.5, 2, 2.25, 2.5, 2.75, 3.0, 3.25],
    'y': [1.5, 2, 2.25, 2.5, 2.75, 3.0, 3.25],
    'z': [1, 1.25, 1.5, 1.75, 2, 2.5, 2.75, 3.0, 3.25]
}

# --- GLOBAL WORKER STATE ---
# We store the pre-calculated raw clusters here to avoid pickling overhead
_worker_raw_clusters = None

def init_worker(data):
    """Stores the raw clusters in the worker process memory."""
    global _worker_raw_clusters
    _worker_raw_clusters = data

# --- Data Loader ---
def cache_simulation_data(max_frames_to_load=None):
    print("--- Phase 1: Caching Simulation Data ---")
    
    # Path Setup
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

    # Load Data Loop
    for frame_count in tqdm(range(1, limit), desc="Loading & Solving"):
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
        if frame_detections: 
            filtered_static_points = filter_static_points(frame_detections)
            cached_frames.append(filtered_static_points)

    return cached_frames

def pre_calculate_clusters(cached_data):
    """
    Run the heavy DBSCAN once using the fixed parameters.
    Returns a list of cluster lists.
    """
    print("--- Phase 2: Pre-calculating Raw Clusters ---")
    all_raw_clusters = []
    
    for frame_detections in tqdm(cached_data, desc="Clustering"):
        clusters, _ = cluster_detections_6d(
            frame_detections,
            eps=FIXED_CLUSTERING_PARAMS['eps'],
            min_samples=FIXED_CLUSTERING_PARAMS['min_samples'],
            velocity_weight=FIXED_CLUSTERING_PARAMS['velocity_weight']
        )
        all_raw_clusters.append(clusters)
        
    return all_raw_clusters

# --- Evaluation Logic (Worker) ---
def evaluate_filter_performance(all_raw_clusters, thresh_x, thresh_y, thresh_z):
    """
    Applies the filter and calculates F1 Score based on 
    Signal Retention (Recall) vs Noise Rejection (Precision).
    """
    tp = 0 # Real kept
    fp = 0 # Ghost kept
    tn = 0 # Ghost removed
    fn = 0 # Real removed (Over-filtering)

    # Prepare tuple for the new filter function
    keep_ratio = (thresh_x, thresh_y, thresh_z)

    for raw_clusters_frame in all_raw_clusters:
        
        # This function is now very fast (just checking absolute differences)
        refined_clusters, purge_noise = filter_clusters_mad(
            raw_clusters_frame,
            std_threshold=keep_ratio
        )
        
        # 1. Analyze what we KEPT (refined_clusters)
        for cluster in refined_clusters:
            for det in cluster:
                # only care about non-static objects
                if det[7] > 0:
                    # det[3] is NoiseType (1=REAL, 0=NOISE)
                    if det[3] == NoiseType.REAL:
                        tp += 1
                    else:
                        fp += 1
                    
        # 2. Analyze what we REMOVED (purge_noise)
        for det in purge_noise:
            # only care about non-static objects
            if det[7] > 0:
                if det[3] == NoiseType.REAL:
                    fn += 1 # Bad: We threw away a real point
                else:
                    tn += 1 # Good: We threw away noise

    # --- Calculate Score ---
    # Precision: Out of everything we kept, how much is real?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: Out of all real points provided, how many did we keep?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return f1, precision, recall, tn

def process_wrapper(params):
    """Unpacks params and runs evaluation using global data."""
    global _worker_raw_clusters
    
    f1, prec, rec, ghosts = evaluate_filter_performance(
        _worker_raw_clusters, 
        params['x'], 
        params['y'], 
        params['z']
    )
    return f1, prec, rec, ghosts, params

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Load Data
    cached_data = cache_simulation_data(max_frames_to_load=None)
    if not cached_data:
        print("No data.")
        sys.exit(1)
        
    # 2. Pre-Calculate Clusters (The Optimization)
    # We do this here so we don't have to re-run DBSCAN 200 times.
    raw_clusters_data = pre_calculate_clusters(cached_data)

    # 3. Generate Search Grid
    keys, values = zip(*PARAM_GRID.items())
    # combinations will look like: [{'x': 1.0, 'y': 3.0, 'z': 0.5}, ...]
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Phase 3: Grid Search ({len(param_combinations)} combinations) ---")
    
    best_score = -1.0
    best_params = None
    best_stats = None

    # 4. Run Parallel Processing
    max_workers = os.cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(raw_clusters_data,) # Pass the clusters, not the raw frames
    ) as executor:
        
        results = list(tqdm(
            executor.map(process_wrapper, param_combinations), 
            total=len(param_combinations), 
            desc="Optimizing Thresholds"
        ))

    # 5. Find Best Result
    for f1, prec, rec, ghosts, params in results:
        if f1 > best_score:
            best_score = f1
            best_params = params
            best_stats = (prec, rec, ghosts)
            # Optional: Print new high scores as they are found
            # print(f"New Best: {f1:.3f} | Params: {params}")

    print("\n" + "="*40)
    print("FILTER OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Best F1 Score: {best_score*100:.2f}%")
    print(f"  - Precision (Purity):    {best_stats[0]*100:.2f}%")
    print(f"  - Recall (Retention):    {best_stats[1]*100:.2f}%")
    print(f"  - Total Ghosts Removed:  {best_stats[2]}")
    print("-" * 20)
    print("Optimal Thresholds:")
    print(f"  X (Lateral):      {best_params['x']} m/s")
    print(f"  Y (Longitudinal): {best_params['y']} m/s")
    print(f"  Z (Vertical):     {best_params['z']} m/s")
    print("="*40)