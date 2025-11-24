import sys
import os
import glob
import numpy as np
import cv2
import itertools
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm

# --- Import your specific modules ---
from modules.clustering import cluster_detections_anisotropic # Use your NEW clustering function
from modules.types import NoiseType, DetectionTuple
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera, RadarDetection

# --- CONFIGURATION: SEARCH RANGES ---
PARAM_GRID = {
    # Center around the current best (2.5)
    'eps': [1.25, 1.5, 1.75], 

    # High density seems good
    'min_samples': [4, 5, 6, 7],

    # Doppler is king. Let it go higher.
    'weight_vz': [4, 5, 6, 7], 

    # TEST: Does a small penalty on transverse velocity help separate 
    # side-by-side cars without losing ghosts?
    'weight_vxy': [2.75, 3, 3.5], 
}

# --- 1. Data Loader (Unchanged) ---
def cache_simulation_data(max_frames_to_load=None):
    """
    Loads flow, poses, and radar data, solves for velocity, 
    and returns a list of ALL frame detections (unclustered).
    """
    print("--- Phase 1: Caching Simulation Data (Running Solver) ---")
    
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
    if not all_image_files:
        print("Error: No images found.")
        sys.exit(1)

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

        if not radar_detections:
            continue

        frame_detections = estimate_velocities_from_data(
            radar_detections, flow, mock_camera, 
            T_A_to_B, T_A_to_R_static, DELTA_T
        )
        
        if frame_detections:
            cached_frames.append(frame_detections)

    print(f"Successfully cached {len(cached_frames)} frames with valid data.")
    return cached_frames

def evaluate_clustering_quality(all_frames_data, eps, min_samples, weight_vz, weight_vxy):
    """
    Evaluates clustering Quality, ignoring objects with < 10 real points.
    """
    total_clusters_generated = 0
    total_healthy_clusters = 0
    
    total_objects_tracked = 0
    total_perfect_matches = 0   
    
    for frame_detections in all_frames_data:
        
        # --- PRE-SCAN: Identify Sparse Objects ---
        real_point_counts = defaultdict(int)
        for det in frame_detections:
            if det[7] > 0 and det[3] == NoiseType.REAL:
                real_point_counts[det[7]] += 1
        
        # Set of IDs that are "Worth Tracking"
        valid_eval_ids = set()
        for obj_id, count in real_point_counts.items():
            if count >= 10:
                valid_eval_ids.add(obj_id)
                
        # --- 1. Run Clustering ---
        clusters, noise_points = cluster_detections_anisotropic(
            detections=frame_detections,
            eps=eps,
            min_samples=min_samples,
            weight_vz=weight_vz,
            weight_vxy=weight_vxy
        )
        
        total_clusters_generated += len(clusters)

        # --- 2. Analyze Each Cluster ---
        id_to_healthy_clusters = defaultdict(list)
        
        for c_idx, cluster in enumerate(clusters):
            
            # Tally Composition
            id_counts = defaultdict(int)
            count_actor_points = 0
            count_wall_points = 0
            count_noise_points = 0
            
            total_points = len(cluster)
            
            for det in cluster:
                obj_id = det[7]
                n_type = det[3]
                
                if obj_id > 0:
                    id_counts[obj_id] += 1
                    count_actor_points += 1
                else:
                    if n_type == NoiseType.REAL:
                        count_wall_points += 1
                    else:
                        count_noise_points += 1
            
            # --- CLASSIFY THE CLUSTER ---
            max_category = max(count_actor_points, count_wall_points, count_noise_points)
            
            if max_category == count_wall_points:
                # WALL -> Neutral
                total_clusters_generated -= 1
                continue
                
            elif max_category == count_noise_points:
                # GHOST -> Bad (Lowers Precision)
                continue
                
            else:
                # ACTOR CLUSTER
                dominant_id = max(id_counts, key=id_counts.get)
                
                # CHECK: Is this a "Sparse Object"?
                if dominant_id not in valid_eval_ids:
                    # This cluster belongs to a far-away object we are ignoring.
                    # Treat as Neutral (remove from denominator).
                    total_clusters_generated -= 1
                    continue
                
                # It is a Valid Target Object. Check Purity.
                valid_signal_count = id_counts[dominant_id]
                
                if valid_signal_count > (total_points / 2):
                    total_healthy_clusters += 1
                    id_to_healthy_clusters[dominant_id].append(c_idx)

        # --- 3. Analyze Objects (Recall) ---
        # Only iterate over VALID IDs (>= 10 points)
        for obj_id in valid_eval_ids:
            total_objects_tracked += 1
            assigned_healthy = id_to_healthy_clusters[obj_id]
            
            if len(assigned_healthy) == 1:
                total_perfect_matches += 1

    # --- C. Calculate Normalized Score ---
    if total_clusters_generated == 0:
        precision = 0.0
    else:
        precision = total_healthy_clusters / total_clusters_generated
        
    if total_objects_tracked == 0:
        recall = 0.0
    else:
        recall = total_perfect_matches / total_objects_tracked
        
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return f1_score, total_perfect_matches, total_objects_tracked

# --- 3. Main Execution ---
if __name__ == "__main__":
    
    cached_data = cache_simulation_data(max_frames_to_load=None) 
    
    if not cached_data:
        print("No data to optimize.")
        sys.exit(1)

    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Phase 2: Grid Search ({len(param_combinations)} combinations) ---")

    best_score = -1.0
    best_params = None

    for params in tqdm(param_combinations, desc="Optimizing"):
        
        score, matches, total = evaluate_clustering_quality(
            cached_data, 
            params['eps'], 
            params['min_samples'], 
            params['weight_vz'],
            params['weight_vxy']
        )
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New Best: {score:.3f} (Matches: {matches}/{total}) | Params: {params}")

    print("\n" + "="*40)
    print("OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Best Integrity Score: {best_score*100:.2f}%")
    print("-" * 20)
    print("Optimal Parameters:")
    print(f"  eps:             {best_params['eps']}")
    print(f"  min_samples:     {best_params['min_samples']}")
    print(f"  weight_vz:       {best_params['weight_vz']}")
    print(f"  weight_vxy:      {best_params['weight_vxy']}")
    print("="*40)