import sys
import os
import glob
import numpy as np
import itertools
import concurrent.futures
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm

# --- Import your specific modules ---
# NOTE: Ensure 'cluster_detections_polar' is in modules/clustering.py
from modules.clustering import cluster_detections_polar
from modules.types import NoiseType, DetectionTuple
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

# --- CONFIGURATION: POLAR SEARCH GRID ---
PARAM_GRID = {
    # 1. Base 'eps' (Range Threshold in meters)
    # Radar is precise in range, so we can be relatively strict here (1.0m - 2.0m).
    'eps': [2.0, 3.0, 4.0, 5.0, 6.0], 

    # 2. Minimum points to form a cluster
    # Lower values help detect far-away objects (which have few points).
    'min_samples': [9, 12, 15, 18, 21],

    # 3. Azimuth Weight
    # Controls the "Cone Width".
    # 10.0 => 1 radian error costs as much as 10 meters of range error.
    # Higher = Stricter on Angle. Lower = Looser on Angle.
    'azimuth_weight': [0.5, 1.0, 3.0, 5.0, 10.0], 

    # 4. Velocity Weight
    # How much 1 m/s difference matters compared to 1 meter of distance.
    'velocity_weight': [1.0, 2.0, 3.0, 4.0, 5.0]
}

# --- GLOBAL WORKER STATE ---
_worker_cached_data = None

def init_worker(data):
    """Initialize the worker process with the dataset."""
    global _worker_cached_data
    _worker_cached_data = data

# --- 1. Data Loader ---
def cache_simulation_data(max_frames_to_load=None):
    """
    Loads flow, poses, and radar data, solves for velocity, 
    and returns a list of ALL frame detections (unclustered).
    """
    print("--- Phase 1: Caching Simulation Data ---")
    
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
        
        if frame_detections:
            cached_frames.append(frame_detections)

    print(f"Successfully cached {len(cached_frames)} frames.")
    return cached_frames

# --- 2. Strict Evaluation Logic ---
def evaluate_polar_quality(all_frames_data, eps, min_samples, az_weight, vel_weight):
    """
    Evaluates Polar Clustering Quality.
    Metric: Retention-Weighted F1 Score.
    """
    total_clusters_generated = 0
    total_healthy_clusters = 0
    total_objects_tracked = 0
    weighted_object_recall_sum = 0.0 
    
    for frame_detections in all_frames_data:
        
        # --- PRE-SCAN: Count REAL points per object ---
        real_point_counts = defaultdict(int)
        for det in frame_detections:
            if det[7] > 0 and det[3] == NoiseType.REAL:
                real_point_counts[det[7]] += 1
        
        valid_eval_ids = set()
        for obj_id, count in real_point_counts.items():
            if count >= 10:
                valid_eval_ids.add(obj_id)
                
        # --- 1. Run Polar Clustering ---
        clusters, noise_points = cluster_detections_polar(
            detections=frame_detections,
            eps=eps,
            min_samples=min_samples,
            azimuth_weight=az_weight,
            velocity_weight=vel_weight
        )
        
        total_clusters_generated += len(clusters)

        # --- 2. Analyze Clusters (Precision) ---
        id_to_healthy_clusters = defaultdict(list)
        
        for c_idx, cluster in enumerate(clusters):
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
            
            # Classify Cluster
            max_category = max(count_actor_points, count_wall_points, count_noise_points)
            
            if max_category == count_wall_points:
                total_clusters_generated -= 1 # Neutral (Background)
                continue
            elif max_category == count_noise_points:
                continue # Bad (Ghost/Clutter)
            else:
                # ACTOR CLUSTER
                dominant_id = max(id_counts, key=id_counts.get)
                
                # If dominant ID is not one we are tracking (e.g., <10 points), ignore.
                if dominant_id not in valid_eval_ids:
                    total_clusters_generated -= 1
                    continue
                
                # Purity Check (>50% must be the dominant ID)
                valid_signal_count = id_counts[dominant_id]
                if valid_signal_count > (total_points / 2):
                    total_healthy_clusters += 1
                    id_to_healthy_clusters[dominant_id].append(c_idx)

        # --- 3. Analyze Objects (Retention-Weighted Recall) ---
        for obj_id in valid_eval_ids:
            total_objects_tracked += 1
            assigned_healthy = id_to_healthy_clusters[obj_id]
            
            # STRICT: Must have exactly ONE cluster for this object
            if len(assigned_healthy) == 1:
                cluster_idx = assigned_healthy[0]
                cluster = clusters[cluster_idx]
                
                # Count points that match ID AND were originally Real
                points_in_cluster = sum(
                    1 for d in cluster 
                    if d[7] == obj_id and d[3] == NoiseType.REAL
                )
                
                points_available = real_point_counts[obj_id]
                
                # Calculate Retention (0.0 to 1.0)
                if points_available > 0:
                    retention_ratio = min(1.0, points_in_cluster / points_available)
                else:
                    retention_ratio = 0.0
                
                weighted_object_recall_sum += retention_ratio

    # --- C. Calculate Scores ---
    if total_clusters_generated == 0:
        precision = 0.0
    else:
        precision = total_healthy_clusters / total_clusters_generated
        
    if total_objects_tracked == 0:
        recall = 0.0
    else:
        recall = weighted_object_recall_sum / total_objects_tracked
        
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return f1_score, weighted_object_recall_sum, total_objects_tracked

# --- Parallel Wrapper ---
def process_wrapper(params):
    global _worker_cached_data
    score, matches, total = evaluate_polar_quality(
        _worker_cached_data, 
        params['eps'], 
        params['min_samples'],
        params['azimuth_weight'],
        params['velocity_weight']
    )
    return score, matches, total, params

# --- 3. Main Execution ---
if __name__ == "__main__":
    
    # 1. Load data once
    cached_data = cache_simulation_data(max_frames_to_load=None) 
    
    if not cached_data:
        print("No data to optimize.")
        sys.exit(1)

    # 2. Prepare Search Grid
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Phase 2: Polar Grid Search ({len(param_combinations)} combinations) ---")
    print(f"Optimizing on {os.cpu_count()} cores...")

    best_score = -1.0
    best_params = None

    # 3. Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(), 
        initializer=init_worker, 
        initargs=(cached_data,)
    ) as executor:
        
        results_iterator = list(tqdm(
            executor.map(process_wrapper, param_combinations), 
            total=len(param_combinations), 
            desc="Optimizing"
        ))

    # 4. Results
    for score, matches, total, params in results_iterator:
        if score > best_score:
            best_score = score
            best_params = params
            # Convert matches (float) to int for cleaner print, or keep float
            print(f"New Best: {score:.3f} (Retained: {matches:.1f}/{total}) | Params: {params}")

    print("\n" + "="*40)
    print("POLAR OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Best Retention F1 Score: {best_score*100:.2f}%")
    print("-" * 20)
    print("Optimal Parameters:")
    if best_params:
        print(f"  eps (Range Thresh):  {best_params['eps']} m")
        print(f"  min_samples:         {best_params['min_samples']}")
        print(f"  azimuth_weight:      {best_params['azimuth_weight']}")
        print(f"  velocity_weight:     {best_params['velocity_weight']}")
    print("="*40)