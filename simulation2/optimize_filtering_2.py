import sys
import os
import glob
import numpy as np
import optuna
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from typing import List
from enum import IntEnum

# --- Imports ---
from modules.types import NoiseType  # Assuming this exists as you described
from modules.clustering import cluster_detections_6d
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

# --- CONFIGURATION ---

# 1. FIXED STAGE 1 PARAMETERS (Baseline)
STAGE_1_PARAMS = {
    'eps': 10.29,
    'min_samples': 19,
    'velocity_weight': 17.99
}

# 2. OPTUNA CONFIG
N_TRIALS = 100

# --- GLOBAL STORAGE ---
# Format: List of matrices. Each matrix is (N, 8)
# Columns: [x, y, z, vx, vy, vz, noise_type, object_id]
_shared_clusters_data: List[np.ndarray] = []

def cache_and_cluster_data(max_frames=None):
    """
    Runs Stage 1 (Coarse Clustering) and caches the resulting clusters 
    so we can repeatedly optimize Stage 2 (Fine Filtering).
    """
    print("--- Phase 1: Pre-calculating Stage 1 Clusters ---")
    
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
        T_A_to_R = np.loadtxt(os.path.join(CALIB_DIR, "extrinsics_radar_from_camera.txt"))
        mock_camera = MockCamera(K_cam)
    except Exception as e:
        print(f"Error loading calib: {e}")
        sys.exit(1)

    all_image_files = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    limit = len(all_image_files) if max_frames is None else min(len(all_image_files), max_frames)
    
    extracted_clusters = []

    for i in tqdm(range(1, limit), desc="Clustering Stage 1"):
        image_path = all_image_files[i]
        frame_id = os.path.basename(image_path).split('.')[0]
        
        pose_file = os.path.join(POSES_DIR, f"{frame_id}_relative_pose.txt")
        ply_file = os.path.join(PLY_DIR, f"{frame_id}.ply")
        flow_file = os.path.join(FLOW_DIR, f"{frame_id}.npy")

        if not (os.path.exists(pose_file) and os.path.exists(ply_file) and os.path.exists(flow_file)):
            continue

        try:
            T_A_to_B = np.loadtxt(pose_file)
            radar_detections = load_radar_ply(ply_file)
            flow = np.load(flow_file)

            if not radar_detections: continue

            frame_detections = estimate_velocities_from_data(
                radar_detections, flow, mock_camera, 
                T_A_to_B, T_A_to_R, DELTA_T
            )
            
            if not frame_detections: continue

            # --- RUN STAGE 1 (FIXED) ---
            clusters, _ = cluster_detections_6d(
                frame_detections,
                eps=STAGE_1_PARAMS['eps'],
                min_samples=STAGE_1_PARAMS['min_samples'],
                velocity_weight=STAGE_1_PARAMS['velocity_weight']
            )

            # Convert to Numpy for speed
            for cluster in clusters:
                if len(cluster) < 3: continue

                # Tuple structure:
                # 0: vel_mag, 1: vel_err, 2: disp_err, 
                # 3: NOISE_TYPE (Enum), 
                # 4: vel_3d (np array), 5: pos_3d (np array), 
                # 6: gt_vel, 
                # 7: OBJECT_ID (int)
                
                mat_list = []
                for d in cluster:
                    # We store: x, y, z, vx, vy, vz, noise_type, object_id
                    # Note: We must cast Enum to int for numpy storage
                    noise_val = d[3].value if hasattr(d[3], 'value') else int(d[3])
                    row = np.concatenate([d[5], d[4], [noise_val], [d[7]]])
                    mat_list.append(row)
                
                extracted_clusters.append(np.array(mat_list, dtype=np.float32))

        except Exception:
            continue

    return extracted_clusters

def objective(trial):
    global _shared_clusters_data

    # --- Hyperparameters to Tune (Stage 2) ---
    s2_eps = trial.suggest_float("eps", 0.01, 4.0) 
    s2_min_samples = trial.suggest_int("min_samples", 3, 15)
    s2_vel_weight = trial.suggest_float("velocity_weight", 0.0, 20.0)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cluster_data in _shared_clusters_data:
        # cluster_data: [x, y, z, vx, vy, vz, noise_type, object_id]
        
        # 1. Filter out background points (ID == 0) from SCORING, 
        # but we usually keep them in CLUSTERING to see if they get filtered out.
        # However, your instruction says "we don't care about object_ids == 0".
        # So we will mask them out during the score calculation.
        
        # DBSCAN Input
        X = cluster_data[:, 0:6].copy()
        X[:, 3:6] *= s2_vel_weight

        # 2. Run Stage 2 DBSCAN (The Refinement)
        try:
            # We treat label -1 as "Noise/Ghost" to be removed
            db = DBSCAN(eps=s2_eps, min_samples=s2_min_samples, n_jobs=1).fit(X)
            labels = db.labels_
        except Exception:
            labels = np.full(len(cluster_data), -1)

        # 3. Scoring Logic
        noise_types = cluster_data[:, 6]
        object_ids = cluster_data[:, 7]

        # Masks
        is_relevant = object_ids != 0  # Ignore background
        is_kept = labels != -1         # Stage 2 kept this point
        is_removed = labels == -1      # Stage 2 removed this point
        
        # Ground Truths (within relevant points)
        # REAL = 0, everything else (1-6) is GHOST/NOISE
        is_gt_real = noise_types == 0 
        is_gt_ghost = noise_types > 0

        # Calculate metrics ONLY on relevant points
        # TP: Real point that was KEPT
        total_tp += np.sum(is_kept & is_gt_real & is_relevant)

        # FP: Ghost point that was KEPT
        total_fp += np.sum(is_kept & is_gt_ghost & is_relevant)

        # FN: Real point that was REMOVED
        total_fn += np.sum(is_removed & is_gt_real & is_relevant)

    # 4. F1 Score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    if (precision + recall) == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

if __name__ == "__main__":
    
    # 1. Prepare Data
    _shared_clusters_data = cache_and_cluster_data(max_frames=None)
    
    if not _shared_clusters_data:
        print("No clusters extracted. Check data paths.")
        sys.exit(1)
        
    print(f"Optimizing Stage 2 over {len(_shared_clusters_data)} clusters.")

    # 2. Run Optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # 3. Results
    print("\n" + "="*50)
    print("STAGE 2: GHOST REMOVAL OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best F1 (Real vs Ghost): {study.best_value*100:.2f}%")
    print("-" * 20)
    print("Optimal Refinement Params:")
    print(f"  EPS:             {study.best_params['eps']:.4f}")
    print(f"  Min Samples:     {study.best_params['min_samples']}")
    print(f"  Velocity Weight: {study.best_params['velocity_weight']:.4f}")
    print("="*50)