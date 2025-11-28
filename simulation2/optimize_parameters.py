import sys
import os
import glob
import numpy as np
import gc
import signal
import optuna
from typing import List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.cluster import DBSCAN

# Keep your custom modules
from modules.clustering import filter_static_points
from modules.types import DetectionTuple
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

# --- CONFIGURATION ---
TIMEOUT_SECONDS = 5

# --- Global Storage ---
_shared_frames_data: List[np.ndarray] = [] 
_shared_frames_gt: List[np.ndarray] = []

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException()

# --- Per-Frame Evaluation ---
def evaluate_single_frame(
    data_matrix: np.ndarray,
    gt_ids: np.ndarray,
    eps: float,
    min_samples: int,
    velocity_weight: float
) -> float:
    if data_matrix.shape[0] == 0:
        return 0.0

    X = data_matrix.copy()
    # Apply velocity weight
    X[:, 3:6] *= velocity_weight

    try:
        # n_jobs=1 is preferred inside a loop or optuna worker to avoid overhead
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit(X)
        pred_labels = db.labels_
    except Exception:
        return 0.0

    mask_valid_gt = gt_ids != 0
    mask_clustered = pred_labels != -1
    final_mask = mask_valid_gt | mask_clustered

    if np.sum(final_mask) == 0:
        return 0.0

    y_true = gt_ids[final_mask]
    y_pred = pred_labels[final_mask]

    if len(y_true) == 0:
        return 0.0

    _, _, v_meas = homogeneity_completeness_v_measure(y_true, y_pred)
    return float(v_meas)

# --- Optuna Objective ---
def objective(trial):
    global _shared_frames_data, _shared_frames_gt

    # --- CHANGE 1: Fine-Tuning Ranges ---
    # Centered around your best results: eps=10.6, min_samples=20, vel=14.1
    eps = trial.suggest_float("eps", 0.1, 13.0) 
    min_samples = trial.suggest_int("min_samples", 6, 30)
    velocity_weight = trial.suggest_float("velocity_weight", 0.1, 18.0)

    scores = []
    
    # Loop through ALL loaded frames
    for i in range(len(_shared_frames_data)):
        frame_data = _shared_frames_data[i]
        frame_gt = _shared_frames_gt[i]
        
        try:
            # Fixed variable name bug here (was vel_weight)
            score = evaluate_single_frame(frame_data, frame_gt, eps, min_samples, velocity_weight)
            scores.append(score)
        except Exception:
            scores.append(0.0)

    if not scores:
        return 0.0
    
    return np.mean(scores)

# --- Data Preparation ---
def prepare_data_lists(frames: List[Any]):
    print("Pre-processing frames into numpy arrays...")
    
    frames_data = []
    frames_gt = []

    all_points = [p for f in frames for p in f]
    obj_counts = defaultdict(int)
    for p in all_points:
        if p[7] > 0: obj_counts[p[7]] += 1
    
    # Valid IDs must appear in at least 10 points across the dataset
    valid_ids = {oid for oid, c in obj_counts.items() if c >= 10}

    for frame in frames:
        if not frame:
            continue
            
        # N x 6 Matrix (x, y, z, vx, vy, vz)
        d_mat = np.array([np.concatenate([p[4], p[5]]) for p in frame], dtype=np.float32)
        
        # Ground Truth Array
        gt_arr = np.array([p[7] if p[7] in valid_ids else 0 for p in frame], dtype=np.int32)
        
        frames_data.append(d_mat)
        frames_gt.append(gt_arr)
        
    return frames_data, frames_gt

# --- Main Optimization Routine ---
def run_optimization(data: List[Any], n_trials: int = 50) -> None:
    global _shared_frames_data, _shared_frames_gt
    
    # --- CHANGE 2: Removed Slicing/Chunking ---
    print(f"Loading ALL {len(data)} frames into memory for optimization...")
    _shared_frames_data, _shared_frames_gt = prepare_data_lists(data)
    
    print(f"Data ready. Dimensions: {len(_shared_frames_data)} frames.")
    gc.collect()

    # --- Run Optuna ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    print(f"Starting {n_trials} trials on FULL dataset...")
    # Consider reducing n_trials slightly if dataset is huge, as it will be slower now
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    if len(study.trials) == 0:
        print("Optimization failed.")
        return

    # --- Results ---
    best = study.best_params
    print("\n" + "="*40)
    print("OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Best Mean V-Measure: {study.best_value:.4f}")
    print("Optimal Parameters:")
    print(f"  EPS:             {best['eps']:.4f}")
    print(f"  Min Samples:     {best['min_samples']}")
    print(f"  Velocity Weight: {best['velocity_weight']:.4f}")

# --- Standard Boilerplate ---
def load_data(max_frames: Optional[int] = None) -> List[Any]:
    print("--- Phase 1: Loading Data ---")
    carla_output_dir = "../carla/output"
    delta_t = 0.05
    cam_dir = os.path.join(carla_output_dir, "camera_rgb")
    ply_dir = os.path.join(carla_output_dir, "radar_ply")
    poses_dir = os.path.join(carla_output_dir, "poses")
    calib_dir = os.path.join(carla_output_dir, "calib")
    flow_dir = os.path.join(carla_output_dir, "flow")

    try:
        k_cam = np.loadtxt(os.path.join(calib_dir, "intrinsics.txt"))
        t_a_to_r = np.loadtxt(os.path.join(calib_dir, "extrinsics_radar_from_camera.txt"))
        mock_camera = MockCamera(k_cam)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    cached_frames = []
    limit = len(image_files) if max_frames is None else min(len(image_files), max_frames)

    for i in tqdm(range(1, limit), desc="Loading Frames"):
        img_path = image_files[i]
        frame_id = os.path.basename(img_path).split('.')[0]
        pose_path = os.path.join(poses_dir, f"{frame_id}_relative_pose.txt")
        ply_path = os.path.join(ply_dir, f"{frame_id}.ply")
        flow_path = os.path.join(flow_dir, f"{frame_id}.npy")

        if not (os.path.exists(pose_path) and os.path.exists(ply_path) and os.path.exists(flow_path)):
            continue
        try:
            t_a_to_b = np.loadtxt(pose_path)
            detections = load_radar_ply(ply_path)
            flow = np.load(flow_path)
            if not detections: continue

            processed_frame = estimate_velocities_from_data(
                detections, flow, mock_camera, t_a_to_b, t_a_to_r, delta_t
            )
            if processed_frame:
                filtered_static_frame = filter_static_points(processed_frame) 
                cached_frames.append(filtered_static_frame)
        except: continue

    
    return cached_frames

if __name__ == "__main__":
    # Removed max_frames limit to ensure we get everything available on disk
    all_data = load_data(max_frames=None)
    if not all_data:
        sys.exit(1)
    
    # Run optimization
    run_optimization(all_data, n_trials=50)