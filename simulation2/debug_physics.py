import sys
import os
import glob
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm

# --- Reuse your existing imports ---
from modules.types import DetectionTuple
from main_carla import load_radar_ply, estimate_velocities_from_data, MockCamera

def print_vector(name, vec):
    """Helper to print vectors clearly"""
    return f"{name}: [{vec[0]:6.3f}, {vec[1]:6.3f}, {vec[2]:6.3f}] (Mag: {np.linalg.norm(vec):.3f})"

def debug_physics_logic(frames_with_pose: List[Tuple[List[DetectionTuple], np.ndarray]], t_a_to_r: np.ndarray):
    
    delta_t = 0.05
    R_cam_to_radar = t_a_to_r[:3, :3]
    
    print("\n" + "="*60)
    print(f"SEARCHING FOR HIGH VELOCITY FRAMES...")
    print("="*60)

    found_moving_frame = False

    for i, (detections, t_a_to_b) in enumerate(frames_with_pose):
        
        # 1. Calculate Ego Motion FIRST
        trans_cam_frame = t_a_to_b[:3, 3]
        
        # Apply your fix: Invert translation to get velocity
        v_ego_cam_frame = -(trans_cam_frame / delta_t) 
        v_ego_radar_frame = R_cam_to_radar @ v_ego_cam_frame
        
        speed = np.linalg.norm(v_ego_radar_frame)

        # SKIP frames where the car is stopped (< 1.0 m/s)
        if speed < 1.0:
            continue
            
        found_moving_frame = True
        
        print(f"\n--- FRAME {i} (Speed: {speed:.2f} m/s) ---")
        print(print_vector("v_ego (Radar Frame)", v_ego_radar_frame))
        print("-" * 30)

        # 2. Analyze Detections (Pick STATIC samples only)
        # We really only care about walls (ID=0) right now
        static_samples = [d for d in detections if d[7] == 0][:3]
        
        if not static_samples:
            print("  (No static points found in this frame)")
            continue

        for j, det in enumerate(static_samples):
            v_rel = det[5]
            v_world = v_rel + v_ego_radar_frame
            v_mag = np.linalg.norm(v_world)
            
            print(f"\n  Point {j} [STATIC]")
            print(f"    " + print_vector("v_rel (Measured)", v_rel))
            print(f"    " + print_vector("+ v_ego (Added)   ", v_ego_radar_frame))
            print(f"    " + "-"*40)
            print(f"    " + print_vector("= v_world (Result)", v_world))
            
            # THE MOMENT OF TRUTH
            if v_mag > 1.0:
                print(f"    >>> FAIL: Result is {v_mag:.2f} m/s. (Should be near 0)")
            else:
                print(f"    >>> PASS: Result is {v_mag:.2f} m/s.")

        # Stop after analyzing 3 moving frames
        if i > 50 and found_moving_frame: # Just grab a few
            break
            
    if not found_moving_frame:
        print("WARNING: Car never exceeded 1.0 m/s in the loaded frames.")

def load_data_debug() -> Tuple[List[Any], np.ndarray]:
    # Simplified loader for debugging
    carla_output_dir = "../carla/output"
    delta_t = 0.05
    ply_dir = os.path.join(carla_output_dir, "radar_ply")
    poses_dir = os.path.join(carla_output_dir, "poses")
    calib_dir = os.path.join(carla_output_dir, "calib")
    flow_dir = os.path.join(carla_output_dir, "flow")

    t_a_to_r = np.loadtxt(os.path.join(calib_dir, "extrinsics_radar_from_camera.txt"))
    k_cam = np.loadtxt(os.path.join(calib_dir, "intrinsics.txt"))
    mock_camera = MockCamera(k_cam)

    image_files = sorted(glob.glob(os.path.join(carla_output_dir, "camera_rgb", "*.png")))
    cached_frames = []
    
    # Just load first 10 frames to find good samples
    print("Scanning first 200 frames for movement...")
    for i in range(1, 200):
        img_path = image_files[i]
        frame_id = os.path.basename(img_path).split('.')[0]
        pose_path = os.path.join(poses_dir, f"{frame_id}_relative_pose.txt")
        ply_path = os.path.join(ply_dir, f"{frame_id}.ply")
        flow_path = os.path.join(flow_dir, f"{frame_id}.npy")

        if os.path.exists(pose_path) and os.path.exists(ply_path):
            t_a_to_b = np.loadtxt(pose_path)
            detections = load_radar_ply(ply_path)
            flow = np.load(flow_path)
            
            processed_frame = estimate_velocities_from_data(detections, flow, mock_camera, t_a_to_b, t_a_to_r, delta_t)
            if processed_frame:
                cached_frames.append((processed_frame, t_a_to_b))

    return cached_frames, t_a_to_r

if __name__ == "__main__":
    frames, extrinsics = load_data_debug()
    if frames:
        debug_physics_logic(frames, extrinsics)
    else:
        print("No frames found.")