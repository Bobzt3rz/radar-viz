import sys
import os
import glob
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# --- Import all the original logic modules ---
from modules.optical_flow import OpticalFlow
from modules.utils import save_image, save_frame_histogram, save_clustering_analysis_plot
from modules.clustering import cluster_detections_6d
from modules.types import NoiseType, DetectionTuple, Vector3, Matrix4x4, FlowField
from modules.velocity_solver import solve_full_velocity, calculate_reprojection_error


# --- NEW: Define the dataclass that bridges PLY to solver ---
@dataclass
class RadarDetection:
    """A simple struct to hold data loaded from a PLY file."""
    position_local: np.ndarray  # 3D (x,y,z) in sensor coords (Paper system)
    radial_velocity: float
    velocity_gt_radar: np.ndarray # 3D (vx,vy,vz) in RADAR coords (Paper system)
    noise_type: NoiseType
    # We also load the world GT for visualization, but don't use it in the solver
    velocity_gt_world: np.ndarray 

# --- NEW: Mock Camera object for the solver ---
class MockCamera:
    """Mocks the Camera object with data loaded from files."""
    def __init__(self, K: np.ndarray):
        self._K = K
        
        self.fx: float = K[0, 0]
        self.fy: float = K[1, 1]
        self.cx: float = K[0, 2]
        self.cy: float = K[1, 2]
        
        self.image_width: int = int(round(self.cx * 2))
        self.image_height: int = int(round(self.cy * 2))

    def get_intrinsics_matrix(self) -> np.ndarray:
        return self._K

# --- NEW: Helper function to load our ASCII PLY files ---
def load_radar_ply(ply_path: str) -> List[RadarDetection]:
    """
    Loads radar detections from our specific ASCII PLY file format
    and converts them into RadarDetection objects.
    """
    try:
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        header_end_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end_index = i
                break
        
        if header_end_index == -1: return []
        data_lines = lines[header_end_index + 1:]
        if not data_lines: return [] 

        # x, y, z, az, el, r, v_rad, vx_gt, vy_gt, vz_gt, vx_w, vy_w, vz_w, noise
        data_array = np.loadtxt(data_lines, dtype=np.float32)

        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
            
        detections = []
        for row in data_array:
            pos_local = np.array([row[0], row[1], row[2]], dtype=np.float32)
            v_rad = float(row[6])
            vel_gt_radar = np.array([row[7], row[8], row[9]], dtype=np.float32)
            vel_gt_world = np.array([row[10], row[11], row[12]], dtype=np.float32)
            noise_type = NoiseType(int(row[13]))
            
            detections.append(RadarDetection(
                position_local=pos_local,
                radial_velocity=v_rad,
                velocity_gt_radar=vel_gt_radar,
                noise_type=noise_type,
                velocity_gt_world=vel_gt_world
            ))
        return detections
    except Exception as e:
        print(f"Error loading PLY file {ply_path}: {e}")
        return []

# --- REFACTORED: Solver logic based on your new plan ---
def estimate_velocities_from_data(
    radar_detections: List[RadarDetection],
    flow: Optional[FlowField],
    camera: MockCamera, # This just holds intrinsics
    T_A_to_B: Matrix4x4, # <-- NEW: Relative pose
    T_A_to_R: Matrix4x4, # <-- NEW: Static extrinsics
    world_delta_t: float
) -> List[DetectionTuple]:
    """
    Calculates full velocity for loaded radar detections.
    """
    frame_results: List[DetectionTuple] = []

    if flow is None:
        return frame_results 

    try:
        T_Cam_from_Radar_static = np.linalg.inv(T_A_to_R)
    except np.linalg.LinAlgError:
        print("  Error inverting T_A_to_R to find static extrinsic.")
        return frame_results

    # 3. Process each detection
    for detection in radar_detections:
        point_radar_coords = detection.position_local
        speed_radial = detection.radial_velocity
        noiseType = detection.noise_type
        # This is the GT we will compare against
        ground_truth_vel_radar = detection.velocity_gt_radar
        # This is just for visualization
        ground_truth_vel_world = detection.velocity_gt_world

        point_rad_h = np.append(point_radar_coords, 1.0)
        point_cam_B_h = T_Cam_from_Radar_static @ point_rad_h
        point_cam_B = point_cam_B_h[:3]
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

        # Call the solver with the inputs directly from our files
        full_vel_vector_radar = solve_full_velocity(
            up=up, vp=vp, uq=uq, vq=vq, d=depth_B, delta_t=world_delta_t,
            T_A_to_B=T_A_to_B, T_A_to_R=T_A_to_R,
            speed_radial=speed_radial, point_radar_coords=point_radar_coords,
            return_in_radar_coords=True
        )
        
        if full_vel_vector_radar is not None:
            full_vel_magnitude = float(np.linalg.norm(full_vel_vector_radar))
            
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
                if noiseType == NoiseType.REAL:
                    # --- FIX: Compare Radar-to-Radar ---
                    velocity_error_magnitude = float(np.linalg.norm(
                        full_vel_vector_radar - ground_truth_vel_radar
                    ))
                    frame_results.append((full_vel_magnitude, 
                                          velocity_error_magnitude, frame_displacement_error, 
                                          noiseType, point_radar_coords, 
                                          full_vel_vector_radar, # Prediction (Radar)
                                          ground_truth_vel_world)) # GT (World) for histogram
                else:
                    frame_results.append((full_vel_magnitude, 
                                          0.0, frame_displacement_error, 
                                          noiseType, point_radar_coords, 
                                          full_vel_vector_radar, # Prediction (Radar)
                                          ground_truth_vel_world)) # GT (World) for histogram
            
    return frame_results


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # --- 1. Define Paths and Constants ---
    CARLA_OUTPUT_DIR = "../carla/output"
    DELTA_T = 0.05 # 20 FPS
    
    CAM_DIR = os.path.join(CARLA_OUTPUT_DIR, "camera_rgb")
    PLY_DIR = os.path.join(CARLA_OUTPUT_DIR, "radar_ply")
    POSES_DIR = os.path.join(CARLA_OUTPUT_DIR, "poses")
    CALIB_DIR = os.path.join(CARLA_OUTPUT_DIR, "calib")

    # --- 2. Load Static Calibration ---
    try:
        K_cam = np.loadtxt(os.path.join(CALIB_DIR, "intrinsics.txt"))
        # --- FIX: Load static extrinsics ---
        T_A_to_R_static = np.loadtxt(os.path.join(CALIB_DIR, "extrinsics_radar_from_camera.txt"))
    except Exception as e:
        print(f"Error: Could not load calibration files. Exiting.")
        print(f"Looked in: {CALIB_DIR}")
        print(e)
        sys.exit(1)

    # --- 3. Initialize Processors and Stats ---
    optical_flow_calculator = OpticalFlow()
    print("\nStarting data processing loop...")
    
    all_real_velocity_abs_errors = []
    all_real_velocity_actual_magnitudes = []
    all_tp, all_fp, all_fn, all_tn = [], [], [], []
    
    # --- 4. Find all frames to process ---
    all_image_files = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    
    if not all_image_files:
        print(f"Error: No images found in {CAM_DIR}. Did you run the Carla script?")
        sys.exit(1)
        
    max_frames = len(all_image_files)

    for frame_count in range(1, max_frames):
        
        # --- A. Get File Paths ---
        image_path_B = all_image_files[frame_count]
        image_path_A = all_image_files[frame_count - 1]
        
        frame_id_B = os.path.basename(image_path_B).split('.')[0]
        frame_id_A = os.path.basename(image_path_A).split('.')[0]

        # --- FIX: Load the correct files based on your plan ---
        pose_file_relative = os.path.join(POSES_DIR, f"{frame_id_B}_relative_pose.txt")
        ply_file_A = os.path.join(PLY_DIR, f"{frame_id_A}.ply")
        
        required_files = [
            image_path_A, image_path_B, 
            pose_file_relative, ply_file_A
        ]
        # --- End ---

        if not all(os.path.exists(f) for f in required_files):
            print(f"Warning: Missing one or more files for frame {frame_id_B}. Skipping.")
            continue

        # --- B. Load all data from disk ---
        try:
            current_frame_rgb = cv2.cvtColor(cv2.imread(image_path_B), cv2.COLOR_BGR2RGB)
            prev_frame_rgb = cv2.cvtColor(cv2.imread(image_path_A), cv2.COLOR_BGR2RGB)
            
            # Load the "diff" (ego-motion)
            T_A_to_B = np.loadtxt(pose_file_relative)
            
            # Load detections from Frame A
            radar_detections: List[RadarDetection] = load_radar_ply(ply_file_A)
            
        except Exception as e:
            print(f"Error loading data for frame {frame_id_B}: {e}. Skipping.")
            continue

        # --- C. Run Algorithms ---
        optical_flow_calculator.inference(prev_frame_rgb)
        flow = optical_flow_calculator.inference(current_frame_rgb)
        
        print(f"--- Frame {frame_count} ({frame_id_B}) ---")
        
        if radar_detections:
            # Mock camera just holds intrinsics now
            mock_camera = MockCamera(K_cam)
            
            # --- FIX: Call the updated solver ---
            current_frame_errors: List[DetectionTuple] = estimate_velocities_from_data(
                radar_detections, 
                flow, 
                mock_camera,
                T_A_to_B,         # Pass relative pose
                T_A_to_R_static,  # Pass static extrinsics
                DELTA_T
            )

            # --- D. Run Analysis (All this code is identical) ---
            
            if current_frame_errors:
                clusters, noise_points = cluster_detections_6d(
                    detections=current_frame_errors,
                    eps=0.7, min_samples=4, velocity_weight=4.0   
                )
                
                gt_real, gt_random, gt_multipath = 0, 0, 0
                for det in current_frame_errors:
                    if det[3] == NoiseType.REAL: gt_real += 1
                    elif det[3] == NoiseType.RANDOM_CLUTTER: gt_random += 1
                    elif det[3] == NoiseType.MULTIPATH_GHOST: gt_multipath += 1
                
                total_real_points = gt_real
                total_noisy_points = gt_random + gt_multipath

                tp, fn = 0, 0
                fp_random, fp_multipath = 0, 0
                tn_random, tn_multipath = 0, 0
                
                for cluster in clusters:
                    for det in cluster:
                        if det[3] == NoiseType.REAL: tp += 1
                        elif det[3] == NoiseType.RANDOM_CLUTTER: fp_random += 1
                        elif det[3] == NoiseType.MULTIPATH_GHOST: fp_multipath += 1

                for det in noise_points:
                    if det[3] == NoiseType.REAL: fn += 1
                    elif det[3] == NoiseType.RANDOM_CLUTTER: tn_random += 1
                    elif det[3] == NoiseType.MULTIPATH_GHOST: tn_multipath += 1
                
                total_fp = fp_random + fp_multipath
                total_tn = tn_random + tn_multipath
                
                precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
                recall = tp / total_real_points if total_real_points > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                all_tp.append(tp); all_fp.append(total_fp); all_fn.append(fn); all_tn.append(total_tn)

                print(f"  Ground Truth: {total_real_points} Real | {total_noisy_points} Noisy (Random:{gt_random}, MP:{gt_multipath})")
                print(f"  Algorithm Output: {len(clusters)} Clusters, {len(noise_points)} Noise Points")
                print(f"  True Positives (TP):  {tp:4d} (Real points found)")
                print(f"  False Negatives (FN): {fn:4d} (Real points missed)")
                print(f"  False Positives (FP): {total_fp:4d} (Total noise clustered)")
                print(f"    - FP Random:    {fp_random:4d} (Filtered {tn_random}/{gt_random})")
                print(f"    - FP Multipath: {fp_multipath:4d} (Filtered {tn_multipath}/{gt_multipath})")
                print(f"  Precision: {precision * 100:6.2f}% | Recall: {recall * 100:6.2f}% | F1-Score: {f1 * 100:6.2f}%")
                print(f"--------------------------------------")
                
                fp_dict = {'random': fp_random, 'mp': fp_multipath}
                tn_dict = {'random': tn_random, 'mp': tn_multipath}
                
                save_clustering_analysis_plot(
                    frame_number=frame_count,
                    clusters=clusters,
                    noise_points=noise_points,
                    tp=tp, fp_dict=fp_dict, fn=fn, tn_dict=tn_dict,
                    precision=precision, recall=recall, f1=f1,
                    output_dir="output/clustering_analysis"
                )

            if current_frame_errors:
                real_velocity_errors, real_displacement_errors, noisy_displacement_errors = [], [], []
                real_vel_magnitudes, noisy_vel_magnitudes = [], []
                real_positions, real_velocities = [], []
                noisy_positions, noisy_velocities = [], []
                
                for vel_mag, vel_err, disp_err, noiseType, pos_3d, vel_3d_radar, vel_3d_world_gt in current_frame_errors:
                    if(noiseType == NoiseType.REAL):
                        real_vel_magnitudes.append(vel_mag)
                        real_velocity_errors.append(vel_err)
                        real_displacement_errors.append(disp_err)
                        real_positions.append(pos_3d)
                        real_velocities.append(vel_3d_world_gt) # Use world for histogram
                    else:
                        noisy_vel_magnitudes.append(vel_mag)
                        noisy_displacement_errors.append(disp_err)
                        noisy_positions.append(pos_3d)
                        noisy_velocities.append(vel_3d_world_gt) # Use world for histogram
                
                all_real_velocity_abs_errors.extend(real_velocity_errors)
                all_real_velocity_actual_magnitudes.extend(real_vel_magnitudes)

                average_real_velocity_error = np.mean(real_velocity_errors) if real_velocity_errors else 0
                average_real_displacement_error = np.mean(real_displacement_errors) if real_displacement_errors else 0
                average_noisy_displacement_error = np.mean(noisy_displacement_errors) if noisy_displacement_errors else 0

                save_frame_histogram(
                        frame_number=frame_count,
                        real_pred_vel_mags=real_vel_magnitudes,
                        real_vel_errors=real_velocity_errors,
                        real_disp_errors=real_displacement_errors,
                        noisy_pred_vel_mags=noisy_vel_magnitudes,
                        noisy_disp_errors=noisy_displacement_errors,
                        real_positions=real_positions,
                        real_velocities=real_velocities,
                        noisy_positions=noisy_positions,
                        noisy_velocities=noisy_velocities,
                        output_dir="output/frame_analysis"
                    )

                print(f"Average Real Velocity Error: {average_real_velocity_error:.6f} m/s")
                print(f"Average Real Displacement Error: {average_real_displacement_error:.6f} pix")
                print(f"Average Noisy Displacement Error: {average_noisy_displacement_error:.6f} pix")
            
        else:
            print("  No radar detections loaded for this frame.")

    # --- E. Final Results (Identical to original) ---
    print("\nProcessing loop finished.")
    print("\n" + "="*40)
    print("--- Overall Simulation Results ---")
    print("="*40)

    print("\n### Clustering Filter Performance (All Frames) ###")
    if all_tp: 
        total_tp = sum(all_tp); total_fp = sum(all_fp); total_fn = sum(all_fn); total_tn = sum(all_tn)
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        total_real = total_tp + total_fn; total_noisy = total_fp + total_tn; total_all = total_real + total_noisy
        
        print(f"  Total Ground Truth: {total_real} Real Points, {total_noisy} Noisy Points ({total_all} total)")
        print(f"  - True Positives (TP): {total_tp:6d} (Real points correctly clustered)")
        print(f"  - False Positives (FP): {total_fp:6d} (Noisy points incorrectly clustered)")
        print(f"  - False Negatives (FN): {total_fn:6d} (Real points incorrectly filtered)")
        print(f"  - True Negatives (TN): {total_tn:6d} (Noisy points correctly filtered)")
        print(f"\n  --- Overall Scores ---")
        print(f"  Precision (Cleanliness): {overall_precision * 100:6.2f}%")
        print(f"  Recall (Completeness):   {overall_recall * 100:6.2f}%")
        print(f"  F1-Score (Balance):      {overall_f1 * 100:6.2f}%")
    else:
        print("No clustering results were recorded.")

    print("\n### Velocity Estimation Performance (on True Positives) ###")
    if all_real_velocity_abs_errors and all_real_velocity_actual_magnitudes:
        errors_array = np.array(all_real_velocity_abs_errors)
        actuals_array = np.array(all_real_velocity_actual_magnitudes)
        global_mae = np.mean(errors_array); mean_actual_speed = np.mean(actuals_array)
        print(f"  Global Mean Absolute Error (MAE):   {global_mae:.6f} m/s")
        print(f"  Mean Actual Object Speed:             {mean_actual_speed:.6f} m/s")
        if mean_actual_speed > 1e-6:
            global_nmae = (global_mae / mean_actual_speed) * 100.0
            print(f"  Normalized MAE (NMAE):              {global_nmae:.2f} %")
        else:
            print("  Normalized MAE (NMAE):              N/A (Mean actual speed is zero)")
        print(f"  (Based on {len(errors_array)} total True Positive detections)")
    else:
        print("  No valid True Positive detections were recorded to calculate an overall average.")

    print("\nProcessing complete.")