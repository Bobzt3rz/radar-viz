import sys
import os
import glob
import numpy as np
import cv2
import concurrent.futures
import itertools
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# --- Import all the original logic modules ---
# Ensure these modules are accessible in the python path
from modules.utils import save_clustering_analysis_plot, save_frame_error_histogram, plot_global_summary, save_frame_projections, GlobalErrorTracker
from modules.clustering import cluster_detections_6d, filter_clusters_quantile, filter_clusters_mad, cluster_detections_perfect, filter_static_points
from modules.types import NoiseType, DetectionTuple, Matrix4x4, FlowField
from modules.velocity_solver import solve_full_velocity, calculate_reprojection_error, calculate_rigid_3d_velocities
# Note: ClusterTracker is stateful (temporal). We cannot parallelize the tracker update itself easily,
# but we can parallelize the detection/velocity steps.

# --- NEW: Define the dataclass that bridges PLY to solver ---
@dataclass
class RadarDetection:
    position_local: np.ndarray
    radial_velocity: float
    velocity_gt_radar: np.ndarray
    noise_type: NoiseType
    velocity_gt_world: np.ndarray
    object_type: int
    object_id: int
    position_gt_local: np.ndarray
    angular_velocity_gt_radar: np.ndarray
    center_gt_radar: np.ndarray

# --- NEW: Mock Camera object for the solver ---
class MockCamera:
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

# --- Helper function to load PLY ---
def load_radar_ply(ply_path: str) -> List[RadarDetection]:
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

        data_array = np.loadtxt(data_lines, dtype=np.float32)
        if data_array.ndim == 1: data_array = data_array.reshape(1, -1)
            
        detections = []
        for row in data_array:
            detections.append(RadarDetection(
                position_local=np.array([row[0], row[1], row[2]], dtype=np.float32),
                radial_velocity=float(row[6]),
                velocity_gt_radar=np.array([row[7], row[8], row[9]], dtype=np.float32),
                noise_type=NoiseType(int(row[13])),
                velocity_gt_world=np.array([row[10], row[11], row[12]], dtype=np.float32),
                object_type=int(row[14]),
                object_id=int(row[15]),
                position_gt_local=np.array([row[16], row[17], row[18]], dtype=np.float32),
                angular_velocity_gt_radar=np.array([row[19], row[20], row[21]], dtype=np.float32),
                center_gt_radar=np.array([row[22], row[23], row[24]], dtype=np.float32)
            ))
        return detections
    except Exception as e:
        # In multiprocessing, print might get messy, better to return empty
        print(e)
        return []

# --- Solver logic ---
def estimate_velocities_from_data(
    radar_detections: List[RadarDetection],
    flow: Optional[FlowField],
    camera: MockCamera,
    T_A_to_B: Matrix4x4,
    T_A_to_R: Matrix4x4,
    world_delta_t: float
) -> List[DetectionTuple]:
    
    frame_results: List[DetectionTuple] = []
    if flow is None: return frame_results 
    
    try:
        T_Cam_from_Radar = np.linalg.inv(T_A_to_R)
        T_B_to_A = np.linalg.inv(T_A_to_B)
    except np.linalg.LinAlgError:
        return frame_results

    for detection in radar_detections:
        point_radar_coords = np.array(detection.position_local, dtype=float)
        noiseType = detection.noise_type
        
        point_rad_h = np.append(point_radar_coords, 1.0)
        point_cam_B = (T_Cam_from_Radar @ point_rad_h)[:3]
        depth_B = point_cam_B[2]
        
        if depth_B <= 1e-3: continue 

        uq = point_cam_B[0] / depth_B
        vq = point_cam_B[1] / depth_B

        xq_pix_f = camera.fx * uq + camera.cx
        yq_pix_f = camera.fy * vq + camera.cy
        xq_pix, yq_pix = int(round(xq_pix_f)), int(round(yq_pix_f))

        if not (0 <= xq_pix < camera.image_width and 0 <= yq_pix < camera.image_height):
            continue
        
        dx, dy = flow[yq_pix, xq_pix]
        xp_pix_f = xq_pix_f - dx
        yp_pix_f = yq_pix_f - dy
        up = (xp_pix_f - camera.cx) / camera.fx
        vp = (yp_pix_f - camera.cy) / camera.fy

        full_vel_vector_radar = solve_full_velocity(
            up=up, vp=vp, uq=uq, vq=vq,
            point_cam_B=point_cam_B, delta_t=world_delta_t,
            T_A_to_B=T_A_to_B, T_B_to_A=T_B_to_A, T_A_to_R=T_A_to_R,
            speed_radial=detection.radial_velocity, point_radar_coords=point_radar_coords
        )
        
        if full_vel_vector_radar is not None:
            full_vel_magnitude = float(np.linalg.norm(full_vel_vector_radar))
            frame_displacement_error = calculate_reprojection_error(
                full_vel_radar=full_vel_vector_radar, point_radar_B=point_radar_coords,
                T_Cam_from_Radar=T_Cam_from_Radar, T_B_to_A=T_B_to_A, flow=flow, 
                camera=camera, xq_pix_f=xq_pix_f, yq_pix_f=yq_pix_f, delta_t=world_delta_t
            )
             
            if frame_displacement_error is not None:
                vel_err = 0.0
                if noiseType == NoiseType.REAL and detection.object_type == 1:
                    vel_err = float(np.linalg.norm(full_vel_vector_radar - detection.velocity_gt_radar))

                frame_results.append((full_vel_magnitude, vel_err, frame_displacement_error, 
                                      noiseType, point_radar_coords, full_vel_vector_radar, 
                                      detection.velocity_gt_radar, detection.object_id, dx, dy, 
                                      detection.position_gt_local, detection.angular_velocity_gt_radar, 
                                      detection.center_gt_radar, detection.velocity_gt_world, np.array([0.0, 0.0, 0.0])))
    return frame_results

# --- WORKER FUNCTION ---
def process_single_frame(
    frame_idx: int,
    image_path_B: str,
    pose_file_relative: str,
    ply_file_B: str,
    flow_file_B: str,
    K_cam: np.ndarray,
    T_A_to_R_static: Matrix4x4,
    delta_t: float
) -> Dict[str, Any]:
    """
    This function runs in a separate process. 
    It loads data, runs the solver, runs clustering, saves plots, 
    and returns the statistics to the main process.
    """
    
    # Result container
    result = {
        'frame_idx': frame_idx,
        'success': False,
        'clusters': [],
        'filtered_clusters': [],
        'cluster_noise_points': [],
        'filter_noise_points': []
    }

    # 1. Check Files
    if not all(os.path.exists(f) for f in [image_path_B, pose_file_relative, ply_file_B, flow_file_B]):
        return result

    try:
        # 2. Load Data
        # Prevent OpenCV from spawning its own threads inside this process
        cv2.setNumThreads(0) 
        
        current_frame_rgb = cv2.cvtColor(cv2.imread(image_path_B), cv2.COLOR_BGR2RGB)
        T_A_to_B = np.loadtxt(pose_file_relative)
        radar_detections = load_radar_ply(ply_file_B)
        flow = np.load(flow_file_B)

        if not radar_detections:
            return result

        # 3. Run Solver
        mock_camera = MockCamera(K_cam)
        detections = estimate_velocities_from_data(
            radar_detections, flow, mock_camera,
            T_A_to_B, T_A_to_R_static, delta_t
        )

        if not detections:
            return result
        
        # filter out static points first
        filtered_static_detections = filter_static_points(detections)

        # 4. Clustering
        clusters, cluster_noise_points = cluster_detections_6d(
            detections=filtered_static_detections, 
            eps=8.7205, min_samples=11, velocity_weight=2.5279
        )

        rigid_clusters = calculate_rigid_3d_velocities(clusters)

        # clusters, cluster_noise_points = cluster_detections_perfect(filtered_static_detections)

        filtered_clusters, filter_noise_points = filter_clusters_mad(rigid_clusters, std_threshold=(2.75, 3.25, 1.75))

        noise_points = cluster_noise_points + filter_noise_points

        result['clusters'] = rigid_clusters
        result['filtered_clusters'] = filtered_clusters
        result['cluster_noise_points'] = cluster_noise_points
        result['filter_noise_points'] = filter_noise_points

        save_frame_projections(frame_number=frame_idx, 
                               detections=filtered_static_detections, 
                               image_rgb=current_frame_rgb, 
                               T_Cam_from_Radar=np.linalg.inv(T_A_to_R_static), 
                               K=K_cam, 
                               output_dir="output/projections")

        flattened_clusters = list(itertools.chain.from_iterable(rigid_clusters))
        save_frame_projections(frame_number=frame_idx, 
                               detections=flattened_clusters, 
                               image_rgb=current_frame_rgb, 
                               T_Cam_from_Radar=np.linalg.inv(T_A_to_R_static), 
                               K=K_cam, 
                               output_dir="output/clustered_projections")
        
        flattened_filtered_clusters = list(itertools.chain.from_iterable(filtered_clusters))
        save_frame_projections(frame_number=frame_idx, 
                               detections=flattened_filtered_clusters, 
                               image_rgb=current_frame_rgb, 
                               T_Cam_from_Radar=np.linalg.inv(T_A_to_R_static), 
                               K=K_cam, 
                               output_dir="output/filtered_projections")

        # 7. Save Visualizations (File I/O is fine here)
        # Note: Matplotlib backends can sometimes be tricky in subprocesses. 
        # Ensure utils uses a non-interactive backend (like Agg) if possible.
        # save_clustering_analysis_plot(
        #     frame_number=frame_idx,
        #     clusters=filtered_clusters,
        #     noise_points=noise_points,
        #     output_dir="output/clustering_analysis"
        # )

        # frame_results_for_histo = save_frame_error_histogram(
        #     frame_idx, filtered_static_detections, current_frame_rgb, 
        #     np.linalg.inv(T_A_to_R_static), K_cam, "output/object_analysis"
        # )

        result['success'] = True
        return result

    except Exception as e:
        print(f"Err Frame {frame_idx}: {e}")
        return result


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Important for multiprocessing on Windows/MacOS
    import multiprocessing
    multiprocessing.freeze_support()

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
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)

    all_image_files = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    if not all_image_files:
        print(f"Error: No images found in {CAM_DIR}.")
        sys.exit(1)
        
    max_frames = len(all_image_files)
    
    # Global Aggregators
    all_detections = []
    all_clusters = []
    all_filtered_clusters = []
    all_cluster_noise_points = []
    all_filter_noise_points = []


    # error_tracker = GlobalErrorTracker()

    print(f"\nStarting Parallel Processing on {os.cpu_count()} cores...")

    # We use ProcessPoolExecutor to manage the worker processes
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Prepare the list of futures
        futures = []
        
        for frame_count in range(1, max_frames):
            image_path_B = all_image_files[frame_count]
            frame_id_B = os.path.basename(image_path_B).split('.')[0]
            pose_file_relative = os.path.join(POSES_DIR, f"{frame_id_B}_relative_pose.txt")
            ply_file_B = os.path.join(PLY_DIR, f"{frame_id_B}.ply")
            flow_file_B = os.path.join(FLOW_DIR, f"{frame_id_B}.npy")
            
            # Submit task
            futures.append(executor.submit(
                process_single_frame,
                frame_count,
                image_path_B,
                pose_file_relative,
                ply_file_B,
                flow_file_B,
                K_cam,
                T_A_to_R_static,
                DELTA_T
            ))

        # Process results as they complete
        # Use as_completed to get results faster, or simply iterate if order matters strictly
        # Here order doesn't matter for aggregation, but usually tracking requires order.
        # Since we aren't running the ClusterTracker (temporal) update in the loop, order is fine.
        
        from tqdm import tqdm # Optional: for progress bar
        
        # If you don't have tqdm, just use: for future in concurrent.futures.as_completed(futures):
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = future.result()
            
            if res['success']:
                all_detections.extend(
                    [det for cluster in res['clusters'] for det in cluster] + 
                    [det for cluster in res['cluster_noise_points'] for det in cluster])
                all_clusters.extend(res['clusters'])
                all_filtered_clusters.extend(res['filtered_clusters'])
                all_cluster_noise_points.extend(res['cluster_noise_points'])
                all_filter_noise_points.extend(res['filter_noise_points'])
            
            # If you want to print per frame (might be noisy with tqdm)
            # print(f"Finished frame {res['frame_idx']}")

    # --- Final Reporting (Identical to original) ---
    plot_global_summary(all_clusters, all_filtered_clusters, all_cluster_noise_points, all_filter_noise_points, 'output')
    
    # if all_real_velocity_abs_errors:
    #     errors_array = np.array(all_real_velocity_abs_errors)
    #     print(f"  Global MAE: {np.mean(errors_array):.6f} m/s")