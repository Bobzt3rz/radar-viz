import numpy as np
from core import Transform
from geometry import project_point

def calculate_reprojection_error(
    estimated_velocity_cam: np.ndarray, # The velocity we want to test (3x1)
    start_point_cam_t: np.ndarray,      # The 3D point at time t (from Radar)
    obs_cam_prev: np.ndarray,           # Actual pixel coordinates at t-1 (GT)
    T_cam_prev_curr: Transform,         # Ego motion: P_prev = T @ P_curr
    K: np.ndarray,                      # Camera Intrinsics
    dt: float
) -> float:
    """
    Calculates the displacement error (in pixels) between:
    1. The actual observed pixel at t-1
    2. The pixel predicted at t-1 by 'rewinding' the point at t using estimated velocity.
    """
    
    # 1. Back-propagate the point to time t-1
    # Logic: P_curr_static = P_curr - (Vel_obj * dt)
    #        P_prev = T_ego * P_curr_static
    
    # Remove object motion
    P_curr_static = start_point_cam_t - (estimated_velocity_cam * dt)
    
    # Apply Ego Motion (Transformation to previous camera frame)
    P_prev_estimated = (T_cam_prev_curr.to_matrix() @ np.append(P_curr_static, [[1]], axis=0))[:3]
    
    # 2. Project to 2D
    uv_projected, _ = project_point(K, P_prev_estimated)
    
    if uv_projected is None:
        # If the estimated velocity pushes the point behind the camera, return high error
        return 1000.0 

    # 3. Calculate L2 Distance
    # obs_cam_prev is shape (2, 1)
    diff = uv_projected - obs_cam_prev
    error_pixels = float(np.linalg.norm(diff))
    
    return error_pixels