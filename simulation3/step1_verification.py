import numpy as np
from core import RealPoint, Transform
from geometry import get_relative_transform, inverse_transform
from sensors import VirtualCamera, VirtualRadar
from solver import solve_pow4r_velocity_corrected
from metrics import calculate_reprojection_error

def run_flaw_analysis() -> None:
    print("=== Step 2: Analyzing the Equation Flaw (Longitudinal Motion) FIXED ===")
    dt = 0.1
    
    # --- Setup ---
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    T_rig_cam = Transform(np.eye(3), np.array([[0],[-0.5],[0]])) 
    T_rig_rad = Transform(np.eye(3), np.array([[0],[0.5],[0]])) 
    
    # T_cam_radar: Maps P_rad -> P_cam
    # P_cam = T_cam_radar @ P_rad
    T_cam_radar = get_relative_transform(T_rig_rad, T_rig_cam) 
    
    cam = VirtualCamera(K, T_rig_cam)
    rad = VirtualRadar(T_rig_rad)
    
    # --- Scenario: Longitudinal Motion ---
    ego_vel_w = np.zeros((3,1))
    obj_pos_start = np.array([[0.0], [0.0], [15.0]]) 
    obj_vel_w = np.array([[0.0], [0.0], [20.0]]) 
    
    point = RealPoint(point_id=1, position_w=obj_pos_start, velocity_w=obj_vel_w)
    
    # --- t-1 ---
    rig_pose_prev = Transform.identity()
    obs_cam_prev = cam.observe(point, rig_pose_prev)
    if obs_cam_prev is None: return

    # --- t ---
    rig_pose_curr = rig_pose_prev
    point.position_w = point.position_w + point.velocity_w * dt
    obs_cam_curr = cam.observe(point, rig_pose_curr)
    if obs_cam_curr is None: return
    obs_radar_curr = rad.observe(point, rig_pose_curr, ego_vel_w)
    
    T_c_prev_c_curr = Transform.identity()

    # --- Honest Depth Estimate (FIXED) ---
    P_r_curr = obs_radar_curr.position_r
    
    # FIX: Use T_cam_radar directly because it is defined as P_rad -> P_cam
    # P_c_curr = T_cam_radar @ P_rad
    P_c_curr = (T_cam_radar.to_matrix() @ np.append(P_r_curr, [[1]], axis=0))[:3]
    
    # --- Run Solver ---
    v_est = solve_pow4r_velocity_corrected(
        obs_cam_curr, obs_cam_prev, obs_radar_curr,
        T_cam_radar, Transform.identity(), dt
    )
    
    # --- Reprojection Error ---
    # We use P_c_curr (calculated correctly above) as the start point
    reproj_error = calculate_reprojection_error(
        v_est, P_c_curr, obs_cam_prev.uv, T_c_prev_c_curr, K, dt
    )

    print(f"GT Velocity:    {obj_vel_w.flatten()}")
    print(f"Est Velocity:   {v_est.flatten()}")
    print(f"Velocity Error: {np.linalg.norm(v_est - obj_vel_w):.4f} m/s")
    print(f"Reproj Error:   {reproj_error:.4f} pixels")

if __name__ == "__main__":
    run_flaw_analysis()