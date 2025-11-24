import numpy as np
from dataclasses import replace
from core import RadarObservation, RealPoint, Transform
from sensors import VirtualCamera, VirtualRadar
from geometry import get_relative_transform
from solver import solve_pow4r_velocity_corrected
from metrics import calculate_reprojection_error


class NoiseGenerator:
    def __init__(self):
        pass
    
    def apply_gaussian_noise(self, obs: RadarObservation, std_pos: float, std_vel: float) -> RadarObservation:
        """Type A: Measurement Noise"""
        new_pos = obs.position_r + np.random.normal(0, std_pos, (3,1))
        new_vel = obs.radial_velocity + np.random.normal(0, std_vel)
        
        # Note: Changing pos changes unit vector technically, but for small noise we can ignore or recompute.
        # Let's strictly recompute unit vector for consistency? 
        # No, usually noise assumes sensor raw data is noisy. 
        # If pos is noisy, unit vec is noisy.
        new_unit = new_pos / np.linalg.norm(new_pos)
        
        return replace(obs, position_r=new_pos, radial_velocity=new_vel, radial_unit_vec=new_unit)

    def apply_doppler_ambiguity(self, obs: RadarObservation, v_max: float = 15.0) -> RadarObservation:
        """Type B: Velocity Aliasing (v_reported = v_true - 2*v_max)"""
        # Simulating a wrap-around error
        new_vel = obs.radial_velocity - 2 * v_max
        return replace(obs, radial_velocity=new_vel)

    def apply_multipath_ghost(self, obs: RadarObservation, offset_dist: float) -> RadarObservation:
        """Type C: Multipath (Ghost Target)"""
        # Ghost is at wrong position (e.g. reflection behind a wall)
        # But often retains similar velocity OR mirrored velocity.
        # Let's simulate a reflection that puts the point 5m deeper and changes direction.
        
        # Reflection logic: Point is further away
        direction = obs.position_r / np.linalg.norm(obs.position_r)
        new_pos = obs.position_r + direction * offset_dist
        
        # Velocity usually projects differently or is messy.
        # Let's assume velocity is preserved but position is wrong (common in simple multipath).
        return replace(obs, position_r=new_pos)

def run_noise_experiment():
    print("\n=== Step 3: Noise Sensitivity Analysis ===")
    
    # Setup (Same as before)
    dt = 0.1
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    T_rig_cam = Transform(np.eye(3), np.array([[0],[-0.5],[0]])) 
    T_rig_rad = Transform(np.eye(3), np.array([[0],[0.5],[0]])) 
    T_cam_radar = get_relative_transform(T_rig_rad, T_rig_cam)
    cam = VirtualCamera(K, T_rig_cam)
    rad = VirtualRadar(T_rig_rad)
    noise_gen = NoiseGenerator()

    # Base Scenario: Diagonal Motion (Good constraints for both Cam and Radar)
    point = RealPoint(1, np.array([[2.0], [0.0], [10.0]]), np.array([[-5.0], [0.0], [10.0]]))
    
    # T-1
    obs_cam_prev = cam.observe(point, Transform.identity())
    if obs_cam_prev is None: return

    # T
    point.position_w += point.velocity_w * dt
    obs_cam_curr = cam.observe(point, Transform.identity())
    obs_radar_gt = rad.observe(point, Transform.identity(), np.zeros((3,1)))
    if obs_cam_curr is None: return
    
    # Honest Depth (Base)
    P_c_curr_gt = (T_cam_radar.to_matrix() @ np.append(obs_radar_gt.position_r, [[1]], axis=0))[:3]
    depth_est = float(P_c_curr_gt[2]) # Approx d(t-1) ~ d(t)

    # Define Tests
    tests = [
        ("Baseline (No Noise)", obs_radar_gt),
        ("Gaussian Noise (Pos=0.2m, Vel=0.5m/s)", noise_gen.apply_gaussian_noise(obs_radar_gt, 0.2, 0.5)),
        ("Doppler Ambiguity (V_max=15)", noise_gen.apply_doppler_ambiguity(obs_radar_gt, v_max=15.0)),
        ("Multipath Ghost (Dist Offset +5m)", noise_gen.apply_multipath_ghost(obs_radar_gt, 5.0))
    ]

    print(f"{'Noise Type':<35} | {'Vel Err (m/s)':<15} | {'Reproj Err (px)':<15} | {'Status'}")
    print("-" * 80)

    for name, noisy_radar in tests:
        P_c_curr_noisy = (T_cam_radar.to_matrix() @ np.append(noisy_radar.position_r, [[1]], axis=0))[:3]

        v_est = solve_pow4r_velocity_corrected(
            obs_cam_curr, obs_cam_prev, noisy_radar,
            T_cam_radar, Transform.identity(), dt
        )
        
        # Metric
        reproj_error = calculate_reprojection_error(
            v_est, P_c_curr_noisy, obs_cam_prev.uv, Transform.identity(), K, dt
        )
        
        # Truth comparison
        vel_err = np.linalg.norm(v_est - point.velocity_w)
        
        # Thresholding (Hypothetical Filter)
        status = "KEEP" if reproj_error < 10.0 else "REJECT"
        
        print(f"{name:<35} | {vel_err:<15.4f} | {reproj_error:<15.4f} | {status}")

if __name__ == "__main__":
    run_noise_experiment()