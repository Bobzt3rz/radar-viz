import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
from dataclasses import dataclass
from typing import List, Optional

# ==========================================
# 1. DATA STRUCTURES
# ==========================================

@dataclass
class RadarDetection:
    position_local: np.ndarray
    radial_velocity: float
    noise_type: int
    object_id: int
    velocity_gt_radar: np.ndarray
    object_type: int

@dataclass
class MockCamera:
    fx: float; fy: float; cx: float; cy: float
    image_width: int; image_height: int

# ==========================================
# 2. THE CORRECTED SOLVER (Exact Match)
# ==========================================

def solve_full_velocity_corrected(
    up: float, vp: float,           # Prev Normalized UV
    uq: float, vq: float,           # Curr Normalized UV
    point_cam_B: np.ndarray,        # 3D Point in Current Frame
    delta_t: float,
    T_A_to_B: np.ndarray,           # Forward Transform (Prev->Curr)
    T_B_to_A: np.ndarray,           # Backward Transform (Curr->Prev)
    T_A_to_R: np.ndarray,           # Extrinsics (Cam -> Radar)
    speed_radial: float,
    point_radar_coords: np.ndarray
) -> Optional[np.ndarray]:
    """
    Solves the linear system exactly. 
    Returns full 3D velocity in Camera Frame.
    """
    # 1. Radar Unit Vector
    dist = np.linalg.norm(point_radar_coords)
    if dist < 1e-6: return None
    u_vec_rad = point_radar_coords / dist
    
    # Rotate Radar Unit Vec to Camera Frame
    R_cam_to_rad = T_A_to_R[0:3, 0:3]
    R_rad_to_cam = R_cam_to_rad.T
    r_vec_cam = R_rad_to_cam @ u_vec_rad

    # 2. Ego Motion
    alpha = T_A_to_B[0:3, 0:3]
    beta  = T_A_to_B[0:3, 3]

    # 3. Depth & Gradient
    point_cam_B_h = np.append(point_cam_B, 1.0)
    p_virtual_prev = (T_B_to_A @ point_cam_B_h)[:3]
    d_static = p_virtual_prev[2]
    k_vec = np.array([0.0, 0.0, -1.0])

    # 4. Build M (LHS)
    M = np.zeros((3, 3), dtype=float)
    for j in range(3):
        M[0, j] = alpha[0, j] - uq * alpha[2, j]
        M[1, j] = alpha[1, j] - vq * alpha[2, j]
    M[2, :] = r_vec_cam

    # 5. Build C (Depth Multiplier)
    p_uv_hom_prev = np.array([up, vp, 1.0])
    comm = np.dot(alpha[2, :], p_uv_hom_prev)
    C = np.zeros(3)
    C[0] = uq * comm - np.dot(alpha[0, :], p_uv_hom_prev)
    C[1] = vq * comm - np.dot(alpha[1, :], p_uv_hom_prev)
    C[2] = 0.0

    # 6. Apply Correction
    M_corrected = M - np.outer(C, k_vec)

    # 7. Build B (RHS)
    B = np.zeros(3, dtype=float)
    offset_1 = uq * beta[2] - beta[0]
    offset_2 = vq * beta[2] - beta[1]
    offset_3 = speed_radial * delta_t
    B[0] = C[0] * d_static + offset_1
    B[1] = C[1] * d_static + offset_2
    B[2] = C[2] * d_static + offset_3 

    # 8. Solve
    try:
        t_vec = np.linalg.solve(M_corrected, B)
        return t_vec / delta_t # Return Velocity
    except np.linalg.LinAlgError:
        return None

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

def analyze_velocity():
    DATA_DIR = "../carla/output_sim"
    if not os.path.exists(DATA_DIR):
        print("Error: output_sim directory not found.")
        return

    # Load Calibration
    K = np.loadtxt(os.path.join(DATA_DIR, "calib", "intrinsics.txt"))
    cam = MockCamera(K[0,0], K[1,1], K[0,2], K[1,2], 1280, 720)
    T_ext = np.loadtxt(os.path.join(DATA_DIR, "calib", "extrinsics.txt"))
    T_Cam_from_Radar = np.linalg.inv(T_ext)

    plys = sorted(glob.glob(os.path.join(DATA_DIR, "radar_ply", "*.ply")))
    imgs = sorted(glob.glob(os.path.join(DATA_DIR, "camera_rgb", "*.png")))
    poses = sorted(glob.glob(os.path.join(DATA_DIR, "poses", "*.txt")))

    # Storage
    vel_real = []
    vel_multipath = []
    vel_random = []
    
    prev_gray = None
    print(f"Analyzing {len(poses)} frames for Velocity Magnitude...")

    for i in range(len(poses)):
        # Load Data
        curr_gray = cv2.cvtColor(cv2.imread(imgs[i+1]), cv2.COLOR_BGR2GRAY)
        pose_path = poses[i]
        try:
            T_A_to_B = np.loadtxt(pose_path)
            T_B_to_A = np.linalg.inv(T_A_to_B)
        except: continue

        if prev_gray is not None:
            # Calculate Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Load Detections
            detections = []
            with open(plys[i+1], 'r') as f:
                lines = f.readlines()
                start = next(k for k,l in enumerate(lines) if "end_header" in l) + 1
                for l in lines[start:]:
                    v = list(map(float, l.split()))
                    detections.append(RadarDetection(
                        np.array(v[0:3]), v[6], int(v[13]), int(v[15]), np.array(v[7:10]), int(v[14])
                    ))

            # Solve per point
            for det in detections:
                pt_cam_B = (T_Cam_from_Radar @ np.append(det.position_local, 1.0))[:3]
                if pt_cam_B[2] <= 0.1: continue
                
                uq, vq = pt_cam_B[0]/pt_cam_B[2], pt_cam_B[1]/pt_cam_B[2]
                ix, iy = int(round(cam.fx*uq+cam.cx)), int(round(cam.fy*vq+cam.cy))
                
                if 0 <= ix < cam.image_width and 0 <= iy < cam.image_height:
                    dx, dy = flow[iy, ix]
                    prev_x, prev_y = (ix - dx), (iy - dy)
                    up, vp = (prev_x - cam.cx)/cam.fx, (prev_y - cam.cy)/cam.fy
                    
                    v_est = solve_full_velocity_corrected(
                        up, vp, uq, vq, pt_cam_B, 0.05,
                        T_A_to_B, T_B_to_A, T_ext, det.radial_velocity, det.position_local
                    )
                    
                    if v_est is not None:
                        # METRIC: Velocity Magnitude
                        mag = np.linalg.norm(v_est)
                        
                        # Sanity check cap for plotting (so outliers don't ruin the scale)
                        if mag > 200: mag = 200 
                        
                        if det.noise_type == 0: vel_real.append(mag)
                        elif det.noise_type == 1: vel_multipath.append(mag)
                        elif det.noise_type == 2: vel_random.append(mag)

        prev_gray = curr_gray
        if i % 10 == 0: print(f"  Frame {i} done.")

    print("Generating Histogram...")
    
    plt.figure(figsize=(10, 6))
    
    # 0 to 100 m/s range
    bins = np.linspace(0, 100, 50)
    
    plt.hist([vel_real, vel_multipath, vel_random], bins=bins, stacked=True, 
             label=['Real Points', 'Multipath (Ghost)', 'Random Noise'],
             color=['green', 'red', 'black'], 
             alpha=0.7, edgecolor='black')
    
    plt.title("Distribution of Estimated Velocity Magnitude (Log Scale)", fontsize=14)
    plt.xlabel("Estimated Velocity (m/s)", fontsize=12)
    plt.ylabel("Count (Log Scale)", fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plot_velocity_magnitude.png')
    print("Saved plot_velocity_magnitude.png")

if __name__ == "__main__":
    analyze_velocity()