import numpy as np

# ======================================================================
# 1. PARAMETER CLASS: CENTRALIZED CONFIGURATION
# ======================================================================

class SimulationParams:
    """Stores all constants and ground truth values for the simulation."""
    def __init__(self, dt, obj_vel, ego_vel, p_a, cam_to_radar_t, cam_to_radar_r=np.identity(3)):
        # Time
        self.DELTA_T = dt 

        # Motion (Absolute, Camera Frame)
        self.OBJ_VEL_GT = np.array(obj_vel)
        self.EGO_VEL = np.array(ego_vel)

        # Initial State (Camera Frame)
        self.P_A = np.array(p_a)
        self.d_A = self.P_A[2] # Initial depth

        # Extrinsic Calibration (Camera A -> Radar R)
        self.T_RA = np.identity(4)
        self.T_RA[:3, :3] = cam_to_radar_r
        self.T_RA[:3, 3] = np.array(cam_to_radar_t)
        self.R_RA = self.T_RA[:3, :3]
        
        # Derived Ego Motion (T_AB: Camera A -> Camera B)
        self.EGO_DISPLACEMENT = self.EGO_VEL * self.DELTA_T
        self.T_AB = np.identity(4)
        self.T_AB[:3, 3] = self.EGO_DISPLACEMENT

        # Derived Object State (P_B / Q)
        self.t_gt_camera_absolute = self.OBJ_VEL_GT * self.DELTA_T
        self.P_B_camera = self.P_A[:3] + self.t_gt_camera_absolute
        
# ======================================================================
# 2. CORE ESTIMATION FUNCTION (remains the same)
# ======================================================================

def estimate_full_displacement(
    dt, T_AB, T_RA, up, vp, d, uq, vq, vx_r, vy_r, vz_r, radial_vel_magnitude
):
    """
    Solves for the object's RELATIVE displacement (t_relative) in the Camera A frame.
    """
    alpha = T_AB[:3, :3]; beta = T_AB[:3, 3]
    a11, a12, a13 = alpha[0, :]; a21, a22, a23 = alpha[1, :]; a31, a32, a33 = alpha[2, :]
    bx, by, bz = beta
    R = T_RA[:3, :3]
    r11, r12, r13 = R[0, :]; r21, r22, r23 = R[1, :]; r31, r32, r33 = R[2, :]

    # LHS Matrix (M)
    M11, M12, M13 = a11 - uq * a31, a12 - uq * a32, a13 - uq * a33
    M21, M22, M23 = a21 - vq * a31, a22 - vq * a32, a23 - vq * a33
    M31 = r11 * vx_r + r21 * vy_r + r31 * vz_r
    M32 = r12 * vx_r + r22 * vy_r + r32 * vz_r
    M33 = r13 * vx_r + r23 * vy_r + r33 * vz_r
    M = np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])

    # RHS Vector (B)
    B1 = ((a31 * up + a32 * vp + a33) * uq - (a11 * up + a12 * vp + a13)) * d + uq * bz - bx
    B2 = ((a31 * up + a32 * vp + a33) * vq - (a21 * up + a22 * vp + a23)) * d + vq * bz - by
    B3 = radial_vel_magnitude * dt
    B = np.array([B1, B2, B3])

    try:
        t = np.linalg.solve(M, B)
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular. Cannot solve for displacement.")
        return np.zeros(3)

    return t

# ======================================================================
# 3. SCENE EXECUTION: MAIN FUNCTION
# ======================================================================

def simulate_scene(params: SimulationParams):
    """Generates inputs, executes estimation, and validates results."""
    
    # ----------------------------------------------------
    # A. SIMULATE SENSOR INPUTS
    # ----------------------------------------------------

    # 1. Optical Flow Inputs: Normalized Image Coordinates
    up, vp = params.P_A[0] / params.P_A[2], params.P_A[1] / params.P_A[2]
    uq, vq = params.P_B_camera[0] / params.P_B_camera[2], params.P_B_camera[1] / params.P_B_camera[2]

    # 2. Unit Radial Vector (Line of sight to Q in Radar Frame)
    Q_R = params.T_RA @ np.append(params.P_B_camera, 1.0)
    Q_R_3d = Q_R[:3]
    unit_radial_vector = Q_R_3d / np.linalg.norm(Q_R_3d)
    vx_r, vy_r, vz_r = unit_radial_vector

    # 3. Ground Truth Radial Velocity Magnitude (V_r_measured)
    # V_r_measured is V_relative projected onto the line of sight (Doppler measurement).
    V_relative_cam = params.OBJ_VEL_GT - params.EGO_VEL
    V_relative_radar = params.R_RA @ V_relative_cam
    radial_vel_magnitude_GT = np.dot(V_relative_radar, unit_radial_vector)
    
    # ----------------------------------------------------
    # B. EXECUTE ESTIMATION
    # ----------------------------------------------------
    
    # Solve for t_relative = t_object - t_ego
    t_est_relative = estimate_full_displacement(
        params.DELTA_T, params.T_AB, params.T_RA, up, vp, params.d_A, uq, vq, 
        vx_r, vy_r, vz_r, radial_vel_magnitude_GT
    )

    # Calculate final estimated velocities
    V_Relative_EST = t_est_relative / params.DELTA_T
    # Final Correction: Calculate the Absolute Velocity by adding Ego-Velocity back.
    V_Absolute_EST = V_Relative_EST + params.EGO_VEL

    # Calculate Ground Truth Relative Displacement for comparison
    t_gt_camera_relative = params.t_gt_camera_absolute - params.EGO_DISPLACEMENT
    
    # ----------------------------------------------------
    # C. VERIFICATION AND OUTPUT
    # ----------------------------------------------------
    
    print("--- Simulation Inputs ---")
    print(f"Time Step (dt): {params.DELTA_T:.3f} s")
    print(f"GT Object Velocity (Absolute, Cam Frame): {params.OBJ_VEL_GT} m/s")
    print(f"Ego Velocity (Cam Frame): {params.EGO_VEL} m/s")
    print(f"Initial Point P_A: {params.P_A[:3]} m, Depth d: {params.d_A} m")
    print(f"Radar-to-Camera Translation: {params.T_RA[:3, 3]} m")
    print(f"GT Radial Velocity Magnitude (v_r): {radial_vel_magnitude_GT:.8f} m/s")

    print("\n--- Estimated Results (Camera A Coordinate Frame) ---")
    print(f"GT Relative Displacement (t_relative): {t_gt_camera_relative}")
    print(f"Estimated Relative Displacement: {t_est_relative.round(8)}")
    print("-" * 50)
    print(f"GT Absolute Velocity: {params.OBJ_VEL_GT}")
    print(f"Estimated Absolute Velocity (Final Result): {V_Absolute_EST.round(12)}")

    error = np.linalg.norm(V_Absolute_EST - params.OBJ_VEL_GT)
    print(f"\nEuclidean Velocity Error: {error:.12f} m/s")
    
    return V_Absolute_EST, error

# ======================================================================
# 4. EXAMPLE USAGE: RUNNING MULTIPLE SCENARIOS
# ======================================================================

if __name__ == '__main__':
    print("--- SCENARIO 1: ORIGINAL VALIDATION CASE ---")
    
    # Define an arbitrary rotation (Pitch down 5 degrees, as discussed earlier)
    theta = np.deg2rad(5)
    R_PITCH = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    # Case 1: Object moving right and away, ego moving forward, NO RADAR ROTATION (Identity)
    params1 = SimulationParams(
        dt=0.033, 
        obj_vel=[1.5, 0.0, 0.5], 
        ego_vel=[0.0, 0.0, 10.0],
        p_a=[3.0, 0.5, 10.0, 1.0], 
        cam_to_radar_t=[0.0, -0.2, 1.0],
        cam_to_radar_r=np.identity(3) # Identity Matrix (No rotation)
    )
    simulate_scene(params1)

    print("\n" + "="*50 + "\n")
    print("--- SCENARIO 2: ARBITRARY RADAR ROTATION (Pitch Down 5 deg) ---")

    # Case 2: Same motion, but with arbitrary radar pitch rotation
    params2 = SimulationParams(
        dt=0.033, 
        obj_vel=[1.5, 0.0, 0.5], 
        ego_vel=[0.0, 0.0, 10.0],
        p_a=[3.0, 0.5, 10.0, 1.0], 
        cam_to_radar_t=[0.0, -0.2, 1.0],
        cam_to_radar_r=R_PITCH # Arbitrary Rotation
    )
    simulate_scene(params2)
    
    print("\n" + "="*50 + "\n")
    print("--- SCENARIO 3: DIFFERENT MOTION (Object moving left/closer) ---")
    
    # Case 3: Object moving Left/Closer (X=-2, Z=-5), Ego moving faster
    params3 = SimulationParams(
        dt=0.033, 
        obj_vel=[-2.0, 0.0, -5.0], 
        ego_vel=[0.0, 0.0, 20.0],
        p_a=[1.0, 0.0, 5.0, 1.0], 
        cam_to_radar_t=[0.0, 0.0, 0.5],
        cam_to_radar_r=np.identity(3) 
    )
    simulate_scene(params3)