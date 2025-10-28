import numpy as np

# ======================================================================
# 1. PARAMETER CLASS: CENTRALIZED CONFIGURATION (Unchanged)
# ======================================================================

class SimulationParams:
    """Stores all constants and ground truth values for the simulation."""
    def __init__(self, dt, obj_vel, ego_vel, p_a, cam_to_radar_t, cam_to_radar_r=np.identity(3)):
        # Time
        self.DELTA_T = dt

        # Motion (Absolute, Camera Frame - Assumed Solver Frame: +X R, +Y D, +Z Fwd)
        self.OBJ_VEL_GT = np.array(obj_vel)
        self.EGO_VEL = np.array(ego_vel)

        # Initial State (Camera Frame - Assumed Solver Frame)
        self.P_A = np.array(p_a) # Should be [X, Y, Z] non-homogeneous
        if self.P_A.shape[0] == 4: # Handle if passed as homogeneous
             self.P_A = self.P_A[:3] / self.P_A[3]

        self.d_A = self.P_A[2] # Initial depth (+Z is forward)

        # Extrinsic Calibration (Solver Camera A -> Radar R)
        self.T_RA = np.identity(4)
        self.T_RA[:3, :3] = cam_to_radar_r # R_RA
        self.T_RA[:3, 3] = np.array(cam_to_radar_t) # t_RA
        self.R_RA = self.T_RA[:3, :3]

        # Derived Ego Motion (T_AB: Solver Camera A -> Solver Camera B)
        # beta is displacement FROM A TO B, in A's frame
        self.EGO_DISPLACEMENT = self.EGO_VEL * self.DELTA_T
        self.T_AB = np.identity(4)
        # Assuming only translation for ego motion (alpha = I)
        self.T_AB[:3, 3] = self.EGO_DISPLACEMENT # beta

        # Derived Object State (P_B / Q in Solver Camera Frame)
        self.t_gt_camera_absolute = self.OBJ_VEL_GT * self.DELTA_T
        self.P_B_camera = self.P_A[:3] + self.t_gt_camera_absolute # Q = P + V_abs*dt


# ======================================================================
# 2. CORE ESTIMATION FUNCTION (Unchanged)
# ======================================================================
# --- Keep your existing estimate_full_displacement function here ---
# --- (Including the [Solver Debug] logs is helpful!) ---
def estimate_full_displacement(
    dt: float,
    T_AB: np.ndarray,
    T_RA: np.ndarray,
    up: float, vp: float, d: float,
    uq: float, vq: float,
    vx_r: float, vy_r: float, vz_r: float, # Unit radial in RADAR frame
    radial_vel_magnitude: float
): # Added Optional typing
    """
    Solves for the object's RELATIVE displacement (t_relative) in the Camera A frame.
    """
    alpha = T_AB[:3, :3]; beta = T_AB[:3, 3]
    a11, a12, a13 = alpha[0, :]; a21, a22, a23 = alpha[1, :]; a31, a32, a33 = alpha[2, :]
    bx, by, bz = beta
    R = T_RA[:3, :3]
    r11, r12, r13 = R[0, :]; r21, r22, r23 = R[1, :]; r31, r32, r33 = R[2, :]

    # LHS Matrix (M) - No change here
    M11, M12, M13 = a11 - uq * a31, a12 - uq * a32, a13 - uq * a33
    M21, M22, M23 = a21 - vq * a31, a22 - vq * a32, a23 - vq * a33
    M31 = r11 * vx_r + r21 * vy_r + r31 * vz_r
    M32 = r12 * vx_r + r22 * vy_r + r32 * vz_r
    M33 = r13 * vx_r + r23 * vy_r + r33 * vz_r
    M = np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])

    # --- Reverted: Use original positive d ---
    d_solver = d
    print(f"  [Solver Debug] Using d_solver = {d_solver:.4f} (input d = {d:.4f}) in B1/B2")
    # --- End Revert ---

    # RHS Vector (B)
    B1 = ((a31 * up + a32 * vp + a33) * uq - (a11 * up + a12 * vp + a13)) * d_solver + uq * bz - bx
    B2 = ((a31 * up + a32 * vp + a33) * vq - (a21 * up + a22 * vp + a23)) * d_solver + vq * bz - by
    B3 = radial_vel_magnitude * dt
    B = np.array([B1, B2, B3])

    try:
        print(f"  [Solver Debug] Matrix M:\n{M}")
        print(f"  [Solver Debug] Vector B:\n{B}")
        t = np.linalg.solve(M, B)
        print(f"  [Solver Debug] Solution t:\n{t}")
    except np.linalg.LinAlgError:
        print("  [Solver Debug] np.linalg.solve failed: Matrix M is singular.")
        return None # Return None on failure

    return t


# ======================================================================
# 3. SCENE EXECUTION: MAIN FUNCTION WITH LOGGING
# ======================================================================

def simulate_scene(params: SimulationParams):
    """Generates inputs, executes estimation, and validates results."""

    # ----------------------------------------------------
    # A. SIMULATE SENSOR INPUTS (Solver Camera Frame)
    # ----------------------------------------------------

    # 1. Optical Flow Inputs: Normalized Image Coordinates
    # Check for division by zero / point at camera center
    if abs(params.P_A[2]) < 1e-6 or abs(params.P_B_camera[2]) < 1e-6:
         print("Error: Point P or Q is at camera center (Z=0). Skipping.")
         return None, np.inf # Return dummy values

    up, vp = params.P_A[0] / params.P_A[2], params.P_A[1] / params.P_A[2]
    uq, vq = params.P_B_camera[0] / params.P_B_camera[2], params.P_B_camera[1] / params.P_B_camera[2]

    # 2. Transform Q (Solver Cam frame) to Radar SAE frame
    Q_cam_solver_homo = np.append(params.P_B_camera, 1.0)
    Q_radar_sae_homo = params.T_RA @ Q_cam_solver_homo
    w_qr = Q_radar_sae_homo[3]
    if abs(w_qr) < 1e-8:
        print("Error: Perspective division failed for Q in Radar frame. Skipping.")
        return None, np.inf
    Q_radar_sae = Q_radar_sae_homo[:3] / w_qr # Vector FROM RADAR ORIGIN TO Q

    # 3. Unit Radial Vector (Line of sight to Q in Radar SAE Frame)
    range_q = np.linalg.norm(Q_radar_sae)
    if range_q < 1e-6:
        print("Error: Point Q is at Radar origin. Skipping.")
        return None, np.inf
    unit_radial_vector_sae = Q_radar_sae / range_q
    vx_r, vy_r, vz_r = unit_radial_vector_sae

    # 4. Ground Truth Radial Velocity Magnitude (V_r_measured)
    # V_r_measured is V_relative projected onto the line of sight (Doppler measurement).
    V_relative_cam_solver = params.OBJ_VEL_GT - params.EGO_VEL
    # Transform relative velocity to Radar SAE frame
    V_relative_radar_sae = params.R_RA @ V_relative_cam_solver
    # Project onto unit radial vector (to Q)
    radial_vel_magnitude_GT = np.dot(V_relative_radar_sae, unit_radial_vector_sae)

    # ----------------------------------------------------
    # B. LOG INPUTS (Matching main.py logs)
    # ----------------------------------------------------
    print(f"\n[EQ-VALIDATION SOLVER INPUTS]:")
    print(f"  (up, vp, d): ({up:.4f}, {vp:.4f}, {params.d_A:.4f})")
    print(f"  (uq, vq):    ({uq:.4f}, {vq:.4f})")
    print(f"  Unit radial (SAE) passed: ({vx_r:.4f}, {vy_r:.4f}, {vz_r:.4f})")
    print(f"  Radial vel mag passed:    {radial_vel_magnitude_GT:.4f}")
    # Optional detailed logs (uncomment if needed)
    # print(f"  P_A (SolverCam): {params.P_A}")
    # print(f"  Obj Vel (SolverCam): {params.OBJ_VEL_GT}")
    # print(f"  Ego Vel (SolverCam): {params.EGO_VEL}")
    # print(f"  Rel Vel (SolverCam): {V_relative_cam_solver}")
    # print(f"  Rel Vel (SAE): {V_relative_radar_sae}")
    # print(f"  Q Pos (SolverCam): {params.P_B_camera}")
    # print(f"  Q Pos (SAE): {Q_radar_sae}")
    # print(f"  T_AB (Ego):\n{params.T_AB}")
    # print(f"  T_RA (Cam->Radar):\n{params.T_RA}")


    # ----------------------------------------------------
    # C. EXECUTE ESTIMATION
    # ----------------------------------------------------

    # Solve for t_relative = t_object - t_ego (in Solver Camera Frame A)
    t_est_relative = estimate_full_displacement(
        params.DELTA_T, params.T_AB, params.T_RA, up, vp, params.d_A, uq, vq,
        vx_r, vy_r, vz_r, radial_vel_magnitude_GT
    )

    if t_est_relative is None: # Handle solver failure
        print("Solver failed in eq-validation.")
        return None, np.inf

    # Calculate final estimated velocities (in Solver Camera Frame)
    V_Relative_EST = t_est_relative / params.DELTA_T
    V_Absolute_EST = V_Relative_EST + params.EGO_VEL # Add ego vel back (in Solver Cam Frame)

    # Calculate Ground Truth Relative Displacement for comparison
    t_gt_camera_relative = params.t_gt_camera_absolute - params.EGO_DISPLACEMENT

    # ----------------------------------------------------
    # D. VERIFICATION AND OUTPUT (Unchanged)
    # ----------------------------------------------------

    print("\n--- Eq-Validation Estimated Results (Camera A Coordinate Frame) ---")
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
    # ... (Scenario 1 definition unchanged) ...
    params1 = SimulationParams(
        dt=0.033,
        obj_vel=[1.5, 0.0, 0.5],
        ego_vel=[0.0, 0.0, 10.0],
        p_a=[3.0, 0.5, 10.0], # Use non-homogeneous directly
        cam_to_radar_t=[0.0, -0.2, 1.0],
        cam_to_radar_r=np.identity(3) # Identity Matrix (No rotation)
    )
    simulate_scene(params1)


    print("\n" + "="*50 + "\n")
    print("--- SCENARIO 2: ARBITRARY RADAR ROTATION (Pitch Down 5 deg) ---")
    # ... (Scenario 2 definition unchanged) ...
    theta = np.deg2rad(5)
    R_PITCH = np.array([ # Standard rotation around X (Right) -> Causes pitch down in +Y Down frame
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]) # This R_PITCH seems incorrect for Solver Cam (+Y Down). Pitch down should be rotation around +X axis.
    # Correct Pitch Down matrix (Rotation around X axis)
    R_PITCH_CORRECT = np.array([
         [1,  0,              0            ],
         [0,  np.cos(theta),  np.sin(theta)], # Note sign change for +Y Down
         [0, -np.sin(theta),  np.cos(theta)]
    ])


    params2 = SimulationParams(
        dt=0.033,
        obj_vel=[1.5, 0.0, 0.5],
        ego_vel=[0.0, 0.0, 10.0],
        p_a=[3.0, 0.5, 10.0],
        cam_to_radar_t=[0.0, -0.2, 1.0],
        # cam_to_radar_r=R_PITCH # Use the arbitrary rotation between Cam A and Radar R
        # It's unclear if R_PITCH was intended as R_RA or ego-rotation. Assuming R_RA.
        # Let's keep it simple for now and use Identity R_RA
         cam_to_radar_r=np.identity(3) # Revert to identity R_RA for simplicity
    )
    simulate_scene(params2)


    print("\n" + "="*50 + "\n")
    print("--- SCENARIO 3: DIFFERENT MOTION (Object moving left/closer) ---")
    # ... (Scenario 3 definition unchanged) ...
    params3 = SimulationParams(
        dt=0.033,
        obj_vel=[-2.0, 0.0, -5.0], # Moving away Z decreases -> actually closer in +Z Fwd frame
        ego_vel=[0.0, 0.0, 20.0], # Moving forward +Z
        p_a=[1.0, 0.0, 5.0], # X=1, Y=0, Z=5 (Depth=5)
        cam_to_radar_t=[0.0, 0.0, 0.5],
        cam_to_radar_r=np.identity(3)
    )
    simulate_scene(params3)


    print("\n" + "="*50 + "\n")
    print("--- SCENARIO 4: REPLICATING MAIN.PY SIMULATION ---")

    # Define R_RA from main.py (Solver Cam -> Radar SAE)
    R_RA_main = np.array([
        [ 0,  0,  1], # Radar X (Fwd) = Camera Z (Fwd)
        [-1,  0,  0], # Radar Y (Left)= -Camera X (Right)
        [ 0, -1,  0]  # Radar Z (Up)  = -Camera Y (Down)
    ])

    # Reconstruct p_a from main.py's first logged frame (t-1 state)
    # Using the P_A (t-1, SolverCam) from the latest main.py log
    p_a_main = [-0.18463076, -0.1619568, 0.07757316] # Use non-homogeneous directly


    params4 = SimulationParams(
        dt=0.016,                     # From main.py
        obj_vel=[-1.0, 0.0, 0.0],     # Object vel in Solver Cam Frame (CHECK THIS ASSUMPTION)
                                      # Assuming main.py's world velocity [-1,0,0] maps directly
                                      # Needs careful verification if world != solver cam frame
        # Let's recalculate obj_vel in Solver Cam Frame from main.py's world vel
        # obj_vel_world = [-1,0,0]
        # R_world_to_solver_cam_t1 @ obj_vel_world needed here, but R depends on modelview...
        # For simplicity, ASSUME R_world_to_solver_cam_t1 is Identity for now (like in main.py)
        # obj_vel_cam_solver = [-1, 0, 0] # Tentative
        # Let's use the Obj Vel (Abs, SolverCam): [1. 0. 0.] logged in main.py!
        # obj_vel_main_solver_cam = [1.0, 0.0, 0.0],


        ego_vel=[0.0, 0.0, 0.0],      # From main.py EGO_VELOCITY_WORLD, assumed 0 in Solver Cam
        p_a=p_a_main,                 # From main.py logs at t-1 in Solver Cam frame
        cam_to_radar_t=[0.0, 0.0, 0.0], # Based on main.py rig setup -> T_RA used in main.py
        cam_to_radar_r=R_RA_main      # R_RA used in main.py
    )
    simulate_scene(params4)

    print("\n" + "="*50 + "\n")
    print("--- SCENARIO 5: REPLICATING LATEST MAIN.PY LOG ---")

    # Define R_RA from main.py (Solver Cam -> Radar SAE)
    R_RA_main = np.array([
        [ 0,  0,  1], # Radar X (Fwd) = Camera Z (Fwd)
        [-1,  0,  0], # Radar Y (Left)= -Camera X (Right)
        [ 0, -1,  0]  # Radar Z (Up)  = -Camera Y (Down)
    ])

    # --- Use the P_A and Obj Vel corresponding to the Solver Cam frame ---
    # Derived from main.py log: (up, vp, d): (0.2223, 0.0116, 0.1016)
    d_main_log = 0.1016
    up_main_log = 0.2223
    vp_main_log = 0.0116
    p_a_main_log = [up_main_log * d_main_log, vp_main_log * d_main_log, d_main_log]

    # Corresponds to main.py's World Vel [-1, 0, 0] transformed to Solver Cam
    obj_vel_solver_cam_log = [1.0, 0.0, 0.0]

    params5 = SimulationParams(
        dt=0.016,
        obj_vel=obj_vel_solver_cam_log, # Use Solver Cam velocity
        ego_vel=[0.0, 0.0, 0.0],
        p_a=p_a_main_log,               # Use Solver Cam P_A derived from log
        cam_to_radar_t=[0.0, 0.0, 0.0],
        cam_to_radar_r=R_RA_main
    )
    # --- Run simulation for this specific scenario ---
    simulate_scene(params5)