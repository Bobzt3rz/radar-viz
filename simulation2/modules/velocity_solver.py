import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from .entity import Entity
from .cube import Cube
from .camera import Camera
from .radar import Radar
from .types import Vector3, Matrix3x3, Matrix4x4, FlowField, NoiseType, DetectionTuple

def solve_full_velocity(
    up: float, vp: float,
    uq: float, vq: float,
    point_cam_B: Vector3,
    delta_t: float,
    T_A_to_B: Matrix4x4,   # Forward (Prev->Curr) for Matrices
    T_B_to_A: Matrix4x4,   # Backward (Curr->Prev) for Depth
    T_A_to_R: Matrix4x4,
    speed_radial: float,
    point_radar_coords: Vector3
) -> Optional[Vector3]:
    """
    Corrected Solver.
    t is defined in Frame A (Previous).
    """
    # --- 1. Radar Constraint Setup ---
    point_radar_coords = np.asarray(point_radar_coords, dtype=float)
    dist = np.linalg.norm(point_radar_coords)
    if dist < 1e-6: return None
    u_vec_rad = point_radar_coords / dist
    
    # Rotate Radar Unit Vec to Camera Frame
    R_cam_to_rad = T_A_to_R[0:3, 0:3]
    R_rad_to_cam = R_cam_to_rad.T
    r_vec_cam = R_rad_to_cam @ u_vec_rad

    # --- 2. Matrix Coefficients (From Forward Transform A->B) ---
    alpha = T_A_to_B[0:3, 0:3]
    beta  = T_A_to_B[0:3, 3]

    # --- 3. Depth & Gradient Calculation ---
    # We need d(t-1). We assume t is defined in Frame A.
    # P_prev = T_B_to_A @ P_curr - t  (Wait. P_curr = T_AB @ (P_prev + t). Correct.)
    # d = P_prev.z
    # P_virtual_prev = T_B_to_A @ P_curr
    # d = P_virtual_prev.z - t_z
    
    point_cam_B_h = np.append(point_cam_B, 1.0)
    p_virtual_prev = (T_B_to_A @ point_cam_B_h)[:3]
    
    d_static = p_virtual_prev[2]
    k_vec = np.array([0.0, 0.0, -1.0]) # Gradient of d w.r.t t=[tx,ty,tz]

    # --- 4. Build LHS Matrix M ---
    M = np.zeros((3, 3), dtype=float)
    # Camera Rows (using alpha from A->B)
    for j in range(3):
        M[0, j] = alpha[0, j] - uq * alpha[2, j]
        M[1, j] = alpha[1, j] - vq * alpha[2, j]
    # Radar Row
    M[2, :] = r_vec_cam

    # --- 5. Build C Vector (Depth Multiplier) ---
    p_uv_hom_prev = np.array([up, vp, 1.0])
    comm = np.dot(alpha[2, :], p_uv_hom_prev)
    
    C = np.zeros(3)
    C[0] = uq * comm - np.dot(alpha[0, :], p_uv_hom_prev)
    C[1] = vq * comm - np.dot(alpha[1, :], p_uv_hom_prev)
    C[2] = 0.0

    # --- 6. Apply Correction (M - C outer k) ---
    correction = np.outer(C, k_vec)
    M_corrected = M - correction

    # --- 7. Build RHS ---
    B = np.zeros(3, dtype=float)
    offset_1 = uq * beta[2] - beta[0]
    offset_2 = vq * beta[2] - beta[1]
    offset_3 = speed_radial * delta_t
    
    B[0] = C[0] * d_static + offset_1
    B[1] = C[1] * d_static + offset_2
    B[2] = C[2] * d_static + offset_3

    # --- 8. Solve ---
    try:
        t_vec = np.linalg.solve(M_corrected, B)
    except np.linalg.LinAlgError:
        return None

    velocity_cam = t_vec / delta_t
    return R_cam_to_rad @ velocity_cam

# # --- Helper Function for Velocity Estimation ---
# def estimate_velocities_for_frame(
#     radar_detections: List[Tuple[Vector3, float, Optional[Entity], NoiseType]],
#     flow: Optional[FlowField],
#     camera: Camera,
#     radar: Radar,
#     prev_poses: Dict[str, Any],
#     world_delta_t: float
# ) -> List[DetectionTuple]:
#     """
#     Calculates full velocity for radar detections using data from two time steps.
#     Returns a list of tuples:
#     (vel_mag, vel_err, disp_err, NoiseType, pos_3d_radar, vel_3d_radar, vel_3d_world)
#     """
#     frame_results: List[DetectionTuple] = []

#     # Check if we have necessary data
#     if flow is None or not prev_poses or 'camera' not in prev_poses:
#         # print("  Waiting for prev state/flow...") # Optional debug
#         return frame_results # Return empty list if prerequisites not met

#     # 1. Get current & previous poses
#     try:
#         current_cam_pose_W2L = camera.get_pose_world_to_local()
#         prev_cam_pose_W2L = prev_poses['camera']
#         prev_radar_pose_W2L = prev_poses['radar']

#         # Check if matrices are invertible before proceeding
#         inv_prev_cam_pose_W2L = np.linalg.inv(prev_cam_pose_W2L)
#         inv_prev_radar_pose_W2L = np.linalg.inv(prev_radar_pose_W2L)

#     except (KeyError, np.linalg.LinAlgError, TypeError):
#         print("  Error accessing or inverting previous poses.")
#         return frame_results # Return empty if poses are missing/invalid

#     # 2. Calculate relative transformations (given these from ego velocity estimation + calibration)
#     T_A_to_B = current_cam_pose_W2L @ inv_prev_cam_pose_W2L
#     T_A_to_R = prev_radar_pose_W2L @ inv_prev_cam_pose_W2L

#     # This is T_Cam(A)_from_Radar(A), which is static and valid for all time.
#     try:
#         T_Cam_from_Radar_static = np.linalg.inv(T_A_to_R)
#     except np.linalg.LinAlgError:
#         print("  Error inverting T_A_to_R to find static extrinsic.")
#         return frame_results

#     # --- SIMULATION-ONLY ---
#     # This part is ONLY for comparing against world-frame ground truth.
#     # A real system would not have this.
#     R_A_radar_to_world = inv_prev_radar_pose_W2L[0:3, 0:3]
#     # --- END SIMULATION-ONLY ---

#     # 3. Process each detection
#     for detection_idx, detection in enumerate(radar_detections):
#         point_radar_coords, speed_radial, source_entity, noiseType = detection

#         # --- Calculate (xq_pix, yq_pix) and (uq, vq) at t+delta_t ---
#         # Convert radar point (t+delta_t) to world (t+delta_t)
#         point_rad_h = np.append(point_radar_coords, 1.0)
#         # print(f"point_rad_h: {point_rad_h}")

#         # Convert radar (B) directly to camera (B) using the static extrinsic
#         point_cam_B_h = T_Cam_from_Radar_static @ point_rad_h
#         # print(f"point_cam_B_h: {point_cam_B_h}")
#         point_cam_B = point_cam_B_h[:3]
#         depth_B = point_cam_B[2]
#         # print(f"depth_B: {depth_B}")

#         if depth_B <= 1e-3: continue # Point is behind or too close to camera B

#         # Normalized coords (t+delta_t)
#         uq = point_cam_B[0] / depth_B
#         vq = point_cam_B[1] / depth_B
#         # print(f"uq: {uq}, vq: {vq}")

#         # Pixel coords (t+delta_t) - careful with int conversion if needed early
#         xq_pix_f = camera.fx * uq + camera.cx
#         yq_pix_f = camera.fy * vq + camera.cy
#         xq_pix = int(round(xq_pix_f))
#         yq_pix = int(round(yq_pix_f))
#         # print(f"xq_pix: {xq_pix}, yq_pix: {yq_pix}")

#         # Check image bounds
#         if not (0 <= xq_pix < camera.image_width and 0 <= yq_pix < camera.image_height):
#             continue

#         # --- Get flow and calculate (up, vp) at t ---
#         dx, dy = flow[yq_pix, xq_pix] # Pixel flow
#         # print(f"dx: {dx}, dy: {dy}")
#         xp_pix_f = xq_pix_f - dx
#         yp_pix_f = yq_pix_f - dy
#         # print(f"xp_pix_f: {xp_pix_f}, yp_pix_f: {yp_pix_f}")

#         # Normalized coords (t)
#         up = (xp_pix_f - camera.cx) / camera.fx
#         vp = (yp_pix_f - camera.cy) / camera.fy
#         # print(f"up: {up}, vp: {vp}")

#         # --- Call the solver ---
#         full_vel_vector_radar = solve_full_velocity(
#             up=up, vp=vp, uq=uq, vq=vq, d=depth_B, delta_t=world_delta_t,
#             T_A_to_B=T_A_to_B, T_A_to_R=T_A_to_R,
#             speed_radial=speed_radial, point_radar_coords=point_radar_coords,
#             return_in_radar_coords=True
#         )

        
#         if full_vel_vector_radar is not None:
#             full_vel_magnitude = float(np.linalg.norm(full_vel_vector_radar))
#             full_vel_world = R_A_radar_to_world @ full_vel_vector_radar
            
#             frame_displacement_error = calculate_reprojection_error(
#                 full_vel_radar_A=full_vel_vector_radar,
#                 point_radar_B=point_radar_coords,
#                 T_Cam_from_Radar=T_Cam_from_Radar_static,
#                 T_CamB_from_CamA=T_A_to_B,
#                 flow=flow, 
#                 camera=camera, 
#                 xq_pix_f=xq_pix_f, 
#                 yq_pix_f=yq_pix_f, 
#                 delta_t=world_delta_t
#             )
             
#             if frame_displacement_error is not None and source_entity is not None:
#                 ground_truth_vel = source_entity.velocity
#                 velocity_error_magnitude = float(np.linalg.norm(full_vel_world - ground_truth_vel))
#                 frame_results.append((full_vel_magnitude, 
#                                       velocity_error_magnitude, frame_displacement_error, 
#                                       noiseType, point_radar_coords, full_vel_vector_radar, full_vel_world))
#                 continue
            
#             if(frame_displacement_error is not None and noiseType is not NoiseType.REAL):
#                 frame_results.append((full_vel_magnitude, 
#                                       0.0, frame_displacement_error, 
#                                       noiseType, point_radar_coords, full_vel_vector_radar, full_vel_world))
#     return frame_results

def calculate_reprojection_error(
    full_vel_radar: Vector3,
    point_radar_B: Vector3,
    T_Cam_from_Radar: Matrix4x4,
    T_B_to_A: Matrix4x4, # Explicit back-transform
    flow: FlowField,
    camera: Camera,
    xq_pix_f: float, yq_pix_f: float,
    delta_t: float
) -> Optional[float]:
    
    # 1. Get Velocity in Camera Frame
    # Note: Solver returns velocity in RADAR frame
    # We need velocity in CAMERA frame
    R_Cam_from_Radar = T_Cam_from_Radar[0:3, 0:3]
    vel_cam = R_Cam_from_Radar @ full_vel_radar

    # 2. Get Point B in Camera Frame
    point_radar_B_h = np.append(point_radar_B, 1.0)
    point_cam_B = (T_Cam_from_Radar @ point_radar_B_h)[:3]

    # 3. Predict Point A
    # The solver assumed t (motion) is in Frame A.
    # P_curr = T_AB @ (P_prev + t)
    # P_prev = T_BA @ P_curr - t
    # P_prev = T_BA @ P_curr - (vel_cam * dt)
    # CAUTION: Is vel_cam defined in Frame A or B?
    # Eq 13: R * t = ... t is motion in Frame A (as defined by P+M->Q)
    # So vel_cam is in Frame A.
    
    point_cam_B_h = np.append(point_cam_B, 1.0)
    p_virtual_prev = (T_B_to_A @ point_cam_B_h)[:3]
    
    point_cam_A_pred = p_virtual_prev - (vel_cam * delta_t)

    # 4. Project
    if point_cam_A_pred[2] <= 1e-3: return None
    
    u = point_cam_A_pred[0] / point_cam_A_pred[2]
    v = point_cam_A_pred[1] / point_cam_A_pred[2]
    
    pred_x = camera.fx * u + camera.cx
    pred_y = camera.fy * v + camera.cy
    
    # 5. Compare with Flow (Forward: prev -> curr)
    # The flow vector at (prev) points to (curr).
    # We are at (curr). We have dx, dy which maps prev -> curr.
    # So x_prev = x_curr - dx
    dx, dy = flow[int(round(yq_pix_f)), int(round(xq_pix_f))]
    actual_x = xq_pix_f - dx
    actual_y = yq_pix_f - dy
    
    return math.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)

def calculate_corrected_angle_error(
    full_vel_radar_A: Vector3,            
    point_radar_A: Vector3, # Note: Input is Point at A (we predict B)
    T_Cam_from_Radar: Matrix4x4,
    T_CamB_from_CamA: Matrix4x4, # EGO MOTION
    flow_dx: float,
    flow_dy: float,
    camera: Camera,
    delta_t: float
) -> float:
    """
    Calculates Angle Error accounting for both Object Velocity AND Ego-Motion.
    """
    # 1. Transform Point and Velocity to Camera A Frame
    point_radar_A_h = np.append(point_radar_A, 1.0)
    point_cam_A_h = T_Cam_from_Radar @ point_radar_A_h
    point_cam_A = point_cam_A_h[:3]
    
    R_Cam_from_Radar = T_Cam_from_Radar[0:3, 0:3]
    vel_cam_A = R_Cam_from_Radar @ full_vel_radar_A
    
    # 2. Predict Point at Time B (in Camera A Frame)
    # Object Motion: P_new = P_old + V_obj * dt
    point_cam_A_at_tB = point_cam_A + (vel_cam_A * delta_t)
    
    # 3. Transform to Camera B Frame (Apply Ego Motion)
    # P_camB = T_B_from_A @ P_camA
    point_cam_A_at_tB_h = np.append(point_cam_A_at_tB, 1.0)
    point_cam_B_h = T_CamB_from_CamA @ point_cam_A_at_tB_h
    point_cam_B = point_cam_B_h[:3]
    
    # 4. Project Start (A) and End (B) to pixels
    # Start Pixel (Re-projecting A ensures we use the exact geometric center)
    # Alternatively, you can use the 'pixel_u' passed in if it corresponds to A.
    # But assuming flow is B->A or A->B, we need the vector.
    
    # Let's verify flow direction: usually Flow is A -> B.
    # Current Pixel (at B) - Previous Pixel (at A) = (dx, dy)
    
    def project(p):
        if p[2] < 0.1: return np.array([0,0])
        u = p[0] / p[2]
        v = p[1] / p[2]
        return np.array([
            camera.fx * u + camera.cx,
            camera.fy * v + camera.cy
        ])

    # Project Prediction
    pix_A_pred = project(point_cam_A) # Where it was
    pix_B_pred = project(point_cam_B) # Where it is now (with Ego + Obj motion)
    
    expected_flow_vector = pix_B_pred - pix_A_pred
    
    # 5. Observed Flow
    # NOTE: Verify if your flow is A->B (forward) or B->A (backward).
    # If flow is "Current(B) - Previous(A)", then use as is.
    observed_flow_vector = np.array([flow_dx, flow_dy])
    
    # 6. Calculate Angle
    norm_exp = np.linalg.norm(expected_flow_vector)
    norm_obs = np.linalg.norm(observed_flow_vector)
    
    # if norm_exp < 0.5 or norm_obs < 0.5:
    #     return 0.0
        
    dot = np.dot(expected_flow_vector, observed_flow_vector)
    cosine = np.clip(dot / (norm_exp * norm_obs), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    
    # 2. Weight by Observation Magnitude
    # "How much visual evidence do we have for this angle?"
    # We use observed flow because that's the "Ground Truth" of image motion.
    weighted_error = angle * (norm_obs * np.linalg.norm(vel_cam_A))
    
    return weighted_error

def calculate_rigid_3d_velocities(clusters: List[List[DetectionTuple]]) -> List[List[DetectionTuple]]:
    refined_clusters = []

    for cluster in clusters:
        points: List[DetectionTuple] = []
        for det in cluster:
            pos_point = det[4]        # p: Position of the point
            vel_point = det[5]        # v_point: Measured 3D velocity of the point
            omega_gt = det[11]        # omega: GT Angular velocity
            center_gt = det[12]       # c: GT Center of object
            
            # 1. Calculate the lever arm (vector from Center to Point)
            # r = p - c
            r_vec = pos_point - center_gt
            
            # 2. Calculate the tangential velocity component caused by rotation
            # v_tan = omega x r
            v_tangential = np.cross(omega_gt, r_vec)
            
            # 3. Calculate the translational velocity of the object (The Vote)
            # v_obj = v_point - v_tan
            v_obj_vote = vel_point - v_tangential

            # shouldn't really mutate it like this, but leave it like this for now
            det[14][:] = v_obj_vote
            
            points.append(det)
        
        refined_clusters.append(points)

    return refined_clusters