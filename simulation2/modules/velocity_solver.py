import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from .entity import Entity
from .cube import Cube
from .camera import Camera
from .radar import Radar
from .types import Vector3, Matrix3x3, Matrix4x4, FlowField

def solve_full_velocity(
    up: float, vp: float,                      # Normalized image coords at time t
    uq: float, vq: float,                      # Normalized image coords at time t+delta_t
    d: float,                                  # Depth of point P in camera coords at time t
    delta_t: float,                            # Time interval
    T_A_to_B: Matrix4x4,                       # Camera motion matrix (Pose B relative to A)
    T_A_to_R: Matrix4x4,                       # Extrinsics: Camera A to Radar R
    # --- New Radar Inputs ---
    speed_radial: float,                       # Scalar radial speed (Doppler measurement)
    point_radar_coords: Vector3,               # 3D position [x,y,z] of the point in RADAR coordinates
    # --- Output Option ---
    return_in_radar_coords: bool = True
) -> Optional[Vector3]:
    """
    Solves for the 3D full velocity vector based on Eq 16a from the POW4R paper,
    using scalar radial speed and point position as radar inputs.

    Parameters:
    ----------
    up, vp : float
        Normalized image plane coordinates of the point at time t (point 'p').
    uq, vq : float
        Normalized image plane coordinates of the point at time t+delta_t (point 'q').
    d : float
        Depth of the point P (in camera coordinates) at time t.
    delta_t : float
        Time interval between frames (in seconds).
    T_A_to_B : Matrix4x4
        4x4 transformation matrix for camera motion, from time t (A) to t+delta_t (B).
    T_A_to_R : Matrix4x4
        4x4 extrinsic transformation matrix from Camera (A) to Radar (R).
    speed_radial : float
        The scalar speed measured by the radar along the line of sight (Doppler speed).
        Positive values typically mean moving away from the radar.
    point_radar_coords : Vector3
        The 3D coordinates [x, y, z] of the detected point in the RADAR's coordinate system.
    return_in_radar_coords : bool, optional
        If True (default), returns the final velocity in radar coordinates (v_f).
        If False, returns the velocity in camera coordinates.

    Returns:
    -------
    Optional[Vector3]
        The 3D full-velocity vector (3-element array), or None if the
        calculation fails (e.g., singular matrix, zero distance to point).
    """

    # --- 1. Calculate the 3D Radial Velocity Vector (v_r) ---
    # Ensure point coords is a numpy array
    point_radar_coords = np.asarray(point_radar_coords, dtype=np.floating)

    # Calculate distance to the point
    distance = np.linalg.norm(point_radar_coords)
    if distance < 1e-6: # Avoid division by zero
        print("Warning: Radar point is too close to the origin, cannot determine direction.")
        return None

    # Calculate the unit direction vector from radar origin to the point
    unit_direction_vector: Vector3 = point_radar_coords / distance

    # Calculate the 3D radial velocity vector in RADAR coordinates
    v_r: Vector3 = speed_radial * unit_direction_vector

    # --- 2. Extract matrix components ---
    a11, a12, a13, bx = T_A_to_B[0, :]
    a21, a22, a23, by = T_A_to_B[1, :]
    a31, a32, a33, bz = T_A_to_B[2, :]
    R_cam_to_radar: Matrix3x3 = T_A_to_R[0:3, 0:3]

    # --- 3. Build the 3x3 matrix M (Eq 16a) ---
    M: Matrix3x3 = np.zeros((3, 3), dtype=np.floating)
    M[0, 0] = a11 - uq * a31
    M[0, 1] = a12 - uq * a32
    M[0, 2] = a13 - uq * a33
    M[1, 0] = a21 - vq * a31
    M[1, 1] = a22 - vq * a32
    M[1, 2] = a23 - vq * a33
    R_T: Matrix3x3 = R_cam_to_radar.T
    M[2, :] = R_T @ v_r # Uses the calculated v_r

    # --- 4. Build the 3x1 vector B (Eq 16a) ---
    B: Vector3 = np.zeros(3, dtype=np.floating)
    term1_d: float = (a31*up + a32*vp + a33) * uq
    term2_d: float = -(a11*up + a12*vp + a13)
    B[0] = (term1_d + term2_d) * d + (uq * bz - bx)
    term3_d: float = (a31*up + a32*vp + a33) * vq
    term4_d: float = -(a21*up + a22*vp + a23)
    B[1] = (term3_d + term4_d) * d + (vq * bz - by)
    B[2] = np.dot(v_r, v_r) * delta_t # Uses the calculated v_r

    # --- 5. Solve the linear system M * t_vec = B ---
    try:
        t_vec: Vector3 = np.linalg.solve(M, B)
    except np.linalg.LinAlgError:
        print("Warning: The matrix M is singular, cannot solve for velocity.")
        return None

    # --- 6. Convert motion vector to velocity ---
    velocity_camera_coords: Vector3 = t_vec / delta_t

    if not return_in_radar_coords:
        return velocity_camera_coords

    # Convert to RADAR coordinates to get v_f (Eq 13)
    velocity_radar_coords: Vector3 = R_cam_to_radar @ velocity_camera_coords

    return velocity_radar_coords

# --- Helper Function for Velocity Estimation ---
def estimate_velocities_for_frame(
    radar_detections: List[Tuple[Vector3, float, Optional[Entity], bool]], # Assuming corner sampler output for now
    flow: Optional[FlowField],
    camera: Camera,
    radar: Radar,
    prev_poses: Dict[str, Any],
    world_delta_t: float
) -> List[Tuple[float, float, float, float]]:
    """
    Calculates full velocity for radar detections using data from two time steps.
    Returns a list of 3d velocity magnitudes, 3d velocity error magnitude, 2d reprojection error magnitudes for successful estimations and flag whether it is a noisy point.
    """
    frame_errors: List[Tuple[float, float, float, float]] = []

    # Check if we have necessary data
    if flow is None or not prev_poses or 'camera' not in prev_poses:
        # print("  Waiting for prev state/flow...") # Optional debug
        return frame_errors # Return empty list if prerequisites not met

    # 1. Get current & previous poses
    try:
        current_cam_pose_W2L = camera.get_pose_world_to_local()
        prev_cam_pose_W2L = prev_poses['camera']
        prev_radar_pose_W2L = prev_poses['radar']

        # Check if matrices are invertible before proceeding
        inv_prev_cam_pose_W2L = np.linalg.inv(prev_cam_pose_W2L)
        inv_prev_radar_pose_W2L = np.linalg.inv(prev_radar_pose_W2L)

    except (KeyError, np.linalg.LinAlgError, TypeError):
        print("  Error accessing or inverting previous poses.")
        return frame_errors # Return empty if poses are missing/invalid

    # 2. Calculate relative transformations (given these from ego velocity estimation + calibration)
    T_A_to_B = current_cam_pose_W2L @ inv_prev_cam_pose_W2L
    T_A_to_R = prev_radar_pose_W2L @ inv_prev_cam_pose_W2L

    # This is T_Cam(A)_from_Radar(A), which is static and valid for all time.
    try:
        T_Cam_from_Radar_static = np.linalg.inv(T_A_to_R)
    except np.linalg.LinAlgError:
        print("  Error inverting T_A_to_R to find static extrinsic.")
        return frame_errors

    # --- SIMULATION-ONLY ---
    # This part is ONLY for comparing against world-frame ground truth.
    # A real system would not have this.
    R_A_radar_to_world = inv_prev_radar_pose_W2L[0:3, 0:3]
    # --- END SIMULATION-ONLY ---

    # 3. Process each detection
    for detection_idx, detection in enumerate(radar_detections):
        point_radar_coords, speed_radial, source_entity, isNoise = detection

        # --- Calculate (xq_pix, yq_pix) and (uq, vq) at t+delta_t ---
        # Convert radar point (t+delta_t) to world (t+delta_t)
        point_rad_h = np.append(point_radar_coords, 1.0)
        # print(f"point_rad_h: {point_rad_h}")

        # Convert radar (B) directly to camera (B) using the static extrinsic
        point_cam_B_h = T_Cam_from_Radar_static @ point_rad_h
        # print(f"point_cam_B_h: {point_cam_B_h}")
        point_cam_B = point_cam_B_h[:3]
        depth_B = point_cam_B[2]
        # print(f"depth_B: {depth_B}")

        if depth_B <= 1e-3: continue # Point is behind or too close to camera B

        # Normalized coords (t+delta_t)
        uq = point_cam_B[0] / depth_B
        vq = point_cam_B[1] / depth_B
        # print(f"uq: {uq}, vq: {vq}")

        # Pixel coords (t+delta_t) - careful with int conversion if needed early
        xq_pix_f = camera.fx * uq + camera.cx
        yq_pix_f = camera.fy * vq + camera.cy
        xq_pix = int(round(xq_pix_f))
        yq_pix = int(round(yq_pix_f))
        # print(f"xq_pix: {xq_pix}, yq_pix: {yq_pix}")

        # Check image bounds
        if not (0 <= xq_pix < camera.image_width and 0 <= yq_pix < camera.image_height):
            continue

        # --- Get flow and calculate (up, vp) at t ---
        dx, dy = flow[yq_pix, xq_pix] # Pixel flow
        # print(f"dx: {dx}, dy: {dy}")
        xp_pix_f = xq_pix_f - dx
        yp_pix_f = yq_pix_f - dy
        # print(f"xp_pix_f: {xp_pix_f}, yp_pix_f: {yp_pix_f}")

        # Normalized coords (t)
        up = (xp_pix_f - camera.cx) / camera.fx
        vp = (yp_pix_f - camera.cy) / camera.fy
        # print(f"up: {up}, vp: {vp}")

        # --- Call the solver ---
        full_vel_vector_radar = solve_full_velocity(
            up=up, vp=vp, uq=uq, vq=vq, d=depth_B, delta_t=world_delta_t,
            T_A_to_B=T_A_to_B, T_A_to_R=T_A_to_R,
            speed_radial=speed_radial, point_radar_coords=point_radar_coords,
            return_in_radar_coords=True
        )

        # --- Calculate and store error ---
        if full_vel_vector_radar is not None:
            full_vel_magnitude = float(np.linalg.norm(full_vel_vector_radar))
            full_vel_world = R_A_radar_to_world @ full_vel_vector_radar
            # print(f"full_vel_radar: {full_vel_vector_radar}, full_vel_world: {full_vel_world}, ground_truth_vel: {ground_truth_vel}")
            frame_displacement_error = calculate_reprojection_error(full_vel_radar_A=full_vel_vector_radar,   # Arg 1: Vel in Radar(A)
                point_radar_B=point_radar_coords,        # Arg 2: Point in Radar(B)
                T_Cam_from_Radar=np.linalg.inv(T_A_to_R),  # Arg 3: Static Extrinsic
                T_CamB_from_CamA=T_A_to_B,               # Arg 4: Relative Motion
                flow=flow, 
                camera=camera, 
                xq_pix_f=xq_pix_f, 
                yq_pix_f=yq_pix_f, 
                delta_t=world_delta_t)
             
            if frame_displacement_error is not None and source_entity is not None:
                ground_truth_vel = source_entity.velocity
                velocity_error_magnitude = float(np.linalg.norm(full_vel_world - ground_truth_vel))
                frame_errors.append((full_vel_magnitude, velocity_error_magnitude, frame_displacement_error, False))
                continue
            
            if(frame_displacement_error is not None and isNoise):
                frame_errors.append((full_vel_magnitude, 0.0, frame_displacement_error, True))

    return frame_errors

def calculate_reprojection_error(
    full_vel_radar_A: Vector3,            # Velocity expressed in Radar(A) frame
    point_radar_B: Vector3,               # Point position in Radar(B) frame
    T_Cam_from_Radar: Matrix4x4,          # Static Extrinsic: T_Cam(A)_from_Radar(A)
    T_CamB_from_CamA: Matrix4x4,          # Relative Motion: T_Cam(B)_from_Cam(A)
    flow: FlowField,                      # Flow field
    camera: Camera,                       # Camera object for intrinsics
    xq_pix_f: float, yq_pix_f: float,     # Projected 3D radar point to image frame at t (B)
    delta_t: float                        # Simulation time step
) -> Optional[float]:
    """
    Calculates the reprojection error using only relative transformations.
    The "world" frame for this calculation is defined as the Camera(A) frame.
    """
    
    # --- 1. Define all state in our "World" frame (Camera A) ---
    
    # 1a. Get predicted velocity in the Cam(A) frame.
    # We must use the 3x3 rotation part of the static extrinsic.
    R_Cam_from_Radar = T_Cam_from_Radar[0:3, 0:3]
    vel_cam_A_predicted = R_Cam_from_Radar @ full_vel_radar_A

    # 1b. Get the point's position at time B, expressed in the Cam(A) frame.
    # This is the most complex step. Path: Radar(B) -> Cam(B) -> Cam(A)
    point_radar_B_h = np.append(point_radar_B, 1.0)
    
    # T_Cam(A)_from_Radar(B) = T_Cam(A)_from_Cam(B) @ T_Cam(B)_from_Radar(B)
    T_CamA_from_CamB = np.linalg.inv(T_CamB_from_CamA)
    # The static extrinsic is the same at time B
    T_CamB_from_RadarB = T_Cam_from_Radar 
    
    T_CamA_from_RadarB = T_CamA_from_CamB @ T_CamB_from_RadarB
    point_cam_A_at_B_h = T_CamA_from_RadarB @ point_radar_B_h

    # We now have P(B) and V(A), both expressed in the Cam(A) frame.
    point_cam_A_at_B = point_cam_A_at_B_h[:3] # This is P(B) in Cam(A) frame
    
    # --- 2. Predict the point's 3D position at time A ---
    
    # P(A) = P(B) - V * dt
    # All terms are now in the same frame (Cam_A), so this physics is valid.
    point_cam_A_predicted = point_cam_A_at_B - vel_cam_A_predicted * delta_t
    
    # --- 3. Project predicted 3D point (A) into pixel (A) ---
    
    try:
        # The point is already in the Cam(A) frame, so we just project it.
        previous_depth_A_predicted = point_cam_A_predicted[2]
        if previous_depth_A_predicted <= 1e-3:
            return None

        # Normalized coords
        previous_up_predicted = point_cam_A_predicted[0] / previous_depth_A_predicted
        previous_vp_predicted = point_cam_A_predicted[1] / previous_depth_A_predicted 

        # Pixel coords
        previous_xp_pred_pix_f = camera.fx * previous_up_predicted + camera.cx
        previous_yp_pred_pix_f = camera.fy * previous_vp_predicted + camera.cy
        
    except Exception as e:
        # print(f" Reprojection math error: {e}") # Debug
        return None

    # --- 4. Get "Observed" pixel position (A) from flow ---
    xq_pix = int(round(xq_pix_f))
    yq_pix = int(round(yq_pix_f))

    dx, dy = flow[yq_pix, xq_pix] # Pixel flow
    flow_previous_xp_pred_pix_f = xq_pix_f - dx
    flow_previous_yp_pred_pix_f = yq_pix_f - dy

    # --- 5. Calculate Euclidean distance error in pixels ---
    error_x = previous_xp_pred_pix_f - flow_previous_xp_pred_pix_f
    error_y = previous_yp_pred_pix_f - flow_previous_yp_pred_pix_f

    reprojection_error = math.sqrt(error_x**2 + error_y**2)

    return reprojection_error