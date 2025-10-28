import numpy as np
from typing import Optional
from .types import Vector3, Matrix3x3, Matrix4x4

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