import numpy as np
from typing import Dict, Tuple, Optional, List
from OpenGL.GLU import gluProject

def estimate_full_displacement(
    dt: float, 
    T_AB: np.ndarray, 
    T_RA: np.ndarray, 
    up: float, vp: float, d: float, 
    uq: float, vq: float, 
    vx_r: float, vy_r: float, vz_r: float, 
    radial_vel_magnitude: float
) -> Optional[np.ndarray]:
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
        return None # Return None on failure

    return t

def calculate_3d_velocity(
    up_vp_d_t1: Tuple[float, float, float],
    unit_radial_vector: np.ndarray,
    radial_vel_magnitude: float,
    pixel_flow_vector: Tuple[float, float],
    pixel_pos_t1: Tuple[float, float],
    intrinsics: dict,
    T_RA: np.ndarray,
    T_AB: np.ndarray,
    dt: float,
    EGO_VELOCITY: np.ndarray
) -> Optional[np.ndarray]:
    """
    Calculates 3D absolute velocity from pure sensor/calibration data.
    This function is portable and knows nothing about the simulation.
    """
    
    # 1. Unpack all inputs
    up, vp, d_t1 = up_vp_d_t1
    vx_r, vy_r, vz_r = unit_radial_vector
    flow_u_pix, flow_v_pix = pixel_flow_vector
    u_pix_t1, v_pix_t1 = pixel_pos_t1
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    # 2. Get T2 Data (uq, vq) - FROM MODEL FLOW
    
    # a) Find T2 pixel position
    u_pix_t2 = u_pix_t1 + flow_u_pix
    v_pix_t2 = v_pix_t1 + flow_v_pix
    
    # b) *** The Approximation ***
    # (This only works because T_RA is identity)
    d_t2_est = d_t1 + (radial_vel_magnitude * dt)
    if d_t2_est == 0: return None
        
    # print(f"d_t1: {d_t1}, d_t2_est: {d_t2_est}")

    # c) Un-project T2 pixel to get normalized (uq, vq)
    uq = u_pix_t2
    vq = v_pix_t2
    
    # 3. Solve!
    t_est_relative = estimate_full_displacement(
        dt, T_AB, T_RA, up, vp, d_t1, uq, vq,
        vx_r, vy_r, vz_r, radial_vel_magnitude
    )
    
    if t_est_relative is not None:
        V_Absolute_EST = (t_est_relative / dt) + EGO_VELOCITY
        return V_Absolute_EST

    return None # Solve failed