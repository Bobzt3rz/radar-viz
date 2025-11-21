import numpy as np
from typing import Tuple, Optional
from core import Transform

def inverse_transform(T: Transform) -> Transform:
    """Computes T_BA given T_AB."""
    R_inv = T.R.T
    t_inv = -R_inv @ T.t
    return Transform(R_inv, t_inv)

def get_relative_transform(T_w_a: Transform, T_w_b: Transform) -> Transform:
    """
    Computes transform FROM A TO B (T_ba).
    P_b = T_ba @ P_a
    Logic: T_ba = inv(T_wb) @ T_wa
    """
    T_bw = inverse_transform(T_w_b)
    mat_bw = T_bw.to_matrix()
    mat_wa = T_w_a.to_matrix()
    mat_ba = mat_bw @ mat_wa
    return Transform.from_matrix(mat_ba)

def project_point(K: np.ndarray, point_c: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Projects a 3D point in Camera Frame to 2D pixels.
    Args:
        K: Intrinsic matrix (3, 3)
        point_c: Point in camera frame (3, 1)
    Returns: 
        (pixel_uv, normalized_uv) or (None, None) if behind camera
    """
    if point_c[2] <= 0:
        return None, None

    # Normalized coordinates (x/z, y/z)
    u_norm = point_c[0] / point_c[2]
    v_norm = point_c[1] / point_c[2]
    normalized_uv = np.array([[u_norm], [v_norm]], dtype=float).reshape(2,1)

    # Pixel coordinates: uv = K @ P (homogenous)
    uv_hom = K @ point_c
    uv = uv_hom[:2] / uv_hom[2]
    
    return uv.reshape(2,1), normalized_uv

def compute_radial_velocity(
    P_r: np.ndarray, 
    V_point_r: np.ndarray, 
    V_radar_r: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Computes radial velocity scalar and unit vector.
    Args:
        P_r: Point position in Radar Frame (3, 1)
        V_point_r: Point velocity in Radar Frame (3, 1)
        V_radar_r: Radar sensor velocity in Radar Frame (3, 1)
    Returns:
        (radial_velocity, unit_vector)
    """
    dist = np.linalg.norm(P_r)
    if dist < 1e-6: 
        return 0.0, np.zeros((3,1))
    
    unit_vec = P_r / dist
    
    if V_radar_r is None:
        V_rel = V_point_r
    else:
        V_rel = V_point_r - V_radar_r
        
    vr = float(np.dot(unit_vec.T, V_rel))
    return vr, unit_vec