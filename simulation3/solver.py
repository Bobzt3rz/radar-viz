import numpy as np
from core import CameraObservation, RadarObservation, Transform

def solve_pow4r_velocity_corrected(
    obs_cam_curr, obs_cam_prev, obs_radar,
    T_cam_radar, T_c2_c1, dt
):
    """
    Solves the POW4R equation EXACTLY by treating depth 'd' as a variable
    dependent on the unknown motion 't'.
    """
    # --- 1. Setup Variables (Same as before) ---
    u_q, v_q = obs_cam_curr.normalized_uv.flatten()
    u_p, v_p = obs_cam_prev.normalized_uv.flatten()
    
    mat_ego = T_c2_c1.to_matrix() # T_prev_curr (P_prev = T @ P_curr)
    
    # Alpha (Rotation-like part of Ego Motion)
    alpha = mat_ego[:3, :3]
    # Beta (Translation part of Ego Motion)
    beta  = mat_ego[:3, 3]

    # Radar Geometry (Radar Unit Vector in Camera Frame)
    R_rad_to_cam = T_cam_radar.R
    r_vec_c = R_rad_to_cam @ obs_radar.radial_unit_vec

    # --- 2. Calculate Static Depth and Gradient ---
    # We need to express d(t-1) = d_static + (Gradient dot t_vec)
    
    # Step A: Get P_curr in Camera Frame
    P_r_curr = obs_radar.position_r
    # P_cam = T_cam_radar @ P_rad (Using our definition T_cam_radar maps Rad->Cam points)
    P_c_curr = (T_cam_radar.to_matrix() @ np.append(P_r_curr, [[1]], axis=0))[:3]
    
    # Step B: Express P_prev in terms of P_curr and t_vec
    # P_prev = T_ego @ (P_curr - t_vec)  <-- CAREFUL WITH SIGNS
    # The paper defines t_vec as motion P -> Q (t-1 -> t).
    # So P_curr = P_prev_unrotated + t_vec_rotated? 
    # Let's stick to the paper's exact definitions:
    # Eq 10: Q~ = T_AB * Q. (Maps Current to Compensated).
    # The "d" in Eq 16a comes from P = (u_p*d, v_p*d, d).
    # The relationship is P_prev = T_ego @ (P_curr - t_vec).
    # d = Z_prev.
    
    # Let P_c_curr = [x, y, z].
    # P_prev = alpha @ (P_c_curr - t_vec) + beta
    # d = (alpha[2,:] dot (P_c_curr - t_vec)) + beta[2]
    # d = (alpha[2,:] dot P_c_curr) + beta[2] - (alpha[2,:] dot t_vec)
    
    # So:
    # d_static = (alpha[2,:] dot P_c_curr) + beta[2]
    # gradient_k = -alpha[2,:]  (The coefficients of t_x, t_y, t_z)
    
    d_static = np.dot(alpha[2, :], P_c_curr) + beta[2]
    k_vec = -alpha[2, :] # Shape (3,)

    # --- 3. Construct Matrices ---
    
    # --- Matrix A (LHS) ---
    # Standard terms from Paper
    A = np.zeros((3,3))
    
    # Rows 1 & 2 (Camera Constraints)
    # A_ij = alpha_ij - u_q * alpha_3j
    for j in range(3):
        A[0, j] = alpha[0, j] - u_q * alpha[2, j]
        A[1, j] = alpha[1, j] - v_q * alpha[2, j]
    
    # Row 3 (Radar Constraint)
    A[2, :] = r_vec_c.flatten()

    # --- Vector C (The "Multiplier" of d on the RHS) ---
    # In the flawed solver, RHS was: C * d + Offset
    # We need to isolate C.
    # Row 1 C: ( (a31*up + a32*vp + a33)*u_q - (a11*up + a12*vp + a13) )
    # Let's simplify:
    # Term 1: u_q * (alpha[2] dot p_uv_hom)
    # Term 2: -1  * (alpha[0] dot p_uv_hom)
    # where p_uv_hom = [u_p, v_p, 1]
    
    p_uv_hom = np.array([u_p, v_p, 1.0])
    
    # Compute the scalar factor "common term" for each row
    # Note: This 'common term' multiplies alpha columns. 
    # Wait, looking at Eq 16a (flawed solver B construction):
    # comm = alpha_31*u_p + alpha_32*v_p + alpha_33
    # B[0] = (comm*u_q - (alpha_11*u_p + ...)) * d
    # So C[0] is exactly that bracket.
    
    comm = np.dot(alpha[2, :], p_uv_hom) # a31*up + a32*vp + a33
    
    C = np.zeros(3)
    # Row 1 Coeff for d:
    # u_q * comm - (alpha[0,:] dot p_uv_hom)
    C[0] = u_q * comm - np.dot(alpha[0, :], p_uv_hom)
    
    # Row 2 Coeff for d:
    # v_q * comm - (alpha[1,:] dot p_uv_hom)
    C[1] = v_q * comm - np.dot(alpha[1, :], p_uv_hom)
    
    # Row 3 Coeff for d:
    # 0 (Radar constraint doesn't depend on d)
    C[2] = 0.0
    
    # --- 4. The Correction Step (Move t terms to LHS) ---
    # We had: A * t = C * (d_static + k * t) + Other_RHS
    #         A * t - C * (k * t) = C * d_static + Other_RHS
    #         (A - outer(C, k)) * t = New_RHS
    
    # Matrix Update
    # We subtract (C column vector) * (k row vector)
    correction_matrix = np.outer(C, k_vec)
    A_corrected = A - correction_matrix
    
    # --- 5. Construct RHS (B_corrected) ---
    B_corrected = np.zeros(3)
    
    # Static part of Previous RHS (The "Offset" terms)
    # Row 1: u_q*beta_z - beta_x
    # Row 2: v_q*beta_z - beta_y
    # Row 3: vr * dt
    
    offset_1 = u_q * beta[2] - beta[0]
    offset_2 = v_q * beta[2] - beta[1]
    offset_3 = obs_radar.radial_velocity * dt
    
    # Combine: C * d_static + Offset
    B_corrected[0] = C[0] * d_static + offset_1
    B_corrected[1] = C[1] * d_static + offset_2
    B_corrected[2] = C[2] * d_static + offset_3 # C[2] is 0, so just offset_3
    
    # --- 6. Solve ---
    try:
        t_vec = np.linalg.solve(A_corrected, B_corrected)
    except np.linalg.LinAlgError:
        return np.zeros((3,1))
        
    return (t_vec / dt).reshape(3,1)