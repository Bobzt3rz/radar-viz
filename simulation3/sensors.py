import numpy as np
from typing import Optional
from core import RealPoint, Transform, RadarObservation, CameraObservation
from geometry import inverse_transform, project_point, compute_radial_velocity

class VirtualCamera:
    def __init__(self, K: np.ndarray, T_rig_cam: Transform):
        self.K = K
        self.T_rc = T_rig_cam     # Transform: Rig -> Camera
        self.T_cr = inverse_transform(T_rig_cam) # Transform: Camera -> Rig

    def observe(self, real_point: RealPoint, rig_pose_w: Transform) -> Optional[CameraObservation]:
        """
        Project a Ground Truth point into the Camera sensor.
        Returns None if the point is behind the camera.
        """
        # Transform World -> Rig -> Camera
        # P_c = T_cr * T_rw * P_w
        T_rw = inverse_transform(rig_pose_w)
        
        P_w_hom = np.append(real_point.position_w, [[1]], axis=0)
        P_r = (T_rw.to_matrix() @ P_w_hom)[:3] # Point in Rig Frame
        P_c = (self.T_cr.to_matrix() @ np.append(P_r, [[1]], axis=0))[:3] # Point in Cam Frame

        uv, norm_uv = project_point(self.K, P_c)
        if uv is None or norm_uv is None: 
            return None

        return CameraObservation(
            point_id=real_point.point_id,
            uv=uv,
            depth=float(P_c[2]),
            normalized_uv=norm_uv
        )

class VirtualRadar:
    def __init__(self, T_rig_rad: Transform):
        self.T_rr = T_rig_rad      # Transform: Rig -> Radar
        # Note: We often need the inverse to map World points into Radar frame
        self.T_r_rig = inverse_transform(T_rig_rad) 

    def observe(self, real_point: RealPoint, rig_pose_w: Transform, rig_vel_w: np.ndarray) -> RadarObservation:
        """
        Project a Ground Truth point into the Radar sensor.
        Calculates position and radial velocity in Radar Frame.
        """
        # 1. Position: World -> Rig -> Radar
        # P_rad = T_rad_rig @ T_rig_world @ P_w
        T_rad_rig = inverse_transform(self.T_rr)
        T_rig_w   = inverse_transform(rig_pose_w)
        
        mat_transform = T_rad_rig.to_matrix() @ T_rig_w.to_matrix()
        P_w_hom = np.append(real_point.position_w, [[1]], axis=0)
        P_rad = (mat_transform @ P_w_hom)[:3]

        # 2. Velocity: Transform vectors to Radar Frame
        # Only Rotation matters for velocity vectors
        R_total = mat_transform[:3, :3]
        V_p_rad = R_total @ real_point.velocity_w
        V_rig_rad = R_total @ rig_vel_w

        # 3. Radial Velocity Calculation
        vr, unit_vec = compute_radial_velocity(P_rad, V_p_rad, V_rig_rad)

        return RadarObservation(
            point_id=real_point.point_id,
            position_r=P_rad,
            radial_velocity=vr,
            radial_unit_vec=unit_vec
        )