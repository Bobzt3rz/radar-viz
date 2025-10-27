import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Tuple, List, Dict

from .world import World
from .entities import Cube, Entity
from .camera import Camera

class Radar(Camera):
    """
    Simulates a radar by acting as a "depth camera."
    It projects all world geometry into a low-resolution
    2D grid and performs a depth test to find the closest
    hit for each "pixel," thus simulating occlusion.
    """
    def __init__(self, position: np.ndarray, target: np.ndarray, up: np.ndarray, 
                 resolution: Tuple[int, int] = (128, 128)):
        
        # 1. Call the Camera's __init__
        super().__init__(position, target, up)

        # 2. Override/set radar-specific properties
        self.fov = 45.0           # Horizontal field of view
        self.min_range = 0.1      # Minimum distance (near clip)
        self.max_range = 100.0    # Maximum distance (far clip)
        
        # (Rename near/far clip for clarity)
        self.near_clip = self.min_range
        self.far_clip = self.max_range
        
        # 3. Resolution of the depth buffer
        self.resolution_w = resolution[0]
        self.resolution_h = resolution[1]

    def simulate_scan(self,
                      world: World,
                      id_to_velocity: Dict[Tuple[int, int, int], np.ndarray],
                      ego_velocity: np.ndarray,
                      dt: float  # <<< ADD dt AS ARGUMENT
                     ) -> np.ndarray:
        """
        Generates a 3D point cloud using the SAE J670 standard, calculating
        radial velocity relative to the predicted position at time t (Q).

        Returns:
            np.ndarray: An (N, 8) array where each row is:
            [x_world_Q, y_world_Q, z_world_Q,           # World pos at t (Q)
             x_local_Q, y_local_Q, z_local_Q,           # Local pos at t (Q) rel to radar at t
             doppler(vr_Q),                             # Radial vel based on Q
             azimuth_Q]                                 # Azimuth based on Q
        """

        # 1. Get maps for time t-1 (as before)
        # These maps are based on the world state *before* the current update
        world_pos_map_P = self.get_world_position_map(
            world, self.resolution_w, self.resolution_h
        )[0]
        id_map = self.get_id_map(world, self.resolution_w, self.resolution_h)

        # --- Use CURRENT radar position (at time t) ---
        radar_pos_t = self.position # Assumes rig.update() was called before this

        # --- 2. Get Radar's Local Coordinate System at time t ---
        #    (Recalculate based on current pose)
        x_axis_local = self.target - self.position
        x_axis_norm = np.linalg.norm(x_axis_local)
        if x_axis_norm > 1e-6:
            x_axis_local /= x_axis_norm
        else:
            x_axis_local = np.array([0, 0, 1]) # Failsafe (should point along Z if target=pos)

        z_axis_local = self.up # Assuming up vector is normalized and constant for now
        z_axis_norm = np.linalg.norm(z_axis_local)
        if z_axis_norm > 1e-6:
             z_axis_local /= z_axis_norm
        else:
             z_axis_local = np.array([0, 1, 0]) # Failsafe

        y_axis_local = np.cross(z_axis_local, x_axis_local)
        # Ensure it's normalized if z and x weren't perfectly orthogonal
        y_axis_norm = np.linalg.norm(y_axis_local)
        if y_axis_norm > 1e-6:
             y_axis_local /= y_axis_norm
        else: # Handle degenerate case (e.g., up aligned with forward)
             # Find an orthogonal vector to x_axis_local
             temp_vec = np.array([0,1,0]) if abs(np.dot(x_axis_local, [0,1,0])) < 0.99 else np.array([1,0,0])
             y_axis_local = np.cross(z_axis_local, x_axis_local) # Should recompute z here based on new y
             y_axis_local /= np.linalg.norm(y_axis_local)
             z_axis_local = np.cross(x_axis_local, y_axis_local) # Recompute z to ensure right-handed system


        # Rotation matrix: World -> Radar (SAE) frame at time t
        # Columns are the local axes in world coordinates
        R_radar_to_world_t = np.stack([x_axis_local, y_axis_local, z_axis_local], axis=1)
        R_world_to_radar_t = R_radar_to_world_t.T # For right-handed orthonormal basis

        hits_data_list = []

        # 3. Iterate over every "pixel" from the t-1 render
        for v in range(self.resolution_h):
            for u in range(self.resolution_w):

                hit_pos_world_P = world_pos_map_P[v, u] # Position at t-1
                if np.all(hit_pos_world_P == 0.0): # Skip background
                    continue

                id_tuple = tuple(id_map[v, u])
                V_object_abs_world = id_to_velocity.get(id_tuple)
                if V_object_abs_world is None: # Should not happen if ID map is correct
                    continue

                # --- 4. Calculate Predicted State at time t (Point Q) ---
                Q_world = hit_pos_world_P + V_object_abs_world * dt

                # --- 5. Calculate Metrics Relative to Q and Radar at t ---

                # a) Line-of-Sight (LOS) vector in world frame (Radar_t -> Q)
                LOS_vector_world_t = Q_world - radar_pos_t

                # b) Calculate Range to Q
                range_t = np.linalg.norm(LOS_vector_world_t)
                if range_t < 1e-6: # Avoid division by zero
                    continue

                # c) Convert LOS_t to Radar's Local (SAE) frame at time t
                #    Q_local_t = R_world_to_radar_t @ LOS_vector_world_t
                x_local_t = np.dot(LOS_vector_world_t, x_axis_local)
                y_local_t = np.dot(LOS_vector_world_t, y_axis_local)
                z_local_t = np.dot(LOS_vector_world_t, z_axis_local)
                Q_local_t = np.array([x_local_t, y_local_t, z_local_t]) # Local position vector of Q

                # d) Calculate Unit Radial Vector (SAE frame) towards Q
                unit_radial_SAE_t = Q_local_t / range_t

                # e) Calculate Relative Velocity
                V_relative_world = V_object_abs_world - ego_velocity

                # f) Transform Relative Velocity to Radar (SAE) frame at time t
                V_relative_radar_t = R_world_to_radar_t @ V_relative_world

                # g) Calculate Doppler / Radial Velocity (Projection onto unit radial to Q)
                vr_t = np.dot(V_relative_radar_t, unit_radial_SAE_t)

                # h) Azimuth angle in the local X-Y plane towards Q
                azimuth_val_t = np.arctan2(y_local_t, x_local_t) # (radians)

                # 6. Append data based on state at time t (Q)
                hits_data_list.append([
                    Q_world[0], Q_world[1], Q_world[2],   # World Position of Q
                    x_local_t, y_local_t, z_local_t,       # Local Position of Q
                    vr_t,                                  # Radial velocity towards Q
                    azimuth_val_t                          # Azimuth towards Q
                ])

        if not hits_data_list:
            return np.empty((0, 8), dtype=float)

        return np.array(hits_data_list, dtype=float)