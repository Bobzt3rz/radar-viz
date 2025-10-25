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
                      ego_velocity: np.ndarray
                     ) -> np.ndarray:
        """
        Generates a 3D point cloud using the SAE J670 standard.
        
        Returns:
            np.ndarray: An (N, 8) array where each row is:
            [x_world, y_world, z_world, 
             x_local(fwd), y_local(left), z_local(up), 
             doppler(vr), azimuth]
        """
        
        # 1. Get maps (as before)
        world_pos_map = self.get_world_position_map(
            world, self.resolution_w, self.resolution_h
        )[0]
        id_map = self.get_id_map(world, self.resolution_w, self.resolution_h)
        radar_pos = self.position
        
        # --- 2. Get Radar's Local Coordinate System (SAE Standard) ---
        
        # X-axis (forward)
        x_axis_local = self.target - self.position
        x_axis_norm = np.linalg.norm(x_axis_local)
        if x_axis_norm > 0:
            x_axis_local /= x_axis_norm
        else:
            x_axis_local = np.array([0, 0, 1]) # Failsafe
            
        # Z-axis (up)
        # Note: We use the rig's 'up' vector directly
        z_axis_local = self.up
        z_axis_norm = np.linalg.norm(z_axis_local)
        if z_axis_norm > 0:
            z_axis_local /= z_axis_norm
        else:
            z_axis_local = np.array([0, 1, 0]) # Failsafe
            
        # Y-axis (left)
        # y = z cross x (to form a right-handed system)
        y_axis_local = np.cross(z_axis_local, x_axis_local)
        
        hits_data_list = []

        # 3. Iterate over every "pixel"
        for v in range(self.resolution_h):
            for u in range(self.resolution_w):
                
                hit_pos_world = world_pos_map[v, u]
                if np.all(hit_pos_world == 0.0):
                    continue
                    
                id_tuple = tuple(id_map[v, u])
                V_object_abs = id_to_velocity.get(id_tuple)
                if V_object_abs is None:
                    continue

                # --- 4. Calculate All Metrics ---
                
                # a) Line-of-Sight (LOS) vector in world frame
                LOS_vector_world = hit_pos_world - radar_pos
                
                # b) Convert LOS to Radar's Local (x,y,z)
                #    using the SAE standard
                x_local = np.dot(LOS_vector_world, x_axis_local)
                y_local = np.dot(LOS_vector_world, y_axis_local)
                z_local = np.dot(LOS_vector_world, z_axis_local)

                # c) Calculate Range and Angles
                range_val = np.linalg.norm(LOS_vector_world)
                if range_val == 0:
                    continue
                
                # Azimuth is the angle in the X-Y (forward-lateral) plane
                azimuth_val = np.arctan2(y_local, x_local) # (radians)
                
                # d) Calculate Doppler / Radial Velocity (as before)
                V_relative = V_object_abs - ego_velocity
                vr = np.dot(V_relative, LOS_vector_world) / range_val

                # 5. Append all data
                hits_data_list.append([
                    hit_pos_world[0], hit_pos_world[1], hit_pos_world[2], # 0, 1, 2
                    x_local, y_local, z_local,                          # 3, 4, 5
                    vr,                                                 # 6
                    azimuth_val                                         # 7
                ])

        if not hits_data_list:
            return np.empty((0, 8), dtype=float)
            
        return np.array(hits_data_list, dtype=float)