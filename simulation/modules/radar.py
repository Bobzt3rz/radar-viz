import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Tuple, List

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
    def __init__(self, position: list, target: list, up: list, 
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

    def simulate_scan(self, world: World) -> np.ndarray:
        """
        Generates a 3D point cloud by un-projecting
        the depth buffer.
        """
        # 1. Call the base class method
        world_pos_map = self.get_world_position_map(
            world, self.resolution_w, self.resolution_h
        )[0]
        
        # 2. Filter out non-hits (which we set to 0,0,0)
        # Reshape to (N, 3)
        hits = world_pos_map.reshape(-1, 3)
        # Filter points where all coords are 0
        hits = hits[np.any(hits != 0, axis=1)] 
        
        return np.array(hits, dtype=float)