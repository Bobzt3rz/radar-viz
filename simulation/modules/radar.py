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

    # ... all other methods (_apply_view_matrix, _apply_projection_matrix)
    # are INHERITED from Camera, so you can delete them if they are identical.
    
    def simulate_scan(self, world: World) -> np.ndarray:
        """
        Renders the world from the radar's perspective to the
        hardware depth buffer, then reads it back and un-projects
        it to a 3D point cloud.
        """
        
        # --- 1. Save Renderer's OpenGL State ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        original_viewport = glGetIntegerv(GL_VIEWPORT)

        # --- 2. Setup Radar's "View" ---
        radar_viewport = (0, 0, self.resolution_w, self.resolution_h)
        glViewport(radar_viewport[0], radar_viewport[1], radar_viewport[2], radar_viewport[3])
        
        self.apply_projection_matrix(self.resolution_w / self.resolution_h)
        self.apply_view_matrix()

        # --- 3. Render Solid Faces to Hardware Depth Buffer ---
        glClear(GL_DEPTH_BUFFER_BIT) 
        for entity in world.entities:
            if isinstance(entity, Cube):
                entity.draw()

        # --- 4. Read the Hardware Depth Buffer ---
        depth_buffer_bytes = glReadPixels(
            0, 0, self.resolution_w, self.resolution_h,
            GL_DEPTH_COMPONENT, GL_FLOAT
        )
        
        # --- THIS IS THE FIX ---
        # Convert bytes to a numpy array of GL_FLOATs (np.float32)
        # and then reshape it to the 2D grid
        depth_buffer = np.frombuffer(
            depth_buffer_bytes, dtype='float32'  # type: ignore
        ).reshape(self.resolution_h, self.resolution_w)
        # ---------------------

        # --- 5. Un-project Depth Image to 3D Point Cloud ---
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        hits = []
        for u in range(self.resolution_w):
            for v in range(self.resolution_h):
                depth = depth_buffer[v, u]
                
                if depth < 1.0:
                    world_x, world_y, world_z = gluUnProject(
                        u, v, depth,
                        modelview, projection, radar_viewport
                    )
                    hits.append([world_x, world_y, world_z])

        # --- 6. Restore Renderer's OpenGL State ---
        glViewport(original_viewport[0], original_viewport[1], original_viewport[2], original_viewport[3])
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        return np.array(hits, dtype=float)