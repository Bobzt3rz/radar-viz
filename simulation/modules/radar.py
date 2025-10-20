import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Tuple, List

from .world import World
from .entities import Cube 
from .camera import Camera  # <-- IMPORT THE BASE CLASS

class Radar(Camera): # <-- INHERIT FROM CAMERA
    """
    Simulates a radar by acting as a "depth camera."
    It projects all world geometry into a low-resolution
    2D grid and performs a depth test to find the closest
    hit for each "pixel," thus simulating occlusion.
    """
    def __init__(self, position: list, target: list, up: list, 
                 resolution: Tuple[int, int] = (64, 64)):
        
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
    
    # We only keep the unique method:
    def simulate_scan(self, world: World) -> np.ndarray:
        """
        Projects all vertices in the world, performs a depth test,
        and returns an (N, 3) numpy array of the visible 3D points.
        """
        
        # --- 1. Save Renderer's OpenGL State ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # --- 2. Setup Radar's "View" ---
        # Use the inherited methods
        self.apply_projection_matrix(self.resolution_w / self.resolution_h)
        self.apply_view_matrix() # This will use glLoadIdentity()
        
        radar_viewport = (0, 0, self.resolution_w, self.resolution_h)

        # --- 3. Create the "Buffers" ---
        depth_buffer = np.full((self.resolution_h, self.resolution_w), 
                                self.max_range, dtype=float)
        hit_buffer = np.full((self.resolution_h, self.resolution_w, 3), 
                              np.nan, dtype=float)

        # --- 4. Project all points and do Depth Test ---
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)

        for entity in world.entities:
            if isinstance(entity, Cube):
                for vertex in entity.get_world_surface_points():
                    try:
                        (u, v, depth) = gluProject(
                            vertex[0], vertex[1], vertex[2],
                            modelview, projection, radar_viewport
                        )
                        
                        u, v = int(round(u)), int(round(v))

                        if (0 <= u < self.resolution_w and 
                            0 <= v < self.resolution_h):
                            if depth < depth_buffer[v, u]:
                                depth_buffer[v, u] = depth
                                hit_buffer[v, u] = vertex
                                
                    except:
                        pass 

        # --- 5. Restore Renderer's OpenGL State ---
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # --- 6. Collect Final Hits ---
        final_hits = hit_buffer[~np.isnan(hit_buffer).any(axis=2)]
        
        return final_hits