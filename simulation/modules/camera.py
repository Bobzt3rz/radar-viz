from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import Tuple

from .world import World
from .entities import Cube

class Camera:
    """
    Represents a camera with its pose (extrinsics) and
    properties (intrinsics) for rendering.
    """
    def __init__(self, position: np.ndarray, target: np.ndarray, up: np.ndarray):
        # Extrinsics (Pose)
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.up = np.array(up, dtype=float)

        # Intrinsics
        self.fov = 45.0
        self.near_clip = 0.1
        self.far_clip = 100.0

    def apply_view_matrix(self):
        """
        Applies the ModelView transformation (gluLookAt)
        based on the camera's extrinsics.
        """
        glLoadIdentity()
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            self.target[0], self.target[1], self.target[2],
            self.up[0], self.up[1], self.up[2]
        )

    def apply_projection_matrix(self, aspect_ratio: float):
        """
        Applies the Projection transformation (gluPerspective)
        based on the camera's intrinsics.
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, aspect_ratio, self.near_clip, self.far_clip)
        
        # Switch back to ModelView mode for the renderer
        glMatrixMode(GL_MODELVIEW)

    def _get_depth_buffer(self, world: World, width: int, height: int) -> np.ndarray:
        """
        Renders the world to a depth buffer of a specific size.
        This is a protected helper method.
        """
        # --- 1. Save Renderer's OpenGL State ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        original_viewport = glGetIntegerv(GL_VIEWPORT)

        # --- 2. Setup Camera's View ---
        glViewport(0, 0, width, height)
        self.apply_projection_matrix(width / height)
        self.apply_view_matrix()

        # --- 3. Render Solid Faces to Hardware Depth Buffer ---
        glClear(GL_DEPTH_BUFFER_BIT) 
        for entity in world.entities:
            # We need to check if draw exists, as Point doesn't have it
            if hasattr(entity, 'draw'):
                entity.draw()

        # --- 4. Read the Hardware Depth Buffer ---
        depth_buffer_bytes = glReadPixels(
            0, 0, width, height,
            GL_DEPTH_COMPONENT, GL_FLOAT
        )
        depth_buffer = np.frombuffer(
            depth_buffer_bytes, dtype='float32'  # type: ignore
        ).reshape(height, width)[::-1, :]

        # ^-- WHY THE FLIP? [::-1, :]
        # glReadPixels has its (0,0) origin at the BOTTOM-LEFT corner
        # of the viewport.
        # NumPy/PIL/OpenCV expect (0,0) at the TOP-LEFT.
        # This flip makes the depth buffer's coordinate system
        # consistent with get_id_map(), which is crucial for
        # calculations in main.py.
        
        # --- 5. Restore Renderer's OpenGL State ---
        glViewport(original_viewport[0], original_viewport[1], original_viewport[2], original_viewport[3])
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        return depth_buffer
    
    def get_id_map(self, world: World, width: int, height: int) -> np.ndarray:
        """
        Renders the world using unique ID colors for each
        object and returns a (H, W, 3) map of these IDs.
        """
        # --- 1. Save Renderer's OpenGL State ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        original_viewport = glGetIntegerv(GL_VIEWPORT)

        # --- 2. Setup Camera's View ---
        glViewport(0, 0, width, height)
        self.apply_projection_matrix(width / height)
        self.apply_view_matrix()

        # --- 3. Render Solid Faces with ID Colors ---
        glClearColor(0.0, 0.0, 0.0, 1.0) # Set background to ID 0
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT)) 
        
        for entity in world.entities:
            # We only care about objects with velocity (Cubes)
            if isinstance(entity, Cube):
                entity.draw_for_id() # <-- Call the new method

        # --- 4. Read the Hardware Color Buffer ---
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        id_buffer_bytes = glReadPixels(
            0, 0, width, height,
            GL_RGB, GL_UNSIGNED_BYTE # <-- Read color (IDs)
        )
        id_map = np.frombuffer(
            id_buffer_bytes, dtype=np.uint8  # type: ignore
        ).reshape(height, width, 3)
        
        # --- 5. Restore Renderer's OpenGL State ---
        glViewport(original_viewport[0], original_viewport[1], original_viewport[2], original_viewport[3])
        glClearColor(0.1, 0.1, 0.1, 1.0) # Restore normal background
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Flip from OpenGL's bottom-left origin
        return id_map[::-1, :, :]
    
    def get_world_position_map(self, world: World, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Renders the world and un-projects the depth buffer
        to create a (H, W, 3) map of world coordinates.
        
        Returns:
            - world_coords (H, W, 3) np.ndarray
            - modelview matrix (np.ndarray)
            - projection matrix (np.ndarray)
            - viewport (np.ndarray)
        """
        # ... (Step 1: Get Depth Buffer) ...
        depth_buffer = self._get_depth_buffer(world, width, height)
        
        # --- 2. Get Matrices ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        self.apply_projection_matrix(width / height)
        self.apply_view_matrix()
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # --- 3. Un-project Depth Image (this block is already here) ---
        world_coords = np.empty((height, width, 3), dtype=float)
        viewport = (0, 0, width, height) # <-- This was already here
        # Create (u, v) coordinates for all pixels
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # Un-project all points at once
        # Note: This is slow and can be optimized further, but
        # it's a direct replacement for your loop.
        world_coords = np.empty((height, width, 3), dtype=float)
        viewport = (0, 0, width, height) # Viewport used for depth capture
        
        for v_idx in range(height):
            for u_idx in range(width):
                depth = depth_buffer[v_idx, u_idx]

                # Get the (u,v) from the grid (top-left)
                u = u_grid[v_idx, u_idx]
                v = v_grid[v_idx, u_idx]

                # Flip v to be bottom-left for gluUnProject
                v_gl = (height - 1) - v
                # --------------------------

                if depth < 1.0:
                    world_x, world_y, world_z = gluUnProject(
                        u, v_gl, depth,  # <-- Use v_gl
                        modelview, projection, viewport
                    )
                    world_coords[v_idx, u_idx] = [world_x, world_y, world_z]
                else:
                    world_coords[v_idx, u_idx] = [0, 0, 0]

        return world_coords, modelview, projection, viewport