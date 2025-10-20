from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class Camera:
    """
    Represents a camera with its pose (extrinsics) and
    properties (intrinsics) for rendering.
    """
    def __init__(self, position: list, target: list, up: list):
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