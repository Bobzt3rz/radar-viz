import numpy as np
from .entity import Entity
from .types import Vector3, Matrix3x3, Matrix4x4

class Camera(Entity):
    """
    An entity representing a calibrated camera with intrinsic parameters.
    Can generate an OpenGL projection matrix.
    """
    def __init__(self,
                 position: Vector3,
                 velocity: Vector3,
                 # --- Intrinsics ---
                 fx: float, fy: float,  # Focal lengths (alpha, beta in docs)
                 cx: float, cy: float,  # Principal point (x0, y0 in docs)
                 # --- Image Dimensions ---
                 image_width: int,
                 image_height: int,
                 skew: float = 0.0,     # Skew factor (s in docs)
                 rotation: Matrix3x3 = np.eye(3),
                 angular_velocity: Vector3 = np.zeros(3)):

        super().__init__(position, velocity, rotation, angular_velocity)

        # Store intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.skew = skew
        self.image_width = image_width
        self.image_height = image_height

    def get_opengl_projection_matrix(self, near_clip: float, far_clip: float) -> Matrix4x4:
        """
        Creates an OpenGL projection matrix from the camera's intrinsics.

        Uses the method described by Kyle Simek:
        https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

        Parameters:
        ----------
        near_clip : float
            Distance to the near clipping plane (must be positive).
        far_clip : float
            Distance to the far clipping plane (must be positive).

        Returns:
        -------
        Matrix4x4
            The 4x4 OpenGL projection matrix (row-major numpy array).
        """
        # 1. Build the Persp matrix (Eq 5 from the HTML source)
        # Note the negations and -1 based on the source's convention
        A = near_clip + far_clip
        B = near_clip * far_clip
        persp_matrix = np.array([
            [self.fx, self.skew, -self.cx,        0],
            [       0, self.fy, -self.cy,        0],
            [       0,        0,      A,        B],
            [       0,        0,     -1,        0]
        ], dtype=np.floating)

        # 2. Build the Ortho/NDC matrix (Eq 7 & 8 from the HTML source)
        # We assume the standard computer vision origin (top-left)
        # This corresponds to glOrtho(0, W, H, 0, near, far)
        left = 0
        right = self.image_width
        bottom = self.image_height # Y points down, so bottom > top
        top = 0

        tx = -(right + left) / (right - left)
        ty = -(top + bottom) / (top - bottom)
        tz = -(far_clip + near_clip) / (far_clip - near_clip)

        ndc_matrix = np.array([
            [2 / (right - left),                  0,                 0, tx],
            [                 0, 2 / (top - bottom),                 0, ty],
            [                 0,                  0, -2 / (far_clip - near_clip), tz],
            [                 0,                  0,                 0, 1]
        ], dtype=np.floating)

        # 3. Combine them: Proj = NDC * Persp (Eq 16 from the HTML source)
        # The result matches the final matrix in Eq 16
        # Note: numpy uses row-major, OpenGL uses column-major.
        # If passing directly to low-level GL functions (like glUniformMatrix4fv),
        # you might need to transpose this matrix first. PyOpenGL often handles this.
        proj_matrix = ndc_matrix @ persp_matrix

        return proj_matrix