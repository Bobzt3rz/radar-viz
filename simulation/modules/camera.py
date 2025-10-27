from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import Tuple, Dict

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

    def apply_projection_matrix(self, width: int, height: int):
        """
        Applies the Projection transformation by manually building the
        matrix from intrinsic parameters, consistent with the blog post:
        https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

        Uses the 'y down' convention suitable for top-left image origins.

        Args:
            width (int): The width of the viewport/image in pixels.
            height (int): The height of the viewport/image in pixels.
        """
        # Get intrinsic parameters based on the current viewport dimensions
        intrinsics = self.get_intrinsics(width, height)
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        znear = self.near_clip
        zfar = self.far_clip

        # Map intrinsics to K matrix elements (assuming K01 skew = 0, x0=y0=0)
        # K = [[fx,  0, cx],
        #      [ 0, fy, cy],
        #      [ 0,  0,  1]]
        K00 = fx
        K01 = 0  # Assuming zero skew
        K02 = cx
        K11 = fy
        K12 = cy
        x0 = 0 # Assuming image origin is 0
        y0 = 0 # Assuming image origin is 0

        # Build the projection matrix using the 'y down' formula from the blog post
        P = np.zeros((4, 4), dtype=np.float32)

        P[0, 0] = 2 * K00 / width
        P[0, 1] = -2 * K01 / width # Skew term (0 here)
        P[0, 2] = (width - 2 * K02 + 2 * x0) / width
        P[0, 3] = 0

        P[1, 1] = 2 * K11 / height # Positive for 'y down'
        P[1, 2] = (-height + 2 * K12 + 2 * y0) / height
        P[1, 3] = 0

        P[2, 2] = (-zfar - znear) / (zfar - znear) # Z mapping
        P[2, 3] = -2 * zfar * znear / (zfar - znear)

        P[3, 2] = -1 # Puts Z into W coordinate

        # --- Debug: Print the calculated matrix ---
        # print("\n--- Manually Calculated Projection Matrix ---")
        # print(P)
        # print("-------------------------------------------\n")

        # Set OpenGL matrix mode and load the calculated matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # OpenGL expects column-major, so we transpose P before loading
        glLoadMatrixf(P.T)

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
        self.apply_projection_matrix(width, height)
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
        self.apply_projection_matrix(width, height)
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
        
        This is a vectorized version of the original loop-based function.
        
        Returns:
            - world_coords (H, W, 3) np.ndarray
            - modelview matrix (np.ndarray)
            - projection matrix (np.ndarray)
            - viewport (np.ndarray)
        """
        # --- 1. Get Depth Buffer (as in original) ---
        # depth_buffer is (H, W), with (0,0) at top-left
        depth_buffer = self._get_depth_buffer(world, width, height)
        
        # --- 2. Get Matrices (as in original) ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        self.apply_projection_matrix(width, height)
        self.apply_view_matrix()
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # --- 3. Vectorized Un-projection ---
        
        # Define the viewport used (as in original)
        viewport = (0, 0, width, height)
        v_x, v_y, v_w, v_h = viewport

        # Create (u, v) grids (as in original)
        # u_grid: [0...W-1]
        # v_grid: [0...H-1] (0 is top)
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flip v to be bottom-left for gluUnProject (as in original)
        # v_gl_grid: [H-1...0]
        v_gl_grid = (height - 1) - v_grid
        
        # --- Replicate gluUnProject Math ---
        
        # 1. Map (u, v_gl, depth) to Normalized Device Coordinates (NDC)
        # x_ndc = (u_pixel - v_x) * (2.0 / v_w) - 1.0
        # y_ndc = (v_gl_pixel - v_y) * (2.0 / v_h) - 1.0
        # z_ndc = depth * 2.0 - 1.0
        x_ndc = (u_grid - v_x) * (2.0 / v_w) - 1.0
        y_ndc = (v_gl_grid - v_y) * (2.0 / v_h) - 1.0
        z_ndc = depth_buffer * 2.0 - 1.0
        
        # 2. Create (H, W, 4) array of homogeneous NDC coordinates
        ndc_coords = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=-1)

        # 3. Get Inverse MVP Matrix
        # As established, the row-major MVP matrix is (modelview @ projection)
        inv_mvp_matrix = np.linalg.inv(modelview @ projection)
        
        # 4. Transform NDC to World (Homogeneous)
        # P_world_row = P_ndc_row @ M_inv_mvp_row
        world_coords_homogeneous = ndc_coords @ inv_mvp_matrix
        
        # 5. Perspective Divide
        # Get the 'w' component (H, W, 1)
        w = world_coords_homogeneous[..., 3, np.newaxis]
        
        # Avoid division by zero
        w[w == 0] = 1.0
        
        # (H, W, 3)
        world_coords = world_coords_homogeneous[..., :3] / w
        
        # --- 4. Handle Background (as in original) ---
        # Create a mask for background pixels (where depth >= 1.0)
        background_mask = (depth_buffer >= 1.0)
        
        # Set background pixels to [0, 0, 0]
        world_coords[background_mask] = [0.0, 0.0, 0.0]

        return world_coords, modelview, projection, viewport
    
    def get_intrinsics(self, width: int, height: int) -> Dict[str, float]:
            """
            Calculates camera intrinsic parameters (fx, fy, cx, cy)
            based on the camera's FOV and the viewport size.
            """
            # Vertical FOV from camera object
            fov_y_rad = self.fov * np.pi / 180.0
            
            # Calculate focal lengths
            fy = (height / 2.0) / np.tan(fov_y_rad / 2.0)
            # Aspect ratio
            aspect = width / float(height)
            fx = fy * aspect # Assuming fov is vertical
            
            # Principal point (image center)
            cx = width / 2.0
            cy = height / 2.0
            
            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}