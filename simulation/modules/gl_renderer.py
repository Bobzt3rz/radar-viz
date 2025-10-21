import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
from typing import Tuple
import numpy as np

# Import the classes it needs to interact with
from .world import World
from .entities import Cube, Point
from .camera import Camera

class OpenGLRenderer:
    """
    Handles all OpenGL and GLFW rendering.
    It is completely separate from the simulation logic and
    can render multiple views into a single window.
    """
    def __init__(self, width: int, height: int, title: str):
        if not glfw.init():
            print("GLFW initialization failed")
            sys.exit(1)

        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            print("GLFW window creation failed")
            sys.exit(1)

        glfw.make_context_current(self.window)
        
        # Store window size
        self.window_width = width
        self.window_height = height
        
        self._setup_opengl()

    def _setup_opengl(self):
        """Sets up the initial *general* OpenGL state."""
        # Note: glViewport is NOT called here anymore, 
        # as it will be set by each view.

        # General OpenGL settings
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST) # Enable depth testing for 3D
        glFrontFace(GL_CCW)

    def should_close(self) -> bool:
        """Checks if the rendering window should close."""
        return glfw.window_should_close(self.window)

    def begin_frame(self):
        """
        Call this once at the beginning of the frame.
        It polls events and clears the entire window.
        """
        glfw.poll_events()
        # Set viewport to the whole window to clear it
        glViewport(0, 0, self.window_width, self.window_height)
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))

    def render_view(self, world: World, camera: Camera, viewport: Tuple[int, int, int, int]):
        """
        Renders the world from a camera's perspective into a
        specific part of the window.
        
        :param viewport: A tuple (x, y, width, height) defining the
                         rectangle in the window to draw to.
        """
        x, y, width, height = viewport
        
        # 1. --- Set the Viewport ---
        # This tells OpenGL to render into this specific rectangle
        glViewport(x, y, width, height)
        
        # 2. --- Setup Projection Matrix (from Camera Intrinsics) ---
        # Aspect ratio must be based on the viewport, not the window
        aspect_ratio = width / height
        camera.apply_projection_matrix(aspect_ratio)

        # 3. --- Setup View Matrix (from Camera Extrinsics) ---
        camera.apply_view_matrix()

        # 4. --- Draw all entities from the world ---
        for entity in world.entities:
            entity.draw()
    
    def read_viewport_pixels(self, viewport: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Reads the pixels from the specified viewport directly from the
        front buffer (after a render) and returns them as a NumPy array.
        
        :param viewport: A tuple (x, y, width, height) to read from.
        :return: A (height, width, 3) NumPy array of RGB pixels.
        """
        x, y, width, height = viewport
        
        # Set the pixel alignment
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        
        # Read the pixel data
        data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert bytes to a numpy array
        image = np.frombuffer(
            data, dtype=np.uint8 # type: ignore
        )
        
        # Reshape to (height, width, 3)
        # Note: OpenGL's origin (0,0) is the bottom-left, so we need
        # to flip the image vertically (using [::-1]) to get the
        # standard top-left origin for images.
        image = image.reshape((height, width, 3))[::-1, :, :]
        
        return image
    
    def draw_image_in_viewport(self, image_data: np.ndarray, viewport: Tuple[int, int, int, int]):
        """
        Draws a NumPy (H, W, 3) RGB image into a specific viewport.
        
        :param image_data: The (H, W, 3) np.uint8 array.
        :param viewport: A tuple (x, y, width, height) to draw into.
        """
        x, y, width, height = viewport
        H, W, _ = image_data.shape
        
        # 1. Set the viewport
        glViewport(x, y, width, height)
        
        # 2. Set the raster position (where to start drawing)
        # We'll just draw at the bottom-left corner (0,0) of the viewport
        glWindowPos2i(x, y) 
        
        # 3. Set pixel alignment (good practice for NumPy)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        
        # 4. Draw the pixels from the NumPy array
        # We must flip the image vertically (image_data[::-1, ...])
        # because OpenGL's (0,0) is bottom-left.
        glDrawPixels(W, H, GL_RGB, GL_UNSIGNED_BYTE, image_data[::-1, ...].tobytes())

    def end_frame(self):
        """
        Call this once at the end of the frame to swap
        the buffers and display the result.
        """
        glfw.swap_buffers(self.window)

    def shutdown(self):
        """Terminates GLFW. (Unchanged)"""
        glfw.terminate()