import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
from typing import Tuple

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
            if isinstance(entity, Cube):
                self._draw_cube(entity)
            elif isinstance(entity, Point):
                self._draw_point(entity)

    def end_frame(self):
        """
        Call this once at the end of the frame to swap
        the buffers and display the result.
        """
        glfw.swap_buffers(self.window)

    def _draw_cube(self, cube_entity: Cube):
        """Draws a single cube entity. (Unchanged)"""
        glPushMatrix() 
        pos = cube_entity.position
        glTranslatef(pos[0], pos[1], pos[2])
        
        glColor3fv(cube_entity.color)
        glBegin(GL_QUADS)
        for face in cube_entity.FACES:
            for vertex_index in face:
                glVertex3fv(cube_entity.VERTICES[vertex_index])
        glEnd()
        
        glPopMatrix()
    
    def _draw_point(self, point_entity: Point):
        """Draws a single point entity."""
        
        # Set point size (how many pixels big)
        glPointSize(5.0)
        
        # Draw the single vertex
        glColor3fv(point_entity.color)
        glBegin(GL_POINTS)
        glVertex3fv(point_entity.position)
        glEnd()

    def shutdown(self):
        """Terminates GLFW. (Unchanged)"""
        glfw.terminate()