import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys

# ======================================================================
# 1. CORE GEOMETRY AND DRAWING (FIXED WINDING ORDER & COLOR)
# ======================================================================

VERTICES = np.array([
    [-1, -1, -1],  # 0
    [ 1, -1, -1],  # 1
    [ 1,  1, -1],  # 2
    [-1,  1, -1],  # 3
    [-1, -1,  1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1,  1],  # 6
    [-1,  1,  1],  # 7
], dtype=np.float32)

FACES = (
    (1, 5, 6, 2),  # Right (+X)
    (4, 0, 3, 7),  # Left (-X)
    (3, 2, 6, 7),  # Top (+Y)
    (4, 5, 1, 0),  # Bottom (-Y)
    (5, 4, 7, 6),  # Front (+Z)
    (0, 1, 2, 3),  # Back (-Z)
)

def draw_cube():
    """Draws the opaque, solid-colored cube."""
    
    # 1. Draw Solid Faces (Opaque part) - SINGLE COLOR FIX
    glColor3f(0.5, 0.5, 0.5) # Set a single color (Gray) for the entire object
    glBegin(GL_QUADS)
    for face in FACES:
        for vertex_index in face:
            glVertex3fv(VERTICES[vertex_index])
    glEnd()

# ======================================================================
# 2. OPENGL AND GLFW INITIALIZATION
# ======================================================================

def initialize_window(width, height):
    """Initializes GLFW and creates the window."""
    if not glfw.init():
        sys.exit(1)


    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)

    window = glfw.create_window(width, height, "GLFW OpenGL Scene (Opaque Cube)", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)
        
    glfw.make_context_current(window)
    return window

def setup_opengl(width, height, fov=45, near=0.1, far=50.0):
    """Sets up the initial OpenGL state, including the Projection Matrix (P)."""
    glViewport(0, 0, width, height)
    
    # --- Projection Matrix (P) ---
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, (width / height), near, far)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # General OpenGL settings
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    # Tell OpenGL to use Counter-Clockwise winding for front faces (standard)
    glFrontFace(GL_CCW) 

# ======================================================================
# 3. MAIN APPLICATION LOOP
# ======================================================================

def main_render_loop(window, width, height, initial_pos_cam):
    """The main loop that handles rendering and events."""
    
    # Setup OpenGL (Must happen after context is current)
    setup_opengl(width, height)
    
    # The cube's initial position (P_A: [X, Y, Z_forward])
    pos_x, pos_y, pos_z = initial_pos_cam[:3]

    while not glfw.window_should_close(window):
        # Poll for and process events
        glfw.poll_events()

        # --- Rendering Step ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glLoadIdentity()
        
        # Place the cube at the simulated P_A position.
        glTranslatef(pos_x, pos_y, -pos_z) 
        
        # Optional: Add rotation for aesthetics
        time = glfw.get_time()
        glRotatef(time * 50.0, 0, 1, 0) # Rotate around Y-axis

        # 4. Draw the object
        draw_cube()
        
        glfw.swap_buffers(window)

    glfw.terminate()

# ======================================================================
# 4. EXECUTION
# ======================================================================

if __name__ == '__main__':
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    
    # P_A from your simulation: [3.0, 0.5, 10.0, 1.0]
    INITIAL_CUBE_POS = np.array([3.0, 1.5, 10.0, 1.0])
    
    window = initialize_window(SCREEN_WIDTH, SCREEN_HEIGHT)
    main_render_loop(window, SCREEN_WIDTH, SCREEN_HEIGHT, INITIAL_CUBE_POS)
