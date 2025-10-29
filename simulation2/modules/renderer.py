import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from typing import Dict, Any
from PIL import Image

from .entity import Entity
from .cube import Cube
from .world import World
from .types import Matrix4x4

# --- Simple Shaders ---
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord; // <-- Added Tex Coord Input

uniform mat4 mvp;

out vec2 TexCoord; // <-- Pass UVs to fragment shader

void main()
{
    gl_Position = mvp * vec4(aPos, 1.0);
    TexCoord = aTexCoord; // <-- Pass through
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord; // <-- Receive UVs from vertex shader

uniform sampler2D texture1; // <-- Texture Sampler uniform

void main()
{
    FragColor = texture(texture1, TexCoord); // <-- Sample texture using UVs
    // FragColor = vec4(TexCoord, 0.0, 1.0); // <-- Uncomment to debug UVs (Red/Green gradient)
}
"""

class Renderer:
    """ Initializes with World state and renders using PyOpenGL. """
    def __init__(self, world: World, width: int = 1280, height: int = 720, title:str = "Simulation"):
        print("Initializing Renderer...")
        self.world = world
        self.width = width
        self.height = height

        # --- Initialize GLFW ---
        if not glfw.init():
            raise Exception("GLFW cannot be initialized!")
        
        # --- Create Window ---
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE) # For MacOS

        self.window = glfw.create_window(self.width, self.height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created!")

        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)

        # --- OpenGL Settings ---
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.2, 1.0) # Dark blue background

        # --- Compile Shaders ---
        try:
            vs = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
            fs = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            self.shader = shaders.compileProgram(vs, fs)
            glDeleteShader(vs)
            glDeleteShader(fs)
        except RuntimeError as e:
            print(f"Shader Error: {e}")
            glfw.terminate()
            raise

        # --- Get Uniform Locations ---
        self.mvp_loc = glGetUniformLocation(self.shader, "mvp")
        self.texture_loc = glGetUniformLocation(self.shader, "texture1") # <-- Get texture uniform location

        # --- OpenGL Coordinate System Conversion ---
        self.gl_conv: Matrix4x4 = np.diag([1,1,-1,1]).astype(np.float32)

        # --- Load Texture ---
        # We now load textures on-demand in _init_buffers
        self.textures: Dict[str, int] = {} # Stores {path: tex_id}

        # --- GL Buffer Initialization ---
        self.render_data: Dict[Entity, Dict[str, Any]] = {}
        self._init_buffers() # This now needs to handle UVs
        print("Renderer Initialization Complete.")

    def _load_texture_file(self, texture_path: str) -> int:
        """
        Loads a texture from a file path, stores its OpenGL ID,
        and returns the ID. If already loaded, returns stored ID.
        """
        # Return the cached ID if we've loaded this texture before
        if texture_path in self.textures:
            return self.textures[texture_path]
        
        # Check for empty path
        if not texture_path:
            return 0 # 0 is an invalid texture ID

        try:
            # This logic is moved from __init__
            texture_image = Image.open(texture_path).convert("RGB")
            texture_data = np.array(texture_image, dtype=np.uint8)
            texture_image.close()

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.width, texture_image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
            
            print(f"  Loaded texture: {texture_path} (ID: {tex_id})")
            self.textures[texture_path] = tex_id # Store the new ID
            return tex_id
        except Exception as e:
            print(f"Error loading texture {texture_path}: {e}")
            self.textures[texture_path] = 0 # Cache 0 to prevent re-trying
            return 0


    def _framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        self.width = width # Update width/height for projection matrix aspect ratio if needed
        self.height = height


    def _init_buffers(self):
        print("  Initializing OpenGL buffers...")
        for e in self.world.entities:
            if isinstance(e, Cube):
                # --- Get vertices AND UVs ---
                v_pos, v_uv, indices = e.get_mesh_data()

                # --- Interleave position and UV data ---
                # Shape: (24, 3) + (24, 2) -> (24, 5) -> (120,) flattened
                interleaved_data = np.hstack((v_pos, v_uv)).astype(np.float32).flatten()
                vertex_size_bytes = (3 + 2) * 4 # 3 pos floats + 2 uv floats, 4 bytes each

                vao,vbo,ebo = glGenVertexArrays(1), glGenBuffers(1), glGenBuffers(1)
                glBindVertexArray(vao)

                # --- VBO with interleaved data ---
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, interleaved_data.nbytes, interleaved_data, GL_STATIC_DRAW)

                # --- EBO ---
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

                # --- Vertex Attributes ---
                # Position Attribute (location = 0)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size_bytes, ctypes.c_void_p(0)) # Offset 0

                # Texture Coordinate Attribute (location = 1)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertex_size_bytes, ctypes.c_void_p(3 * 4)) # Offset after 3 pos floats (12 bytes)

                # Load the texture specified by the cube
                texture_id_to_use = 0
                if e.texture_path:
                    texture_id_to_use = self._load_texture_file(e.texture_path)

                # --- Unbind ---
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glBindVertexArray(0)

                self.render_data[e] = {
                    "vao": vao, "vbo": vbo, "ebo": ebo,
                    "indices_count": len(indices),
                    "texture_id": texture_id_to_use
                 }
                print(f"    Created textured buffers for Cube (VAO: {vao})")
        print("  Buffer initialization complete.")

    def should_close(self) -> bool:
        """ Check if the window should close. """
        return glfw.window_should_close(self.window)

    def render_scene(self, near=0.1, far=100.0):
        cam = self.world.camera; assert cam is not None
        glClear(int(GL_COLOR_BUFFER_BIT)| int(GL_DEPTH_BUFFER_BIT))
        view = self.gl_conv @ cam.get_pose_world_to_local()
        proj = cam.get_opengl_projection_matrix(near, far)

        glUseProgram(self.shader)

        # --- Set the texture sampler uniform ONCE (assumes only unit 0 used) ---
        glUniform1i(self.texture_loc, 0) # Tell shader texture1 is on texture unit 0

        for e, data in self.render_data.items():
            if "vao" in data and "texture_id" in data: # Check we have data for this entity
                model = e.get_pose_local_to_world(); mvp = proj @ view @ model
                glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp.T)

                # --- Bind Texture ---
                texture_id = data["texture_id"]
                if texture_id > 0: # Check if a valid texture was loaded
                    glActiveTexture(GL_TEXTURE0) # Activate texture unit 0
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                # ------------------

                # --- Draw Call ---
                glBindVertexArray(data["vao"])
                glDrawElements(GL_TRIANGLES, data["indices_count"], GL_UNSIGNED_INT, None)

                # --- Unbind ---
                glBindVertexArray(0)
                if texture_id > 0:
                    glBindTexture(GL_TEXTURE_2D, 0) # Optional unbind

        glUseProgram(0)

    def capture_frame(self) -> np.ndarray:
        """ Reads the current framebuffer pixels into a NumPy array. """
        # Ensure we're reading from the back buffer (the one we just drew to)
        glReadBuffer(GL_BACK)
        # Get viewport dimensions to read the correct size
        # Using self.width/height assumes they are up-to-date via callback
        width, height = self.width, self.height
        # Allocate buffer for pixel data
        # GL_UNSIGNED_BYTE is standard for 8-bit color channels
        # GL_RGB means 3 channels
        buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

        # Convert the raw buffer to a NumPy array
        # Reshape to (height, width, 3 channels)
        image = np.frombuffer(buffer, dtype=np.uint8 # type: ignore
                ).reshape(height, width, 3)

        # OpenGL's origin (0,0) is at the bottom-left, while most image libraries
        # (and NumPy indexing for images) assume top-left. So, flip vertically.
        image_flipped = np.flipud(image)

        return image_flipped

    def swap_buffers_and_poll_events(self):
         glfw.swap_buffers(self.window)
         glfw.poll_events()

    def cleanup(self):
        """ Release OpenGL resources and terminate GLFW. """
        print("Cleaning up Renderer...")
        # Delete buffers
        for entity, data in self.render_data.items():
            if "vao" in data: glDeleteVertexArrays(1, [data["vao"]])
            if "vbo" in data: glDeleteBuffers(1, [data["vbo"]])
            if "ebo" in data: glDeleteBuffers(1, [data["ebo"]])
        
        # --- Delete Textures ---
        texture_ids = [tex_id for tex_id in self.textures.values() if tex_id > 0]
        if texture_ids:
            glDeleteTextures(texture_ids)
            print(f"  Deleted {len(texture_ids)} textures.")
        self.textures = {}

        if hasattr(self, 'shader') and self.shader: glDeleteProgram(self.shader)
        glfw.terminate()
        print("Cleanup Complete.")