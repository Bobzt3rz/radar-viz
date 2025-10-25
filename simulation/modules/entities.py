import numpy as np
from typing import List, Union, Optional
from OpenGL.GL import *
from PIL import Image

class Entity:
    """
    Represents a single object in the simulation world.

    This is a base class that defines the core properties of any object that
    can exist in our world, namely its position and velocity.
    """
    def __init__(self, position: Union[List[float], np.ndarray], velocity: Union[List[float], np.ndarray]):
        """
        Initializes an Entity.
        Args:
            position: A 3D vector [x, y, z] for the initial position.
            velocity: A 3D vector [vx, vy, vz] for the initial velocity.
        """
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)

    def update(self, dt: float):
        """
        Updates the entity's state over a time step 'dt'.
        This is the core physics calculation for the object itself.
        """
        self.position += self.velocity * dt
    
    def draw(self):
        """
        Draws the entity using the current OpenGL matrix state.
        This must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class Cube(Entity):
    """
    A specific type of Entity that has the geometry of a cube.
    It inherits position and velocity from the base Entity class.
    """
    VERTICES = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)

    FACES = (
        (1, 5, 6, 2), (4, 0, 3, 7), (3, 2, 6, 7),
        (4, 5, 1, 0), (5, 4, 7, 6), (0, 1, 2, 3),
    )

    # These map the 4 corners of a texture to the 4 vertices of a quad
    TEX_COORDS = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    def __init__(self, position: List[float], velocity: List[float], 
                 texture_path: Optional[str] = None,
                 scale: List[float] = [1.0, 1.0, 1.0]):
        # Call the parent class (Entity) constructor
        super().__init__(position=position, velocity=velocity)
        # Set to white so texture isn't tinted
        self.color = [1.0, 1.0, 1.0]
        # A unique color to identify this object in a render
        self.id_color = [0.0, 0.0, 0.0]
        self.scale = np.array(scale, dtype=np.float32)

        if texture_path:
            self.tex_id = self._load_texture(texture_path)
        else:
            self.tex_id = None

    def _load_texture(self, texture_path: str) -> int:
        """Loads a texture from a file and returns its OpenGL ID."""
        try:
            img = Image.open(texture_path)
            img.load()
            img_data = np.array(list(
                img.getdata() # type: ignore
            ), np.uint8)
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Upload the texture data
            if img.mode == "RGB":
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            elif img.mode == "RGBA":
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            else:
                print(f"Warning: Texture '{texture_path}' is not RGB/RGBA. Skipping.")
                return -1
                
            print(f"Loaded texture: {texture_path}")
            return tex_id
            
        except Exception as e:
            print(f"Error loading texture {texture_path}: {e}")
            return -1 # Return an invalid ID

    def draw(self):
        """ Draws the cube at its position with texture. """
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glScalef(self.scale[0], self.scale[1], self.scale[2])
        
        if self.tex_id is not None and self.tex_id != -1:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.tex_id)
            glColor3fv(self.color) # Use white to show full texture color
        else:
            # Fallback for no texture
            glColor3fv([0.5, 0.5, 0.5]) # Default gray

        glBegin(GL_QUADS)
        for i_face, face in enumerate(self.FACES):
            for i_vertex, vertex_index in enumerate(face):
                if self.tex_id:
                    # Apply texture coordinate *before* vertex
                    glTexCoord2fv(self.TEX_COORDS[i_vertex])
                glVertex3fv(self.VERTICES[vertex_index])
        glEnd()
        
        if self.tex_id is not None and self.tex_id != -1:
            glDisable(GL_TEXTURE_2D)
            
        glPopMatrix()

    def draw_for_id(self):
        """ Draws the cube at its position with its unique ID color. """
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glScalef(self.scale[0], self.scale[1], self.scale[2])
        
        glColor3fv(self.id_color) # <-- Uses ID color
        glBegin(GL_QUADS)
        for face in self.FACES:
            for vertex_index in face:
                glVertex3fv(self.VERTICES[vertex_index])
        glEnd()

        glPopMatrix()

class Point(Entity):
    """
    A simple 3D point entity, e.g., for a radar hit.
    """
    def __init__(self, position: np.ndarray, color: list = [1.0, 0.0, 0.0]):
        self.position = position
        self.color = color
    
    def draw(self):
        """ Draws the point at its position. """
        glPointSize(1.0)
        glColor3fv(self.color)
        glBegin(GL_POINTS)
        glVertex3fv(self.position) # Points don't need translate/push/pop
        glEnd()
