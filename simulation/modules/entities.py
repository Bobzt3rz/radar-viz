import numpy as np
from typing import List, Union
from OpenGL.GL import *

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

    def __init__(self, position: List[float], velocity: List[float]):
        # Call the parent class (Entity) constructor
        super().__init__(position=position, velocity=velocity)
        # You could add cube-specific properties here, like color or size
        self.color = [0.5, 0.5, 0.5]

    def draw(self):
        """ Draws the cube at its position. """
        glPushMatrix() # Save the current matrix state
        
        # Apply the entity's position
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Draw the geometry
        glColor3fv(self.color)
        glBegin(GL_QUADS)
        for face in self.FACES:
            for vertex_index in face:
                glVertex3fv(self.VERTICES[vertex_index])
        glEnd()
        
        glPopMatrix() # Restore the matrix state

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
