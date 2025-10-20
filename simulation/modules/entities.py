import numpy as np
from typing import List, Union

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

    def get_vertices(self) -> np.ndarray:
        """Returns the static, local-space vertices."""
        return self.VERTICES
    
    def get_world_vertices(self) -> np.ndarray:
        """
        Returns the 8 vertices of the cube in their
        final world-space positions.
        """
        return self.VERTICES + self.position
    
    def get_world_surface_points(self, density: int = 5) -> np.ndarray:
        """
        Generates a dense grid of points on all 6 faces of the cube.
        
        :param density: How many points per edge (e.g., 5 -> 5x5 grid per face)
        :return: A numpy array of (N, 3) world-space points.
        """
        all_points = []
        # `np.linspace` creates an array of `density` evenly spaced
        # numbers between -0.5 and 0.5 (the cube's local bounds)
        s = np.linspace(-0.5, 0.5, density)
        
        # Create the 6 faces
        # (X, Y, Z)
        
        # 1. Positive Z face (+0.5 in Z)
        for x in s:
            for y in s:
                all_points.append([x, y, 0.5])
                
        # 2. Negative Z face (-0.5 in Z)
        for x in s:
            for y in s:
                all_points.append([x, y, -0.5])
        
        # 3. Positive X face (+0.5 in X)
        for y in s:
            for z in s:
                all_points.append([0.5, y, z])
                
        # 4. Negative X face (-0.5 in X)
        for y in s:
            for z in s:
                all_points.append([-0.5, y, z])
        
        # 5. Positive Y face (+0.5 in Y)
        for x in s:
            for z in s:
                all_points.append([x, 0.5, z])
                
        # 6. Negative Y face (-0.5 in Y)
        for x in s:
            for z in s:
                all_points.append([x, -0.5, z])
        
        # Convert to a single (N, 3) numpy array and
        # add the cube's world position, just like before.
        return np.array(all_points, dtype=float) + self.position

class Point(Entity):
    """
    A simple 3D point entity, e.g., for a radar hit.
    """
    def __init__(self, position: np.ndarray, color: list = [1.0, 0.0, 0.0]):
        self.position = position
        self.color = color
