import numpy as np
from typing import List, Tuple

from .entity import Entity
from .types import Vector3, Matrix3x3

class Cube(Entity):
    """
    A simple cube entity that inherits its position, velocity,
    and pose from the base Entity class.
    """
    def __init__(self, 
                 position: Vector3, 
                 velocity: Vector3, 
                 rotation: Matrix3x3 = np.eye(3), 
                 size: float = 1.0,
                 texture_repeat: float = 1.0):
        
        # --- Call the parent class's constructor ---
        # This initializes self.position, self.velocity, self.rotation,
        # and all the pose matrices.
        super().__init__(position, velocity, rotation)
        
        # --- Cube-specific properties ---
        self.size: float = size
        self.texture_repeat: float = texture_repeat

    def get_local_vertices(self) -> List[Vector3]:
        """
        Returns the 8 vertices of the cube in its
        own local coordinate system (centered at its origin).
        """
        s = self.size / 2.0
        return [
            np.array([-s, -s, -s]),
            np.array([ s, -s, -s]),
            np.array([ s,  s, -s]),
            np.array([-s,  s, -s]),
            np.array([-s, -s,  s]),
            np.array([ s, -s,  s]),
            np.array([ s,  s,  s]),
            np.array([-s,  s,  s]),
        ]
    
    def get_mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Returns vertices (pos), texture coords (uv), and indices. """
        s = np.float32(self.size / 2.0)
        # Unique vertex positions (8 corners)
        v = np.array([
            [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s], # Back face corners (0-3)
            [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s]  # Front face corners (4-7)
        ], dtype=np.float32)

        # Unique texture coordinates (e.g., corners of a square texture)
        # Decide how the texture maps to the cube faces
        t = np.array([
            [0.0, 0.0], [self.texture_repeat, 0.0], [self.texture_repeat, self.texture_repeat], [0.0, self.texture_repeat] # Standard square UV coords
        ], dtype=np.float32)

        # Define vertices and UVs *per face* (4 vertices per face, 6 faces = 24 vertices total)
        # Each line: [vertex_index_from_v, texcoord_index_from_t]
        # Order: Back, Front, Left, Right, Bottom, Top
        face_vertex_indices = [
            [0,0], [1,1], [2,2], [3,3], # Back
            [4,0], [5,1], [6,2], [7,3], # Front (Use same UVs for simplicity, or flip?)
            [4,0], [0,1], [3,2], [7,3], # Left
            [1,0], [5,1], [6,2], [2,3], # Right
            [4,0], [5,1], [1,2], [0,3], # Bottom
            [3,0], [2,1], [6,2], [7,3]  # Top
        ]

        vertices_textured = np.array([v[vi[0]] for vi in face_vertex_indices], dtype=np.float32)
        uvs_textured = np.array([t[vi[1]] for vi in face_vertex_indices], dtype=np.float32)

        # Indices for drawing triangles (2 triangles per face, 6 faces = 36 indices)
        # Vertices are now arranged per-face (0-3: Back, 4-7: Front, etc.)
        indices = np.array([
             0,  1,  2,  0,  2,  3, # Back
             4,  5,  6,  4,  6,  7, # Front
             8,  9, 10,  8, 10, 11, # Left
            12, 13, 14, 12, 14, 15, # Right
            16, 17, 18, 16, 18, 19, # Bottom
            20, 21, 22, 20, 22, 23  # Top
        ], dtype=np.uint32)

        # Return separate arrays for positions and UVs
        return vertices_textured, uvs_textured, indices

    def get_world_vertices(self) -> List[Vector3]:
        """
        Returns the 8 vertices of the cube in
        World Coordinates.
        """
        local_verts = self.get_local_vertices()
        world_verts = []
        
        # Get the matrix that transforms local points to the world
        M_L_to_W = self.get_pose_local_to_world()
        
        for vert in local_verts:
            # Convert to 4x1 homogeneous coordinates for matrix multiplication
            vert_h = np.append(vert, 1.0)
            
            # Transform to world
            world_vert_h = M_L_to_W @ vert_h
            
            # Convert back to 3D and add to list
            world_verts.append(world_vert_h[0:3])
            
        return world_verts