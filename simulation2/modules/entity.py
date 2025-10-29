from .types import Vector3, Matrix3x3, Matrix4x4
import numpy as np

class Entity:
    """
    A base class for any object in the virtual world (e.g., car, camera, radar).
    
    Its state (position, velocity) is defined in the World Coordinate System.
    """
    def __init__(self, 
                 position: Vector3, 
                 velocity: Vector3, 
                 rotation: Matrix3x3 = np.eye(3),
                 angular_velocity: Vector3 = np.zeros(3)):
        
        # --- State in World Coordinates ---
        
        # Position (x, y, z) in the world
        self.position: Vector3 = np.asarray(position, dtype=np.floating)
        
        # Linear velocity (vx, vy, vz) in the world
        self.velocity: Vector3 = np.asarray(velocity, dtype=np.floating)
        
        # Orientation (3x3 rotation matrix) in the world
        self.rotation: Matrix3x3 = np.asarray(rotation, dtype=np.floating)

        # Angular velocity (wx, wy, wz) in world coordinates (rad/s)
        self.angular_velocity: Vector3 = np.asarray(angular_velocity, dtype=np.floating)
        
        # --- 4x4 Pose Matrices ---
        
        # Places the entity INTO the world (Local -> World)
        self.M_local_to_world: Matrix4x4 = np.eye(4)
        
        # Brings the world into the entity's frame (World -> Local)
        # This is the "View Matrix" for cameras
        self.M_world_to_local: Matrix4x4 = np.eye(4)
        
        # Initialize the pose matrices
        self._update_pose_matrices()

    def _update_pose_matrices(self):
        """
        Private helper to rebuild the 4x4 pose matrices from the
        current position and rotation.
        """
        # Build M_local_to_world (R | T)
        self.M_local_to_world = np.eye(4)
        self.M_local_to_world[0:3, 0:3] = self.rotation
        self.M_local_to_world[0:3, 3] = self.position
        
        # Build M_world_to_local (R.T | -R.T @ T)
        # This is the inverse of M_local_to_world
        self.M_world_to_local = np.linalg.inv(self.M_local_to_world)

    def update(self, delta_t: float):
        """
        Updates the entity's state based on its velocity.
        This is a simple linear physics model.
        """
        # 1. Update position based on velocity
        self.position += self.velocity * delta_t

        angular_speed = np.linalg.norm(self.angular_velocity)
        
        if angular_speed > 1e-6: # Only rotate if there's non-zero velocity
            # Amount to rotate this frame
            theta = angular_speed * delta_t
            
            # Normalized axis of rotation
            axis = self.angular_velocity / angular_speed
            
            # Create the 3x3 rotation matrix for this delta_t
            # Using Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            R_delta = np.eye(3) + (sin_theta * K) + ((1 - cos_theta) * (K @ K))
            
            # Apply the rotation: R_new = R_delta @ R_old
            # This rotates the entity in the world frame
            self.rotation = R_delta @ self.rotation
        
        # 2. Rebuild the pose matrices with the new position
        self._update_pose_matrices()

    def get_pose_world_to_local(self) -> Matrix4x4:
        """
        Returns the 4x4 matrix that transforms points
        from the World to this Entity's local frame.
        (e.g., the Camera's View Matrix)
        """
        return self.M_world_to_local

    def get_pose_local_to_world(self) -> Matrix4x4:
        """
        Returns the 4x4 matrix that transforms points
        from this Entity's local frame to the World.
        """
        return self.M_local_to_world