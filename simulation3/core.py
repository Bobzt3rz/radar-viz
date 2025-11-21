import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Transform:
    """
    Represents a 6DOF pose transformation.
    Conventions: T_AB transforms a point in Frame B to Frame A.
    P_A = T_AB @ P_B
    """
    R: np.ndarray  # Shape (3, 3)
    t: np.ndarray  # Shape (3, 1)

    def to_matrix(self) -> np.ndarray:
        """Returns the 4x4 homogeneous transformation matrix."""
        mat = np.eye(4)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.t.flatten()
        return mat

    @staticmethod
    def identity() -> 'Transform':
        return Transform(np.eye(3), np.zeros((3,1)))

    @staticmethod
    def from_matrix(mat: np.ndarray) -> 'Transform':
        return Transform(mat[:3, :3], mat[:3, 3].reshape(3,1))

@dataclass
class RealPoint:
    """Ground Truth simulated point (God View)."""
    point_id: int
    position_w: np.ndarray  # World Frame (x, y, z), Shape (3, 1)
    velocity_w: np.ndarray  # World Frame (vx, vy, vz), Shape (3, 1)

@dataclass
class RadarObservation:
    """Simulated Radar Return (Sensor View)."""
    point_id: int
    position_r: np.ndarray      # Point position in Radar Frame, Shape (3, 1)
    radial_velocity: float      # Scalar Doppler velocity (m/s)
    radial_unit_vec: np.ndarray # Unit vector from Radar Origin to Point, Shape (3, 1)

@dataclass
class CameraObservation:
    """Simulated Camera Return (Sensor View)."""
    point_id: int
    uv: np.ndarray            # Pixel coordinates, Shape (2, 1)
    depth: float              # Depth in Camera Z (m)
    normalized_uv: np.ndarray # Normalized image plane coordinates (u, v), Shape (2, 1)