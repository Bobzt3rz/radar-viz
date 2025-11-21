from numpy.typing import NDArray
import numpy as np
from enum import Enum
from typing import Tuple

# --- Type Aliases for clarity ---
# Use np.floating to be general for float32, float64, etc.
Vector3 = NDArray[np.floating]
Matrix3x3 = NDArray[np.floating]
Matrix4x4 = NDArray[np.floating]
# float32 with 2 channels (dx, dy)
FlowField = NDArray[np.float32]

# note swap random/multipath enum if using opengl
class NoiseType(Enum):
    REAL = 0              # A real point from an object
    MULTIPATH_GHOST = 1   # A "ghost" reflection of a real object
    RANDOM_CLUTTER = 2    # Uniformly random noise
    

# (vel_mag, vel_err, disp_err, noise_type, pos_3d_radar, vel_3d_radar, vel_3d_world, object_id)
DetectionTuple = Tuple[float, float, float, NoiseType, np.ndarray, np.ndarray, np.ndarray, int]