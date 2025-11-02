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

class NoiseType(Enum):
    REAL = 0              # A real point from an object
    RANDOM_CLUTTER = 1    # Uniformly random noise
    MULTIPATH_GHOST = 2   # A "ghost" reflection of a real object
    ROAD_CLUTTER = 3      # Spatially/velocit-correlated road noise

# (vel_mag, vel_err, disp_err, noise_type, pos_3d_radar, vel_3d_radar, vel_3d_world)
DetectionTuple = Tuple[float, float, float, NoiseType, np.ndarray, np.ndarray, np.ndarray]