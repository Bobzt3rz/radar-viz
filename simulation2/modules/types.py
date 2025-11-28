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
    SHIFTX = 3            # Shifted X by +1 or -1
    SHIFTY = 4            # Shifted Y by +1 or -1
    SHIFTZ = 5            # Shifted Z by +1 or -1
    SHIFTRADIAL = 6       # Shifted Radial Velocity by +1 or -1
    

# (0:vel_mag, 1:vel_err, 2:disp_err, 3:noise_type, 4:pos_3d_radar, 5:vel_3d_radar, 6:vel_3d_radar_gt, 7:object_id, 8:dx, 9:dy, 10:pos_3d_radar_gt, 11:angular_velocity_radar_gt, 12:center_gt_radar, 13:vel_3d_world_gt, 14:vel_3d_rigid)
DetectionTuple = Tuple[float, float, float, NoiseType, np.ndarray, np.ndarray, np.ndarray, int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]