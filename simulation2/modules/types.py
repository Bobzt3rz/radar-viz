from numpy.typing import NDArray
import numpy as np

# --- Type Aliases for clarity ---
# Use np.floating to be general for float32, float64, etc.
Vector3 = NDArray[np.floating]
Matrix3x3 = NDArray[np.floating]
Matrix4x4 = NDArray[np.floating]
# float32 with 2 channels (dx, dy)
FlowField = NDArray[np.float32]