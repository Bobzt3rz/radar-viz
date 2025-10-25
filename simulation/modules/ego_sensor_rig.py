import numpy as np

from .camera import Camera
from .radar import Radar

class EgoSensorRig:
    """
    Represents the sensor suite on the "ego" vehicle.
    It owns the camera and radar objects and knows their
    static relationship to each other.
    """
    def __init__(self):
        # 1. Store the rig's own pose, todo, make so radar is not in same position
        self.position = np.array([0, 1, -10], dtype=float)
        self.target = np.array([0, 0, 0], dtype=float)
        self.up = np.array([0, 1, 0], dtype=float)

        # 2. Create sensors using this pose
        self.camera = Camera(
            position=self.position,
            target=self.target,
            up=self.up
        )
        
        self.radar = Radar(
            position=self.position,
            target=self.target,
            up=self.up
        )

    def get_camera(self) -> Camera:
        """Returns the primary camera for rendering."""
        return self.camera
    
    def get_radar(self) -> Radar:
        """Returns the radar for rendering."""
        return self.radar

    def update(self, dt: float, translation_velocity: np.ndarray):
        """
        Updates the entire rig's pose based on velocity.
        
        :param dt: Time step.
        :param translation_velocity: A (3,) np.array for [vx, vy, vz].
        """
        # --- 1. Calculate Translational Change ---
        displacement = translation_velocity * dt
        
        # --- 2. Apply Change to Rig's Pose ---
        # For translation, both position and target move together.
        self.position += displacement
        self.target += displacement
        
        # --- 3. Propagate Pose to Sensors ---
        # This updates the state for the *next* render.
        self.camera.position = self.position
        self.camera.target = self.target
        self.camera.up = self.up # (Stays the same for translation)

        self.radar.position = self.position
        self.radar.target = self.target
        self.radar.up = self.up