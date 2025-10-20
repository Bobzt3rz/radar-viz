# sensor_rig.py
from .camera import Camera
from .radar import Radar

class EgoSensorRig:
    """
    Represents the sensor suite on the "ego" vehicle.
    It owns the camera and radar objects and knows their
    static relationship to each other.
    """
    def __init__(self):
        # Create a camera, positioned relative to the rig's origin.
        # For simplicity, we'll start with the camera you had.
        self.camera = Camera(
            position=[0, 5, -20],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
        
        self.radar = Radar(
            position=[0, 5, -20],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )

    def get_camera(self) -> Camera:
        """Returns the primary camera for rendering."""
        return self.camera
    
    def get_radar(self) -> Radar:
        """Returns the radar for rendering."""
        return self.radar

    def update_rig_pose(self, new_position, new_target, new_up):
        """
        This method would be called by your simulation to move the
        entire rig. It updates the state of all sensors.
        """
        # In a real system, you'd apply a transformation matrix.
        # For this example, we just set the camera's pose directly.
        self.camera.position = new_position
        self.camera.target = new_target
        self.camera.up = new_up

        self.radar.position = new_position
        self.radar.target = new_target
        self.radar.up = new_up