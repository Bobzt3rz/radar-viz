import numpy as np
from typing import Tuple

from .world import World
from .camera import Camera
from .radar import Radar
from .cube import Cube

TEXTURE_CHECKERBOARD = "/home/bobberman/programming/radar/radar-viz/simulation2/assets/checkerboard.png"
TEXTURE_OPTICAL_FLOW = "/home/bobberman/programming/radar/radar-viz/simulation2/assets/optical_flow_texture.png"
TEXTURE_BACKGROUND = "/home/bobberman/programming/radar/radar-viz/simulation2/assets/background-buildings.jpg"
TEXTURE_FLOOR = "/home/bobberman/programming/radar/radar-viz/simulation2/assets/road.jpg"

def setup_simulation(delta_t: float = 1/60.0) -> Tuple[World, Camera, Radar]:
    """Initializes the world and creates simulation entities."""
    world = World(delta_t=delta_t)

    rig_linear_velocity = np.array([0.0, 0.0, 2.0]) 
    
    # Define a new angular velocity
    # Let's yaw (turn) 10 degrees/sec around the Y-axis (Y is down)
    yaw_rate_rad_s = np.radians(10.0)
    rig_angular_velocity = np.array([0.0, yaw_rate_rad_s, 0.0])

    camera = Camera(
        position=np.array([0.0, 0.0, 0.0]), velocity=rig_linear_velocity, angular_velocity=rig_angular_velocity,
        # Adjust cx, cy if Y-down origin was used during calibration
        fx=800.0, fy=800.0, cx=1280/2, cy=720/2, image_width=1280, image_height=720
    )
    radar = Radar(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=rig_linear_velocity,
        angular_velocity=rig_angular_velocity,
        fov_azimuth_deg=90, fov_elevation_deg=40, max_range=40
        
    )
    angle_radians = np.radians(45)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    rotation_matrix = np.array([
        [ cos_a,  0,  sin_a],
        [     0,  1,      0],
        [-sin_a,  0,  cos_a]
    ], dtype=np.float32)
    target_cube = Cube(
        position=np.array([0.0, 0.0, 5.0]), # Closer
        velocity=np.array([2.0, 0.0, 0.0]), # Slower
        rotation=rotation_matrix,
        size=0.5,
        texture_path=TEXTURE_CHECKERBOARD
    )
    static_cube = Cube(
        position=np.array([-1.5, -1, 5.0]),
        velocity=np.array([0.001, 0.0, 0.0]),
        size=0.5,
        texture_path=TEXTURE_CHECKERBOARD
    )
    cube_2 = Cube(
        position=np.array([0.0, 1.0, 5.0]),
        velocity=np.array([-2.0, 0.0, 0.0]),
        rotation=rotation_matrix,
        size=0.5,
        texture_path=TEXTURE_CHECKERBOARD
    )

    background_wall = Cube(
        position=np.array([0.0, 0.0, 500.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size=500.0,
        texture_repeat=1.0,
        texture_path=TEXTURE_BACKGROUND
    )

    floor = Cube(
        position=np.array([0.0, 300.0, 200.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size=500.0,
        texture_repeat=40.0,
        texture_path=TEXTURE_FLOOR
    )

    world.add_entity(camera)
    world.add_entity(radar)
    world.add_entity(target_cube)
    world.add_entity(static_cube)
    world.add_entity(cube_2)
    world.add_entity(background_wall)
    world.add_entity(floor)


    return world, camera, radar