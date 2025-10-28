import numpy as np
from typing import Tuple

from .world import World
from .camera import Camera
from .radar import Radar
from .cube import Cube

def setup_simulation(delta_t: float = 1/60.0) -> Tuple[World, Camera, Radar]:
    """Initializes the world and creates simulation entities."""
    world = World(delta_t=delta_t)

    camera = Camera(
        position=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 1.0]),
        # Adjust cx, cy if Y-down origin was used during calibration
        fx=800.0, fy=800.0, cx=1280/2, cy=720/2, image_width=1280, image_height=720
    )
    radar = Radar(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([0.0, 0.0, 1.0]),
        fov_azimuth_deg=90, fov_elevation_deg=40, max_range=100
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
        position=np.array([0.0, 0.0, 3.0]), # Closer
        velocity=np.array([1.0, 0.0, 0.0]), # Slower
        rotation=rotation_matrix,
        size=0.5
    )
    static_cube = Cube(
        position=np.array([-1.5, -0.5, 3.0]),
        velocity=np.array([0.001, 0.0, 0.0]),
        size=0.5
    )
    cube_2 = Cube(
        position=np.array([0.0, 1.0, 3.0]),
        velocity=np.array([-1.0, 0.0, 0.0]),
        rotation=rotation_matrix,
        size=0.5
    )

    world.add_entity(camera)
    world.add_entity(radar)
    world.add_entity(target_cube)
    world.add_entity(static_cube)
    world.add_entity(cube_2)

    return world, camera, radar