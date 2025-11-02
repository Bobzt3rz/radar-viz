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

    # --- 1. Define Ego Vehicle Motion ---
    # Moving forward at 10 m/s (36 kph)
    rig_linear_velocity = np.array([0.0, 0.0, 10.0]) 
    
    # Slightly turning (yawing) to the right at 2 deg/sec
    yaw_rate_rad_s = np.radians(2.0)
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
        fov_azimuth_deg=90, fov_elevation_deg=40, max_range=40.0
    )

   # Re-usable rotation matrix for moving cars
    angle_radians = np.radians(15) # 15 degree rotation
    cos_a, sin_a = np.cos(angle_radians), np.sin(angle_radians)
    rotation_matrix = np.array([
        [ cos_a,  0,  sin_a],
        [     0,  1,      0],
        [-sin_a,  0,  cos_a]
    ], dtype=np.float32)

    # --- 3. Define Scene Objects ---
    
    CAR_SIZE = 1.0
    CAR_Y_POS = CAR_SIZE / 2.0 # Place them at y=0.75 to sit on the floor

    # --- 5 MOVING OBJECTS ---

    # Moving 1: Slow car in right lane (you are overtaking)
    car_slow = Cube(
        position=np.array([4.0, CAR_Y_POS, 15.0]),  # Right lane, 40m ahead
        velocity=np.array([0.0, 0.0, 8.0]),   # 8 m/s (slower than you)
        rotation=rotation_matrix,
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )

    # Moving 2: Fast car in left lane (overtaking you)
    car_fast = Cube(
        position=np.array([-2.0, CAR_Y_POS, 5.0]), # Left lane, 25m ahead
        velocity=np.array([0.0, 0.0, 13.0]),  # 13 m/s (faster than you)
        rotation=rotation_matrix,
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )
    
    # Moving 3: Oncoming car in far-left lane
    car_oncoming = Cube(
        position=np.array([0.0, CAR_Y_POS, 20.0]), # Far left, 100m away
        velocity=np.array([0.0, 0.0, 9.0]), # High relative velocity
        rotation=rotation_matrix,
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )
    
    # Moving 4: Car changing lanes (right-to-left)
    car_lane_change = Cube(
        position=np.array([4.0, CAR_Y_POS, 20.0]),  # Starts in right lane, 60m ahead
        velocity=np.array([-1.0, 0.0, 9.0]),   # Moves left at 1 m/s, 9 m/s fwd
        rotation=rotation_matrix,
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )
    
    # Moving 5: Car matching your speed
    car_pacing = Cube(
        position=np.array([-6.0, CAR_Y_POS, 15.0]), # Left lane, 35m ahead
        velocity=np.array([0.0, 0.0, 10.0]),  # Same 10 m/s speed as you
        rotation=rotation_matrix,
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )

    # --- 5 STATIC OBJECTS ---
    
    # Static 1: Small sign, right shoulder
    static_sign_1 = Cube(
        position=np.array([2.0, CAR_Y_POS, 15.0]), # y=1.0 for 2m tall sign
        velocity=np.array([0.0, 0.0, 0.0]),
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )

    # Static 2: Small barrier, left shoulder
    static_barrier_1 = Cube(
        position=np.array([-6.0, CAR_Y_POS, 10.0]), # y=0.5 for 1m cube
        velocity=np.array([0.0, 0.0, 0.0]),
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )
    
    # Static 3: Large sign, right shoulder
    static_sign_2 = Cube(
        position=np.array([4.0, CAR_Y_POS, 10.0]), # y=1.5 for 3m tall sign
        velocity=np.array([0.0, 0.0, 0.0]),
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )
    
    # Static 4: Debris in center median (between lanes)
    static_debris = Cube(
        position=np.array([1.5, CAR_Y_POS, 10.0]), # y=0.25 for 0.5m cube
        velocity=np.array([0.0, 0.0, 0.0]),
        size=CAR_SIZE,
        texture_path=TEXTURE_CHECKERBOARD
    )
    
    # Static 5: Barrier, left shoulder, far away
    static_barrier_2 = Cube(
        position=np.array([-4.0, CAR_Y_POS, 15.0]), # y=0.5 for 1m cube
        velocity=np.array([0.0, 0.0, 0.0]),
        size=CAR_SIZE,
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
    world.add_entity(background_wall)
    world.add_entity(floor)

    # Add the 5 moving
    world.add_entity(car_slow)
    world.add_entity(car_fast)
    world.add_entity(car_oncoming)
    world.add_entity(car_lane_change)
    world.add_entity(car_pacing)
    
    # Add the 5 static
    world.add_entity(static_sign_1)
    world.add_entity(static_barrier_1)
    world.add_entity(static_sign_2)
    world.add_entity(static_debris)
    world.add_entity(static_barrier_2)
    



    return world, camera, radar