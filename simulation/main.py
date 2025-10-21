from modules.world import World
from modules.entities import Cube, Point
from modules.gl_renderer import OpenGLRenderer
from modules.ego_sensor_rig import EgoSensorRig
from modules.utils import save_as_ply
from modules.optical_flow import OpticalFlow

import sys

sys.path.append('..') # Add parent directory to path

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720

if __name__ == "__main__":
    # 1. Simulation Setup
    world = World()

    # Radar world
    radar_world = World()
    rig = EgoSensorRig()

    # Camera
    optical_flow = OpticalFlow()
    optical_flow.inference()
    
    # Add a cube moving in the +X direction
    cube1 = Cube(position=[-5.0, 0.0, 0.0], velocity=[2.0, 0.0, 0.0])
    world.add_entity(cube1)

    # Add a cube moving in the -Z direction
    cube2 = Cube(position=[3.0, 2.0, 5.0], velocity=[0.0, 0.0, -1.5])
    cube2.color = [0.8, 0.2, 0.2] # Make it red
    world.add_entity(cube2)

    renderer = OpenGLRenderer(1280, 720, "Sensor Rig Sim")

    # 2. Define your viewports
    # Left half of the window
    viewport_left = (0, 0, WINDOW_WIDTH // 2, WINDOW_HEIGHT)
    # Right half of the window
    viewport_right = (WINDOW_WIDTH // 2, 0, WINDOW_WIDTH // 2, WINDOW_HEIGHT)

    # 3. Main Loop
    dt = 0.016 # Aim for ~60 FPS
    saved_frame = False
    frame_count = 0

    while not renderer.should_close():
        # Update the simulation state
        world.update(dt)

        radar_points = rig.get_radar().simulate_scan(world)

        # Clear the old points
        radar_world.clear_entities()

        # Add the new points
        for point_3d in radar_points:
            radar_world.add_entity(Point(position=point_3d, color=[0.0, 1.0, 0.0]))
        
        # --- 6. Save on first frame ---
        if frame_count % 100 == 0 and radar_points.size > 0:
            save_as_ply(radar_points, str(frame_count) + ".ply")
            saved_frame = True
        
        # Render the current world state
        renderer.begin_frame()
    
        # Render the left camera's view into the left viewport
        renderer.render_view(world, rig.get_camera(), viewport_left)
        # Render the radar view into the right viewport
        renderer.render_view(radar_world, rig.get_radar(), viewport_right)
        
        renderer.end_frame()

        frame_count += 1

    renderer.shutdown()