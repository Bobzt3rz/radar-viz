import carla
import time

try:
    # 1. Connect to the server
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0) # 5-second timeout

    # 2. Get the world object
    world = client.get_world()

    # 3. Get the spectator (the "camera")
    spectator = world.get_spectator()
    
    spawn_points = world.get_map().get_spawn_points()

    print(spawn_points)


except Exception as e:
    print(f"Error connecting to CARLA: {e}")