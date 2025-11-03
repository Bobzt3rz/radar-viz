import carla
import time

try:
    # 1. Connect to the server
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0) # 5-second timeout

    # 2. Get the world object
    world = client.load_world('Town02')

    # 3. Get the spectator (the "camera")
    spectator = world.get_spectator()
    
    print("Success! Connected to CARLA server.")
    print(f"Current map: {world.get_map().name}")

    print(client.get_available_maps())


except Exception as e:
    print(f"Error connecting to CARLA: {e}")