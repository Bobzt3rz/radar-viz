import carla
import random
import time
import queue
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Use the backend we know works
import matplotlib.pyplot as plt

# Import specific types for annotation
from typing import List, Tuple, Optional, Any
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# (Helper functions are unchanged and correct)
def build_camera_intrinsics(image_w: int, image_h: int, fov: float) -> NDArray[np.float32]:
    f: float = image_w / (2 * np.tan(fov * np.pi / 360))
    cx: float = image_w / 2.0
    cy: float = image_h / 2.0
    K: NDArray[np.float32] = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K

def process_image(image_data: carla.Image) -> NDArray[np.uint8]:
    array: NDArray[np.uint8] = np.frombuffer(image_data.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image_data.height, image_data.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def radar_to_camera_projection(
    radar_data: carla.RadarMeasurement, 
    camera: carla.Actor, 
    camera_intrinsics: NDArray[np.float32], 
    radar_transform: carla.Transform
) -> List[Tuple[Tuple[float, float], float]]:
    
    projected_points: List[Tuple[Tuple[float, float], float]] = []
    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix: List[List[float]] = T_world_camera.get_inverse_matrix()
    T_world_radar: carla.Transform = radar_transform
    M: List[List[float]] = T_camera_world_matrix

    for d in radar_data:
        x: float = d.depth * np.cos(d.altitude) * np.cos(d.azimuth)
        y: float = d.depth * np.cos(d.altitude) * np.sin(d.azimuth)
        z: float = d.depth * np.sin(d.altitude)
        p_radar_local: carla.Location = carla.Location(x=x, y=y, z=z)
        p_world: carla.Location = T_world_radar.transform(p_radar_local)
        
        x_cam: float = M[0][0]*p_world.x + M[0][1]*p_world.y + M[0][2]*p_world.z + M[0][3]
        y_cam: float = M[1][0]*p_world.x + M[1][1]*p_world.y + M[1][2]*p_world.z + M[1][3]
        z_cam: float = M[2][0]*p_world.x + M[2][1]*p_world.y + M[2][2]*p_world.z + M[2][3]

        if x_cam > 0:
            u_norm: float = y_cam / x_cam
            v_norm: float = -z_cam / x_cam
            K: NDArray[np.float32] = camera_intrinsics
            pixel_u: float = K[0, 0] * u_norm + K[0, 2]
            pixel_v: float = K[1, 1] * v_norm + K[1, 2]

            if 0 <= pixel_u < K[0, 2] * 2 and 0 <= pixel_v < K[1, 2] * 2:
                projected_points.append(
                    ((pixel_u, pixel_v), d.velocity)
                )
    return projected_points


def main() -> None:
    
    actor_list: List[carla.Actor] = []
    client: Optional[carla.Client] = None
    vehicle: Optional[carla.Actor] = None
    camera: Optional[carla.Actor] = None
    radar: Optional[carla.Actor] = None
    tm_port: int = 8000
    original_settings: Optional[carla.WorldSettings] = None
    
    # --- Plot variables ---
    fig: Optional[Figure] = None
    
    try:
        # 1. Connect and get world
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world: carla.World = client.get_world()
        blueprint_library: carla.BlueprintLibrary = world.get_blueprint_library()
        original_settings = world.get_settings()
        
        # 2. Spawn Vehicle
        vehicle_bp: carla.ActorBlueprint = blueprint_library.find('vehicle.tesla.model3')
        spawn_point: carla.Transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f'Spawned vehicle: {vehicle.type_id}')

        # 3. Set vehicle to autopilot
        tm: carla.TrafficManager = client.get_trafficmanager(tm_port)
        tm_port = tm.get_port()
        vehicle.set_autopilot(True, tm_port)
        print(f'Vehicle set to autopilot on port {tm_port}')

        # 4. Spawn Camera
        camera_bp: carla.ActorBlueprint = blueprint_library.find('sensor.camera.rgb')
        cam_w: int = 1280
        cam_h: int = 720
        cam_fov: float = 90.0
        camera_bp.set_attribute('image_size_x', str(cam_w))
        camera_bp.set_attribute('image_size_y', str(cam_h))
        camera_bp.set_attribute('fov', str(cam_fov))
        camera_transform: carla.Transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print(f'Spawned sensor: {camera.type_id}')
        K: NDArray[np.float32] = build_camera_intrinsics(cam_w, cam_h, cam_fov)

        # 5. Spawn Radar
        radar_bp: carla.ActorBlueprint = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('points_per_second', '5000')
        radar_bp.set_attribute('horizontal_fov', '60')
        radar_bp.set_attribute('vertical_fov', '20')
        radar_bp.set_attribute('range', '150')
        radar_relative_transform: carla.Transform = carla.Transform(carla.Location(x=2.7, z=1.0))
        radar = world.spawn_actor(radar_bp, radar_relative_transform, attach_to=vehicle)
        actor_list.append(radar)
        print(f'Spawned sensor: {radar.type_id}')

        # 6. Set up Synchronous Mode
        settings: carla.WorldSettings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)

        # 7. Create data queues and start listeners
        camera_queue: "queue.Queue[carla.Image]" = queue.Queue()
        radar_queue: "queue.Queue[carla.RadarMeasurement]" = queue.Queue()
        camera.listen(camera_queue.put)
        radar.listen(radar_queue.put)

        # --- NEW: Setup the plot window *before* the loop ---
        print("Setup complete. Opening plot window...")
        plt.ion() # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(12, 7))
        # Create empty plots that we can update
        im_plot = ax.imshow(np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
        scatter = ax.scatter([], [], s=5, c=[])
        ax.set_title("Radar-Camera Projection (Waiting for data...)")
        ax.set_axis_off()
        fig.tight_layout()
        fig.show() # Show the window once
        # ----------------------------------------------------
        
        print("Running... Press Ctrl+C in this terminal to stop.")
        
        # 8. Main simulation loop
        while True:
            # 1. Tell the server to advance one frame
            frame_id: int = world.tick()

            try:
                # --- Get the Camera Image, clearing all stale ones ---
                image_data: Optional[carla.Image] = None
                while True:
                    # Get an image. Wait if queue is empty.
                    img: carla.Image = camera_queue.get(timeout=1.0) 
                    
                    if img.frame == frame_id:
                        # This is the one we want!
                        image_data = img
                        break # Exit inner camera loop
                    
                    if img.frame > frame_id:
                        # We overshot. This is a problem.
                        # Discard and break to restart main loop.
                        print(f"Sync miss: Img {img.frame} > World {frame_id}")
                        break # Exit inner camera loop
                    
                    # If img.frame < frame_id, we are stale.
                    # Do nothing, just let the loop run again
                    # to pull the *next* item. This is the "draining" process.
                
                # Check if we broke without finding an image
                if not image_data:
                    continue # Restart main loop
                
                # --- We have image(N). Now find radar(N) ---
                radar_data: Optional[carla.RadarMeasurement] = None
                while True:
                    # Get radar data. Wait if queue is empty.
                    rad: carla.RadarMeasurement = radar_queue.get(timeout=1.0)
                    
                    if rad.frame == frame_id:
                        # This is the one we want!
                        radar_data = rad
                        break # Exit inner radar loop
                    
                    if rad.frame > frame_id:
                        print(f"Sync miss: Radar {rad.frame} > World {frame_id}")
                        break # Overshot
                    
                    # If rad.frame < frame_id, loop again to drain.
                
                # Check if we found both
                if not radar_data:
                    continue # Restart main loop
                    
                # --- WE HAVE A MATCH! ---
                print(f"Success! Plotting frame {frame_id}") # Add a success message
                
                # 9. Process Data
                img_rgb: NDArray[np.uint8] = process_image(image_data)
                radar_world_transform: carla.Transform = radar.get_transform()
                projected_data: List[Tuple[Tuple[float, float], float]] = radar_to_camera_projection(
                    radar_data, camera, K, radar_world_transform
                )

                # 10. Update Plot
                im_plot.set_data(img_rgb)
                
                if projected_data:
                    points_2d: NDArray[np.float64] = np.array([p[0] for p in projected_data])
                    velocities: List[float] = [p[1] for p in projected_data]
                    colors: List[str] = ['r' if v < -0.1 else ('b' if v > 0.1 else 'gray') for v in velocities]
                    
                    scatter.set_offsets(points_2d)
                    scatter.set_color(colors)
                else:
                    scatter.set_offsets(np.array([]))
                
                ax.set_title(f"Radar-Camera Projection (Frame {frame_id}) | {len(projected_data)} points")
                
                # 11. Redraw Canvas and Pause
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001) # Smallest possible pause

            except queue.Empty:
                # This happens if get(timeout=1.0) fails.
                # This is okay, it just means the server is slow.
                print(f"Frame {frame_id}: Sensors timed out. Skipping...")
                continue

    except KeyboardInterrupt:
        print("\nCaught Ctrl+C! Stopping simulation.")
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        import traceback
        traceback.print_exc()
        print("---------------------------\n")

    finally:
        print('\nCleaning up...')
        if original_settings and client:
            print("Restoring original world settings...")
            world: carla.World = client.get_world()
            world.apply_settings(original_settings)
            
        if camera and hasattr(camera, 'is_listening') and camera.is_listening:
            camera.stop()
        if radar and hasattr(radar, 'is_listening') and radar.is_listening:
            radar.stop()
            
        if client and vehicle and vehicle.is_alive:
            vehicle.set_autopilot(False, tm_port)
            
        if client and actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            print(f'Destroyed {len(actor_list)} actors.')
        
        if fig:
            print("Closing plot window...")
            plt.ioff()
            plt.close(fig)
        
        print('Cleanup complete.')


if __name__ == '__main__':
    main()