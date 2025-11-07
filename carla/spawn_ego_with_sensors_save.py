import carla
import random
import time
import queue
import numpy as np
import matplotlib
matplotlib.use('Agg') # <-- Use non-interactive backend for speed
import matplotlib.pyplot as plt

# --- NEW IMPORTS ---
import os
import math
from enum import Enum
from dataclasses import dataclass

# Import specific types for annotation
from typing import List, Tuple, Optional, Any, Dict
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# --- Enums for clarity ---
class NoiseType(Enum):
    REAL = 0
    MULTIPATH = 1
    RANDOM = 2

class ObjectType(Enum):
    STATIC = 0  # Road, buildings, poles, etc.
    ACTOR = 1   # Vehicles and Walkers

# --- MODIFIED: Dataclass for Radar Point ---
@dataclass
class RadarPoint:
    """A single radar detection with all its associated data."""
    # --- Data for PLY file ---
    local_x: float
    local_y: float
    local_z: float
    azimuth: float   # rad
    elevation: float # rad
    range: float     # meters
    radial_velocity: float # m/s
    vx_gt: float     # m/s
    vy_gt: float     # m/s
    vz_gt: float     # m/s
    noise_type: NoiseType
    # --- Data for internal logic ---
    pixel_uv: Tuple[float, float]
    world_location: carla.Location
    object_type: ObjectType


# --- Helper functions ---
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

def process_instance_image(image_data: carla.Image) -> NDArray[np.uint8]:
    array: NDArray[np.uint8] = np.frombuffer(image_data.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image_data.height, image_data.width, 4))
    return array

# --- MODIFIED: radar_to_camera_projection (BUG FIX) ---
def radar_to_camera_projection(
    radar_data: carla.RadarMeasurement,
    camera: carla.Actor,
    camera_intrinsics: NDArray[np.float32],
    radar_transform: carla.Transform,
    instance_image: NDArray[np.uint8],
    world: carla.World,
    ego_vehicle_id: int,
    ego_vehicle_velocity: carla.Vector3D # <-- NEW ARGUMENT
) -> List[RadarPoint]:
    """
    Project REAL radar points to camera image plane and find their true 3D velocity.
    """
    
    projected_points: List[RadarPoint] = []
    
    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix: List[List[float]] = T_world_camera.get_inverse_matrix()
    T_world_radar: carla.Transform = radar_transform
    M: List[List[float]] = T_camera_world_matrix
    
    actor_velocity_cache: Dict[int, Tuple[carla.Vector3D, ObjectType]] = {}
    
    img_height, img_width = instance_image.shape[:2]
    
    # --- *** START OF FIX *** ---
    # Get the inverse transform matrix as a numpy array
    T_world_radar_inv_np = np.array(T_world_radar.get_inverse_matrix())
    # Get the 3x3 inverse rotation matrix
    R_world_radar_inv_np = T_world_radar_inv_np[:3, :3]
    # --- *** END OF FIX *** ---


    for d in radar_data:
        local_x: float = d.depth * np.cos(d.altitude) * np.cos(d.azimuth)
        local_y: float = d.depth * np.cos(d.altitude) * np.sin(d.azimuth)
        local_z: float = d.depth * np.sin(d.altitude)
        p_radar_local: carla.Location = carla.Location(x=local_x, y=local_y, z=local_z)
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

            u_int: int = int(round(pixel_u))
            v_int: int = int(round(pixel_v))

            if (0 <= u_int < img_width) and (0 <= v_int < img_height):
                true_3d_velocity = carla.Vector3D(0.0, 0.0, 0.0)
                object_type = ObjectType.STATIC
                
                b_val = int(instance_image[v_int, u_int, 0])
                g_val = int(instance_image[v_int, u_int, 1])
                object_id: int = (b_val * 256) + g_val
                
                if object_id != 0 and object_id != ego_vehicle_id:
                    if object_id in actor_velocity_cache:
                        true_3d_velocity, object_type = actor_velocity_cache[object_id]
                    else:
                        actor: Optional[carla.Actor] = world.get_actor(object_id)
                        if actor:
                            if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
                                true_3d_velocity = actor.get_velocity()
                                object_type = ObjectType.ACTOR
                            
                            actor_velocity_cache[object_id] = (true_3d_velocity, object_type)
                
                # --- *** START OF FIX *** ---
                
                if d.depth > 0:
                    unit_vector_local = p_radar_local / d.depth
                else:
                    unit_vector_local = carla.Vector3D(1.0, 0, 0)
                
                relative_velocity_world = true_3d_velocity - ego_vehicle_velocity

                # --- BUG FIX IS HERE ---
                # Convert to numpy and manually apply 3x3 inverse rotation
                v_rel_world_np = np.array([relative_velocity_world.x, 
                                           relative_velocity_world.y, 
                                           relative_velocity_world.z])
                v_rel_local_np = R_world_radar_inv_np @ v_rel_world_np
                relative_velocity_local = carla.Vector3D(x=v_rel_local_np[0], 
                                                         y=v_rel_local_np[1], 
                                                         z=v_rel_local_np[2])
                
                correct_radial_velocity = - (relative_velocity_local.dot(unit_vector_local))
                
                # --- *** END OF FIX *** ---

                projected_points.append(
                    RadarPoint(
                        pixel_uv=(pixel_u, pixel_v),
                        local_x=local_x,
                        local_y=local_y,
                        local_z=local_z,
                        azimuth=d.azimuth,
                        elevation=d.altitude,
                        range=d.depth,
                        radial_velocity=correct_radial_velocity,
                        vx_gt=true_3d_velocity.x,
                        vy_gt=true_3d_velocity.y,
                        vz_gt=true_3d_velocity.z,
                        world_location=p_world,
                        noise_type=NoiseType.REAL,
                        object_type=object_type
                    )
                )
    return projected_points

# --- MODIFIED: generate_multipath_points (BUG FIX) ---
def generate_multipath_points(
    real_points: List[RadarPoint],
    world: carla.World,
    camera: carla.Actor, # This is the co-located instance_camera
    camera_intrinsics: NDArray[np.float32], # This is K_instance
    ego_vehicle_velocity: carla.Vector3D
) -> List[RadarPoint]:
    """
    Generates ground-reflection multipath (ghost) points for dynamic objects.
    """
    ghost_points: List[RadarPoint] = []
    carla_map = world.get_map()
    
    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix_np = np.array(T_world_camera.get_inverse_matrix())
    M: List[List[float]] = T_camera_world_matrix_np.tolist()
    
    K: NDArray[np.float32] = camera_intrinsics 
    img_height, img_width = K[1, 2] * 2, K[0, 2] * 2
    
    # --- *** START OF FIX *** ---
    # Get the 3x3 inverse rotation matrix
    R_world_camera_inv_np = T_camera_world_matrix_np[:3, :3]
    # --- *** END OF FIX *** ---

    for real_point in real_points:
        if real_point.object_type == ObjectType.ACTOR:
            try:
                p_real = real_point.world_location
                waypoint = carla_map.get_waypoint(p_real)
                z_road = waypoint.transform.location.z
                
                dist_to_road = p_real.z - z_road
                if dist_to_road < 0: 
                    continue
                
                p_virtual_world = carla.Location(
                    p_real.x,
                    p_real.y,
                    z_road - dist_to_road
                )
                
                p_virt_h = np.array([p_virtual_world.x, p_virtual_world.y, p_virtual_world.z, 1.0])
                p_virt_local_h = T_camera_world_matrix_np @ p_virt_h
                p_virt_local = p_virt_local_h[:3]
                
                local_x, local_y, local_z = p_virt_local[0], p_virt_local[1], p_virt_local[2]
                
                ghost_range = float(np.linalg.norm(p_virt_local))
                if ghost_range == 0: continue
                
                ghost_azimuth = float(np.arctan2(local_y, local_x))
                ghost_elevation = float(np.arcsin(local_z / ghost_range))
                
                # --- *** START OF FIX FOR GHOST RADIAL VELOCITY *** ---
                true_3d_velocity = carla.Vector3D(real_point.vx_gt, real_point.vy_gt, real_point.vz_gt)
                relative_velocity_world = true_3d_velocity - ego_vehicle_velocity
                
                # --- BUG FIX IS HERE ---
                # Convert to numpy and manually apply 3x3 inverse rotation
                v_rel_world_np = np.array([relative_velocity_world.x, 
                                           relative_velocity_world.y, 
                                           relative_velocity_world.z])
                v_rel_local_np = R_world_camera_inv_np @ v_rel_world_np
                relative_velocity_local = carla.Vector3D(x=v_rel_local_np[0], 
                                                         y=v_rel_local_np[1], 
                                                         z=v_rel_local_np[2])

                unit_vector_local_ghost = carla.Vector3D(local_x, local_y, local_z) / ghost_range

                ghost_radial_velocity = - (relative_velocity_local.dot(unit_vector_local_ghost))
                # --- *** END OF FIX *** ---

                x_cam: float = M[0][0]*p_virtual_world.x + M[0][1]*p_virtual_world.y + M[0][2]*p_virtual_world.z + M[0][3]
                y_cam: float = M[1][0]*p_virtual_world.x + M[1][1]*p_virtual_world.y + M[1][2]*p_virtual_world.z + M[1][3]
                z_cam: float = M[2][0]*p_virtual_world.x + M[2][1]*p_virtual_world.y + M[2][2]*p_virtual_world.z + M[2][3]

                if x_cam > 0: 
                    u_norm: float = y_cam / x_cam
                    v_norm: float = -z_cam / x_cam
                    pixel_u: float = K[0, 0] * u_norm + K[0, 2]
                    pixel_v: float = K[1, 1] * v_norm + K[1, 2]

                    if (0 <= pixel_u < img_width) and (0 <= pixel_v < img_height):
                        ghost_points.append(
                            RadarPoint(
                                pixel_uv=(pixel_u, pixel_v),
                                local_x=local_x,
                                local_y=local_y,
                                local_z=local_z,
                                azimuth=ghost_azimuth,
                                elevation=ghost_elevation,
                                range=ghost_range,
                                radial_velocity=ghost_radial_velocity,
                                vx_gt=real_point.vx_gt,
                                vy_gt=real_point.vy_gt,
                                vz_gt=real_point.vz_gt,
                                world_location=p_virtual_world,
                                noise_type=NoiseType.MULTIPATH,
                                object_type=real_point.object_type
                            )
                        )
            except Exception as e:
                # print(f"Could not generate multipath point: {e}")
                pass
                
    return ghost_points

# --- NEW: Function to save ego velocity ---
def save_ego_velocity(filepath: str, vehicle: carla.Actor) -> None:
    """Saves the ego vehicle's 3D velocity to a text file."""
    vel = vehicle.get_velocity()
    try:
        with open(filepath, 'w') as f:
            f.write(f"vx: {vel.x}\n")
            f.write(f"vy: {vel.y}\n")
            f.write(f"vz: {vel.z}\n")
    except Exception as e:
        print(f"Error saving velocity: {e}")

# --- NEW: Function to save radar data as PLY ---
def save_radar_ply(filepath: str, points: List[RadarPoint]) -> None:
    """
    Saves a list of RadarPoint objects to a PLY file in the requested format.
    """
    try:
        with open(filepath, 'w') as f:
            # Write PLY Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float azimuth\n")
            f.write("property float elevation\n")
            f.write("property float range\n")
            f.write("property float radial_velocity\n")
            f.write("property float vx_gt\n")
            f.write("property float vy_gt\n")
            f.write("property float vz_gt\n")
            f.write("property int noise_type\n")
            f.write("end_header\n")
            
            # Write data
            for p in points:
                f.write(f"{p.local_x} {p.local_y} {p.local_z} "
                        f"{p.azimuth} {p.elevation} {p.range} "
                        f"{p.radial_velocity} "
                        f"{p.vx_gt} {p.vy_gt} {p.vz_gt} "
                        f"{p.noise_type.value}\n")
                
    except Exception as e:
        print(f"Error saving PLY file: {e}")


def main() -> None:
    
    actor_list: List[carla.Actor] = []
    client: Optional[carla.Client] = None
    vehicle: Optional[carla.Actor] = None
    camera: Optional[carla.Actor] = None
    radar: Optional[carla.Actor] = None
    instance_camera: Optional[carla.Actor] = None
    
    tm_port: int = 8000
    original_settings: Optional[carla.WorldSettings] = None
    
    OUTPUT_DIR = "output"
    VEL_DIR = os.path.join(OUTPUT_DIR, "velocities")
    PLY_DIR = os.path.join(OUTPUT_DIR, "radar_ply")
    
    os.makedirs(VEL_DIR, exist_ok=True)
    os.makedirs(PLY_DIR, exist_ok=True)
    
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
        ego_vehicle_id = vehicle.id
        print(f'Spawned vehicle: {vehicle.type_id} (ID: {ego_vehicle_id})')

        # 3. Set vehicle to autopilot
        tm: carla.TrafficManager = client.get_trafficmanager(tm_port)
        tm_port = tm.get_port()
        vehicle.set_autopilot(True, tm_port)
        print(f'Vehicle set to autopilot on port {tm_port}')

        co_located_transform: carla.Transform = carla.Transform(carla.Location(x=2.7, z=1.0))

        # 4. Spawn RGB Camera
        camera_bp: carla.ActorBlueprint = blueprint_library.find('sensor.camera.rgb')
        cam_w: int = 1280
        cam_h: int = 720
        cam_fov: float = 90.0
        camera_bp.set_attribute('image_size_x', str(cam_w))
        camera_bp.set_attribute('image_size_y', str(cam_h))
        camera_bp.set_attribute('fov', str(cam_fov))
        camera = world.spawn_actor(camera_bp, co_located_transform, attach_to=vehicle)
        actor_list.append(camera)
        print(f'Spawned sensor: {camera.type_id}')

        # 5. Spawn Radar (using your settings)
        radar_bp: carla.ActorBlueprint = blueprint_library.find('sensor.other.radar')
        radar_fov_horiz: float = 70.0
        radar_bp.set_attribute('points_per_second', '15000')
        radar_bp.set_attribute('horizontal_fov', str(radar_fov_horiz))
        radar_bp.set_attribute('vertical_fov', '40')
        radar_bp.set_attribute('range', '150')
        radar = world.spawn_actor(radar_bp, co_located_transform, attach_to=vehicle)
        actor_list.append(radar)
        print(f'Spawned sensor: {radar.type_id}')
        
        # 6. Spawn Instance Camera (for data)
        instance_camera_bp: carla.ActorBlueprint = blueprint_library.find('sensor.camera.instance_segmentation')
        instance_camera_bp.set_attribute('image_size_x', str(cam_w))
        instance_camera_bp.set_attribute('image_size_y', str(cam_h))
        instance_cam_fov: float = 90.0
        instance_camera_bp.set_attribute('fov', str(instance_cam_fov))
        instance_camera = world.spawn_actor(instance_camera_bp, co_located_transform, attach_to=vehicle)
        actor_list.append(instance_camera)
        print(f'Spawned sensor: {instance_camera.type_id}')
        K_instance: NDArray[np.float32] = build_camera_intrinsics(cam_w, cam_h, instance_cam_fov)

        # 7. Set up Synchronous Mode (using 20 FPS)
        settings: carla.WorldSettings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)
        print(f"Set simulation to {1.0/settings.fixed_delta_seconds:.0f} FPS (timestep = {settings.fixed_delta_seconds}s)")


        # 8. Create data queues and start listeners
        camera_queue: "queue.Queue[carla.Image]" = queue.Queue()
        radar_queue: "queue.Queue[carla.RadarMeasurement]" = queue.Queue()
        instance_camera_queue: "queue.Queue[carla.Image]" = queue.Queue()
        
        camera.listen(camera_queue.put) 
        radar.listen(radar_queue.put)
        instance_camera.listen(instance_camera_queue.put)

        # 9. Setup complete
        print("Setup complete. Running simulation to save data...")
        print("Press Ctrl+C in this terminal to stop.")
        
        # 10. Main simulation loop
        while True:
            frame_id: int = world.tick()

            try:
                # --- *** START OF FIX *** ---
                # Increase timeout to 5 seconds. This forces the script
                # to wait for the data, preventing frame skips.
                timeout_seconds = 5.0
                # --- *** END OF FIX *** ---

                # 1. Get Camera Data
                while True:
                    image_data: carla.Image = camera_queue.get(timeout=timeout_seconds)
                    if image_data.frame == frame_id:
                        break # We found the correct frame
                    if image_data.frame > frame_id:
                        # This should not happen, but if it does, skip this tick
                        print(f"Sync miss: Camera {image_data.frame} > World {frame_id}")
                        raise queue.Empty # Use exception to skip
                
                # 2. Get Radar Data
                while True:
                    radar_data: carla.RadarMeasurement = radar_queue.get(timeout=timeout_seconds)
                    if radar_data.frame == frame_id:
                        break # We found the correct frame
                    if radar_data.frame > frame_id:
                        print(f"Sync miss: Radar {radar_data.frame} > World {frame_id}")
                        raise queue.Empty
                    
                # 3. Get Instance Camera Data
                while True:
                    instance_image_data: carla.Image = instance_camera_queue.get(timeout=timeout_seconds)
                    if instance_image_data.frame == frame_id:
                        break # We found the correct frame
                    if instance_image_data.frame > frame_id:
                        print(f"Sync miss: Instance Img {instance_image_data.frame} > World {frame_id}")
                        raise queue.Empty
                
                # --- *** END OF FIX *** ---
                
                # ... (rest of the file-saving logic is the same) ...
                
                # 11. Process Data
                instance_image_array: NDArray[np.uint8] = process_instance_image(instance_image_data)
                radar_world_transform: carla.Transform = radar.get_transform()
                
                ego_vehicle_velocity = vehicle.get_velocity()
                
                real_points: List[RadarPoint] = radar_to_camera_projection(
                    radar_data,
                    instance_camera,
                    K_instance,
                    radar_world_transform,
                    instance_image_array,
                    world,
                    ego_vehicle_id,
                    ego_vehicle_velocity
                )
                
                ghost_points: List[RadarPoint] = generate_multipath_points(
                    real_points,
                    world,
                    instance_camera,
                    K_instance,
                    ego_vehicle_velocity
                )
                
                all_points = real_points + ghost_points
                
                # --- 12. NEW: Save data to disk ---
                filename_id = f"{frame_id:08d}" 
                
                vel_filepath = os.path.join(VEL_DIR, f"{filename_id}.txt")
                save_ego_velocity(vel_filepath, vehicle)
                
                ply_filepath = os.path.join(PLY_DIR, f"{filename_id}.ply")
                save_radar_ply(ply_filepath, all_points)
                
                print(f"Frame {frame_id}: Saved {len(all_points)} points ({len(real_points)} real, {len(ghost_points)} ghost)")

            except queue.Empty:
                # This will now only happen if the server is truly stuck (5+ seconds)
                print(f"Frame {frame_id}: Sensors timed out (5.0s). Skipping...")
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
        if instance_camera and hasattr(instance_camera, 'is_listening') and instance_camera.is_listening:
            instance_camera.stop()
            
        if client and vehicle and vehicle.is_alive:
            vehicle.set_autopilot(False, tm_port)
            
        if client and actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            print(f'Destroyed {len(actor_list)} actors.')
        
        print('Cleanup complete.')


if __name__ == '__main__':
    main()