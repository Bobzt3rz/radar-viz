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
    vx_gt: float     # m/s (In RADAR coordinates)
    vy_gt: float     # m/s (In RADAR coordinates)
    vz_gt: float     # m/s (In RADAR coordinates)
    
    # --- ADD THESE 3 LINES ---
    vx_gt_world: float # m/s (In WORLD coordinates)
    vy_gt_world: float # m/s (In WORLD coordinates)
    vz_gt_world: float # m/s (In WORLD coordinates)
    noise_type: NoiseType
    # --- Data for internal logic ---
    pixel_uv: Tuple[float, float]
    world_location: carla.Vector3D
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
) -> Tuple[List[RadarPoint], int]:
    """
    Project REAL radar points to camera image plane and find their true 3D velocity.
    """
    
    projected_points: List[RadarPoint] = []
    
    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix: List[List[float]] = T_world_camera.get_inverse_matrix()
    T_world_radar: carla.Transform = radar_transform
    M: List[List[float]] = T_camera_world_matrix
        
    img_height, img_width = instance_image.shape[:2]

    mistag_count: int = 0  # <-- 1. INITIALIZE COUNTER

    for d in radar_data:
        local_x: float = d.depth * np.cos(d.altitude) * np.cos(d.azimuth)
        local_y: float = d.depth * np.cos(d.altitude) * np.sin(d.azimuth)
        local_z: float = d.depth * np.sin(d.altitude)
        p_radar_local: carla.Vector3D = carla.Vector3D(x=local_x, y=local_y, z=local_z)
        p_world: carla.Vector3D = T_world_radar.transform(p_radar_local)
        
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
                true_3d_velocity_world = carla.Vector3D(0.0, 0.0, 0.0)
                # We get the raw 3x3 rotation matrix
                R_rad_from_world_np = np.array(
                    radar_transform.get_inverse_matrix()
                )[:3, :3]
                object_type = ObjectType.STATIC
                
                b_val = int(instance_image[v_int, u_int, 0])
                g_val = int(instance_image[v_int, u_int, 1])
                object_id: int = (b_val * 256) + g_val
                
                if object_id != 0 and object_id != ego_vehicle_id:
                    actor: Optional[carla.Actor] = world.get_actor(object_id)
                    if actor:
                        if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
                            true_3d_velocity_world = actor.get_velocity()
                            object_type = ObjectType.ACTOR

               # 1. Calculate the object's velocity RELATIVE to the ego vehicle in world coords
                relative_velocity_world = true_3d_velocity_world - ego_vehicle_velocity

                # 2. Convert GT velocity from Carla World to Carla Radar
                #    (Use the new relative_velocity_world)
                vel_world_np = np.array([
                    relative_velocity_world.x, 
                    relative_velocity_world.y, 
                    relative_velocity_world.z
                ])
                vel_radar_np = R_rad_from_world_np @ vel_world_np
                
                # 3. Store the (Carla Radar) velocity
                vx_gt_carla_radar = vel_radar_np[0]
                vy_gt_carla_radar = vel_radar_np[1]
                vz_gt_carla_radar = vel_radar_np[2]
                
                ego_speed_m_s = ego_vehicle_velocity.length() 
                
                if object_type == ObjectType.STATIC and ego_speed_m_s > 0.1:
                    epsilon = 0.01
                    
                    carla_gt_v_rad = d.velocity
                    
                    if abs(carla_gt_v_rad) > ego_speed_m_s + epsilon:
                        mistag_count += 1  # <-- 2. INCREMENT COUNTER
                        # print(f"--- WARNING: VELOCITY CHECK FAILED (Carla's GT) ---")
                        # print(f"  Frame: {radar_data.frame}")

                projected_points.append(
                    RadarPoint(
                        pixel_uv=(pixel_u, pixel_v),
                        local_x=local_x,
                        local_y=local_y,
                        local_z=local_z,
                        azimuth=d.azimuth,
                        elevation=d.altitude,
                        range=d.depth,
                        radial_velocity=d.velocity,
                        vx_gt=vx_gt_carla_radar,   # (Carla Radar Coords)
                        vy_gt=vy_gt_carla_radar,   # (Carla Radar Coords)
                        vz_gt=vz_gt_carla_radar,   # (Carla Radar Coords)
                        # (true_3d_velocity_world is from actor.get_velocity())
                        vx_gt_world=true_3d_velocity_world.x, # (Carla World Coords)
                        vy_gt_world=true_3d_velocity_world.y, # (Carla World Coords)
                        vz_gt_world=true_3d_velocity_world.z, # (Carla World Coords)
                        world_location=p_world,
                        noise_type=NoiseType.REAL,
                        object_type=object_type
                    )
                )
    return projected_points, mistag_count

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
                                radial_velocity=real_point.radial_velocity,
                                vx_gt=real_point.vx_gt,
                                vy_gt=real_point.vy_gt,
                                vz_gt=real_point.vz_gt,
                                vx_gt_world=real_point.vx_gt_world,
                                vy_gt_world=real_point.vy_gt_world,
                                vz_gt_world=real_point.vz_gt_world,
                                world_location=p_virtual_world,
                                noise_type=NoiseType.MULTIPATH,
                                object_type=real_point.object_type
                            )
                        )
            except Exception as e:
                # print(f"Could not generate multipath point: {e}")
                pass
                
    return ghost_points

def save_debug_image(
    filepath: str, 
    image_data: carla.Image, 
    points: List[RadarPoint],
    vel_color_map: Dict[Tuple[float, float, float], str],
    color_cycle: List[str],
    ghost_color: str,
    static_color: str
) -> None:
    """
    Saves a debug image with radar points plotted on it,
    color-coded by 3D ground truth velocity.
    """
    try:
        # 1. Process the RGB image
        img_rgb: NDArray[np.uint8] = process_image(image_data)
        img_height, img_width = img_rgb.shape[:2]
        
        # 2. Create a Matplotlib figure
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)
        ax.imshow(img_rgb)
        
        # 3. Collect points and colors
        plot_points_2d: List[Tuple[float, float]] = []
        plot_colors: List[str] = []
        
        if points:
            for p in points:
                plot_points_2d.append(p.pixel_uv)
                
                # --- START: NEW COLOR LOGIC ---
                color = ''
                if p.noise_type == NoiseType.MULTIPATH:
                    color = ghost_color
                else:
                    # It's a REAL point, color by 3D velocity
                    v_tuple = (round(p.vx_gt, 2), round(p.vy_gt, 2), round(p.vz_gt, 2))
                    
                    if v_tuple == (0.0, 0.0, 0.0):
                        color = static_color
                    elif v_tuple in vel_color_map:
                        color = vel_color_map[v_tuple]
                    else:
                        # New velocity, assign a new color
                        new_color_index = len(vel_color_map) % len(color_cycle)
                        color = color_cycle[new_color_index]
                        # The map is mutable, so this change will persist
                        # back in the main() function's dictionary.
                        vel_color_map[v_tuple] = color
                
                plot_colors.append(color)
                # --- END: NEW COLOR LOGIC ---
            
            ax.scatter(
                [uv[0] for uv in plot_points_2d], 
                [uv[1] for uv in plot_points_2d], 
                s=5, 
                c=plot_colors,
                alpha=0.7
            )

        # 4. Save the figure
        ax.set_title(f"Frame {image_data.frame} - {len(points)} points")
        ax.set_axis_off()
        fig.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
        
        # 5. Close the figure to free memory
        plt.close(fig)
        
    except Exception as e:
        print(f"Error saving debug image: {e}")

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
            f.write("property float vx_gt_world\n")
            f.write("property float vy_gt_world\n")
            f.write("property float vz_gt_world\n")
            f.write("property int noise_type\n")
            f.write("property int object_type\n")
            f.write("end_header\n")
            
            # Write data
            for p in points:
                f.write(f"{p.local_x} {p.local_y} {p.local_z} "
                        f"{p.azimuth} {p.elevation} {p.range} "
                        f"{p.radial_velocity} "
                        f"{p.vx_gt} {p.vy_gt} {p.vz_gt} "
                        f"{p.vx_gt_world} {p.vy_gt_world} {p.vz_gt_world} "
                        f"{p.noise_type.value} {p.object_type.value}\n")
                
    except Exception as e:
        print(f"Error saving PLY file: {e}")

def transform_points_to_paper_coords(
    points: List[RadarPoint]
) -> List[RadarPoint]:
    """
    Transforms the coordinate systems of a list of RadarPoint objects
    from Carla's conventions (X-Fwd, Y-Right, Z-Up)
    to the paper's conventions (Z-fwd, Y-down, X-right).
    """
    transformed_points: List[RadarPoint] = []
    for p in points:
        # 1. Transform Local Sensor Coordinates (Carla Sensor -> Paper Sensor)
        # Carla Sensor: +X Fwd, +Y Right, +Z Up
        # Paper Sensor: +X Right, +Y Down, +Z Fwd
        paper_local_x = p.local_y   # Paper X (Right) = Carla Y (Right)
        paper_local_y = -p.local_z  # Paper Y (Down)  = -Carla Z (Up)
        paper_local_z = p.local_x   # Paper Z (Fwd)   = Carla X (Fwd)
        
        # 2. Transform World Ground Truth Velocity (Carla World -> Paper World)
        # Carla World: +X Fwd, +Y Right, +Z Up
        # Paper World: +X Right, +Y Down, +Z Fwd
        paper_vx_gt = p.vy_gt       # Paper VX (Right) = Carla VY (Right)
        paper_vy_gt = -p.vz_gt      # Paper VY (Down)  = -Carla VZ (Up)
        paper_vz_gt = p.vx_gt       # Paper VZ (Fwd)   = Carla VX (Fwd)

        # 3. Transform GT World Velocity (Carla World -> Paper World) ---
        # Carla World: +X Fwd, +Y Right, +Z Up
        # Paper World: +X Right, +Y Down, +Z Fwd
        # (This also uses the same transform)
        paper_vx_gt_world = p.vy_gt_world   # Paper X (Right) = Carla Y (Right)
        paper_vy_gt_world = -p.vz_gt_world  # Paper Y (Down)  = -Carla Z (Up)
        paper_vz_gt_world = p.vx_gt_world   # Paper Z (Fwd)   = Carla X (Fwd)

        # 4. Re-calculate Azimuth and Elevation for the new system
        # In the paper's Z-fwd, X-right, Y-down system:
        # Azimuth is rotation around Y-axis: atan2(x, z)
        # Elevation is rotation around X-axis: asin(y / range)
        
        paper_azimuth: float
        paper_elevation: float
        
        if p.range > 1e-6: # Avoid division by zero
            paper_azimuth = math.atan2(paper_local_x, paper_local_z)
            asin_arg = max(-1.0, min(1.0, paper_local_y / p.range))
            paper_elevation = math.asin(asin_arg)
        else:
            paper_azimuth = 0.0
            paper_elevation = 0.0

        # 4. Create a new RadarPoint with all transformed values
        transformed_points.append(RadarPoint(
            pixel_uv=p.pixel_uv,
            local_x=paper_local_x,   # <-- Transformed
            local_y=paper_local_y,   # <-- Transformed
            local_z=paper_local_z,   # <-- Transformed
            azimuth=paper_azimuth,     # <-- Re-calculated
            elevation=paper_elevation, # <-- Re-calculated
            range=p.range,           # <-- Unchanged scalar
            radial_velocity=p.radial_velocity, # <-- Unchanged scalar
            vx_gt=paper_vx_gt,       # <-- Transformed
            vy_gt=paper_vy_gt,       # <-- Transformed
            vz_gt=paper_vz_gt,       # <-- Transformed
            vx_gt_world=paper_vx_gt_world,
            vy_gt_world=paper_vy_gt_world,
            vz_gt_world=paper_vz_gt_world,
            world_location=p.world_location, 
            noise_type=p.noise_type,
            object_type=p.object_type
        ))
    return transformed_points


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
    CAM_DIR = os.path.join(OUTPUT_DIR, "camera_rgb")
    DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
    CALIB_DIR = os.path.join(OUTPUT_DIR, "calib")
    POSES_DIR = os.path.join(OUTPUT_DIR, "poses")
    
    os.makedirs(VEL_DIR, exist_ok=True)
    os.makedirs(PLY_DIR, exist_ok=True)
    os.makedirs(CAM_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)
    os.makedirs(POSES_DIR, exist_ok=True)

    # Color definitions for debug images
    global_velocity_to_color_map: Dict[Tuple[float, float, float], str] = {}
    color_cycle_list: List[str] = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF',
        '#FF00FF', '#FFA500', '#800080', '#FFC0CB', '#008000'
    ]
    MULTIPATH_COLOR = '#FF00FF' # Fuchsia / Magenta
    STATIC_COLOR = '#FFFFFF'    # White
    
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

        K_camera: NDArray[np.float32] = build_camera_intrinsics(cam_w, cam_h, cam_fov)

        try:
            # 1. Save Camera Intrinsics (K_camera)
            intrinsics_path = os.path.join(CALIB_DIR, "intrinsics.txt")
            np.savetxt(intrinsics_path, K_camera, fmt='%.6f')
            print(f"Saved camera intrinsics to {intrinsics_path}")

            # --- 2. Calculate and Save Extrinsics (T_A_to_R) ---
            # We must save the extrinsics in the same "Paper" coordinate
            # system as our poses and PLY files.
            
            # CoB matrix: Carla (Xf, Yr, Zu) -> Paper (Xr, Yd, Zf)
            C_paper_from_carla = np.array([
                [0, 1,  0, 0], # Paper X (Right) = Carla Y (Right)
                [0, 0, -1, 0], # Paper Y (Down)  = -Carla Z (Up)
                [1, 0,  0, 0], # Paper Z (Fwd)   = Carla X (Fwd)
                [0, 0,  0, 1]
            ], dtype=np.float32)
            C_carla_from_paper = np.linalg.inv(C_paper_from_carla)

            # Get the raw WORLD-TO-LOCAL poses from Carla
            cam_pose_carla_np = np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float32)
            rad_pose_carla_np = np.array(radar.get_transform().get_inverse_matrix(), dtype=np.float32)

            # Convert these poses to the Paper's coordinate system
            cam_pose_paper_np = C_paper_from_carla @ cam_pose_carla_np @ C_carla_from_paper
            rad_pose_paper_np = C_paper_from_carla @ rad_pose_carla_np @ C_carla_from_paper
            
            # Now, calculate the extrinsics using the formula from your coordinate_systems.md
            # T_A_to_R = RadarPose @ inv(CamPose_A)
            # This gives us the static transform from PaperCamera to PaperRadar
            extrinsics_matrix = rad_pose_paper_np @ np.linalg.inv(cam_pose_paper_np)
            
            extrinsics_path = os.path.join(CALIB_DIR, "extrinsics_radar_from_camera.txt")
            np.savetxt(extrinsics_path, extrinsics_matrix, fmt='%.8f')
            print(f"Saved extrinsics (Radar from Camera) to {extrinsics_path}")

        except Exception as e:
            print(f"--- ERROR SAVING CALIBRATION ---")
            print(e)
            print("---------------------------------")

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

        prev_cam_pose_paper: Optional[NDArray[np.float64]] = None
        
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
                
                real_points: List[RadarPoint]
                mistag_count: int
                real_points, mistag_count = radar_to_camera_projection(
                    radar_data,
                    instance_camera,
                    K_instance,
                    radar_world_transform,
                    instance_image_array,
                    world,
                    ego_vehicle_id,
                    ego_vehicle_velocity
                )
                # --- END MODIFIED CALL ---
                
                ghost_points: List[RadarPoint] = generate_multipath_points(
                    real_points,
                    world,
                    instance_camera,
                    K_instance,
                    ego_vehicle_velocity
                )
                
                all_points = real_points + ghost_points
                
                # --- 12. Save data to disk ---
                filename_id = f"{frame_id:08d}"

                cam_filepath = os.path.join(CAM_DIR, f"{filename_id}.png")
                image_data.save_to_disk(cam_filepath)

                # Save Debug Image
                debug_filepath = os.path.join(DEBUG_DIR, f"{filename_id}.png")
                save_debug_image(
                    debug_filepath, 
                    image_data, 
                    all_points,
                    global_velocity_to_color_map,
                    color_cycle_list,
                    MULTIPATH_COLOR,
                    STATIC_COLOR
                )
                
                vel_filepath = os.path.join(VEL_DIR, f"{filename_id}.txt")
                save_ego_velocity(vel_filepath, vehicle)
                
                # transformed coordinate system for further processing
                final_points_for_ply = transform_points_to_paper_coords(all_points)
                ply_filepath = os.path.join(PLY_DIR, f"{filename_id}.ply")
                save_radar_ply(ply_filepath, final_points_for_ply)

                try:
                    # --- 1. Define the Change of Basis (CoB) matrices ---
                    C_paper_from_carla = np.array([
                        [0, 1,  0, 0], # Paper X (Right) = Carla Y (Right)
                        [0, 0, -1, 0], # Paper Y (Down)  = -Carla Z (Up)
                        [1, 0,  0, 0], # Paper Z (Fwd)   = Carla X (Fwd)
                        [0, 0,  0, 1]
                    ], dtype=np.float32)
                    C_carla_from_paper = np.linalg.inv(C_paper_from_carla)

                    # --- 2. Get CURRENT poses in Paper's system ---
                    # T_CarlaCam_from_CarlaWorld
                    cam_pose_carla_np = np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float32)
                    # T_PaperCam_from_PaperWorld (This is "Pose_B")
                    curr_cam_pose_paper = C_paper_from_carla @ cam_pose_carla_np @ C_carla_from_paper

                    # --- 3. Calculate and Save the "diff" (T_A_to_B) ---
                    if prev_cam_pose_paper is not None:
                        # This is "Pose_A"
                        Pose_A = prev_cam_pose_paper 
                        # This is "Pose_B"
                        Pose_B = curr_cam_pose_paper
                        
                        # T_A_to_B = Pose_B @ inv(Pose_A)
                        relative_pose_matrix = Pose_B @ np.linalg.inv(Pose_A)
                        
                        # Save the relative pose
                        rel_pose_path = os.path.join(POSES_DIR, f"{filename_id}_relative_pose.txt")
                        np.savetxt(rel_pose_path, relative_pose_matrix, fmt='%.8f')
                    
                    # --- 4. Store the current pose for the NEXT frame ---
                    prev_cam_pose_paper = curr_cam_pose_paper

                except Exception as e:
                    print(f"--- ERROR SAVING POSES for frame {frame_id} ---")
                    print(e)
                # --- END: SAVE POSES ---
                
                print(f"Frame {frame_id}: Saved {len(all_points)} points ({len(real_points)} real, {len(ghost_points)} ghost). Mis-tags: {mistag_count}")

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