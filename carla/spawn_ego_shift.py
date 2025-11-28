import carla
import random
import time
import queue
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import os
import math
from enum import Enum
from dataclasses import dataclass

from typing import List, Tuple, Optional, Any, Dict
from numpy.typing import NDArray

# --- Enums ---
class NoiseType(Enum):
    REAL = 0
    MULTIPATH = 1
    RANDOM = 2
    SHIFTX = 3
    SHIFTY = 4
    SHIFTZ = 5
    SHIFTRADIAL = 6

class ObjectType(Enum):
    STATIC = 0 
    ACTOR = 1   

# --- Dataclass ---
@dataclass
class RadarPoint:
    local_x: float
    local_y: float
    local_z: float
    azimuth: float   
    elevation: float 
    range: float     
    radial_velocity: float 
    vx_gt: float     
    vy_gt: float     
    vz_gt: float     
    vx_gt_world: float 
    vy_gt_world: float 
    vz_gt_world: float 
    noise_type: NoiseType
    actor_id: int  
    pixel_uv: Tuple[float, float]
    world_location: carla.Vector3D
    object_type: ObjectType
    gt_local_x: float
    gt_local_y: float
    gt_local_z: float
    gt_av_x: float
    gt_av_y: float
    gt_av_z: float
    gt_center_x: float
    gt_center_y: float
    gt_center_z: float

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

def project_world_point_to_image(
    world_point: carla.Vector3D,
    camera_transform: carla.Transform,
    K: NDArray[np.float32],
    img_width: int,
    img_height: int
) -> Optional[Tuple[float, float]]:
    
    T_world_camera = camera_transform
    M = T_world_camera.get_inverse_matrix()
    p_world = world_point
    
    x_cam = M[0][0]*p_world.x + M[0][1]*p_world.y + M[0][2]*p_world.z + M[0][3]
    y_cam = M[1][0]*p_world.x + M[1][1]*p_world.y + M[1][2]*p_world.z + M[1][3]
    z_cam = M[2][0]*p_world.x + M[2][1]*p_world.y + M[2][2]*p_world.z + M[2][3]

    if x_cam > 0: 
        u_norm = y_cam / x_cam
        v_norm = -z_cam / x_cam
        pixel_u = K[0, 0] * u_norm + K[0, 2]
        pixel_v = K[1, 1] * v_norm + K[1, 2]

        if (0 <= pixel_u < img_width) and (0 <= pixel_v < img_height):
            return (pixel_u, pixel_v)
    return None

# --- UPDATED FUNCTION: Independent Probability for Each Noise Type ---
def generate_actor_noise_types(
    real_points: List[RadarPoint],
    world: carla.World,
    radar_transform: carla.Transform,
    camera_transform: carla.Transform,
    K: NDArray[np.float32],
    img_width: int,
    img_height: int,
    instance_image: NDArray[np.uint8], 
    percentage: float = 0.8
) -> List[RadarPoint]:
    
    noise_points: List[RadarPoint] = []
    actor_cache: Dict[int, Any] = {}
    
    # Filter only actor points
    actor_points = [p for p in real_points if p.object_type == ObjectType.ACTOR and p.actor_id != 0]

    # Define the types we want to attempt for EACH point
    noise_types_to_generate = [
        NoiseType.SHIFTX,
        NoiseType.SHIFTY,
        NoiseType.SHIFTZ,
        NoiseType.SHIFTRADIAL
    ]

    for p in actor_points:
        
        # Ensure actor is valid
        if p.actor_id not in actor_cache:
            actor = world.get_actor(p.actor_id)
            actor_cache[p.actor_id] = actor
        actor = actor_cache[p.actor_id]
        if not actor: continue

        # --- ITERATE THROUGH EVERY NOISE TYPE INDEPENDENTLY ---
        for n_type in noise_types_to_generate:
            
            # 1. Independent Probability Check
            if random.random() > percentage:
                continue # Skip this specific noise type for this point
            
            # 2. Random Direction (+1 or -1)
            direction = 1.0 if random.random() < 0.5 else -1.0

            # --- SPATIAL SHIFTS ---
            if n_type in [NoiseType.SHIFTX, NoiseType.SHIFTY, NoiseType.SHIFTZ]:
                
                # Map NoiseType (Target/Paper) to Axis Index (Source/Carla)
                # Paper X (Right)   -> Carla Y (1)
                # Paper Y (Down)    -> Carla Z (2)
                # Paper Z (Forward) -> Carla X (0)
                if n_type == NoiseType.SHIFTX:
                    axis_idx = 1
                elif n_type == NoiseType.SHIFTY:
                    axis_idx = 2
                else: # SHIFTZ
                    axis_idx = 0

                curr_local = [p.local_x, p.local_y, p.local_z]
                step_count = 0
                max_steps = 50 
                
                while step_count < max_steps:
                    # Move point along the CARLA axis
                    curr_local[axis_idx] += (1.0 * direction)
                    
                    # To World
                    loc_vec = carla.Location(x=curr_local[0], y=curr_local[1], z=curr_local[2])
                    p_world_check = radar_transform.transform(loc_vec)
                    
                    # Project to 2D
                    new_uv = project_world_point_to_image(
                        p_world_check, camera_transform, K, img_width, img_height
                    )

                    is_on_actor_mask = False
                    if new_uv:
                        u, v = int(new_uv[0]), int(new_uv[1])
                        # Check Pixel ID
                        b_val = int(instance_image[v, u, 0])
                        g_val = int(instance_image[v, u, 1])
                        pixel_object_id = (b_val * 256) + g_val
                        
                        if pixel_object_id == p.actor_id:
                            is_on_actor_mask = True
                        else:
                            is_on_actor_mask = False 
                    else:
                        is_on_actor_mask = False 

                    if not is_on_actor_mask:
                        # Found a spot outside the mask
                        new_range = math.sqrt(curr_local[0]**2 + curr_local[1]**2 + curr_local[2]**2)
                        new_az = math.atan2(curr_local[1], curr_local[0])
                        new_el = math.asin(curr_local[2] / new_range) if new_range > 0 else 0
                        
                        if new_uv:
                            noise_points.append(RadarPoint(
                                local_x=curr_local[0], local_y=curr_local[1], local_z=curr_local[2],
                                azimuth=new_az, elevation=new_el, range=new_range,
                                radial_velocity=p.radial_velocity,
                                vx_gt=p.vx_gt, vy_gt=p.vy_gt, vz_gt=p.vz_gt,
                                vx_gt_world=p.vx_gt_world, vy_gt_world=p.vy_gt_world, vz_gt_world=p.vz_gt_world,
                                noise_type=n_type, # Use the loop variable
                                actor_id=p.actor_id,
                                pixel_uv=new_uv, world_location=p_world_check,
                                object_type=ObjectType.ACTOR,
                                gt_local_x=p.local_x,
                                gt_local_y=p.local_y,
                                gt_local_z=p.local_z,
                                gt_av_x=p.gt_av_x,
                                gt_av_y=p.gt_av_y,
                                gt_av_z=p.gt_av_z,
                                gt_center_x=p.gt_center_x,
                                gt_center_y=p.gt_center_y,
                                gt_center_z=p.gt_center_z,
                            ))
                        break 
                    step_count += 1

            # --- RADIAL SHIFT ---
            else: # n_type == NoiseType.SHIFTRADIAL
                new_vel = p.radial_velocity + (random.uniform(0.1, 0.7) * direction)
                noise_points.append(RadarPoint(
                    local_x=p.local_x, local_y=p.local_y, local_z=p.local_z,
                    azimuth=p.azimuth, elevation=p.elevation, range=p.range,
                    radial_velocity=new_vel, 
                    vx_gt=p.vx_gt, vy_gt=p.vy_gt, vz_gt=p.vz_gt,
                    vx_gt_world=p.vx_gt_world, vy_gt_world=p.vy_gt_world, vz_gt_world=p.vz_gt_world,
                    noise_type=NoiseType.SHIFTRADIAL, actor_id=p.actor_id,
                    pixel_uv=p.pixel_uv, world_location=p.world_location,
                    object_type=ObjectType.ACTOR,
                    gt_local_x=p.local_x,
                    gt_local_y=p.local_y,
                    gt_local_z=p.local_z,
                    gt_av_x=p.gt_av_x,
                    gt_av_y=p.gt_av_y,
                    gt_av_z=p.gt_av_z,
                    gt_center_x=p.gt_center_x,
                    gt_center_y=p.gt_center_y,
                    gt_center_z=p.gt_center_z,
                ))

    return noise_points

def radar_to_camera_projection(
    radar_data: carla.RadarMeasurement,
    camera: carla.Actor,
    camera_intrinsics: NDArray[np.float32],
    radar_transform: carla.Transform,
    instance_image: NDArray[np.uint8],
    world: carla.World,
    ego_vehicle_id: int,
    ego_vehicle_velocity: carla.Vector3D 
) -> Tuple[List[RadarPoint], int]:
    
    projected_points: List[RadarPoint] = []
    
    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix: List[List[float]] = T_world_camera.get_inverse_matrix()
    T_world_radar: carla.Transform = radar_transform
    M: List[List[float]] = T_camera_world_matrix
        
    img_height, img_width = instance_image.shape[:2]

    mistag_count: int = 0 

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
                R_rad_from_world_np = np.array(
                    radar_transform.get_inverse_matrix()
                )[:3, :3]
                object_type = ObjectType.STATIC
                
                b_val = int(instance_image[v_int, u_int, 0])
                g_val = int(instance_image[v_int, u_int, 1])
                object_id: int = (b_val * 256) + g_val

                av_x_radar, av_y_radar, av_z_radar = 0.0, 0.0, 0.0
                center_x_radar, center_y_radar, center_z_radar = 0.0, 0.0, 0.0
                
                if object_id != 0 and object_id != ego_vehicle_id:
                    actor: Optional[carla.Actor] = world.get_actor(object_id)
                    if actor:
                        if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
                            true_3d_velocity_world = actor.get_velocity()
                            object_type = ObjectType.ACTOR

                            # 2. Angular Velocity (World Frame -> Radar Frame)
                            # CARLA returns degrees/s, convert to radians/s
                            av_world_deg = actor.get_angular_velocity()
                            av_world_rad_np = np.radians(np.array([av_world_deg.x, av_world_deg.y, av_world_deg.z]))
                            
                            # Rotate Angular Velocity vector into Radar Coordinates
                            # We use the same rotation matrix R_rad_from_world_np you defined earlier
                            av_radar_np = R_rad_from_world_np @ av_world_rad_np
                            av_x_radar, av_y_radar, av_z_radar = av_radar_np[0], av_radar_np[1], av_radar_np[2]

                            # 3. Center Position (World Frame -> Radar Frame)
                            # We use the actor's origin (pivot) to be consistent with get_velocity()
                            actor_center_world = actor.get_location()
                            
                            # Transform Point from World -> Radar Coordinates
                            # inverse_transform handles the translation and rotation relative to the radar
                            actor_center_radar = radar_transform.inverse_transform(actor_center_world)
                            center_x_radar = actor_center_radar.x
                            center_y_radar = actor_center_radar.y
                            center_z_radar = actor_center_radar.z

                if object_type == ObjectType.STATIC:
                    final_saved_id = 0
                else:
                    final_saved_id = object_id

                relative_velocity_world = true_3d_velocity_world - ego_vehicle_velocity

                vel_world_np = np.array([
                    relative_velocity_world.x, 
                    relative_velocity_world.y, 
                    relative_velocity_world.z
                ])
                vel_radar_np = R_rad_from_world_np @ vel_world_np
                
                vx_gt_carla_radar = vel_radar_np[0]
                vy_gt_carla_radar = vel_radar_np[1]
                vz_gt_carla_radar = vel_radar_np[2]

                ego_speed_m_s = ego_vehicle_velocity.length() 
                
                if object_type == ObjectType.STATIC and ego_speed_m_s > 0.1:
                    epsilon = 0.01
                    carla_gt_v_rad = d.velocity
                    if abs(carla_gt_v_rad) > ego_speed_m_s + epsilon:
                        mistag_count += 1 

                # add random noise 80% of the time (drop rate)
                if random.random() > 0.8:
                    continue

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
                        vx_gt=vx_gt_carla_radar,   
                        vy_gt=vy_gt_carla_radar,   
                        vz_gt=vz_gt_carla_radar,   
                        vx_gt_world=true_3d_velocity_world.x, 
                        vy_gt_world=true_3d_velocity_world.y, 
                        vz_gt_world=true_3d_velocity_world.z, 
                        world_location=p_world,
                        noise_type=NoiseType.REAL,
                        object_type=object_type,
                        actor_id=final_saved_id,
                        gt_local_x=local_x,
                        gt_local_y=local_y,
                        gt_local_z=local_z,
                        gt_av_x=av_x_radar,
                        gt_av_y=av_y_radar,
                        gt_av_z=av_z_radar,
                        gt_center_x=center_x_radar,
                        gt_center_y=center_y_radar,
                        gt_center_z=center_z_radar,
                    )
                )
    return projected_points, mistag_count

def generate_multipath_points(
    real_points: List[RadarPoint],
    world: carla.World,
    camera: carla.Actor, 
    camera_intrinsics: NDArray[np.float32],
    ego_vehicle_velocity: carla.Vector3D
) -> List[RadarPoint]:
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
                                object_type=real_point.object_type,
                                actor_id=real_point.actor_id,
                                gt_local_x=local_x,
                                gt_local_y=local_y,
                                gt_local_z=local_z,
                                gt_av_x=0.0,
                                gt_av_y=0.0,
                                gt_av_z=0.0,
                                gt_center_x=0.0,
                                gt_center_y=0.0,
                                gt_center_z=0.0,
                            )
                        )
            except Exception as e:
                pass
                
    return ghost_points

def generate_actor_scattering_points(
    real_points: List[RadarPoint],
    world: carla.World,
    camera: carla.Actor,
    camera_intrinsics: NDArray[np.float32],
    scattering_ratio: float = 0.8, 
    scatter_margin: float = 1,   
    velocity_noise: float = 0.5    
) -> List[RadarPoint]:
    scattering_points: List[RadarPoint] = []
    
    actor_velocities: Dict[int, List[float]] = {}
    detected_actor_ids = set()

    for p in real_points:
        if p.object_type == ObjectType.ACTOR and p.actor_id != 0:
            if p.actor_id not in actor_velocities:
                actor_velocities[p.actor_id] = []
            actor_velocities[p.actor_id].append(p.radial_velocity)
            detected_actor_ids.add(p.actor_id)

    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix_np = np.array(T_world_camera.get_inverse_matrix())
    K = camera_intrinsics
    img_height, img_width = K[1, 2] * 2, K[0, 2] * 2

    for actor_id in detected_actor_ids:
        actor = world.get_actor(actor_id)
        if not actor:
            continue
            
        vels = actor_velocities[actor_id]
        if not vels: continue
        
        num_real_points = len(vels)
        points_to_generate = int(num_real_points * scattering_ratio)
        
        if points_to_generate == 0:
            continue

        avg_radial_vel = sum(vels) / len(vels)

        bbox: carla.BoundingBox = actor.bounding_box
        transform: carla.Transform = actor.get_transform()
        
        bx, by, bz = bbox.extent.x, bbox.extent.y, bbox.extent.z
        loc_center = bbox.location 

        for _ in range(points_to_generate):
            sx = 1 if random.random() < 0.5 else -1
            sy = 1 if random.random() < 0.5 else -1
            sz = 1 if random.random() < 0.5 else -1
            
            rx = (bx + random.uniform(0, scatter_margin)) * sx
            ry = (by + random.uniform(0, scatter_margin)) * sy
            rz = (bz + random.uniform(0, scatter_margin)) * sz
            
            p_local_actor = carla.Location(
                x = loc_center.x + rx,
                y = loc_center.y + ry,
                z = loc_center.z + rz
            )
            
            p_world = transform.transform(p_local_actor)
            
            p_world_h = np.array([p_world.x, p_world.y, p_world.z, 1.0])
            p_cam_local = T_camera_world_matrix_np @ p_world_h
            
            if p_cam_local[0] <= 0: continue 

            x_cam, y_cam, z_cam = p_cam_local[0], p_cam_local[1], p_cam_local[2]
            
            u_norm = y_cam / x_cam
            v_norm = -z_cam / x_cam
            pixel_u: float = K[0, 0] * u_norm + K[0, 2]
            pixel_v: float = K[1, 1] * v_norm + K[1, 2]

            if (0 <= pixel_u < img_width) and (0 <= pixel_v < img_height):
                
                final_vel = avg_radial_vel + random.gauss(0, velocity_noise)
                
                r_dist = np.linalg.norm([x_cam, y_cam, z_cam])
                
                scattering_points.append(
                    RadarPoint(
                        pixel_uv=(pixel_u, pixel_v),
                        local_x=x_cam, 
                        local_y=y_cam,
                        local_z=z_cam,
                        azimuth=np.arctan2(y_cam, x_cam),
                        elevation=np.arcsin(z_cam / r_dist),
                        range=r_dist,
                        radial_velocity=final_vel,
                        vx_gt=0.0, vy_gt=0.0, vz_gt=0.0,
                        vx_gt_world=0.0, vy_gt_world=0.0, vz_gt_world=0.0,
                        world_location=p_world,
                        noise_type=NoiseType.MULTIPATH,
                        object_type=ObjectType.ACTOR,
                        actor_id=actor_id,
                        gt_local_x=x_cam,
                        gt_local_y=y_cam,
                        gt_local_z=z_cam,
                        gt_av_x=0.0,
                        gt_av_y=0.0,
                        gt_av_z=0.0,
                        gt_center_x=0.0,
                        gt_center_y=0.0,
                        gt_center_z=0.0,
                    )
                )

    return scattering_points

def generate_static_clutter(
    camera: carla.Actor, 
    camera_intrinsics: NDArray[np.float32], 
    radar_transform: carla.Transform,
    ego_vehicle_velocity: carla.Vector3D,
    radar_range: float,
    radar_fov_horiz: float, 
    radar_fov_vert: float,  
    points_to_generate: int = 1000,
    velocity_std_dev: float = 0.3, 
) -> List[RadarPoint]:
    random_points: List[RadarPoint] = []
    
    T_world_radar = radar_transform

    T_world_camera: carla.Transform = camera.get_transform()
    T_camera_world_matrix_np = np.array(T_world_camera.get_inverse_matrix())
    M: List[List[float]] = T_camera_world_matrix_np.tolist()
    
    K: NDArray[np.float32] = camera_intrinsics 
    
    ego_vel_np = np.array([
        ego_vehicle_velocity.x, 
        ego_vehicle_velocity.y, 
        ego_vehicle_velocity.z
    ])
    
    h_fov_rad = np.deg2rad(radar_fov_horiz)
    v_fov_rad = np.deg2rad(radar_fov_vert)

    for i in range(points_to_generate):
        r = random.uniform(5.0, radar_range) 
        az = random.uniform(-h_fov_rad / 2.0, h_fov_rad / 2.0)
        el = random.uniform(-v_fov_rad / 2.0, v_fov_rad / 2.0)
        
        local_x = r * np.cos(el) * np.cos(az)
        local_y = r * np.cos(el) * np.sin(az)
        local_z = r * np.sin(el)
        
        p_radar_local = carla.Vector3D(x=local_x, y=local_y, z=local_z)
        p_world: carla.Vector3D = T_world_radar.transform(p_radar_local)

        r_vector_world_carla = T_world_radar.location - p_world 
        r_vector_world_np = np.array([
            r_vector_world_carla.x, 
            r_vector_world_carla.y, 
            r_vector_world_carla.z
        ])
        
        r_unit_np = r_vector_world_np / (np.linalg.norm(r_vector_world_np) + 1e-6)

        expected_radial_velocity_static = -float(np.dot(r_unit_np, ego_vel_np))
        
        noisy_radial_velocity = expected_radial_velocity_static + random.gauss(0, velocity_std_dev)

        x_cam: float = M[0][0]*p_world.x + M[0][1]*p_world.y + M[0][2]*p_world.z + M[0][3]
        y_cam: float = M[1][0]*p_world.x + M[1][1]*p_world.y + M[1][2]*p_world.z + M[1][3]
        z_cam: float = M[2][0]*p_world.x + M[2][1]*p_world.y + M[2][2]*p_world.z + M[2][3]

        if x_cam > 0: 
            u_norm: float = y_cam / x_cam
            v_norm: float = -z_cam / x_cam
            pixel_u: float = K[0, 0] * u_norm + K[0, 2]
            pixel_v: float = K[1, 1] * v_norm + K[1, 2]
            
            random_points.append(
                RadarPoint(
                    pixel_uv=(pixel_u, pixel_v),
                    local_x=local_x,
                    local_y=local_y,
                    local_z=local_z,
                    azimuth=az,
                    elevation=el,
                    range=r,
                    radial_velocity=noisy_radial_velocity,
                    vx_gt=0.0, vy_gt=0.0, vz_gt=0.0, 
                    vx_gt_world=0.0, vy_gt_world=0.0, vz_gt_world=0.0,
                    world_location=p_world,
                    noise_type=NoiseType.RANDOM,
                    object_type=ObjectType.STATIC, 
                    actor_id=0,
                    gt_local_x=x_cam,
                    gt_local_y=y_cam,
                    gt_local_z=z_cam,
                    gt_av_x=0.0,
                    gt_av_y=0.0,
                    gt_av_z=0.0,
                    gt_center_x=0.0,
                    gt_center_y=0.0,
                    gt_center_z=0.0,
                )
        )
            
    return random_points

def save_debug_image(
    filepath: str, 
    image_data: carla.Image, 
    points: List[RadarPoint],
    vel_color_map: Dict[Tuple[float, float, float], str],
    color_cycle: List[str],
    ghost_color: str,
    static_color: str,
    random_color: str
) -> None:
    try:
        img_rgb: NDArray[np.uint8] = process_image(image_data)
        img_height, img_width = img_rgb.shape[:2]
        
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)
        ax.imshow(img_rgb)
        
        plot_points_2d: List[Tuple[float, float]] = []
        plot_colors: List[str] = []
        
        if points:
            for p in points:
                plot_points_2d.append(p.pixel_uv)
                
                # --- UPDATED COLOR LOGIC ---
                color = ''
                if p.noise_type == NoiseType.MULTIPATH:
                    color = ghost_color
                elif p.noise_type == NoiseType.RANDOM:
                    color = random_color
                # --- New Noise Colors ---
                elif p.noise_type == NoiseType.SHIFTX:
                    color = '#FFA500' # Orange
                elif p.noise_type == NoiseType.SHIFTY:
                    color = '#00FFFF' # Cyan
                elif p.noise_type == NoiseType.SHIFTZ:
                    color = '#FFFF00' # Yellow
                elif p.noise_type == NoiseType.SHIFTRADIAL:
                    color = '#00FF00' # Lime Green
                else:
                    # REAL point
                    v_tuple = (round(p.vx_gt, 2), round(p.vy_gt, 2), round(p.vz_gt, 2))
                    
                    if v_tuple == (0.0, 0.0, 0.0):
                        color = static_color
                    elif v_tuple in vel_color_map:
                        color = vel_color_map[v_tuple]
                    else:
                        new_color_index = len(vel_color_map) % len(color_cycle)
                        color = color_cycle[new_color_index]
                        vel_color_map[v_tuple] = color
                
                plot_colors.append(color)
            
            ax.scatter(
                [uv[0] for uv in plot_points_2d], 
                [uv[1] for uv in plot_points_2d], 
                s=5, 
                c=plot_colors,
                alpha=0.7
            )

        ax.set_title(f"Frame {image_data.frame} - {len(points)} points")
        ax.set_axis_off()
        fig.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error saving debug image: {e}")

def save_ego_velocity(filepath: str, vehicle: carla.Actor) -> None:
    vel = vehicle.get_velocity()
    try:
        with open(filepath, 'w') as f:
            f.write(f"vx: {vel.x}\n")
            f.write(f"vy: {vel.y}\n")
            f.write(f"vz: {vel.z}\n")
    except Exception as e:
        print(f"Error saving velocity: {e}")

def save_radar_ply(filepath: str, points: List[RadarPoint]) -> None:
    try:
        with open(filepath, 'w') as f:
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
            f.write("property int actor_id\n")
            f.write("property float gt_x\n")
            f.write("property float gt_y\n")
            f.write("property float gt_z\n")
            f.write("property float gt_av_x\n")
            f.write("property float gt_av_y\n")
            f.write("property float gt_av_z\n")
            f.write("property float gt_center_x\n")
            f.write("property float gt_center_y\n")
            f.write("property float gt_center_z\n")
            f.write("end_header\n")

            for p in points:
                f.write(f"{p.local_x} {p.local_y} {p.local_z} "
                        f"{p.azimuth} {p.elevation} {p.range} "
                        f"{p.radial_velocity} "
                        f"{p.vx_gt} {p.vy_gt} {p.vz_gt} "
                        f"{p.vx_gt_world} {p.vy_gt_world} {p.vz_gt_world} "
                        f"{p.noise_type.value} {p.object_type.value} "
                        f"{p.actor_id} "
                        f"{p.gt_local_x} {p.gt_local_y} {p.gt_local_z} "
                        f"{p.gt_av_x} {p.gt_av_y} {p.gt_av_z} "
                        f"{p.gt_center_x} {p.gt_center_y} {p.gt_center_z}\n")
                
    except Exception as e:
        print(f"Error saving PLY file: {e}")

def transform_points_to_paper_coords(points: List[RadarPoint]) -> List[RadarPoint]:
    transformed_points: List[RadarPoint] = []
    for p in points:
        paper_local_x = p.local_y   
        paper_local_y = -p.local_z  
        paper_local_z = p.local_x

        paper_gt_local_x = p.gt_local_y   
        paper_gt_local_y = -p.gt_local_z   
        paper_gt_local_z = p.gt_local_x   

        paper_gt_center_x = p.gt_center_y
        paper_gt_center_y = -p.gt_center_z
        paper_gt_center_z = p.gt_center_x
        
        paper_vx_gt = p.vy_gt       
        paper_vy_gt = -p.vz_gt      
        paper_vz_gt = p.vx_gt       

        paper_vx_gt_world = p.vy_gt_world   
        paper_vy_gt_world = -p.vz_gt_world  
        paper_vz_gt_world = p.vx_gt_world   

        paper_av_x = -p.gt_av_y       # -(  y )
        paper_av_y = p.gt_av_z        # -(- z ) -> +z
        paper_av_z = -p.gt_av_x       # -(  x )

        paper_azimuth: float
        paper_elevation: float
        
        if p.range > 1e-6: 
            paper_azimuth = math.atan2(paper_local_x, paper_local_z)
            asin_arg = max(-1.0, min(1.0, paper_local_y / p.range))
            paper_elevation = math.asin(asin_arg)
        else:
            paper_azimuth = 0.0
            paper_elevation = 0.0

        transformed_points.append(RadarPoint(
            pixel_uv=p.pixel_uv,
            local_x=paper_local_x,   
            local_y=paper_local_y,   
            local_z=paper_local_z,   
            azimuth=paper_azimuth,     
            elevation=paper_elevation, 
            range=p.range,           
            radial_velocity=p.radial_velocity, 
            vx_gt=paper_vx_gt,       
            vy_gt=paper_vy_gt,       
            vz_gt=paper_vz_gt,       
            vx_gt_world=paper_vx_gt_world,
            vy_gt_world=paper_vy_gt_world,
            vz_gt_world=paper_vz_gt_world,
            world_location=p.world_location, 
            noise_type=p.noise_type,
            object_type=p.object_type,
            actor_id=p.actor_id,
            gt_local_x=paper_gt_local_x,
            gt_local_y=paper_gt_local_y,
            gt_local_z=paper_gt_local_z,
            gt_av_x=paper_av_x,
            gt_av_y=paper_av_y,
            gt_av_z=paper_av_z,
            gt_center_x=paper_gt_center_x,
            gt_center_y=paper_gt_center_y,
            gt_center_z=paper_gt_center_z
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

    global_velocity_to_color_map: Dict[Tuple[float, float, float], str] = {}
    color_cycle_list: List[str] = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF',
        '#FF00FF', '#FFA500', '#800080', '#FFC0CB', '#008000'
    ]
    MULTIPATH_COLOR = '#FF00FF' 
    STATIC_COLOR = '#FFFFFF'    
    RANDOM_COLOR = '#000000' 
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world: carla.World = client.get_world()
        blueprint_library: carla.BlueprintLibrary = world.get_blueprint_library()
        original_settings = world.get_settings()
        
        vehicle_bp: carla.ActorBlueprint = blueprint_library.find('vehicle.tesla.model3')
        spawn_point: carla.Transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        ego_vehicle_id = vehicle.id
        print(f'Spawned vehicle: {vehicle.type_id} (ID: {ego_vehicle_id})')

        tm: carla.TrafficManager = client.get_trafficmanager(tm_port)
        tm_port = tm.get_port()
        vehicle.set_autopilot(True, tm_port)
        # 1. Ignore Traffic Lights (100% chance)
        tm.ignore_lights_percentage(vehicle, 100.0)

        print(f'Vehicle set to autopilot on port {tm_port}')

        co_located_transform: carla.Transform = carla.Transform(carla.Location(x=2.7, z=1.0))

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

        radar_bp: carla.ActorBlueprint = blueprint_library.find('sensor.other.radar')
        radar_fov_horiz: float = 70.0
        radar_fov_vert: float = 40.0
        radar_range: float = 40.0
        radar_bp.set_attribute('points_per_second', '45000')
        radar_bp.set_attribute('horizontal_fov', str(radar_fov_horiz))
        radar_bp.set_attribute('vertical_fov', str(radar_fov_vert))
        radar_bp.set_attribute('range', str(radar_range))
        radar = world.spawn_actor(radar_bp, co_located_transform, attach_to=vehicle)
        actor_list.append(radar)
        print(f'Spawned sensor: {radar.type_id}')
        
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
            intrinsics_path = os.path.join(CALIB_DIR, "intrinsics.txt")
            np.savetxt(intrinsics_path, K_camera, fmt='%.6f')
            print(f"Saved camera intrinsics to {intrinsics_path}")
            
            C_paper_from_carla = np.array([
                [0, 1,  0, 0], 
                [0, 0, -1, 0], 
                [1, 0,  0, 0], 
                [0, 0,  0, 1]
            ], dtype=np.float32)
            C_carla_from_paper = np.linalg.inv(C_paper_from_carla)

            cam_pose_carla_np = np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float32)
            rad_pose_carla_np = np.array(radar.get_transform().get_inverse_matrix(), dtype=np.float32)

            extrinsics_carla_np = rad_pose_carla_np @ np.linalg.inv(cam_pose_carla_np)

            extrinsics_matrix = C_paper_from_carla @ extrinsics_carla_np @ C_carla_from_paper
            
            extrinsics_path = os.path.join(CALIB_DIR, "extrinsics_radar_from_camera.txt")
            np.savetxt(extrinsics_path, extrinsics_matrix, fmt='%.8f')
            print(f"Saved extrinsics (Radar from Camera) to {extrinsics_path}")

        except Exception as e:
            print(f"--- ERROR SAVING CALIBRATION ---")
            print(e)
            print("---------------------------------")

        settings: carla.WorldSettings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 
        world.apply_settings(settings)
        print(f"Set simulation to {1.0/settings.fixed_delta_seconds:.0f} FPS")

        camera_queue: "queue.Queue[carla.Image]" = queue.Queue()
        radar_queue: "queue.Queue[carla.RadarMeasurement]" = queue.Queue()
        instance_camera_queue: "queue.Queue[carla.Image]" = queue.Queue()
        
        camera.listen(camera_queue.put) 
        radar.listen(radar_queue.put)
        instance_camera.listen(instance_camera_queue.put)

        print("Setup complete. Running simulation to save data...")
        print("Press Ctrl+C in this terminal to stop.")

        prev_cam_pose_carla_np: Optional[NDArray[np.float32]] = None
        
        while True:
            frame_id: int = world.tick()

            try:
                timeout_seconds = 5.0
                while True:
                    image_data: carla.Image = camera_queue.get(timeout=timeout_seconds)
                    if image_data.frame == frame_id: break 
                    if image_data.frame > frame_id: raise queue.Empty 
                
                while True:
                    radar_data: carla.RadarMeasurement = radar_queue.get(timeout=timeout_seconds)
                    if radar_data.frame == frame_id: break 
                    if radar_data.frame > frame_id: raise queue.Empty
                    
                while True:
                    instance_image_data: carla.Image = instance_camera_queue.get(timeout=timeout_seconds)
                    if instance_image_data.frame == frame_id: break 
                    if instance_image_data.frame > frame_id: raise queue.Empty
                
                instance_image_array: NDArray[np.uint8] = process_instance_image(instance_image_data)
                radar_world_transform: carla.Transform = radar.get_transform()
                camera_world_transform: carla.Transform = camera.get_transform()
                
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

                # ghost_points = generate_actor_scattering_points(
                #     real_points,
                #     world,
                #     instance_camera,
                #     K_instance,
                #     scattering_ratio=0.7, 
                #     scatter_margin=0.5,   
                #     velocity_noise=0.5
                # )
                
                # random_clutter_points: List[RadarPoint] = generate_static_clutter(
                #     instance_camera,
                #     K_instance,
                #     radar_world_transform,
                #     ego_vehicle_velocity, 
                #     radar_range,
                #     radar_fov_horiz,
                #     radar_fov_vert,
                #     points_to_generate=500, 
                #     velocity_std_dev=0.3 
                # )

                # --- NEW: Generate Noise based on 2D Contour ---
                actor_noise_points = generate_actor_noise_types(
                    real_points,
                    world,
                    radar_world_transform,
                    camera_world_transform,
                    K_instance,
                    cam_w, 
                    cam_h,
                    instance_image_array, 
                    percentage=0.123
                )

                all_points = real_points + actor_noise_points
                
                # --- Save data to disk ---
                filename_id = f"{frame_id:08d}"

                cam_filepath = os.path.join(CAM_DIR, f"{filename_id}.png")
                image_data.save_to_disk(cam_filepath)

                debug_filepath = os.path.join(DEBUG_DIR, f"{filename_id}.png")
                save_debug_image(
                    debug_filepath, 
                    image_data, 
                    all_points,
                    global_velocity_to_color_map,
                    color_cycle_list,
                    MULTIPATH_COLOR,
                    STATIC_COLOR,
                    RANDOM_COLOR
                )
                
                vel_filepath = os.path.join(VEL_DIR, f"{filename_id}.txt")
                save_ego_velocity(vel_filepath, vehicle)
                
                final_points_for_ply = transform_points_to_paper_coords(all_points)
                ply_filepath = os.path.join(PLY_DIR, f"{filename_id}.ply")
                save_radar_ply(ply_filepath, final_points_for_ply)

                try:
                    C_paper_from_carla = np.array([
                        [0, 1,  0, 0], 
                        [0, 0, -1, 0], 
                        [1, 0,  0, 0], 
                        [0, 0,  0, 1]
                    ], dtype=np.float32)
                    C_carla_from_paper = np.linalg.inv(C_paper_from_carla)

                    curr_cam_pose_carla_np = np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float32)

                    if prev_cam_pose_carla_np is not None:
                        Pose_A_carla = prev_cam_pose_carla_np
                        Pose_B_carla = curr_cam_pose_carla_np
                        
                        relative_pose_carla = Pose_B_carla @ np.linalg.inv(Pose_A_carla)
                        
                        relative_pose_matrix = C_paper_from_carla @ relative_pose_carla @ C_carla_from_paper
                        
                        rel_pose_path = os.path.join(POSES_DIR, f"{filename_id}_relative_pose.txt")
                        np.savetxt(rel_pose_path, relative_pose_matrix, fmt='%.8f')
                    
                    prev_cam_pose_carla_np = curr_cam_pose_carla_np

                except Exception as e:
                    print(f"--- ERROR SAVING POSES for frame {frame_id} ---")
                    print(e)
                
                print(f"Frame {frame_id}: Saved {len(all_points)} points.")

            except queue.Empty:
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