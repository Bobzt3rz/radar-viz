import carla
import random
import queue
import numpy as np
import os
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict
from numpy.typing import NDArray

# --- Enums ---
class NoiseType(Enum):
    REAL = 0
    MULTIPATH = 1
    RANDOM = 2

class ObjectType(Enum):
    STATIC = 0
    ACTOR = 1

@dataclass
class RadarPoint:
    local_x: float; local_y: float; local_z: float
    azimuth: float; elevation: float; range: float
    radial_velocity: float
    vx_gt: float; vy_gt: float; vz_gt: float
    vx_gt_world: float; vy_gt_world: float; vz_gt_world: float
    noise_type: NoiseType; actor_id: int
    pixel_uv: Tuple[float, float]
    world_location: carla.Vector3D; object_type: ObjectType

# --- Helper Functions ---
def build_camera_intrinsics(image_w: int, image_h: int, fov: float) -> NDArray[np.float32]:
    f = image_w / (2 * np.tan(fov * np.pi / 360))
    K = np.array([[f, 0.0, image_w/2.0], [0.0, f, image_h/2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def process_instance_image(image_data: carla.Image) -> NDArray[np.uint8]:
    array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
    return np.reshape(array, (image_data.height, image_data.width, 4))

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
    
    projected_points = []
    # 1. Get Transform World -> Camera Actor (UE4 coords: X-Fwd, Y-Right, Z-Up)
    T_cam_world = np.array(camera.get_transform().get_inverse_matrix())
    img_h, img_w = instance_image.shape[:2]
    mistag_count = 0
    
    # Rotation Matrix for GT Velocity (World -> Radar)
    R_rad_world = np.array(radar_transform.get_inverse_matrix())[:3, :3]

    for d in radar_data:
        # 2. Radar Local (Spherical) -> World
        # CARLA Radar local: X is forward.
        lx = d.depth * np.cos(d.altitude) * np.cos(d.azimuth)
        ly = d.depth * np.cos(d.altitude) * np.sin(d.azimuth)
        lz = d.depth * np.sin(d.altitude)
        p_local = carla.Location(lx, ly, lz)
        p_world = radar_transform.transform(p_local)
        
        # 3. World -> Camera Actor (UE4 Coords)
        p_cam_ue_h = np.dot(T_cam_world, [p_world.x, p_world.y, p_world.z, 1.0])
        
        # 4. UE4 Camera -> CV Camera (Standard Pinhole)
        # UE4: X=Fwd, Y=Right, Z=Up
        # CV:  Z=Fwd, X=Right, Y=Down
        p_cam_cv = np.array([
            p_cam_ue_h[1],   # CV X (Right) = UE4 Y
            -p_cam_ue_h[2],  # CV Y (Down)  = -UE4 Z
            p_cam_ue_h[0]    # CV Z (Fwd)   = UE4 X
        ])
        
        # 5. Project
        if p_cam_cv[2] > 0: # Check Z-forward > 0
            u_norm = p_cam_cv[0] / p_cam_cv[2]
            v_norm = p_cam_cv[1] / p_cam_cv[2]
            
            u = camera_intrinsics[0,0] * u_norm + camera_intrinsics[0,2]
            v = camera_intrinsics[1,1] * v_norm + camera_intrinsics[1,2]

            if 0 <= int(u) < img_w and 0 <= int(v) < img_h:
                # --- GT Logic (Same as before) ---
                obj_type = ObjectType.STATIC; final_id = 0
                gt_vel_world = carla.Vector3D(0,0,0)
                
                b, g = instance_image[int(v), int(u), 0], instance_image[int(v), int(u), 1]
                oid = (int(b) * 256) + int(g)
                
                if oid != 0 and oid != ego_vehicle_id:
                    actor = world.get_actor(oid)
                    if actor and ('vehicle' in actor.type_id or 'walker' in actor.type_id):
                        gt_vel_world = actor.get_velocity()
                        obj_type = ObjectType.ACTOR; final_id = oid

                # Relative Velocity
                rel_vel_world = gt_vel_world - ego_vehicle_velocity
                rel_vel_world_np = np.array([rel_vel_world.x, rel_vel_world.y, rel_vel_world.z])
                vel_rad_local = R_rad_world @ rel_vel_world_np
                
                # Static Check
                if obj_type == ObjectType.STATIC and ego_vehicle_velocity.length() > 0.1:
                    if abs(d.velocity - vel_rad_local[0]) > 1.0: 
                        mistag_count += 1

                projected_points.append(RadarPoint(
                    local_x=lx, local_y=ly, local_z=lz,
                    azimuth=d.azimuth, elevation=d.altitude, range=d.depth,
                    radial_velocity=d.velocity,
                    vx_gt=vel_rad_local[0], vy_gt=vel_rad_local[1], vz_gt=vel_rad_local[2],
                    vx_gt_world=gt_vel_world.x, vy_gt_world=gt_vel_world.y, vz_gt_world=gt_vel_world.z,
                    noise_type=NoiseType.REAL, actor_id=final_id,
                    pixel_uv=(u, v), world_location=p_world, object_type=obj_type
                ))
    return projected_points, mistag_count

def generate_multipath_points(real_points, world, camera, K, ego_vel):
    ghosts = []
    # 1. Get Transform World -> Camera Actor (UE4 coords: X-Fwd, Y-Right, Z-Up)
    T_cam_world = np.array(camera.get_transform().get_inverse_matrix())
    img_h, img_w = K[1,2]*2, K[0,2]*2
    
    for p in real_points:
        if p.object_type == ObjectType.ACTOR:
            try:
                # 2. Reflection Logic: Mirror point under the road surface
                wpt = world.get_map().get_waypoint(p.world_location)
                road_z = wpt.transform.location.z
                
                # Height of real point above road
                height = p.world_location.z - road_z
                
                # If point is already below road (weird), skip
                # Note: We mirror it to be 'height' distance BELOW the road_z
                p_ghost_world = carla.Location(p.world_location.x, p.world_location.y, road_z - height)
                
                # 3. Project World -> UE4 Camera Frame
                p_cam_ue_h = np.dot(T_cam_world, [p_ghost_world.x, p_ghost_world.y, p_ghost_world.z, 1.0])
                
                # 4. UE4 Camera -> CV Camera (The Critical Fix)
                # UE4: X=Fwd, Y=Right, Z=Up
                # CV:  Z=Fwd, X=Right, Y=Down
                p_cam_cv = np.array([
                    p_cam_ue_h[1],   # CV X (Right) = UE4 Y
                    -p_cam_ue_h[2],  # CV Y (Down)  = -UE4 Z
                    p_cam_ue_h[0]    # CV Z (Fwd)   = UE4 X
                ])
                
                # 5. Project using CV Coordinates
                if p_cam_cv[2] > 0: # Forward > 0
                    u_norm = p_cam_cv[0] / p_cam_cv[2]
                    v_norm = p_cam_cv[1] / p_cam_cv[2]
                    
                    u = K[0,0] * u_norm + K[0,2]
                    v = K[1,1] * v_norm + K[1,2]
                    
                    if 0 <= int(u) < img_w and 0 <= int(v) < img_h:

                        # --- FIX STARTS HERE ---
                        # We must convert the p_ghost_world back to Radar Local coordinates
                        # so the solver sees it at the wrong position.
                        
                        # 1. World -> Radar Transform (Inverse of Radar->World)
                        # We can get this from the camera since they are co-located, 
                        # or pass radar_transform into this function.
                        # Let's infer it: Local = T_world_radar_inv * World
                        
                        # Easier way: We know the radar is at (0,0,0) relative to itself. 
                        # But calculating the inverse transform matrix is safer.
                        T_world_radar_inv = np.array(camera.get_transform().get_inverse_matrix()) # Assuming co-located
                        
                        p_ghost_local_h = np.dot(T_world_radar_inv, [p_ghost_world.x, p_ghost_world.y, p_ghost_world.z, 1.0])
                        
                        # Convert back to spherical for the RadarPoint (Optional, but good for consistency)
                        # But RadarPoint takes local_x, y, z cartesian.
                        # Note: CARLA Local X is Forward.
                        gx, gy, gz = p_ghost_local_h[0], p_ghost_local_h[1], p_ghost_local_h[2]
                        
                        # Recompute Range/Az/El for the ghost
                        g_range = math.sqrt(gx**2 + gy**2 + gz**2)
                        g_az = math.atan2(gy, gx)
                        g_el = math.asin(gz / (g_range + 1e-6))

                        ghosts.append(RadarPoint(
                            # USE THE GHOST COORDINATES
                            local_x=gx, local_y=gy, local_z=gz, 
                            azimuth=g_az, elevation=g_el, range=g_range,
                            
                            radial_velocity=p.radial_velocity, 
                            vx_gt=p.vx_gt, vy_gt=p.vy_gt, vz_gt=p.vz_gt,
                            vx_gt_world=p.vx_gt_world, vy_gt_world=p.vy_gt_world, vz_gt_world=p.vz_gt_world,
                            noise_type=NoiseType.MULTIPATH, actor_id=p.actor_id,
                            pixel_uv=(u, v), world_location=p_ghost_world, object_type=p.object_type
                        ))
                        
            except Exception as e: 
                pass
    return ghosts

def generate_random_noise(
    num_points: int,
    radar_transform: carla.Transform,
    camera: carla.Actor,
    K: NDArray[np.float32],
    image_w: int, 
    image_h: int,
    fov_h_deg: float = 70.0,
    fov_v_deg: float = 40.0,
    max_range: float = 100.0,
    min_range: float = 5.0
) -> List[RadarPoint]:
    
    noise_points = []
    
    # 1. Precompute Transforms
    # World -> UE4 Camera
    T_cam_world = np.array(camera.get_transform().get_inverse_matrix())
    
    # Convert FOV to radians
    h_lim = np.deg2rad(fov_h_deg) / 2.0
    v_lim = np.deg2rad(fov_v_deg) / 2.0
    
    for _ in range(num_points):
        # 2. Generate Random Spherical Coords (Radar Frame)
        r = random.uniform(min_range, max_range)
        az = random.uniform(-h_lim, h_lim)
        el = random.uniform(-v_lim, v_lim)
        
        # 3. Spherical -> Cartesian (Radar Local - UE4: X-Fwd)
        lx = r * np.cos(el) * np.cos(az)
        ly = r * np.cos(el) * np.sin(az)
        lz = r * np.sin(el)
        p_local = carla.Location(lx, ly, lz)
        
        # 4. Local -> World
        p_world = radar_transform.transform(p_local)
        
        # 5. World -> UE4 Camera
        p_cam_ue_h = np.dot(T_cam_world, [p_world.x, p_world.y, p_world.z, 1.0])
        
        # 6. UE4 Camera -> CV Camera (Fix)
        p_cam_cv = np.array([p_cam_ue_h[1], -p_cam_ue_h[2], p_cam_ue_h[0]])
        
        # 7. Project
        if p_cam_cv[2] > 0:
            u_norm = p_cam_cv[0] / p_cam_cv[2]
            v_norm = p_cam_cv[1] / p_cam_cv[2]
            
            u = K[0,0] * u_norm + K[0,2]
            v = K[1,1] * v_norm + K[1,2]
            
            if 0 <= int(u) < image_w and 0 <= int(v) < image_h:
                # 8. Assign Random Velocity
                # Random velocity between -30 m/s and +30 m/s to simulate noise
                rand_vel = random.uniform(-30.0, 30.0)
                
                noise_points.append(RadarPoint(
                    local_x=lx, local_y=ly, local_z=lz,
                    azimuth=az, elevation=el, range=r,
                    radial_velocity=rand_vel,
                    # GT Velocity is technically 0 or undefined for pure noise
                    vx_gt=0.0, vy_gt=0.0, vz_gt=0.0,
                    vx_gt_world=0.0, vy_gt_world=0.0, vz_gt_world=0.0,
                    noise_type=NoiseType.RANDOM, actor_id=0,
                    pixel_uv=(u, v), world_location=p_world, object_type=ObjectType.STATIC
                ))
                
    return noise_points

def transform_to_paper_coords(points):
    """
    Converts CARLA (Left-Handed: X-Fwd, Y-Right, Z-Up) 
    to PAPER (Right-Handed: Z-Fwd, X-Right, Y-Down)
    """
    out = []
    for p in points:
        # Pos: Paper(X,Y,Z) -> Carla(Y, -Z, X)
        # Vel: Paper(X,Y,Z) -> Carla(VY, -VZ, VX)
        # Recompute azimuth/elevation for new frame
        paper_x, paper_y, paper_z = p.local_y, -p.local_z, p.local_x
        paper_az = math.atan2(paper_x, paper_z)
        paper_el = math.asin(paper_y / (p.range + 1e-6))
        
        out.append(RadarPoint(
            local_x=paper_x, local_y=paper_y, local_z=paper_z,
            azimuth=paper_az, elevation=paper_el, range=p.range,
            radial_velocity=p.radial_velocity,
            vx_gt=p.vy_gt, vy_gt=-p.vz_gt, vz_gt=p.vx_gt,
            vx_gt_world=p.vy_gt_world, vy_gt_world=-p.vz_gt_world, vz_gt_world=p.vx_gt_world,
            noise_type=p.noise_type, actor_id=p.actor_id,
            pixel_uv=p.pixel_uv, world_location=p.world_location, object_type=p.object_type
        ))
    return out

def save_radar_ply(filepath, points):
    with open(filepath, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        props = ["x","y","z","azimuth","elevation","range","radial_velocity",
                 "vx_gt","vy_gt","vz_gt","vx_gt_world","vy_gt_world","vz_gt_world",
                 "noise_type","object_type","actor_id"]
        for p in props: f.write(f"property float {p}\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p.local_x} {p.local_y} {p.local_z} {p.azimuth} {p.elevation} {p.range} "
                    f"{p.radial_velocity} {p.vx_gt} {p.vy_gt} {p.vz_gt} "
                    f"{p.vx_gt_world} {p.vy_gt_world} {p.vz_gt_world} "
                    f"{p.noise_type.value} {p.object_type.value} {p.actor_id}\n")

# --- MAIN ---
def main():
    try:
        client = carla.Client('localhost', 2000); client.set_timeout(10.0)
        world = client.get_world(); bp_lib = world.get_blueprint_library()
        OUT = "output_sim"
        for d in ["velocities", "radar_ply", "camera_rgb", "poses", "calib"]:
            os.makedirs(os.path.join(OUT, d), exist_ok=True)

        # 1. Spawn Actors
        ego = world.spawn_actor(bp_lib.find('vehicle.tesla.model3'), random.choice(world.get_map().get_spawn_points()))
        ego.set_autopilot(True)
        
        tf = carla.Transform(carla.Location(x=1.5, z=2.0)) # Mounted high
        
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280'); cam_bp.set_attribute('image_size_y', '720')
        cam = world.spawn_actor(cam_bp, tf, attach_to=ego)
        
        rad_bp = bp_lib.find('sensor.other.radar')
        rad_bp.set_attribute('horizontal_fov', '70'); rad_bp.set_attribute('vertical_fov', '40')
        rad_bp.set_attribute('range', '100')
        rad = world.spawn_actor(rad_bp, tf, attach_to=ego)
        
        seg_bp = bp_lib.find('sensor.camera.instance_segmentation')
        seg_bp.set_attribute('image_size_x', '1280'); seg_bp.set_attribute('image_size_y', '720')
        seg = world.spawn_actor(seg_bp, tf, attach_to=ego)

        # 2. Calibration
        K = build_camera_intrinsics(1280, 720, 90.0)
        np.savetxt(os.path.join(OUT, "calib", "intrinsics.txt"), K)
        
        # Change of Basis: Carla(LH) -> Paper(RH)
        CoB = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        
        # Extrinsics (Paper Frame)
        T_cam_carla = np.array(cam.get_transform().get_inverse_matrix())
        T_rad_carla = np.array(rad.get_transform().get_inverse_matrix())
        # T_rad_from_cam = T_rad_world * inv(T_cam_world)
        # Note: get_inverse_matrix returns World->Sensor. So T_sensor_world.
        # P_cam = T_cam_world * P_w => P_w = inv(T_cam_world) * P_cam
        # P_rad = T_rad_world * P_w => P_rad = T_rad_world * inv(T_cam_world) * P_cam
        T_rel_carla = T_rad_carla @ np.linalg.inv(T_cam_carla)
        T_rel_paper = CoB @ T_rel_carla @ np.linalg.inv(CoB)
        np.savetxt(os.path.join(OUT, "calib", "extrinsics.txt"), T_rel_paper)

        # 3. Queues & Sync
        q_cam = queue.Queue(); cam.listen(q_cam.put)
        q_rad = queue.Queue(); rad.listen(q_rad.put)
        q_seg = queue.Queue(); seg.listen(q_seg.put)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        prev_view_matrix = None
        print("Running Simulation...")
        
        for i in range(100):
            world.tick()
            img = q_cam.get(); rad_msmt = q_rad.get(); seg_img = q_seg.get()
            img_arr = process_instance_image(seg_img)
            ego_vel = ego.get_velocity()
            
            # Processing
            real_pts, _ = radar_to_camera_projection(rad_msmt, cam, K, rad.get_transform(), img_arr, world, ego.id, ego_vel)
            ghost_pts = generate_multipath_points(real_pts, world, cam, K, ego_vel)
            # Generate 50 random noise points per frame
            noise_pts = generate_random_noise(
                num_points=50,
                radar_transform=rad.get_transform(),
                camera=cam,
                K=K,
                image_w=1280, image_h=720,
                fov_h_deg=70.0, fov_v_deg=40.0
            )
            
            # 4. Combine all
            all_pts_ue4 = real_pts + ghost_pts + noise_pts
            paper_pts = transform_to_paper_coords(all_pts_ue4)
            
            # Save
            id_str = f"{img.frame:06d}"
            img.save_to_disk(os.path.join(OUT, "camera_rgb", f"{id_str}.png"))
            save_radar_ply(os.path.join(OUT, "radar_ply", f"{id_str}.ply"), paper_pts)
            
            # 4. Save Pose T_Prev_to_Curr (in Paper Frame)
            curr_view_matrix = np.array(cam.get_transform().get_inverse_matrix())
            if prev_view_matrix is not None:
                # T_prev_to_curr = V_curr * inv(V_prev)
                # Check: P_curr = V_curr * P_w; P_prev = V_prev * P_w => P_w = inv(V_prev) * P_prev
                # P_curr = V_curr * inv(V_prev) * P_prev. CORRECT.
                T_A_to_B_carla = curr_view_matrix @ np.linalg.inv(prev_view_matrix)
                T_A_to_B_paper = CoB @ T_A_to_B_carla @ np.linalg.inv(CoB)
                np.savetxt(os.path.join(OUT, "poses", f"{id_str}.txt"), T_A_to_B_paper)
            
            prev_view_matrix = curr_view_matrix
            print(f"Frame {id_str}: Saved {len(paper_pts)} points")

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in [cam, rad, seg, ego]: a.destroy()

if __name__ == "__main__":
    main()