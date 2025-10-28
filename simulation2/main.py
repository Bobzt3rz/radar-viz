import sys
import numpy as np
import cv2

from modules.velocity_solver import solve_full_velocity
from modules.world import World
from modules.camera import Camera
from modules.radar import Radar
from modules.cube import Cube
from modules.renderer import Renderer
from modules.optical_flow import OpticalFlow
from modules.utils import save_image

if __name__ == "__main__":

    world = World(delta_t=1/60.0)

    # Add entities BEFORE creating the renderer
    camera = Camera(
        position=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.5]),
        # Adjust cx, cy if Y-down origin was used during calibration
        fx=800.0, fy=800.0, cx=1280/2, cy=720/2, image_width=1280, image_height=720
    )
    radar = Radar(
        position=np.array([0.0, 0.5, 0.0]),
        velocity=np.array([0.0, 0.0, 0.5]),
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
        position=np.array([0.0, 0.4, 3.0]), # Closer
        velocity=np.array([1.0, -1.5, 0.5]), # Slower
        rotation=rotation_matrix,
        size=0.5
    )
    static_cube = Cube(
        position=np.array([-2.0, 0.0, 7.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size=1.0
    )

    world.add_entity(camera)
    world.add_entity(radar)
    world.add_entity(target_cube)
    # world.add_entity(static_cube)

    # Initialize Renderer
    try:
        renderer = Renderer(world, width=1280, height=720)
    except Exception as e:
        print(f"Failed to initialize Renderer: {e}")
        exit()

    optical_flow_calculator = OpticalFlow()

    # --- Main Loop ---
    print("\nStarting simulation loop...")
    frame_count = 0
    max_frames = 300 # Run for 5 seconds at 60fps

    # --- Store state from the previous frame ---
    prev_poses = {} # Store world_to_local poses from time t

    while not renderer.should_close() and frame_count < max_frames:

        # --- A. Store State from Time 't' (before step) ---
        # Store poses needed for T_A_to_B, T_A_to_R, and depth calculation
        prev_poses = {
            'camera': camera.get_pose_world_to_local(),
             'radar': radar.get_pose_world_to_local(),
             # Store entity states if needed for depth calculation
             'entities': {entity: {'position': entity.position.copy(),
                                     'rotation': entity.rotation.copy()}
                           for entity in world.entities if isinstance(entity, Cube)}
        }
        # If generate_point_cloud doesn't return entity/corner info,
        # you might need a more complex way to associate points across time.

        # Update simulation state
        world.step()

        radar_detections = radar.generate_point_cloud(world)

        # Render the current state
        renderer.render_scene(near=0.1, far=100.0)

        current_frame_rgb = renderer.capture_frame()

        save_image(current_frame_rgb, f"output/frame_{frame_count:04d}.png")

        flow = optical_flow_calculator.inference(current_frame_rgb)

        # --- F. Calculate Full Velocity (if possible) ---
        if frame_count > 0 and flow is not None and prev_poses: # Need previous state and flow
            # 1. Get current poses (t+delta_t)
            current_cam_pose_W2L = camera.get_pose_world_to_local()
            # current_radar_pose_W2L = radar.get_pose_world_to_local() # Might not be needed directly

            # 2. Get previous poses (t)
            prev_cam_pose_W2L = prev_poses['camera']
            prev_radar_pose_W2L = prev_poses['radar']

            print(f"DETECTION FRAME {frame_count}")

            # 3. Calculate transformation matrices relative to Camera A (time t)
            # T_A_to_B: Pose B relative to A = W_B^-1 @ W_A
            T_A_to_B = current_cam_pose_W2L @ np.linalg.inv(prev_cam_pose_W2L)
            # T_A_to_R: Pose R (at t) relative to A = W_R^-1 @ W_A
            T_A_to_R = prev_radar_pose_W2L @ np.linalg.inv(prev_cam_pose_W2L)
            print(f"T_A_to_B: {T_A_to_B}")
            print(f"T_A_to_R: {T_A_to_R}")

            for detection in radar_detections:
                 # Unpack detection info (assuming modified generate_point_cloud)
                 point_radar_coords, speed_radial, source_entity, corner_idx = detection

                 # --- Calculate (xq_pix, yq_pix) and (uq, vq) at t+delta_t ---
                 # Convert radar point (t+delta_t) to world (t+delta_t)
                 point_rad_h = np.append(point_radar_coords, 1.0)
                 print(f"point_rad_h: {point_rad_h}")
                 # Need M_radar_to_world at t+delta_t (inverse of current radar W2L)
                 M_radar_to_world_B = np.linalg.inv(radar.get_pose_world_to_local())
                 print(f"M_radar_to_world_B: {M_radar_to_world_B}")
                 point_world_B_h = M_radar_to_world_B @ point_rad_h
                 print(f"point_world_B_h: {point_world_B_h}")

                 # Convert world (t+delta_t) to camera (t+delta_t)
                 point_cam_B_h = current_cam_pose_W2L @ point_world_B_h
                 print(f"point_cam_B_h: {point_cam_B_h}")
                 point_cam_B = point_cam_B_h[:3]
                 depth_B = point_cam_B[2]
                 print(f"depth_B: {depth_B}")

                 if depth_B <= 1e-3: continue # Point is behind or too close to camera B

                 # Normalized coords (t+delta_t)
                 uq = point_cam_B[0] / depth_B
                 vq = point_cam_B[1] / depth_B
                 print(f"uq: {uq}, vq: {vq}")

                 # Pixel coords (t+delta_t) - careful with int conversion if needed early
                 xq_pix_f = camera.fx * uq + camera.cx
                 yq_pix_f = camera.fy * -1 * vq + camera.cy
                 xq_pix = int(round(xq_pix_f))
                 yq_pix = int(round(yq_pix_f))
                 print(f"xq_pix: {xq_pix}, yq_pix: {yq_pix}")

                 # Check image bounds
                 if not (0 <= xq_pix < camera.image_width and 0 <= yq_pix < camera.image_height):
                     continue

                 # --- Get flow and calculate (up, vp) at t ---
                 dx, dy = flow[yq_pix, xq_pix] # Pixel flow
                 print(f"dx: {dx}, dy: {dy}")
                 xp_pix_f = xq_pix_f - dx
                 yp_pix_f = yq_pix_f - dy
                 print(f"xp_pix_f: {xp_pix_f}, yp_pix_f: {yp_pix_f}")

                 # Normalized coords (t)
                 up = (xp_pix_f - camera.cx) / camera.fx
                 vp = -(yp_pix_f - camera.cy) / camera.fy
                 print(f"up: {up}, vp: {vp}")

                 # --- Call the solver ---
                 full_vel_vector_radar = solve_full_velocity(
                     up=up, vp=vp, uq=uq, vq=vq, d=depth_B, delta_t=world.delta_t,
                     T_A_to_B=T_A_to_B, T_A_to_R=T_A_to_R,
                     speed_radial=speed_radial, point_radar_coords=point_radar_coords,
                     return_in_radar_coords=True # Or False if you want camera coords
                 )

                 if full_vel_vector_radar is not None:
                     print(f"  > Solved Vel (Radar Coords): {full_vel_vector_radar.round(3)} for corner {corner_idx} of {type(source_entity).__name__}")
                     # Optional: Convert to world and compare to ground truth source_entity.velocity
                     R_radar_to_world_A = np.linalg.inv(prev_radar_pose_W2L[0:3,0:3]) # Rotation of Radar at time t
                     full_vel_world = R_radar_to_world_A @ full_vel_vector_radar
                     print(f"    GT Vel (World Coords):   {source_entity.velocity.round(3)}")
                     print(f"    Calc Vel (World Coords): {full_vel_world.round(3)}")

                 # just print out first point
                 break

        # Handle window events and display frame
        renderer.swap_buffers_and_poll_events()
        
        frame_count += 1
        # print(f"Frame: {frame_count}, Time: {world.current_time:.2f}") # Optional debug print

    # Cleanup
    print("\nSimulation loop finished.")
    renderer.cleanup()