import sys
import time
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

from modules.world import World
from modules.entities import Cube, Point
from modules.gl_renderer import OpenGLRenderer
from modules.ego_sensor_rig import EgoSensorRig
from modules.optical_flow import OpticalFlow
# --- UPDATED IMPORT ---
from modules.velocity_solver import estimate_full_displacement
from modules.utils import (
    save_as_ply, 
    flow_to_image, 
    save_flow_map, 
    # save_flow_histogram, 
    build_velocity_map,
    project_to_2d,
    save_grayscale_map,
    save_radar_data_as_ply
    # --- REMOVED REDUNDANT IMPORTS ---
)

sys.path.append('..') # Add parent directory to path

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 360
TEXTURE_FILE = "/home/bobberman/programming/radar/radar-viz/simulation/assets/checkerboard.png"

# --- NOTE: EGO_VELOCITY IS STILL ACTIVE ---
EGO_VELOCITY = np.array([0.0, 0.0, 0.0], dtype=float)

if __name__ == "__main__":
    # 1. Simulation Setup
    world = World()
    radar_world = World()
    rig = EgoSensorRig()
    optical_flow = OpticalFlow()
    renderer = OpenGLRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, "Sensor Rig Sim")
    
    # --- SIMPLIFIED SCENE ---
    # Add a cube moving in the -X direction
    cube2 = Cube(
        position=[3.0, 2.0, 2.0],
        velocity=[-1.0, 0.0, 0.0], # True Absolute Velocity
        texture_path = TEXTURE_FILE
    )
    cube2.id_color = [2.0 / 255.0, 0.0, 0.0] # ID Color 2
    world.add_entity(cube2)

    id_to_velocity_map = {
        (2, 0, 0): cube2.velocity,
        (0, 0, 0): np.array([0.0, 0.0, 0.0]) # Background
    }
    # --- END SIMPLIFIED SCENE ---

    # 2. Define viewports
    viewport_cam = (0, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    viewport_radar = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    viewport_flow = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT // 2)

    CAM_H, CAM_W = viewport_cam[3], viewport_cam[2]
    cam_intrinsics = rig.get_camera().get_intrinsics(CAM_W, CAM_H)
    # --- GET INTRINSICS FOR CONVERSION ---
    fx, fy, cx, cy = cam_intrinsics['fx'], cam_intrinsics['fy'], cam_intrinsics['cx'], cam_intrinsics['cy']

    pixel_pos_map_prev = np.dstack([np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))])

    # 3. Main Loop
    dt = 0.016 # Aim for ~60 FPS
    frame_count = 0
    previous_image = None
    previous_world_pos_map = None
    previous_id_map = None
    previous_radar_points_data = np.empty((0, 8), dtype=float)
    previous_modelview = None
    previous_projection = None
    previous_viewport = None
    flow_image = None
    
    while not renderer.should_close():
        # 4. SIMULATION STEP (Calculate state at t)
        world.update(dt)
        rig.update(dt, EGO_VELOCITY) # <-- Rig is moving
        radar_points_data = rig.get_radar().simulate_scan(world, id_to_velocity_map, EGO_VELOCITY)

        radar_world.clear_entities()
        for point_data in radar_points_data:
            radar_world.add_entity(Point(position=point_data[:3], color=[0.0, 1.0, 0.0]))

        cam = rig.get_camera()
        current_world_pos_map, current_modelview, current_projection, current_viewport = \
            cam.get_world_position_map(world, CAM_W, CAM_H)
        current_id_map = cam.get_id_map(world, CAM_W, CAM_H)
        
        # 5. RENDER (Draws state at t)
        renderer.begin_frame()
        renderer.render_view(world, rig.get_camera(), viewport_cam)
        renderer.render_view(radar_world, rig.get_radar(), viewport_radar)
        if flow_image is not None:
            flow_h, flow_w, _ = flow_image.shape
            viewport_flow_adjusted = (0, 0, flow_w, flow_h)
            renderer.draw_image_in_viewport(flow_image, viewport_flow_adjusted)
        current_image = renderer.read_viewport_pixels(viewport_cam)
        renderer.end_frame()

        # 6. RUN INFERENCE & COMPARE (Using t-1 and t)
        if previous_image is not None and \
            previous_radar_points_data.size > 0 and \
            previous_modelview is not None:
            
            # A. Get Estimated Flow (from model)
            estimated_flow_map = optical_flow.inference(previous_image, current_image)

            # --- E. 3D VELOCITY ESTIMATION (WITH FULL DEBUG) ---
            print("--- Estimating 3D Velocity for t-1 Radar Points ---")
            
            # Get Transformation Data
            T_RA = np.identity(4) 
            ego_displacement = EGO_VELOCITY * dt
            T_AB = np.identity(4)
            T_AB[:3, 3] = ego_displacement

            points_printed_count = 0

            for point_data_t1 in previous_radar_points_data:
                
                # --- Step 1: Get Radar Point Data (from t-1) ---
                world_pos_t1 = point_data_t1[0:3]
                local_pos_radar_t1 = point_data_t1[3:6]
                radial_vel_magnitude = point_data_t1[6]
                
                norm = np.linalg.norm(local_pos_radar_t1)
                if norm == 0: continue 
                unit_radial_vector = local_pos_radar_t1 / norm
                vx_r, vy_r, vz_r = unit_radial_vector

                # --- Step 2: Simplified Projection (Get up, vp, d AND u_pix, v_pix) ---
                
                # 2a. World -> Camera Coords
                world_pos_t1_homo = np.append(world_pos_t1, 1.0)

                # Use the corrected matrix for the transformation
                cam_pos_t1_homo = world_pos_t1_homo @ previous_modelview
                
                w = cam_pos_t1_homo[3]
                if w == 0: continue
                cam_pos_t1 = cam_pos_t1_homo[:3] / w # (X_cam, Y_cam, d_t1)

                d_t1 = cam_pos_t1[2]
                if d_t1 == 0: continue
                    
                # 2b. Camera Coords -> Normalized Coords (Inputs for Solver)
                up_t1 = cam_pos_t1[0] / d_t1
                vp_t1 = cam_pos_t1[1] / d_t1
                
                # 2c. Normalized Coords -> Pixel Coords (Inputs for Flow Lookup)
                u_pix_t1 = (up_t1 * fx) + cx
                v_pix_t1 = (vp_t1 * fy) + cy

                # --- Step 3: Get Optical Flow Data (t-1 -> t) ---
                u_idx, v_idx = int(round(u_pix_t1)), int(round(v_pix_t1))
                if not (0 <= v_idx < CAM_H and 0 <= u_idx < CAM_W):
                    continue 
                
                pixel_flow_vector = estimated_flow_map[v_idx, u_idx]
                flow_u_pix, flow_v_pix = pixel_flow_vector
                
                # --- Step 4: Get (uq, vq) using THE CORRECT FORMULA ---
                u_pix_t2 = u_pix_t1 + flow_u_pix
                v_pix_t2 = v_pix_t1 + flow_v_pix

                uq = (u_pix_t2 - cx) / fx
                vq = (v_pix_t2 - cy) / fy
                
                # --- START FULL DEBUG PRINT ---
                if points_printed_count < 1:
                    print(f"\n--- DEBUG POINT {points_printed_count + 1} ---")
                    print(f"  world_pos_t1: {world_pos_t1.round(4)}")
                    print(f"  cam_pos_t1 (X_cam, Y_cam, d_t1): {cam_pos_t1.round(4)}")
                    print(f"  pixel_pos_t1 (u_pix_t1, v_pix_t1): ({u_pix_t1:.4f}, {v_pix_t1:.4f})")
                    print(f"  pixel_flow_vector (flow_u, flow_v): {pixel_flow_vector.round(4)}")
                    print(f"  pixel_pos_t2 (u_pix_t2, v_pix_t2): ({u_pix_t2:.4f}, {v_pix_t2:.4f})")
                    print("  --- FINAL SOLVER INPUTS ---")
                    print(f"  (up, vp, d): ({up_t1:.4f}, {vp_t1:.4f}, {d_t1:.4f})")
                    print(f"  (uq, vq): ({uq:.4f}, {vq:.4f})")
                    print(f"  unit_radial_vector (vx_r, vy_r, vz_r): {unit_radial_vector.round(4)}")
                    print(f"  radial_vel_magnitude: {radial_vel_magnitude:.4f}")
                    print(f"  T_AB (Ego Motion): \n{T_AB}")
                    print(f"  EGO_VELOCITY: {EGO_VELOCITY}")
                # --- END FULL DEBUG PRINT ---
                
                # --- Step 5: Solve for 3D Velocity ---
                t_est_relative = estimate_full_displacement(
                    dt, T_AB, T_RA,
                    up_t1, vp_t1, d_t1,
                    uq, vq,
                    vx_r, vy_r, vz_r,
                    radial_vel_magnitude
                )
                
                if t_est_relative is not None:
                    V_Relative_EST = t_est_relative / dt
                    V_Absolute_EST = V_Relative_EST + EGO_VELOCITY
                    
                    if points_printed_count < 3: 
                        print(f"  >>> V_Relative_EST: {V_Relative_EST.round(2)}")
                        print(f"  >>> V_Absolute_EST: {V_Absolute_EST.round(2)}")
                        print(f"  ------------------------------")
                        points_printed_count += 1
                
            # F. Update visualization for *next* loop
            flow_image = flow_to_image(estimated_flow_map)

        # 7. Update State for Next Loop (t becomes t-1)
        previous_image = current_image
        previous_world_pos_map = current_world_pos_map
        previous_id_map = current_id_map
        previous_radar_points_data = radar_points_data
        previous_modelview = current_modelview
        previous_projection = current_projection
        previous_viewport = current_viewport
        
        frame_count += 1

    renderer.shutdown()