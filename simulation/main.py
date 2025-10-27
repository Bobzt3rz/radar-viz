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
from modules.velocity_solver import estimate_full_displacement
from modules.utils import (
    save_as_ply, 
    flow_to_image, 
    save_flow_map, 
    build_velocity_map,
    project_to_2d,
    save_grayscale_map,
    save_radar_data_as_ply
)

sys.path.append('..') # Add parent directory to path

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 360
TEXTURE_FILE = "/home/bobberman/programming/radar/radar-viz/simulation/assets/checkerboard.png"

EGO_VELOCITY = np.array([0.0, 0.0, 0.0], dtype=float)

if __name__ == "__main__":
    # 1. Simulation Setup
    world = World()
    radar_world = World()
    rig = EgoSensorRig()
    optical_flow = OpticalFlow()
    renderer = OpenGLRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, "Sensor Rig Sim")
    
    # --- SIMPLIFIED SCENE ---
    cube2 = Cube(
        position=[3.0, 2.0, 2.0],
        velocity=[-1.0, 0.0, 0.0], # True Absolute Velocity
        texture_path = TEXTURE_FILE
    )
    cube2.id_color = [2.0 / 255.0, 0.0, 0.0]
    world.add_entity(cube2)

    id_to_velocity_map = {
        (2, 0, 0): cube2.velocity,
        (0, 0, 0): np.array([0.0, 0.0, 0.0])
    }

    # 2. Define viewports
    viewport_cam = (0, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    viewport_radar = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    viewport_flow = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT // 2)

    CAM_H, CAM_W = viewport_cam[3], viewport_cam[2]
    cam_intrinsics = rig.get_camera().get_intrinsics(CAM_W, CAM_H)
    fx, fy, cx, cy = cam_intrinsics['fx'], cam_intrinsics['fy'], cam_intrinsics['cx'], cam_intrinsics['cy']

    pixel_pos_map_prev = np.dstack(np.meshgrid(np.arange(CAM_W), np.arange(CAM_H)))

    # 3. Main Loop
    dt = 0.016
    frame_count = 0
    previous_image = None
    previous_world_pos_map = None
    previous_id_map = None
    previous_radar_points_data = np.empty((0, 8), dtype=float)
    previous_modelview = None
    previous_projection = None
    previous_viewport = None
    previous_pixel_positions = {}  # Track pixel positions for ground truth flow
    flow_image = None
    
    while not renderer.should_close():
        # 4. SIMULATION STEP
        world.update(dt)
        rig.update(dt, EGO_VELOCITY)
        radar_points_data = rig.get_radar().simulate_scan(world, id_to_velocity_map, EGO_VELOCITY, dt)

        radar_world.clear_entities()
        for point_data in radar_points_data:
            radar_world.add_entity(Point(position=point_data[:3], color=[0.0, 1.0, 0.0]))

        cam = rig.get_camera()
        current_world_pos_map, current_modelview, current_projection, current_viewport = \
            cam.get_world_position_map(world, CAM_W, CAM_H)
        current_id_map = cam.get_id_map(world, CAM_W, CAM_H)
        
        # Track ground truth pixel positions for flow validation
        for point_idx, point_data in enumerate(radar_points_data):
            world_pos = point_data[:3]
            try:
                px, py_gl, pz = gluProject(
                    world_pos[0], world_pos[1], world_pos[2],
                    current_modelview, current_projection, current_viewport
                )
                u_pix = px
                v_pix = (CAM_H - 1) - py_gl
            except:
                pass
        
        # 5. RENDER
        renderer.begin_frame()
        renderer.render_view(world, rig.get_camera(), viewport_cam)
        renderer.render_view(radar_world, rig.get_radar(), viewport_radar)
        if flow_image is not None:
            flow_h, flow_w, _ = flow_image.shape
            viewport_flow_adjusted = (0, 0, flow_w, flow_h)
            renderer.draw_image_in_viewport(flow_image, viewport_flow_adjusted)
        current_image = renderer.read_viewport_pixels(viewport_cam)
        renderer.end_frame()

        # 6. RUN INFERENCE & COMPARE
        if previous_image is not None and \
            previous_radar_points_data.size > 0 and \
            previous_id_map is not None and \
            previous_world_pos_map is not None and \
            previous_modelview is not None:
            
            # A. Get Estimated Flow
            estimated_flow_map_raw = optical_flow.inference(previous_image, current_image)

            # 1. Get velocity map for Frame N-1 (t-1)
            world_vel_map_prev = build_velocity_map(previous_id_map, id_to_velocity_map)
            # 2. Find true 3D pos for Frame N (t) by moving Frame N-1
            world_pos_map_curr_gt = previous_world_pos_map + (world_vel_map_prev * dt)
            # 3. Project these new 3D points back to 2D pixels using *current* camera matrices
            pixel_pos_map_curr_gt = project_to_2d(
                world_pos_map_curr_gt, 
                current_modelview, 
                current_projection, 
                current_viewport
            )
            # 4. Calculate the true flow map
            ground_truth_flow_map = pixel_pos_map_curr_gt - pixel_pos_map_prev

            # --- E. 3D VELOCITY ESTIMATION ---
            print("\n" + "="*80)
            print("=== VELOCITY ESTIMATION WITH FLOW DIAGNOSTICS ===")
            print("="*80)
            
            # Coordinate transformation matrices
            T_RA = np.identity(4)
            T_RA[:3, :3] = np.array([
                [ 0,  0, -1],  # SAE_X (forward) = -Camera_Z
                [-1,  0,  0],  # SAE_Y (left)    = -Camera_X
                [ 0,  1,  0]   # SAE_Z (up)      =  Camera_Y
            ])
            R_RA = T_RA[:3, :3]
            R_SAE_to_Camera = R_RA.T
            
            ego_displacement = EGO_VELOCITY * dt
            T_AB = np.identity(4)
            T_AB[:3, 3] = ego_displacement

            velocity_estimates = []
            valid_point_count = 0

            for point_idx, point_data_t1 in enumerate(previous_radar_points_data):
                
                if valid_point_count >= 1:
                    break
    
                print(f"\n{'─'*80}")
                print(f"POINT {point_idx + 1} - FLOW DIRECTION CHECK")
                print(f"{'─'*80}")
                
                # Get radar data
                world_pos_t1 = point_data_t1[0:3]
                local_pos_radar_t1 = point_data_t1[3:6]
                radial_vel_magnitude = point_data_t1[6]
                
                norm = np.linalg.norm(local_pos_radar_t1)
                if norm == 0: continue
                unit_radial_SAE = local_pos_radar_t1 / norm
                unit_radial_camera = R_SAE_to_Camera @ unit_radial_SAE
                vx_r, vy_r, vz_r = unit_radial_SAE
                
                # Project to image
                try:
                    px, py_gl, pz_norm = gluProject(
                        world_pos_t1[0], world_pos_t1[1], world_pos_t1[2],
                        previous_modelview, previous_projection, previous_viewport
                    )
                except:
                    continue

                u_pix_t1 = px
                v_pix_t1 = (CAM_H - 1) - py_gl

                if not (0 <= u_pix_t1 < CAM_W and 0 <= v_pix_t1 < CAM_H):
                    continue

                # Get camera depth
                cam_pos_gl_homo = np.append(world_pos_t1, 1.0) @ previous_modelview
                w = cam_pos_gl_homo[3]
                if abs(w) < 1e-8: continue
                cam_pos_gl = cam_pos_gl_homo[:3] / w
                d_t1 = -cam_pos_gl[2]
                if d_t1 <= 1e-3: continue
                
                # Normalized coordinates
                up_t1 = (u_pix_t1 - cx) / fx
                vp_t1 = (v_pix_t1 - cy) / fy

                # GET OPTICAL FLOW
                u_idx, v_idx = int(round(u_pix_t1)), int(round(v_pix_t1))
                if not (0 <= v_idx < CAM_H and 0 <= u_idx < CAM_W):
                    continue 
                
                # Get the PERFECT flow vector from our new GT map
                pixel_flow_vector = ground_truth_flow_map[v_idx, u_idx]
                flow_u_pix, flow_v_pix = pixel_flow_vector
                
                # Get the model's flow just for logging
                model_flow_raw = estimated_flow_map_raw[v_idx, u_idx]
                
                
                print(f"\n[OPTICAL FLOW ANALYSIS]:")
                print(f"  Object velocity (world): {cube2.velocity}")
                print(f"  Object moving LEFT in world (negative X)")
                print(f"  → Object is on LEFT of screen, moving towards camera's RIGHT vector, so flow should be POSITIVE u")
                
                print(f"\n  Pixel position at t-1: ({u_pix_t1:.2f}, {v_pix_t1:.2f})")
                
                print(f"\n  Model flow (RAW): ({model_flow_raw[0]:.4f}, {model_flow_raw[1]:.4f})")
                print(f"  PERFECT GT Flow:       ({flow_u_pix:.4f}, {flow_v_pix:.4f})")
                print(f"  → Using PERFECT GT flow for this test.")
                
                # Calculate (uq, vq)
                u_pix_t2 = u_pix_t1 + flow_u_pix
                v_pix_t2 = v_pix_t1 + flow_v_pix
                uq = (u_pix_t2 - cx) / fx
                vq = (v_pix_t2 - cy) / fy
                
                print(f"\n[SOLVER INPUTS]:")
                print(f"  (up, vp, d): ({up_t1:.4f}, {vp_t1:.4f}, {d_t1:.4f})")
                print(f"  (uq, vq): ({uq:.4f}, {vq:.4f})")
                print(f"  Unit radial (SAE) passed to solver: ({vx_r:.4f}, {vy_r:.4f}, {vz_r:.4f})")
                print(f"  Unit radial (camera, for reference): ({unit_radial_camera[0]:.4f}, {unit_radial_camera[1]:.4f}, {unit_radial_camera[2]:.4f})")
                print(f"  Radial vel magnitude: {radial_vel_magnitude:.4f}")
                
                # Solve
                t_est_relative = estimate_full_displacement(
                    dt, T_AB, T_RA,
                    up_t1, vp_t1, d_t1,
                    uq, vq,
                    vx_r, vy_r, vz_r,
                    radial_vel_magnitude
                )
                
                if t_est_relative is None:
                    print(f"\n[RESULT]: ✗ Solver failed")
                    continue
                
                V_Relative_EST = t_est_relative / dt
                V_Absolute_EST = V_Relative_EST + EGO_VELOCITY
                
                error = V_Absolute_EST - cube2.velocity
                error_magnitude = np.linalg.norm(error)
                
                velocity_estimates.append(V_Absolute_EST)
                valid_point_count += 1
                
                print(f"\n[RESULTS]:")
                print(f"  Displacement (camera): {t_est_relative}")
                print(f"  Velocity (absolute):   {V_Absolute_EST}")
                print(f"  Ground Truth:          {cube2.velocity}")
                print(f"  Error:                 {error}")
                print(f"  Error magnitude:       {error_magnitude:.4f} m/s")
                
                # Diagnostics
                print(f"\n[DIAGNOSTICS]:")
                radial_vel_check = np.dot(V_Relative_EST, unit_radial_camera)
                print(f"  Radial constraint check:")
                print(f"    V_rel · unit_radial = {radial_vel_check:.4f}")
                print(f"    Measured radial     = {radial_vel_magnitude:.4f}")
                print(f"    Difference:         = {abs(radial_vel_check - radial_vel_magnitude):.4f}")
                print(f"    Status: {'✓ PASS' if abs(radial_vel_check - radial_vel_magnitude) < 0.5 else '✗ FAIL'}")

            if len(velocity_estimates) > 0:
                mean_velocity = np.mean(velocity_estimates, axis=0)
                median_velocity = np.median(velocity_estimates, axis=0)
                mean_error = mean_velocity - cube2.velocity
                median_error = median_velocity - cube2.velocity
                mean_error_mag = np.linalg.norm(mean_error)
                median_error_mag = np.linalg.norm(median_error)
                
                print(f"\n{'═'*80}")
                print(f"SUMMARY")
                print(f"{'═'*80}")
                print(f"Ground Truth:    {cube2.velocity}")
                print(f"Mean Estimate:   {mean_velocity}")
                print(f"  → Mean Error Mag:  {mean_error_mag:.4f} m/s ({(mean_error_mag / (np.linalg.norm(cube2.velocity) + 1e-9) * 100):.1f}%)")
                print(f"Median Estimate:   {median_velocity}")
                print(f"  → Median Error Mag: {median_error_mag:.4f} m/s ({(median_error_mag / (np.linalg.norm(cube2.velocity) + 1e-9) * 100):.1f}%)")
                print(f"{'═'*80}\n")
                
            flow_image = flow_to_image(estimated_flow_map_raw)

        # 7. Update State
        previous_image = current_image
        previous_world_pos_map = current_world_pos_map
        previous_id_map = current_id_map
        previous_radar_points_data = radar_points_data
        previous_modelview = current_modelview
        previous_projection = current_projection
        previous_viewport = current_viewport

        previous_world_pos_map = current_world_pos_map
        previous_id_map = current_id_map
        
        frame_count += 1

    renderer.shutdown()