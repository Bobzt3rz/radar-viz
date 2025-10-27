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
from modules.velocity_solver import estimate_full_displacement # Use only the solver
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

WINDOW_WIDTH, WINDOW_HEIGHT = 720, 360
TEXTURE_FILE = "/home/bobberman/programming/radar/radar-viz/simulation/assets/checkerboard.png"
# TEXTURE_FILE = "/home/bobberman/programming/radar/radar-viz/simulation/assets/optical_flow_texture.png"
# --- EGO_VELOCITY in OpenGL World Frame ---
EGO_VELOCITY_WORLD = np.array([0.0, 0.0, 0.0], dtype=float)

if __name__ == "__main__":
    # 1. Simulation Setup (Same as before)
    world = World()
    radar_world = World()
    rig = EgoSensorRig()
    optical_flow = OpticalFlow()
    renderer = OpenGLRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, "Sensor Rig Sim")
    cube2 = Cube(
        position=[1.0, 0.25, -7.0], # World Coordinates (OpenGL: +X R, +Y U, +Z Out)
        velocity=[-1.0, 0.0, 0.0], # World Coordinates (OpenGL)
        texture_path = TEXTURE_FILE
    )
    cube2.id_color = [2.0 / 255.0, 0.0, 0.0]
    world.add_entity(cube2)
    id_to_velocity_map = {
        (2, 0, 0): cube2.velocity,
        (0, 0, 0): np.array([0.0, 0.0, 0.0])
    }

    # 2. Viewports and Intrinsics (Same as before)
    viewport_cam = (0, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    viewport_radar = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    viewport_flow = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT // 2)
    CAM_H, CAM_W = viewport_cam[3], viewport_cam[2]
    cam_intrinsics = rig.get_camera().get_intrinsics(CAM_W, CAM_H)
    fx, fy, cx, cy = cam_intrinsics['fx'], cam_intrinsics['fy'], cam_intrinsics['cx'], cam_intrinsics['cy']

    u_coords, v_coords = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))
    pixel_pos_map_prev = np.stack([u_coords, v_coords], axis=-1) # (H, W, 2)

    # 3. Main Loop Setup (Same as before)
    dt = 0.016
    frame_count = 0
    previous_image = None
    previous_world_pos_map = None
    previous_id_map = None
    previous_modelview = None
    previous_projection = None
    previous_viewport = None
    flow_image = None


    # --- Define Coordinate System Transformations (Same as before) ---
    R_RA = np.array([
        [ 0,  0,  1], [ -1, 0,  0], [ 0, -1,  0]
    ])
    t_RA = np.array([0.0, 0.0, 0.0])
    T_RA = np.identity(4); T_RA[:3, :3] = R_RA; T_RA[:3, 3] = t_RA
    T_opengl_to_solver = np.array([
        [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]
    ])
    T_solver_to_opengl = np.linalg.inv(T_opengl_to_solver)


    while not renderer.should_close():
        # 4. SIMULATION STEP (Same)
        world.update(dt)
        rig.update(dt, EGO_VELOCITY_WORLD)

        # --- RADAR DATA FOR VISUALIZATION ONLY (Same) ---
        _radar_points_data_vis = rig.get_radar().simulate_scan(world, id_to_velocity_map, EGO_VELOCITY_WORLD, dt)
        radar_world.clear_entities()
        if _radar_points_data_vis.shape[0] > 0:
             for point_data in _radar_points_data_vis:
                  radar_world.add_entity(Point(position=point_data[0:3], color=[0.0, 1.0, 0.0]))
        # --- END RADAR VISUALIZATION ---

        cam = rig.get_camera()
        current_world_pos_map, current_modelview, current_projection, current_viewport = \
            cam.get_world_position_map(world, CAM_W, CAM_H) # OpenGL coords
        current_id_map = cam.get_id_map(world, CAM_W, CAM_H)

        # 5. RENDER (Same)
        renderer.begin_frame()
        renderer.render_view(world, rig.get_camera(), viewport_cam)
        renderer.render_view(radar_world, rig.get_radar(), viewport_radar)
        if flow_image is not None:
             flow_h, flow_w, _ = flow_image.shape
             viewport_flow_adjusted = (0, 0, flow_w, flow_h)
             renderer.draw_image_in_viewport(flow_image, viewport_flow_adjusted)
        current_image = renderer.read_viewport_pixels(viewport_cam)
        renderer.end_frame()


        # 6. SOLVER INPUT CALCULATION & EXECUTION
        if previous_image is not None and \
            previous_id_map is not None and \
            previous_world_pos_map is not None and \
            previous_projection is not None and \
            previous_modelview is not None:

            print(f"\n[OPTICAL FLOW MODEL]")
            estimated_flow_map_raw = optical_flow.inference(previous_image, current_image)

            # B. Calculate PERFECT Ground Truth Flow
            print(f"\n[GROUND TRUTH FLOW CALC]: Calculating perfect GT flow map...")
            world_vel_map_prev = build_velocity_map(previous_id_map, id_to_velocity_map)
            world_pos_map_curr_gt = previous_world_pos_map + (world_vel_map_prev * dt)
            pixel_pos_map_curr_gt = project_to_2d(
                world_pos_map_curr_gt, current_modelview, current_projection, current_viewport
            )
            ground_truth_flow_map = pixel_pos_map_curr_gt - pixel_pos_map_prev
            print(f"[GROUND TRUTH FLOW CALC]: Done.")


            print("\n" + "="*80)
            print("=== VELOCITY ESTIMATION (Using inference + Correct Transforms) ===")
            print("="*80)

            # --- Define World-to-SolverCam Transform for time t-1 (Same) ---
            T_world_to_solver_cam_t1 = previous_modelview @ T_opengl_to_solver
            R_world_to_solver_cam_t1 = T_world_to_solver_cam_t1[:3, :3]

            # --- Calculate Ego Motion T_AB in Solver Camera Frame (Same) ---
            ego_displacement_world = EGO_VELOCITY_WORLD * dt
            ego_displacement_cam_solver = R_world_to_solver_cam_t1 @ (-ego_displacement_world)
            T_AB = np.identity(4)
            T_AB[:3, 3] = ego_displacement_cam_solver


            velocity_estimates = []
            processed_points = 0

            # --- Iterate through PIXELS of the PREVIOUS frame ---
            for v_idx in range(CAM_H):
                 for u_idx in range(CAM_W):
                    if processed_points >= 1: break # Limit logs

                    id_tuple_t1 = tuple(previous_id_map[v_idx, u_idx])
                    if id_tuple_t1 == (0, 0, 0): continue # Skip background

                    world_pos_t1 = previous_world_pos_map[v_idx, u_idx] # OpenGL World Coords
                    obj_vel_world_t1 = id_to_velocity_map.get(id_tuple_t1)
                    if obj_vel_world_t1 is None: continue

                    # --- Transform World Pos (t-1) to Solver Camera Frame (t-1) (Same) ---
                    world_pos_t1_homo = np.append(world_pos_t1, 1.0)
                    P_A_solver_homo = T_world_to_solver_cam_t1 @ world_pos_t1_homo
                    w_pa = P_A_solver_homo[3]
                    if abs(w_pa) < 1e-8: continue
                    P_A_solver = P_A_solver_homo[:3] / w_pa
                    d_t1 = P_A_solver[2]
                    if d_t1 <= 1e-3: continue

                    # --- Calculate up/vp directly from P_A_solver (Same) ---
                    if abs(P_A_solver[2]) < 1e-6: continue
                    up_t1 = P_A_solver[0] / P_A_solver[2]
                    vp_t1 = P_A_solver[1] / P_A_solver[2]

                    # --- GET ESTIMATED FLOW ---
                    pixel_flow_vector = ground_truth_flow_map[v_idx, u_idx]
                    flow_u_pix, flow_v_pix = pixel_flow_vector

                    # --- GET PERFECT GT FLOW (for comparison) ---
                    perfect_flow_vector = ground_truth_flow_map[v_idx, u_idx]
                    perfect_flow_u, perfect_flow_v = perfect_flow_vector


                    u_pix_t1 = u_idx # Pixel index at t-1
                    v_pix_t1 = v_idx
                    u_pix_t2 = u_pix_t1 + flow_u_pix # Estimated pixel pos at t
                    v_pix_t2 = v_pix_t1 + flow_v_pix
                    uq = (u_pix_t2 - cx) / fx
                    vq = (v_pix_t2 - cy) / fy

                    # --- Calculate Radial Velocity Inputs (using Solver Frames - Same logic) ---
                    obj_vel_cam_solver = R_world_to_solver_cam_t1 @ obj_vel_world_t1
                    ego_velocity_cam_solver = R_world_to_solver_cam_t1 @ EGO_VELOCITY_WORLD
                    V_relative_cam_solver = obj_vel_cam_solver - ego_velocity_cam_solver
                    V_relative_radar_sae = R_RA @ V_relative_cam_solver
                    t_abs_cam_solver = obj_vel_cam_solver * dt
                    Q_cam_solver = P_A_solver + t_abs_cam_solver
                    Q_cam_solver_homo = np.append(Q_cam_solver, 1.0)
                    Q_radar_sae_homo = T_RA @ Q_cam_solver_homo
                    w_qr = Q_radar_sae_homo[3]
                    if abs(w_qr) < 1e-8: continue
                    Q_radar_sae = Q_radar_sae_homo[:3] / w_qr
                    range_q = np.linalg.norm(Q_radar_sae)
                    if range_q < 1e-6: continue
                    unit_radial_sae = Q_radar_sae / range_q
                    vx_r, vy_r, vz_r = unit_radial_sae
                    radial_vel_magnitude = np.dot(V_relative_radar_sae, unit_radial_sae)

                    # TODO: THIS WORKS
                    # # --- Calculate uq/vq directly from Q_cam_solver ---
                    if abs(Q_cam_solver[2]) < 1e-6: continue
                    correct_uq = Q_cam_solver[0] / Q_cam_solver[2]
                    correct_vq = Q_cam_solver[1] / Q_cam_solver[2]
                    print(f"[CORRECT]:")
                    print(f"  (correct_uq, correct_vq):    ({correct_uq:.4f}, {correct_vq:.4f})")
                    # uq = Q_cam_solver[0] / Q_cam_solver[2] # Q_x / Q_z
                    # vq = Q_cam_solver[1] / Q_cam_solver[2] # Q_y / Q_z
                    


                    # --- Log Inputs ---
                    print(f"\n--- Processing Pixel ({u_idx}, {v_idx}) ---")
                    print(f"[OPTICAL FLOW]:")
                    print(f"  Model flow: ({flow_u_pix:.4f}, {flow_v_pix:.4f})")
                    # --- ADDED LOG ---
                    print(f"  Perfect GT Flow:     ({perfect_flow_u:.4f}, {perfect_flow_v:.4f})")
                    print(f"[SOLVER INPUTS]:")
                    print(f"  (up, vp, d): ({up_t1:.4f}, {vp_t1:.4f}, {d_t1:.4f})")
                    print(f"  (uq, vq):    ({uq:.4f}, {vq:.4f})")
                    print(f"  Unit radial (SAE) passed: ({vx_r:.4f}, {vy_r:.4f}, {vz_r:.4f})")
                    print(f"  Radial vel mag passed:    {radial_vel_magnitude:.4f}")


                    # --- Solve (Same) ---
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

                    # --- Convert Result back to World Frame for Comparison (Same) ---
                    V_Relative_EST_cam_solver = t_est_relative / dt
                    V_Absolute_EST_cam_solver = V_Relative_EST_cam_solver + ego_velocity_cam_solver
                    R_solver_cam_to_world_t1 = np.linalg.inv(R_world_to_solver_cam_t1)
                    V_Absolute_EST_world = R_solver_cam_to_world_t1 @ V_Absolute_EST_cam_solver
                    error = V_Absolute_EST_world - obj_vel_world_t1
                    error_magnitude = np.linalg.norm(error)
                    velocity_estimates.append(V_Absolute_EST_world)
                    processed_points += 1

                    # --- Results Logging (Same) ---
                    print(f"\n[RESULTS]:")
                    print(f"  Velocity (abs, world):    {V_Absolute_EST_world}")
                    print(f"  Ground Truth (world):     {obj_vel_world_t1}")
                    print(f"  Error (world):            {error}")
                    print(f"  Error magnitude:          {error_magnitude:.6f} m/s")


                 if processed_points >= 1: break # Break outer loop


            # --- Summary (Same) ---
            if len(velocity_estimates) > 0:
                mean_velocity = np.mean(velocity_estimates, axis=0)
                median_velocity = np.median(velocity_estimates, axis=0)
                mean_error = mean_velocity - cube2.velocity # world GT
                median_error = median_velocity - cube2.velocity # world GT
                mean_error_mag = np.linalg.norm(mean_error)
                median_error_mag = np.linalg.norm(median_error)

                print(f"\n{'═'*80}")
                print(f"SUMMARY (Using {processed_points} valid points, inference_cv)")
                print(f"{'═'*80}")
                print(f"Ground Truth (World): {cube2.velocity}")
                print(f"Mean Estimate (World):{mean_velocity}")
                print(f"  → Mean Error Mag:  {mean_error_mag:.6f} m/s ({(mean_error_mag / (np.linalg.norm(cube2.velocity) + 1e-9) * 100):.1f}%)")
                print(f"Median Estimate (World):{median_velocity}")
                print(f"  → Median Error Mag: {median_error_mag:.6f} m/s ({(median_error_mag / (np.linalg.norm(cube2.velocity) + 1e-9) * 100):.1f}%)")
                print(f"{'═'*80}\n")

            # Update flow image using OpenCV's output
            flow_image = flow_to_image(estimated_flow_map_raw)


        # 7. Update State (Same as before)
        previous_image = current_image
        previous_world_pos_map = current_world_pos_map
        previous_id_map = current_id_map
        previous_modelview = current_modelview
        previous_projection = current_projection
        previous_viewport = current_viewport

        frame_count += 1

        if frame_count > 1:
             print("Pausing after first successful solve. Close window to exit.")
             # time.sleep(10) # Uncomment to pause automatically
             break         # Uncomment to exit after first solve

    renderer.shutdown()