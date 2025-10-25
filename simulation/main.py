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
from modules.velocity_solver import calculate_3d_velocity
from modules.utils import (
    save_as_ply, 
    flow_to_image, 
    save_flow_map, 
    save_flow_histogram,
    build_velocity_map,
    project_to_2d,
    save_grayscale_map,
    save_radar_data_as_ply
)

sys.path.append('..') # Add parent directory to path

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 360
TEXTURE_FILE = "/home/bobberman/programming/radar/radar-viz/simulation/assets/checkerboard.png"

EGO_VELOCITY = np.array([0.0, 0.0, 1.0], dtype=float)

if __name__ == "__main__":
    # 1. Simulation Setup
    world = World()

    # Radar world
    radar_world = World()
    rig = EgoSensorRig()

    # Camera
    optical_flow = OpticalFlow()

    # make sure to initialize renderer before objects since objects need to load textures
    renderer = OpenGLRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, "Sensor Rig Sim")
    
    # Add a cube moving in the +X direction
    cube1 = Cube(
        position=[-5.0, 0.0, 0.0],
        velocity=[2.0, 1.0, 0.5],
        texture_path = TEXTURE_FILE
    )
    cube1.id_color = [1.0 / 255.0, 0.0, 0.0] # ID Color 1
    world.add_entity(cube1)

    # Add a cube moving in the -Z direction
    cube2 = Cube(
        position=[3.0, 2.0, 2.0],
        velocity=[0.0, 0.0, -1.0],
        texture_path = TEXTURE_FILE
    )
    cube2.id_color = [2.0 / 255.0, 0.0, 0.0] # ID Color 2
    world.add_entity(cube2)

    static_cube = Cube(
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        texture_path = TEXTURE_FILE
    )
    static_cube.id_color = [3.0 / 255.0, 0.0, 0.0] # ID 3 (static object)
    world.add_entity(static_cube)

    static_cube2 = Cube(
        position=[2.0, 0.0, -2.0],
        velocity=[0.0, 0.0, 0.0],
        texture_path = TEXTURE_FILE
    )
    static_cube2.id_color = [4.0 / 255.0, 0.0, 0.0] # ID 4 (static object)
    world.add_entity(static_cube2)

    id_to_velocity_map = {
        # (R, G, B) : [vx, vy, vz]
        (1, 0, 0): cube1.velocity,
        (2, 0, 0): cube2.velocity,
        (3, 0, 0): static_cube.velocity,
        (4, 0, 0): static_cube2.velocity,
        (0, 0, 0): np.array([0.0, 0.0, 0.0]) # Background
    }

    # 2. Define viewports
    # Top-left (Camera)
    viewport_cam = (0, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    # Top-right (Radar)
    viewport_radar = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    # Bottom (Optical Flow)
    viewport_flow = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT // 2)

    # Get camera viewport dimensions
    CAM_H, CAM_W = viewport_cam[3], viewport_cam[2]
    # Get camera intrinsics
    cam_intrinsics = rig.get_camera().get_intrinsics(CAM_W, CAM_H)

    # Create a base (u,v) grid for flow calculation
    u_coords, v_coords = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))
    pixel_pos_map_prev = np.dstack([u_coords, v_coords])

    # 3. Main Loop
    dt = 0.016 # Aim for ~60 FPS
    saved_frame = False
    frame_count = 0

    previous_image = None
    previous_world_pos_map = None
    previous_id_map = None

    flow_image = None
    
    while not renderer.should_close():
        # Update the simulation state
        world.update(dt)
        rig.update(dt, EGO_VELOCITY)

        # radar_points_data is now an (N, 4) NumPy array [x, y, z, vr]
        radar_points_data = rig.get_radar().simulate_scan(
            world, 
            id_to_velocity_map,
            EGO_VELOCITY
        )

        # Clear the old points
        radar_world.clear_entities()

        # Add the new points
        # We only need the position for visualization
        for point_data in radar_points_data:
            # point_data is [x, y, z, vr]
            # We only need the first 3 elements for visualization
            radar_world.add_entity(Point(position=point_data[:3], color=[0.0, 1.0, 0.0]))

        # 2. Get Ground Truth Data for Frame N
        cam = rig.get_camera()
        # --- UNPACK NEW RETURN VALUES ---
        current_world_pos_map, current_modelview, current_projection, current_viewport = \
            cam.get_world_position_map(world, CAM_W, CAM_H)
        current_id_map = cam.get_id_map(world, CAM_W, CAM_H)
        
        # Render the current world state
        renderer.begin_frame()
    
        # Render the left camera's view into the left viewport
        renderer.render_view(world, rig.get_camera(), viewport_cam)
        # Render the radar view into the right viewport
        renderer.render_view(radar_world, rig.get_radar(), viewport_radar)

        if flow_image is not None:
            # We draw the flow map from the PREVIOUS iteration
            # This is now perfectly synced (1-frame delay)
            
            # (Adjust viewport_flow to match flow_image_to_draw.shape)
            flow_h, flow_w, _ = flow_image.shape
            viewport_flow_adjusted = (0, 0, flow_w, flow_h)
            
            renderer.draw_image_in_viewport(flow_image, viewport_flow_adjusted)

        # save the camera frames for optical flow
        current_image = renderer.read_viewport_pixels(viewport_cam)

        renderer.end_frame()

        # --- 7. RUN INFERENCE & COMPARE (Using Frame N-1 and Frame N) ---
        if previous_image is not None and \
            previous_id_map is not None and \
            previous_world_pos_map is not None and \
            radar_points_data.size > 0 and \
            frame_count % 1 == 0:
            print(f"\n--- Running inference & validation on frame {frame_count} ---")
            
            # A. Get Estimated Flow (from model)
            estimated_flow_map = optical_flow.inference_cv(previous_image, current_image)

            print(f"Saved GT flow, Estimated flow, and EPE map for frame {frame_count}")
            
            # B. Calculate Ground Truth Flow
            
            # B1. Get velocity map for Frame N-1
            world_vel_map_prev = build_velocity_map(previous_id_map, id_to_velocity_map)
            
            # B2. Find true 3D pos for Frame N (by moving Frame N-1)
            world_pos_map_curr_gt = previous_world_pos_map + (world_vel_map_prev * dt)
            
            # --- USE THE STORED MATRICES ---
            pixel_pos_map_curr_gt = project_to_2d(
                world_pos_map_curr_gt, 
                current_modelview, 
                current_projection, 
                current_viewport
            )

            # B4. Calculate the flow vector
            ground_truth_flow_map = pixel_pos_map_curr_gt - pixel_pos_map_prev
            
            # C. Compare Estimated vs. Ground Truth
            error_map = ground_truth_flow_map - estimated_flow_map
            epe_map = np.linalg.norm(error_map, axis=2)

            # --- SAVE DEBUG IMAGES ---
            gt_flow_image = flow_to_image(ground_truth_flow_map)
            estimated_flow_image = flow_to_image(estimated_flow_map)
            # Save flow visualizations
            Image.fromarray(gt_flow_image).save(f"gt_flow_vis_{frame_count}.png")
            Image.fromarray(estimated_flow_image).save(f"est_flow_vis_{frame_count}.png")
            # Save the EPE map (clipped for visibility)
            save_grayscale_map(epe_map, f"epe_map_{frame_count}.png", vmax=50) # vmax=50 means errors > 50px appear white

            print(f"Saved GT flow, Estimated flow, and EPE map for frame {frame_count}")

            # 1. Save the Radar PLY file
            output_filename_pcd = f"radar_data_{frame_count}.ply"
            save_radar_data_as_ply(radar_points_data, output_filename_pcd)
            # -------------------------
            
            # Only calculate error on pixels that actually moved
            # 1. Find pixels that are *supposed* to be moving
            gt_is_moving = np.linalg.norm(ground_truth_flow_map, axis=2) > 0.01
            
            # 2. Find pixels whose "landing spot" is still *inside* the frame
            # (pixel_pos_map_curr_gt was calculated in step B3)
            u_landed = pixel_pos_map_curr_gt[..., 0]
            v_landed = pixel_pos_map_curr_gt[..., 1]
            is_in_frame = (u_landed >= 0) & (u_landed < CAM_W) & \
                          (v_landed >= 0) & (v_landed < CAM_H)
            
            # 3. The final mask: Pixels that are moving AND land inside the frame
            moving_pixels_mask = gt_is_moving & is_in_frame

            if np.any(moving_pixels_mask):
                avg_epe = np.mean(epe_map[moving_pixels_mask])
                median_epe = np.median(epe_map[moving_pixels_mask])
                print(f"=============================================")
                print(f"  MODEL ACCURACY (Average EPE): {avg_epe:.4f} pixels, Median EPE: {median_epe:.4f} pixels")
                print(f"=============================================")
            else:
                print("No moving pixels detected in ground truth.")

            # D. Update visualization for *next* loop
            flow_image = flow_to_image(estimated_flow_map)


        # --- 8. Update State for Next Loop ---
        previous_image = current_image
        previous_world_pos_map = current_world_pos_map
        previous_id_map = current_id_map
        
        frame_count += 1

    renderer.shutdown()