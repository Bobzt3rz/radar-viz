import sys
import numpy as np
import cv2
from typing import List

from modules.velocity_solver import estimate_velocities_for_frame
from modules.setup import setup_simulation
from modules.radar import visualize_radar_points, save_radar_point_cloud_ply
from modules.cube import Cube
from modules.renderer import Renderer
from modules.optical_flow import OpticalFlow
from modules.utils import save_image, save_frame_histogram



if __name__ == "__main__":

    world, camera, radar = setup_simulation(delta_t=1/60.0)

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
    max_frames = 90

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

        # Update simulation state
        world.step()

        radar_detections = radar.generate_point_cloud(world)

        # Render the current state
        renderer.render_scene(near=0.1, far=1000.0)

        current_frame_rgb = renderer.capture_frame()

        save_image(current_frame_rgb, f"output/camera/{frame_count:04d}.png")

        if radar_detections:
            save_radar_point_cloud_ply(radar_detections, f"output/radar_cloud/{frame_count:04d}.ply")
            radar_image = visualize_radar_points(
                detections=radar_detections,
                radar_fov_az_rad=radar.fov_azimuth_rad,
                radar_fov_el_rad=radar.fov_elevation_rad,
                output_width=640,
                output_height=360,
                color_map_range=(-5, 5) # Example speed range for color
            )
            # Save the radar visualization
            save_image(cv2.cvtColor(radar_image, cv2.COLOR_BGR2RGB), # Convert BGR to RGB for Pillow
                       f"output/radar_image/{frame_count:04d}.png")

        flow = optical_flow_calculator.inference(current_frame_rgb)

        print(f"--- Frame {frame_count} (Time: {world.current_time:.2f}) ---")
        if frame_count > 0: # Only estimate after first frame
            current_frame_errors = estimate_velocities_for_frame(
                radar_detections, flow, camera, radar, prev_poses, world.delta_t
            )

            if current_frame_errors:
                real_velocity_errors: List[float] = []
                real_displacement_errors: List[float] = []
                noisy_displacement_errors: List[float] = []
                real_vel_magnitudes: List[float] = []
                noisy_vel_magnitudes: List[float] = []
                
                real_positions: List[np.ndarray] = []
                real_velocities: List[np.ndarray] = []
                noisy_positions: List[np.ndarray] = []
                noisy_velocities: List[np.ndarray] = []
                
                for vel_mag, vel_err, disp_err, isNoise, pos_3d, vel_3d_world in current_frame_errors:
                    if(isNoise == False):
                        real_vel_magnitudes.append(vel_mag)
                        real_velocity_errors.append(vel_err)
                        real_displacement_errors.append(disp_err)
                        real_positions.append(pos_3d)
                        real_velocities.append(vel_3d_world)
                    else:
                        noisy_vel_magnitudes.append(vel_mag)
                        noisy_displacement_errors.append(disp_err)
                        noisy_positions.append(pos_3d)
                        noisy_velocities.append(vel_3d_world)
                
                average_real_velocity_error = np.mean(real_velocity_errors) if real_velocity_errors else 0
                average_real_displacement_error = np.mean(real_displacement_errors) if real_displacement_errors else 0
                average_noisy_displacement_error = np.mean(noisy_displacement_errors) if noisy_displacement_errors else 0

                save_frame_histogram(
                        frame_number=frame_count,
                        real_pred_vel_mags=real_vel_magnitudes,
                        real_vel_errors=real_velocity_errors,
                        real_disp_errors=real_displacement_errors,
                        noisy_pred_vel_mags=noisy_vel_magnitudes,
                        noisy_disp_errors=noisy_displacement_errors,
                        real_positions=real_positions,
                        real_velocities=real_velocities,
                        noisy_positions=noisy_positions,
                        noisy_velocities=noisy_velocities,
                        output_dir="output/frame_analysis"
                    )

                print(f"Average Real Velocity Error: {average_real_velocity_error:.6f} m/s")
                print(f"Average Real Displacement Error: {average_real_displacement_error:.6f} pix")
                print(f"Average Noisy Displacement Error: {average_noisy_displacement_error:.6f} pix")
            elif radar_detections:
                 print("  No valid velocity estimates calculated this frame.")
        else:
            print("  Waiting for next frame for velocity estimation...")

        # Handle window events and display frame
        renderer.swap_buffers_and_poll_events()
        
        frame_count += 1

    # Cleanup
    print("\nSimulation loop finished.")
    renderer.cleanup()