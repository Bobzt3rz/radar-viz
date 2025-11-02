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
from modules.utils import save_image, save_frame_histogram, save_clustering_analysis_plot
from modules.clustering import cluster_detections_6d
from modules.types import NoiseType

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
    max_frames = 51

    all_real_velocity_abs_errors = []
    all_real_velocity_actual_magnitudes = []
    all_tp = []
    all_fp = []
    all_fn = []
    all_tn = []

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

            # --- NEW CLUSTERING STEP ---
            if current_frame_errors:
                # Run the clustering function
                clusters, noise_points = cluster_detections_6d(
                    detections=current_frame_errors,
                    eps=0.7,              # Tune this 6D distance (1.0 is a good start)
                    min_samples=4,        # Same as before
                    velocity_weight=4.0   
                )

                # --- NEW PERFORMANCE ANALYSIS (THE "FULL PICTURE") ---
                
                # 1. Get Ground Truth Counts
                gt_real = 0
                gt_random = 0
                gt_multipath = 0
                for det in current_frame_errors:
                    if det[3] == NoiseType.REAL: gt_real += 1
                    elif det[3] == NoiseType.RANDOM_CLUTTER: gt_random += 1
                    elif det[3] == NoiseType.MULTIPATH_GHOST: gt_multipath += 1
                
                total_real_points = gt_real
                total_noisy_points = gt_random + gt_multipath

                # 2. Calculate TP, FP, FN, TN
                tp, fn = 0, 0
                fp_random, fp_multipath = 0, 0
                tn_random, tn_multipath = 0, 0
                
                for cluster in clusters:
                    for det in cluster:
                        if det[3] == NoiseType.REAL:
                            tp += 1
                        elif det[3] == NoiseType.RANDOM_CLUTTER:
                            fp_random += 1
                        elif det[3] == NoiseType.MULTIPATH_GHOST:
                            fp_multipath += 1

                for det in noise_points:
                    if det[3] == NoiseType.REAL:
                        fn += 1
                    elif det[3] == NoiseType.RANDOM_CLUTTER:
                        tn_random += 1
                    elif det[3] == NoiseType.MULTIPATH_GHOST:
                        tn_multipath += 1
                
                # 3. Aggregate Totals and Calculate Key Metrics
                total_fp = fp_random + fp_multipath
                total_tn = tn_random + tn_multipath
                
                precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
                
                recall = tp / total_real_points if total_real_points > 0 else 0.0

                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # 4. Save metrics for overall results
                all_tp.append(tp)
                all_fp.append(total_fp)
                all_fn.append(fn)
                all_tn.append(total_tn)

                # 5. Print the Full Report
                # (This part was already correct and will now display the right values)
                print(f"  Ground Truth: {total_real_points} Real | {total_noisy_points} Noisy (Random:{gt_random}, MP:{gt_multipath})")
                print(f"  Algorithm Output: {len(clusters)} Clusters, {len(noise_points)} Noise Points")
                print(f"  True Positives (TP):  {tp:4d} (Real points found)")
                print(f"  False Negatives (FN): {fn:4d} (Real points missed)")
                print(f"  False Positives (FP): {total_fp:4d} (Total noise clustered)")
                print(f"    - FP Random:    {fp_random:4d} (Filtered {tn_random}/{gt_random})")
                print(f"    - FP Multipath: {fp_multipath:4d} (Filtered {tn_multipath}/{gt_multipath})")
                print(f"  Precision: {precision * 100:6.2f}% | Recall: {recall * 100:6.2f}% | F1-Score: {f1 * 100:6.2f}%")
                print(f"--------------------------------------")
                
                # 6. Save the plot
                fp_dict = {'random': fp_random, 'mp': fp_multipath}
                tn_dict = {'random': tn_random, 'mp': tn_multipath}
                
                save_clustering_analysis_plot(
                    frame_number=frame_count,
                    clusters=clusters,
                    noise_points=noise_points,
                    tp=tp, fp_dict=fp_dict, fn=fn, tn_dict=tn_dict,
                    precision=precision, recall=recall, f1=f1,
                    output_dir="output/clustering_analysis"
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
                
                for vel_mag, vel_err, disp_err, noiseType, pos_3d, vel_3d_radar, vel_3d_world in current_frame_errors:
                    if(noiseType == NoiseType.REAL):
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
                
                all_real_velocity_abs_errors.extend(real_velocity_errors)
                all_real_velocity_actual_magnitudes.extend(real_vel_magnitudes)

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

    print("\n" + "="*40)
    print("--- Overall Simulation Results ---")
    print("="*40)

    print("\n### Clustering Filter Performance (All Frames) ###")
    if all_tp: # Check if any frames were processed
        total_tp = sum(all_tp)
        total_fp = sum(all_fp)
        total_fn = sum(all_fn)
        total_tn = sum(all_tn)

        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        total_real = total_tp + total_fn
        total_noisy = total_fp + total_tn
        total_all = total_real + total_noisy
        
        print(f"  Total Ground Truth: {total_real} Real Points, {total_noisy} Noisy Points ({total_all} total)")
        print(f"  - True Positives (TP): {total_tp:6d} (Real points correctly clustered)")
        print(f"  - False Positives (FP): {total_fp:6d} (Noisy points incorrectly clustered)")
        print(f"  - False Negatives (FN): {total_fn:6d} (Real points incorrectly filtered)")
        print(f"  - True Negatives (TN): {total_tn:6d} (Noisy points correctly filtered)")
        print(f"\n  --- Overall Scores ---")
        print(f"  Precision (Cleanliness): {overall_precision * 100:6.2f}%")
        print(f"  Recall (Completeness):   {overall_recall * 100:6.2f}%")
        print(f"  F1-Score (Balance):      {overall_f1 * 100:6.2f}%")
    else:
        print("No clustering results were recorded.")

    # --- Section 2: Velocity Estimation Performance (MODIFIED) ---
    print("\n### Velocity Estimation Performance (on True Positives) ###")
    if all_real_velocity_abs_errors and all_real_velocity_actual_magnitudes:
        
        errors_array = np.array(all_real_velocity_abs_errors)
        actuals_array = np.array(all_real_velocity_actual_magnitudes)
        
        # 1. Calculate Global MAE (Mean Absolute Error)
        global_mae = np.mean(errors_array)
        
        # 2. Calculate Mean Actual Speed
        mean_actual_speed = np.mean(actuals_array)

        print(f"  Global Mean Absolute Error (MAE):   {global_mae:.6f} m/s")
        print(f"  Mean Actual Object Speed:             {mean_actual_speed:.6f} m/s")
        
        # 3. Calculate NMAE (and check for division by zero)
        if mean_actual_speed > 1e-6:
            global_nmae = (global_mae / mean_actual_speed) * 100.0 # As a percentage
            print(f"  Normalized MAE (NMAE):              {global_nmae:.2f} %")
        else:
            print("  Normalized MAE (NMAE):              N/A (Mean actual speed is zero)")

        print(f"  (Based on {len(errors_array)} total True Positive detections)")
    else:
        print("  No valid True Positive detections were recorded to calculate an overall average.")

    renderer.cleanup()