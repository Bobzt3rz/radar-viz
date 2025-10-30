from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def save_image(img_array: np.ndarray, file_path: str) -> bool:
    """
    Saves a NumPy array representing an image to a specified file path.

    Args:
        img_array: The NumPy array containing image data (e.g., HxWx3 for RGB).
                   Assumes data type is compatible with Pillow (like uint8).
        file_path: The full path (including filename and extension, e.g.,
                   'output/images/frame_001.png') where the image should be saved.

    Returns:
        True if the image was saved successfully, False otherwise.
    """
    if not isinstance(img_array, np.ndarray) or img_array.size == 0:
        print(f"Error: Invalid image data provided for saving to {file_path}.")
        return False
    if not isinstance(file_path, str) or not file_path:
        print("Error: Invalid file path provided for saving image.")
        return False

    try:
        # --- 1. Ensure Directory Exists ---
        directory = os.path.dirname(file_path)
        if directory: # Check if directory part exists (not just filename)
            os.makedirs(directory, exist_ok=True) # Create directories if they don't exist

        # --- 2. Convert NumPy array to Pillow Image ---
        # Assumes img_array is in a format Pillow understands (like HxWx3 RGB uint8)
        img = Image.fromarray(img_array)

        # --- 3. Save the Image ---
        # Pillow infers format from the file extension in file_path
        img.save(file_path)
        # print(f"Image saved successfully to: {file_path}") # Optional confirmation
        return True

    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")
        return False
    
def save_frame_histogram(
    frame_number: int,
    real_pred_vel_mags: List[float],
    real_vel_errors: List[float],
    real_disp_errors: List[float],
    noisy_pred_vel_mags: List[float],
    noisy_disp_errors: List[float],
    real_positions: List[np.ndarray],
    real_velocities: List[np.ndarray],
    noisy_positions: List[np.ndarray],
    noisy_velocities: List[np.ndarray],
    output_dir: str = "output/debug_histograms"
):
    """
    Saves a 2x4 analysis plot for a SINGLE frame.
    - Row 1: Vel Err (Hist) | Real Disp (Hist) | Noisy Disp (Hist) | 3D Position (Cluster)
    - Row 2: Vel Err (Scatter) | Real Disp (Scatter) | Noisy Disp (Scatter) | 3D Velocity (Cluster)
    """
    
    # --- 1. Create Directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Convert to NumPy arrays ---
    real_pred_vel_np = np.array(real_pred_vel_mags)
    real_vel_err_np = np.array(real_vel_errors)
    real_disp_np = np.array(real_disp_errors)
    
    noisy_pred_vel_np = np.array(noisy_pred_vel_mags)
    noisy_disp_np = np.array(noisy_disp_errors)

    # --- 3. Create Plot (2 rows, 4 columns) ---
    plt.style.use('ggplot')
    # --- MODIFIED: 2x4 layout, wider figure ---
    fig, axes = plt.subplots(2, 4, figsize=(34, 16)) 
    fig.suptitle(f'Frame {frame_number} - Error and Cluster Analysis', fontsize=16)

    # === ROW 1: HISTOGRAMS AND POSITION ===

    # --- Row 1, Col 1: Real Velocity Error (Histogram) ---
    ax = axes[0, 0]
    if real_vel_err_np.size > 0:
        ax.hist(real_vel_err_np, bins=20, color='blue', alpha=0.7, edgecolor='black')
        mean_val = np.mean(real_vel_err_np)
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f"Real Velocity Error (N={len(real_vel_err_np)})\nMean: {mean_val:.4f} m/s")
        ax.set_xlabel("Velocity Error (m/s)")
    else:
        ax.set_title("Real Velocity Error\n(No Data)")
    ax.set_ylabel("Count")

    # --- Shared Bins for Displacement Histograms ---
    disp_bins = 20
    all_disp_data = np.concatenate((real_disp_np, noisy_disp_np))
    if all_disp_data.size > 0:
        global_min, global_max = np.min(all_disp_data), np.max(all_disp_data)
        if global_min == global_max:
            global_min -= 0.5; global_max += 0.5
        disp_bins = np.linspace(global_min, global_max, 21) # 21 edges = 20 bins

    # --- Row 1, Col 2: Real Displacement Error (Histogram) ---
    ax = axes[0, 1]
    if real_disp_np.size > 0:
        ax.hist(real_disp_np, bins=disp_bins, color='green', alpha=0.7, edgecolor='black')
        mean_val = np.mean(real_disp_np)
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f"Real Displacement Error (N={len(real_disp_np)})\nMean: {mean_val:.4f} pix")
        ax.set_xlabel("Displacement Error (pixels)")
        if all_disp_data.size > 0:
            ax.set_xlim(disp_bins[0], disp_bins[-1]) # Use shared limits
    else:
        ax.set_title("Real Displacement Error\n(No Data)")
    ax.set_ylabel("Count")

    # --- Row 1, Col 3: Noisy Displacement Error (Histogram) ---
    ax = axes[0, 2]
    if noisy_disp_np.size > 0:
        ax.hist(noisy_disp_np, bins=disp_bins, color='purple', alpha=0.7, edgecolor='black')
        mean_val = np.mean(noisy_disp_np)
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f"Noisy Displacement Error (N={len(noisy_disp_np)})\nMean: {mean_val:.4f} pix")
        ax.set_xlabel("Displacement Error (pixels)")
        if all_disp_data.size > 0:
            ax.set_xlim(disp_bins[0], disp_bins[-1]) # Use shared limits
    else:
        ax.set_title("Noisy Displacement Error\n(No Data)")
    ax.set_ylabel("Count")

    # --- Row 1, Col 4: 3D Position (Radar Coords) ---
    ax = axes[0, 3]
    ax.remove() # Remove the 2D axes
    ax_pos = fig.add_subplot(2, 4, 4, projection='3d') # Add 3D subplot
    
    if real_positions:
        real_pos_np = np.array(real_positions)
        ax_pos.scatter(real_pos_np[:, 0], real_pos_np[:, 1], real_pos_np[:, 2], 
                       c='blue', s=10, label=f'Real (N={len(real_pos_np)})', alpha=0.5)
    if noisy_positions:
        noisy_pos_np = np.array(noisy_positions)
        ax_pos.scatter(noisy_pos_np[:, 0], noisy_pos_np[:, 1], noisy_pos_np[:, 2], 
                       c='red', s=5, label=f'Noisy (N={len(noisy_pos_np)})', alpha=0.1)
    
    ax_pos.set_title("3D Position (Radar Coordinates)")
    ax_pos.set_xlabel("X (m)")
    ax_pos.set_ylabel("Y (m)")
    ax_pos.set_zlabel("Z (m)")
    ax_pos.legend()


    # === ROW 2: SCATTER PLOTS AND VELOCITY ===

    # --- Shared X-Axis for Scatter Plots ---
    scatter_x_min, scatter_x_max = 0, 1
    all_pred_vel_data = np.concatenate((real_pred_vel_np, noisy_pred_vel_np))
    if all_pred_vel_data.size > 0:
        scatter_x_min, scatter_x_max = np.min(all_pred_vel_data), np.max(all_pred_vel_data)
        if scatter_x_min == scatter_x_max:
            scatter_x_min -= 0.5; scatter_x_max += 0.5
        padding = (scatter_x_max - scatter_x_min) * 0.05
        scatter_x_min -= padding
        scatter_x_max += padding

    # --- Shared Y-Axis for Displacement Scatters ---
    scatter_y_min, scatter_y_max = 0, 1
    if all_disp_data.size > 0:
        scatter_y_min, scatter_y_max = np.min(all_disp_data), np.max(all_disp_data)
        if scatter_y_min == scatter_y_max:
            scatter_y_min -= 0.5; scatter_y_max += 0.5
        padding = (scatter_y_max - scatter_y_min) * 0.05
        scatter_y_min -= padding
        scatter_y_max += padding


    # --- Row 2, Col 1: Real Pred Vel Mag vs. Velocity Error (Scatter) ---
    ax = axes[1, 0]
    if real_pred_vel_np.size > 0 and real_vel_err_np.size > 0:
        ax.scatter(real_pred_vel_np, real_vel_err_np, color='blue', alpha=0.6, s=10)
        ax.set_title(f"Real: Pred. Vel Mag vs. Vel Error (N={len(real_pred_vel_np)})")
        ax.set_xlabel("Predicted Velocity Mag (m/s)")
        ax.set_ylabel("Velocity Error (m/s)")
        if all_pred_vel_data.size > 0:
             ax.set_xlim(scatter_x_min, scatter_x_max)
    else:
        ax.set_title("Real: Pred. Vel Mag vs. Vel Error\n(No Data)")


    # --- Row 2, Col 2: Real Pred Vel Mag vs. Displacement Error (Scatter) ---
    ax = axes[1, 1]
    if real_pred_vel_np.size > 0 and real_disp_np.size > 0:
        ax.scatter(real_pred_vel_np, real_disp_np, color='green', alpha=0.6, s=10)
        ax.set_title(f"Real: Pred. Vel Mag vs. Disp. Error (N={len(real_pred_vel_np)})")
        ax.set_xlabel("Predicted Velocity Mag (m/s)")
        ax.set_ylabel("Displacement Error (pixels)")
        if all_pred_vel_data.size > 0:
             ax.set_xlim(scatter_x_min, scatter_x_max)
        if all_disp_data.size > 0:
             ax.set_ylim(scatter_y_min, scatter_y_max)
    else:
        ax.set_title("Real: Pred. Vel Mag vs. Disp. Error\n(No Data)")

    # --- Row 2, Col 3: Noisy Pred Vel Mag vs. Displacement Error (Scatter) ---
    ax = axes[1, 2]
    if noisy_pred_vel_np.size > 0 and noisy_disp_np.size > 0:
        ax.scatter(noisy_pred_vel_np, noisy_disp_np, color='purple', alpha=0.6, s=10)
        ax.set_title(f"Noisy: Pred. Vel Mag vs. Disp. Error (N={len(noisy_pred_vel_np)})")
        ax.set_xlabel("Predicted Velocity Mag (m/s)")
        ax.set_ylabel("Displacement Error (pixels)")
        if all_pred_vel_data.size > 0:
             ax.set_xlim(scatter_x_min, scatter_x_max)
        if all_disp_data.size > 0:
             ax.set_ylim(scatter_y_min, scatter_y_max)
    else:
        ax.set_title("Noisy: Pred. Vel Mag vs. Disp. Error\n(No Data)")

    # --- Row 2, Col 4: 3D Solved Velocity (World Coords) ---
    ax = axes[1, 3]
    ax.remove() # Remove the 2D axes
    ax_vel = fig.add_subplot(2, 4, 8, projection='3d') # Add 3D subplot

    if real_velocities:
        real_vel_np = np.array(real_velocities)
        ax_vel.scatter(real_vel_np[:, 0], real_vel_np[:, 1], real_vel_np[:, 2], 
                       c='blue', s=10, label=f'Real (N={len(real_vel_np)})', alpha=0.5)
    if noisy_velocities:
        noisy_vel_np = np.array(noisy_velocities)
        ax_vel.scatter(noisy_vel_np[:, 0], noisy_vel_np[:, 1], noisy_vel_np[:, 2], 
                       c='red', s=5, label=f'Noisy (N={len(noisy_vel_np)})', alpha=0.1)

    ax_vel.set_title("3D Solved Velocity (World Coords)")
    ax_vel.set_xlabel("Vx (m/s)")
    ax_vel.set_ylabel("Vy (m/s)")
    ax_vel.set_zlabel("Vz (m/s)")
    ax_vel.legend()
    ax_vel.set_xlim([-5, 5])
    ax_vel.set_ylim([-5, 5])
    ax_vel.set_zlim([-5, 5])

    # --- 4. Save and Close ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"frame_{frame_number:04d}_analysis.png")
    
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error saving analysis plot: {e}")
    
    plt.close(fig) # Close the figure to free up memory