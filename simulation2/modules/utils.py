from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import cv2
import textwrap
from scipy.spatial import KDTree

from .types import DetectionTuple, NoiseType

def calculate_avg_intra_set_distance(
    data_set: List[np.ndarray] 
) -> float:
    """
    Calculates the average closest distance from every point in the set 
    to its nearest neighbor within the same set (self-referential clustering metric).
    
    Args:
        data_set: The list of points (e.g., Real positions or Real velocities).
    
    Returns:
        The average distance to the *second* nearest neighbor (k=2) to exclude self-reference.
    """
    if len(data_set) < 2:
        # Cannot calculate distance if there are 0 or 1 points
        return 0.0

    # 1. Convert lists to NumPy array
    np_data = np.array(data_set)
    
    # 2. Build a KD-Tree from the data set itself
    tree = KDTree(np_data)
    
    # 3. Query the *second* nearest neighbor (k=2) for every point in the set.
    # The first neighbor (k=1) will always be the point itself (distance 0).
    # We select the second neighbor's distance.
    # We only need the distances array, distances[:, 1] gives the distance to the second neighbor.
    
    # NOTE: The query result is structured as (distances, indices). 
    distances, _ = tree.query(np_data, k=2)
    
    # 4. Calculate the average of the distances to the nearest *unique* neighbor (index 1)
    return float(np.mean(distances[:, 1]))

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

    # --- Row 2, Col 4: 3D Solved Velocity (Radar Coords) ---
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

    ax_vel.set_title("3D Solved Velocity (Radar Coords)")
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


def save_clustering_analysis_plot(
    frame_number: int,
    clusters: List[List[DetectionTuple]],
    noise_points: List[DetectionTuple],
    output_dir: str = "output/clustering_analysis"
):
    """
    Saves a 2x3 analysis plot showing clustering performance. 
    Metrics (TP, FP, F1) and error visualizations (TP/FP/FN/TN) are calculated internally.
    Plot 5 specifically filters out Object ID 0.
    """
    
    # --- 1. Data Reconstruction and Metric Calculation ---
    
    # Reconstruct the complete set
    clustered_points_flat = [det for cluster in clusters for det in cluster]
    all_detections = clustered_points_flat + noise_points 
    
    # Initialize point category lists
    gt_real, gt_multipath, gt_random = 0, 0, 0
    tp_points, fp_points, fn_points, tn_points = [], [], [], []

    # Iterate once to calculate GT counts and categorize all points
    for det in all_detections:
        is_clustered = det in clustered_points_flat # Check if the point was put into a cluster
        
        # 1. Ground Truth Counts
        if det[3] == NoiseType.REAL:
            gt_real += 1
            if is_clustered:
                tp_points.append(det) # True Positive (Real and Clustered)
            else:
                fn_points.append(det) # False Negative (Real and Filtered)

        elif det[3] == NoiseType.RANDOM_CLUTTER:
            gt_random += 1
            if is_clustered:
                fp_points.append(det) # False Positive (Noise and Clustered)
            else:
                tn_points.append(det) # True Negative (Noise and Filtered)
                
        else:
            gt_multipath += 1
            if is_clustered:
                fp_points.append(det) # False Positive (Noise and Clustered)
            else:
                tn_points.append(det) # True Negative (Noise and Filtered)
                
        
            
    # Final counts
    tp, fn = len(tp_points), len(fn_points)
    total_fp, total_tn = len(fp_points), len(tn_points)
    total_real_points = gt_real
    
    # d. Calculate final scores
    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
    recall = tp / total_real_points if total_real_points > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- 2. Data Extraction for Plots ---
    
    real_pos = [det[4] for det in all_detections if det[3] == NoiseType.REAL]
    random_pos = [det[4] for det in all_detections if det[3] == NoiseType.RANDOM_CLUTTER]
    mp_pos = [det[4] for det in all_detections if det[3] is not NoiseType.REAL and det[3] is not NoiseType.RANDOM_CLUTTER]
    all_points_pos = [det[4] for det in all_detections] # For axis limits

    # --- 3. Plotting Setup (2x3 Grid) ---
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('ggplot')
    # Use 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(26, 16)) 
    
    metric_text = (
        f"Real GT: {total_real_points} | MP GT: {gt_multipath} | Rand GT: {gt_random}\n"
        f"Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1-Score: {f1 * 100:.2f}%"
    )
    
    fig.suptitle(f'Frame {frame_number} - Clustering Analysis', fontsize=16, weight='bold')
    # Adjusted text box location for 2x3 header
    fig.text(0.5, 0.94, metric_text, ha='center', fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # --- Axis Limits Setup (for all 3D plots) ---
    if all_points_pos:
        all_pos_np = np.array(all_points_pos)
        min_lim, max_lim = np.min(all_pos_np, axis=0), np.max(all_pos_np, axis=0)
        padding = (max_lim - min_lim) * 0.1 
    
    def set_3d_limits(ax):
        if all_points_pos:
            ax.set_xlim(min_lim[0]-padding[0], max_lim[0]+padding[0]); 
            ax.set_ylim(min_lim[1]-padding[1], max_lim[1]+padding[1]); 
            ax.set_zlim(min_lim[2]-padding[2], max_lim[2]+padding[2])
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")


    # === PLOT 1: Ground Truth (Position by GT Type) ===
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if random_pos:
        pos_np = np.array(random_pos)
        ax1.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], c='red', s=5, label=f'Random (N={len(random_pos)})', alpha=0.1)
    if mp_pos:
        pos_np = np.array(mp_pos)
        ax1.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], c='purple', s=15, label=f'Multipath (N={len(mp_pos)})', alpha=0.5)
    if real_pos:
        pos_np = np.array(real_pos)
        ax1.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], c='blue', s=25, label=f'Real (N={len(real_pos)})', alpha=1.0)
    ax1.set_title("1. Ground Truth (All Detections)")
    set_3d_limits(ax1)
    ax1.legend(fontsize=8)
    
    # === PLOT 2: Clustering Result (Position by Cluster ID) ===
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    if not clusters:
        ax2.set_title("2. Filter Result (No Clusters Found)")
    else:
        colors = plt.cm.get_cmap('tab10', len(clusters))
        for i, cluster in enumerate(clusters):
            object_id = cluster[0][7]
            cluster_points_pos = [det[4] for det in cluster]
            cluster_pos_np = np.array(cluster_points_pos)
            
            ax2.scatter(cluster_pos_np[:, 0], cluster_pos_np[:, 1], cluster_pos_np[:, 2], 
                        c=[colors(i)], s=20, label=f'Clust {i} (ID:{object_id}, N={len(cluster_pos_np)})')
        ax2.set_title(f"2. Filter Result ({len(clusters)} Clusters Found)")
    set_3d_limits(ax2)
    ax2.legend(fontsize=8)
    
    # === PLOT 3: Classification Error (Position by TP/FP/FN/TN) ---
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Plot in order of error prominence: FN (Worst), FP, TP, TN (Best)
    if fn_points:
        fn_pos = np.array([det[4] for det in fn_points])
        ax3.scatter(fn_pos[:, 0], fn_pos[:, 1], fn_pos[:, 2], c='yellow', s=50, marker='X', label=f'FN - Missed Real (N={len(fn_points)})', alpha=1.0, edgecolor='black')
    if fp_points:
        fp_pos = np.array([det[4] for det in fp_points])
        ax3.scatter(fp_pos[:, 0], fp_pos[:, 1], fp_pos[:, 2], c='red', s=25, label=f'FP - Clustered Noise (N={len(fp_points)})', alpha=0.8)
    if tp_points:
        tp_pos = np.array([det[4] for det in tp_points])
        ax3.scatter(tp_pos[:, 0], tp_pos[:, 1], tp_pos[:, 2], c='green', s=15, label=f'TP - Clustered Real (N={len(tp_points)})', alpha=0.6)
    if tn_points:
        tn_pos = np.array([det[4] for det in tn_points])
        ax3.scatter(tn_pos[:, 0], tn_pos[:, 1], tn_pos[:, 2], c='gray', s=5, label=f'TN - Filtered Noise (N={len(tn_points)})', alpha=0.2)
        
    ax3.set_title("3. Classification Error Breakdown (TP/FP/FN/TN)")
    set_3d_limits(ax3)
    ax3.legend(fontsize=8)

    # === PLOT 4: Overall Detection Counts (Bar Chart) ===
    ax4 = axes[1, 0]
    count_labels = ['TP', 'FP', 'FN', 'TN']
    count_values = [tp, total_fp, fn, total_tn]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E']
    bars = ax4.bar(count_labels, count_values, color=colors)
    ax4.bar_label(bars); ax4.set_title("4. Overall Detection Counts"); ax4.set_ylabel("Number of Points")

    # === PLOT 5: Detection Counts (Excluding Object ID 0) ===
    ax5 = axes[1, 1]
    
    # Filter detections where det[7] (Object ID) is NOT 0
    tp_no0 = sum(1 for det in tp_points if det[7] != 0)
    fp_no0 = sum(1 for det in fp_points if det[7] != 0)
    fn_no0 = sum(1 for det in fn_points if det[7] != 0)
    tn_no0 = sum(1 for det in tn_points if det[7] != 0)

    count_labels_no0 = ['TP (No 0)', 'FP (No 0)', 'FN (No 0)', 'TN (No 0)']
    count_values_no0 = [tp_no0, fp_no0, fn_no0, tn_no0]
    
    bars5 = ax5.bar(count_labels_no0, count_values_no0, color=colors)
    ax5.bar_label(bars5)
    ax5.set_title("5. Detection Counts (Ignoring ID 0)")
    ax5.set_ylabel("Number of Points")
    
    # === PLOT 6: Placeholder (empty in 2x3) ===
    axes[1, 2].axis('off')

    # --- 4. Save and Close ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 
    save_path = os.path.join(output_dir, f"frame_{frame_number:04d}_analysis.png")
    
    try:
        fig.savefig(save_path)
    except Exception as e:
        print(f"Error saving analysis plot: {e}")
    
    plt.close(fig)


def draw_points_by_noise(
    image: np.ndarray, 
    detections: List[DetectionTuple],
    points_2d: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Draws 2D points on an image, color-coded by NoiseType (Real/Multipath/random).
    Real points are Green, Mutipath points are Red, and random points are black
    """
    if points_2d.shape[0] == 0:
        return image

    img_with_points = image.copy()
    h, w = img_with_points.shape[:2]
    
    # Define colors (BGR format for OpenCV)
    COLOR_REAL = (0, 255, 0)   # Green
    COLOR_MULTIPATH = (0, 0, 0)  # Black
    COLOR_RANDOM = (0, 0, 255)

    # Filter detections to only include the points that were successfully projected (valid_mask)
    valid_detections = [det for det, is_valid in zip(detections, valid_mask) if is_valid]
    
    num_points = points_2d.shape[0]
    
    if num_points != len(valid_detections):
        # This should not happen if logic is correct, but acts as a guard.
        print("Warning: Mismatch between 2D points and valid detections.")
        return image

    for i in range(num_points):
        u, v = points_2d[i]
        
        # det[3] is NoiseType
        noiseType = valid_detections[i][3] 
        
        color = COLOR_REAL if noiseType == NoiseType.REAL else COLOR_MULTIPATH if noiseType == NoiseType.MULTIPATH_GHOST else COLOR_RANDOM
            
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_with_points, (int(u), int(v)), 3, color, -1)
            
    return img_with_points

def project_and_get_depths(
    detections: List[DetectionTuple],
    T_Cam_from_Radar: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # <-- RETURNS valid_mask
    """
    Extracts all 3D points from the list, projects them,
    and returns 2D points, their corresponding depths, and a mask of valid points.
    """
    # 1. Extract all 3D points (det_tuple[4] is pos_3d)
    points_3d_radar = [det[4] for det in detections]
    
    if not points_3d_radar:
        return np.array([]), np.array([]), np.array([False] * len(detections))
        
    points_3d_radar = np.array(points_3d_radar) # (N, 3)
    
    # 2. Project them (standard projection logic)
    # NOTE: The implementation of transform_points_3d is missing, assuming it transforms (N, 3) -> (N, 4) or (N, 3)
    # We must assume T_Cam_from_Radar is 4x4 and the input needs to be homogeneous.
    points_3d_radar_h = np.hstack((points_3d_radar, np.ones((points_3d_radar.shape[0], 1))))
    points_cam_h = T_Cam_from_Radar @ points_3d_radar_h.T
    points_cam = points_cam_h[:3, :].T # (N, 3) in camera coordinates
    
    if points_cam.shape[0] == 0:
        return np.array([]), np.array([]), np.array([False] * len(detections))
    
    depths = points_cam[:, 2] # Camera Z-depths
    
    # 3. Create Mask
    valid_mask = depths > 0.1  # 10cm threshold
    
    if not np.any(valid_mask):
        return np.array([]), np.array([]), valid_mask
        
    points_cam_valid = points_cam[valid_mask]
    depths_valid = depths[valid_mask]
    
    u_norm = points_cam_valid[:, 0] / depths_valid
    v_norm = points_cam_valid[:, 1] / depths_valid
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[1, 1], K[1, 2] # NOTE: Assuming K is 3x3 and K[1,1]=fy, K[1,2]=cy. 
                              # Also fixing u_pix = fx*u_norm + cx, v_pix = fy*v_norm + cy
    
    # Fixing standard camera intrinsics usage
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u_pix = fx * u_norm + cx
    v_pix = fy * v_norm + cy
    
    points_2d = np.vstack((u_pix, v_pix)).T
    
    return points_2d, depths_valid, valid_mask # <-- RETURNING MASK

class GlobalErrorTracker:
    def __init__(self):
        # Structure: [NoiseType][Axis_Index (0-3)] -> List of errors
        self.noise_errors = {
            'ShiftX': [[], [], [], []],
            'ShiftY': [[], [], [], []],
            'ShiftZ': [[], [], [], []],
            'Radial': [[], [], [], []],
            # We omit 'Random' here if we only care about Actor noise models in the summary
        }
        # We will only store "Real" errors for Actors (Obj ID > 0)
        self.actor_real_errors = [[], [], [], []] # [Vx, Vy, Vz, Mag]

    def add_data(self, noise_map, real_vel_np, gt_vel_radar, object_id):
        """
        Accumulates error data. 
        CRITICAL FIX: Only adds 'Real' data if object_id > 0 (Actors).
        """
        gt_mag = np.linalg.norm(gt_vel_radar)
        
        # 1. Store Real Errors (STRICTLY FOR ACTORS)
        if object_id > 0:
            for axis in range(3):
                if len(real_vel_np) > 0:
                    errs = np.abs(real_vel_np[:, axis] - gt_vel_radar[axis])
                    self.actor_real_errors[axis].extend(errs)
            if len(real_vel_np) > 0:
                mag_errs = np.abs(np.linalg.norm(real_vel_np, axis=1) - gt_mag)
                self.actor_real_errors[3].extend(mag_errs)

        # 2. Store Noise Errors (These are already implicitly Actor-only)
        for name, data in noise_map.items():
            if name not in self.noise_errors: continue
            
            for axis in range(3):
                if len(data) > 0:
                    errs = np.abs(data[:, axis] - gt_vel_radar[axis])
                    self.noise_errors[name][axis].extend(errs)
            if len(data) > 0:
                mag_errs = np.abs(np.linalg.norm(data, axis=1) - gt_mag)
                self.noise_errors[name][3].extend(mag_errs)

def save_frame_projections(frame_number: int,
                           detections: List[DetectionTuple],
                           image_rgb: np.ndarray,
                           T_Cam_from_Radar: np.ndarray,
                           K: np.ndarray,
                           output_dir: str = "output/frame_projections"):
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        image_bgr_to_draw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        non_static_detections = [d for d in detections if d[7] != 0]
        
        points_2d, _, full_object_mask = project_and_get_depths(non_static_detections, T_Cam_from_Radar, K)
        
        image_with_points = draw_points_by_noise(image_bgr_to_draw, non_static_detections, points_2d, full_object_mask)
        
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_number:08d}_proj.png"), image_with_points)
    except Exception as e:
        # print(f"Projection failed for frame {frame_number}, obj {object_id}: {e}")
        pass

def save_frame_error_histogram(
    frame_number: int,
    all_frame_detections: List[Tuple], 
    image_rgb: np.ndarray,                    
    T_Cam_from_Radar: np.ndarray,       
    K: np.ndarray,                      
    output_dir: str = "output/object_analysis"
):
    """
    Plots Absolute Error (|Pred - GT|) with Global Isotropic Scaling.
    
    RETURNS:
        frame_data_list (List): A list where each item is tuple:
        (noise_map, real_vel_np, gt_vel_radar)
        You can iterate this list to update your GlobalErrorTracker.
    """
    
    # Return container
    frame_data_collection = []

    # 1. Group detections
    grouped_detections: Dict[int, List[Tuple]] = {}
    for det in all_frame_detections:
        obj = det[7]
        if obj not in grouped_detections: grouped_detections[obj] = []
        grouped_detections[obj].append(det)

    random_noise_detections = [det for det in all_frame_detections if det[3] == NoiseType.RANDOM_CLUTTER]
    
    VEL_COLS = [(0, "Err Vx"), 
                (1, "Err Vy"), 
                (2, "Err Vz"), 
                (None, "Err |V|"), 
                ('max_comp', "Max(Err X,Y,Z)")
            ]

    # --- ITERATE THROUGH ALL OBJECTS ---
    for object_id, detections in grouped_detections.items():
        
        object_output_dir = os.path.join(output_dir, f"object_{object_id:04d}")
        os.makedirs(object_output_dir, exist_ok=True)

        # --- PART A: PROJECTION LOGIC (PRESERVED) ---
        try:
            image_bgr_to_draw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            # Include random noise only for background (Obj 0), otherwise just object points
            dets_to_draw = detections + random_noise_detections if object_id == 0 else detections
            
            # (Assuming project_and_get_depths is available in your scope)
            points_2d, _, full_object_mask = project_and_get_depths(dets_to_draw, T_Cam_from_Radar, K)
            
            # (Assuming draw_points_by_noise is available in your scope)
            image_with_points = draw_points_by_noise(image_bgr_to_draw, dets_to_draw, points_2d, full_object_mask)
            
            cv2.imwrite(os.path.join(object_output_dir, f"frame_{frame_number:08d}_proj.png"), image_with_points)
        except Exception as e:
            # print(f"Projection failed for frame {frame_number}, obj {object_id}: {e}")
            pass

        # --- PART B: EXTRACT DATA ---
        gt_vel_radar = detections[0][5]
        gt_mag = np.linalg.norm(gt_vel_radar)

        # Real Data
        real_vels = [det[5] for det in detections if det[3] == NoiseType.REAL]
        real_vel_np = np.array(real_vels) if real_vels else np.empty((0, 3))

        # Noise Map Construction
        if object_id == 0:
            noise_map = {'Random': np.array([d[5] for d in random_noise_detections]) if random_noise_detections else np.empty((0,3))}
            row_configs = [('Random', 'purple')]
        else:
            temp_noise = {nt: [] for nt in [NoiseType.SHIFTX, NoiseType.SHIFTY, NoiseType.SHIFTZ, NoiseType.SHIFTRADIAL]}
            for det in detections:
                if det[3] in temp_noise: temp_noise[det[3]].append(det[5])
            
            noise_map = {
                'ShiftX': np.array(temp_noise[NoiseType.SHIFTX]) if temp_noise[NoiseType.SHIFTX] else np.empty((0,3)),
                'ShiftY': np.array(temp_noise[NoiseType.SHIFTY]) if temp_noise[NoiseType.SHIFTY] else np.empty((0,3)),
                'ShiftZ': np.array(temp_noise[NoiseType.SHIFTZ]) if temp_noise[NoiseType.SHIFTZ] else np.empty((0,3)),
                'Radial': np.array(temp_noise[NoiseType.SHIFTRADIAL]) if temp_noise[NoiseType.SHIFTRADIAL] else np.empty((0,3)),
            }
            row_configs = [
                ('ShiftX', 'orange'), ('ShiftY', 'cyan'), ('ShiftZ', 'gold'), ('Radial', 'lime')
            ]

        # --- SAVE DATA FOR GLOBAL TRACKER ---
        # We append this tuple to our list, to be returned AFTER the loop
        frame_data_collection.append((noise_map, real_vel_np, gt_vel_radar, object_id))

        # --- PART C: CALCULATE LIMITS (Global Error Scaling) ---
        max_errors = []
        
        # Helper to get error array
        def get_errs(arr, ax_idx):
            if len(arr) == 0: return np.array([])
            
            if ax_idx == 'max_comp':
                # Calculate Abs Error for ALL 3 axes, then take the Max per point
                # arr shape: (N, 3), gt_vel_radar shape: (3,)
                diffs = np.abs(arr - gt_vel_radar)
                return np.max(diffs, axis=1) # Returns shape (N,)
            
            elif ax_idx is None:
                # Magnitude error
                val = np.linalg.norm(arr, axis=1)
                gt = gt_mag
                return np.abs(val - gt)
            
            else:
                # Single Axis error
                val = arr[:, ax_idx] 
                gt = gt_vel_radar[ax_idx]
                return np.abs(val - gt)

        # Scan columns for max error
        for col_idx, (axis_idx, _) in enumerate(VEL_COLS):
            col_max = 0
            e_real = get_errs(real_vel_np, axis_idx)
            if len(e_real): col_max = max(col_max, np.max(e_real))
            for n_key in noise_map:
                e_noise = get_errs(noise_map[n_key], axis_idx)
                if len(e_noise): col_max = max(col_max, np.max(e_noise))
            max_errors.append(col_max)

        global_max_err = max(max_errors) if max_errors else 0
        plot_limit = max(global_max_err * 1.1, 1.0) # Min 1.0m/s width
        
        shared_bins = np.linspace(0, plot_limit, 50) 

        # --- PART D: PLOTTING ---
        plt.style.use('ggplot')
        rows = len(row_configs)
        fig, axes = plt.subplots(rows, 5, figsize=(50, 4 * rows))
        
        title_prefix = "Background (Obj 0)" if object_id == 0 else f"Object {object_id}"
        fig.suptitle(f'Frame {frame_number} - {title_prefix} - Absolute Error Analysis', fontsize=26)
        
        if rows == 1: axes = np.expand_dims(axes, axis=0)

        for row_idx, (noise_label, noise_color) in enumerate(row_configs):
            curr_noise_data = noise_map[noise_label]
            
            for col_idx, (axis_idx, col_label) in enumerate(VEL_COLS):
                ax = axes[row_idx, col_idx]
                
                r_err = get_errs(real_vel_np, axis_idx)
                n_err = get_errs(curr_noise_data, axis_idx)

                if len(r_err) > 0:
                    ax.hist(r_err, bins=shared_bins, color='blue', alpha=0.5, density=True, label=f'Real (N={len(r_err)})')
                    ax.axvline(np.mean(r_err), color='darkblue', ls=':', lw=2, label=f'μ={np.mean(r_err):.2f}')
                
                if len(n_err) > 0:
                    ax.hist(n_err, bins=shared_bins, color=noise_color, alpha=0.5, density=True, label=f'{noise_label} (N={len(n_err)})')
                    ax.axvline(np.mean(n_err), color=noise_color, ls=':', lw=2, label=f'μ={np.mean(n_err):.2f}')

                ax.axvline(0, color='green', ls='-', lw=3, label='GT (0)')
                ax.set_xlim(0, plot_limit)
                
                if row_idx == 0: ax.set_title(col_label, fontsize=20, fontweight='bold')
                if col_idx == 0: ax.set_ylabel(f"vs {noise_label}", fontsize=18, fontweight='bold')
                ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(object_output_dir, f"frame_{frame_number:08d}_error_analysis.png"), dpi=150)
        plt.close(fig)

    # --- RETURN DATA FOR ALL OBJECTS ---
    return frame_data_collection

def plot_global_summary(
    clusters: List[List['DetectionTuple']], 
    filtered_clusters: List[List['DetectionTuple']], 
    cluster_noise: List['DetectionTuple'], 
    filtered_cluster_noise: List['DetectionTuple'],  
    output_dir: str = "output"
):
    """
    Generates a 2x2 Global Summary Plot (Ablation Study).
    Fixed to avoid hashing errors with numpy arrays.
    
    Layout:
    1. Normal Clusters (All IDs)    | 2. Normal Clusters (ID > 0)
    3. Filtered Clusters (All IDs)  | 4. Filtered Clusters (ID > 0)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Helper: Metric Calculation (Optimized) ---
    def get_metrics(input_clusters, input_noise_flat, exclude_id_zero=False):
        """
        Calculates TP/FP/FN/TN by iterating lists directly.
        Avoids set() lookups on unhashable numpy arrays.
        """
        tp, fp, fn, tn = 0, 0, 0, 0
        real_gt_count = 0
        
        # 1. Analyze Clustered Points (Predicted = Positive)
        # input_clusters is List[List[DetectionTuple]]
        for cluster in input_clusters:
            for det in cluster:
                # Filter ID
                if exclude_id_zero and det[7] == 0:
                    continue
                
                # Check Ground Truth
                if det[3] == NoiseType.REAL:
                    tp += 1
                    real_gt_count += 1
                else:
                    fp += 1 # Noise that got clustered

        # 2. Analyze Noise Points (Predicted = Negative)
        # input_noise_flat is List[DetectionTuple]
        for det in input_noise_flat:
            # Filter ID
            if exclude_id_zero and det[7] == 0:
                continue
                
            # Check Ground Truth
            if det[3] == NoiseType.REAL:
                fn += 1 # Real point that was discarded as noise
                real_gt_count += 1
            else:
                tn += 1 # Noise point correctly discarded
                    
        # Stats Calculation
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / real_gt_count if real_gt_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'counts': [tp, fp, fn, tn],
            'stats': (precision, recall, f1)
        }

    # --- Helper: Bar Plotter ---
    def plot_bar_subplot(ax, title, data):
        counts = data['counts'] # [TP, FP, FN, TN]
        prec, rec, f1 = data['stats']
        
        labels = ['TP: Correctly Kept Real Points', 'FP: Incorrectly Filtered Real Points', 'FN: Incorrectly Kept Noise Points', 'TN: Correctly Filtered Noise Points']
        colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E'] # Green, Red, Orange, Gray

        labels = ['\n'.join(textwrap.wrap(l, width=12)) for l in labels]
        
        bars = ax.bar(labels, counts, color=colors)
        ax.bar_label(bars, fmt='%d', padding=3)
        
        # Formatting
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_ylabel("Count")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add Stats Text Box
        stats_text = (
            f"Prec:   {prec*100:.1f}%\n"
            f"Recall: {rec*100:.1f}%\n"
            f"F1:     {f1*100:.1f}%"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=11, family='monospace', verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # --- 1. Prepare Data ---
    
    # Helper to flatten noise if it comes as List[List] (e.g. grouped by frame)
    def ensure_flat(noise_input):
        if noise_input and isinstance(noise_input[0], list):
             return [item for sublist in noise_input for item in sublist]
        return noise_input

    flat_noise_normal = ensure_flat(cluster_noise)
    flat_noise_filtered = ensure_flat(filtered_cluster_noise)

    # 1. Normal - All IDs
    res_norm_all = get_metrics(clusters, flat_noise_normal, exclude_id_zero=False)
    # 2. Normal - ID > 0
    res_norm_filter = get_metrics(clusters, flat_noise_normal, exclude_id_zero=True)
    # 3. Filtered - All IDs
    res_filt_all = get_metrics(filtered_clusters, flat_noise_filtered, exclude_id_zero=False)
    # 4. Filtered - ID > 0
    res_filt_filter = get_metrics(filtered_clusters, flat_noise_filtered, exclude_id_zero=True)

    # --- 2. Plotting ---
    
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Global Clustering Performance Summary (Ablation Test)', fontsize=18, weight='bold', y=0.98)
    
    # Plot 1: Normal (All IDs)
    plot_bar_subplot(axes[0, 0], "1. Normal Clustering (Score With Static Background Points)", res_norm_all)
    
    # Plot 2: Normal (ID > 0)
    plot_bar_subplot(axes[0, 1], "2. Normal Clustering (Score Without Static Background Points)", res_norm_filter)
    
    # Plot 3: Filtered (All IDs)
    plot_bar_subplot(axes[1, 0], "3. Filtered Clustering (Score With Static Background Points)", res_filt_all)
    
    # Plot 4: Filtered (ID > 0)
    plot_bar_subplot(axes[1, 1], "4. Filtered Clustering (Score Without Static Background Points)", res_filt_filter)
    
    # Final Adjustments
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, "global_summary_ablation_2.png")
    # print(f"Saving global summary to: {save_path}") # Optional print
    fig.savefig(save_path)
    plt.close(fig)

# TODO:
# def print_noise_injection_summary(detections: List[DetectionTuple]):
#     shifted_x_noise = [det for det in detections if det[3] == NoiseType.SHIFTX]
#     for noise in shifted_x_noise:
        

    
def save_cluster_survivor_analysis(
    frame_number: int,
    clusters: List[List[DetectionTuple]],
    output_dir: str = "output/survivor_error_analysis"
):
    """
    Plots the error distribution ONLY for points that survived clustering.
    Broken down by:
      1. Real Static (Background, ID=0)
      2. Real Moving (Objects, ID!=0)
      3. Multipath Ghosts
      4. Random Noise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Flatten the survivors
    survivors = [det for cluster in clusters for det in cluster]
    
    if not survivors:
        print(f"Frame {frame_number}: No clusters found to analyze.")
        return

    # 2. Extract Errors by Detailed Category
    # Index 2 = Error Metric
    # Index 3 = NoiseType
    # Index 7 = Object ID (0 = Static/Background)
    
    # Split Real into Static vs Moving
    errors_real_static = [
        det[2] for det in survivors 
        if det[3] == NoiseType.REAL and det[7] == 0
    ]
    errors_real_moving = [
        det[2] for det in survivors 
        if det[3] == NoiseType.REAL and det[7] != 0
    ]
    
    errors_multipath = [det[2] for det in survivors if det[3] == NoiseType.MULTIPATH_GHOST]
    errors_random = [det[2] for det in survivors if det[3] == NoiseType.RANDOM_CLUTTER]
    
    # Combine for bin calculation
    all_errors = errors_real_static + errors_real_moving + errors_multipath + errors_random

    # 3. Setup Plot (2x2 Grid)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Frame {frame_number} - Cluster Survivor Error Analysis (Split Static/Moving)', fontsize=16, weight='bold')
    
    # Shared Binning Logic
    if all_errors:
        max_val = np.percentile(all_errors, 95) if len(all_errors) > 0 else 1.0
        max_val = max(max_val, 0.1)
        bins = np.linspace(0, max_val, 40)
    else:
        bins = 40

    # --- Plot 1: Real Points (Split by Motion) ---
    ax1 = axes[0, 0]
    data_real = []
    colors_real = []
    labels_real = []
    
    if errors_real_static:
        data_real.append(errors_real_static)
        colors_real.append('steelblue') # Blue-ish for static
        labels_real.append(f'Static (N={len(errors_real_static)})')
        
    if errors_real_moving:
        data_real.append(errors_real_moving)
        colors_real.append('limegreen') # Green for moving
        labels_real.append(f'Moving (N={len(errors_real_moving)})')
        
    if data_real:
        ax1.hist(data_real, bins=bins, color=colors_real, label=labels_real, 
                 stacked=True, alpha=0.7, edgecolor='black')
        ax1.legend()
        
    ax1.set_title("1. Surviving Real Points (Static vs Moving)")
    ax1.set_xlabel("Error Metric")
    ax1.set_ylabel("Count")

    # --- Plot 2: Multipath Ghosts ---
    ax2 = axes[0, 1]
    if errors_multipath:
        ax2.hist(errors_multipath, bins=bins, color='purple', alpha=0.7, edgecolor='black')
        mean_val = np.mean(errors_multipath)
        ax2.axvline(mean_val, color='indigo', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax2.legend()
    ax2.set_title(f"2. Surviving Multipath Ghosts (N={len(errors_multipath)})")
    ax2.set_xlabel("Error Metric")

    # --- Plot 3: Random Noise ---
    ax3 = axes[1, 0]
    if errors_random:
        ax3.hist(errors_random, bins=bins, color='red', alpha=0.7, edgecolor='black')
        mean_val = np.mean(errors_random)
        ax3.axvline(mean_val, color='darkred', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No Random Noise Survived", ha='center', va='center')
    ax3.set_title(f"3. Surviving Random Noise (N={len(errors_random)})")
    ax3.set_xlabel("Error Metric")

    # --- Plot 4: Combined Separation View (All 4 Categories) ---
    ax4 = axes[1, 1]
    data_list = []
    color_list = []
    label_list = []
    
    # Add in order of "Good" to "Bad"
    if errors_real_static:
        data_list.append(errors_real_static)
        color_list.append('steelblue')
        label_list.append('Real Static')
        
    if errors_real_moving:
        data_list.append(errors_real_moving)
        color_list.append('limegreen')
        label_list.append('Real Moving')
        
    if errors_multipath:
        data_list.append(errors_multipath)
        color_list.append('purple')
        label_list.append('Multipath')
        
    if errors_random:
        data_list.append(errors_random)
        color_list.append('red')
        label_list.append('Random')
        
    if data_list:
        # Density=False to see counts, alpha for overlap visibility
        ax4.hist(data_list, bins=bins, color=color_list, label=label_list, 
                 stacked=False, alpha=0.5, edgecolor='black')
        ax4.legend()
        ax4.set_title("4. Combined Comparison (Counts)")
    
    ax4.set_xlabel("Error Metric")
    
    # 4. Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, f"frame_{frame_number:04d}_survivors_split.png")
    try:
        fig.savefig(filename)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig)