from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import cv2
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
                
        elif det[3] == NoiseType.MULTIPATH_GHOST:
            gt_multipath += 1
            if is_clustered:
                fp_points.append(det) # False Positive (Noise and Clustered)
            else:
                tn_points.append(det) # True Negative (Noise and Filtered)
                
        elif det[3] == NoiseType.RANDOM_CLUTTER:
            gt_random += 1
            if is_clustered:
                fp_points.append(det) # False Positive (Noise and Clustered)
            else:
                tn_points.append(det) # True Negative (Noise and Filtered)
            
    # Final counts
    tp, fn = len(tp_points), len(fn_points)
    total_fp, total_tn = len(fp_points), len(tn_points)
    total_real_points = gt_real
    
    # Count breakdown for FP/TN Bar Chart
    fp_rand = sum(1 for det in fp_points if det[3] == NoiseType.RANDOM_CLUTTER)
    fp_mp = total_fp - fp_rand
    tn_rand = sum(1 for det in tn_points if det[3] == NoiseType.RANDOM_CLUTTER)
    tn_mp = total_tn - tn_rand
    
    # d. Calculate final scores
    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
    recall = tp / total_real_points if total_real_points > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Prepare dictionaries for plotting
    fp_dict = {'random': fp_rand, 'mp': fp_mp}
    tn_dict = {'random': tn_rand, 'mp': tn_mp}

    # --- 2. Data Extraction for Plots ---
    
    real_pos = [det[4] for det in all_detections if det[3] == NoiseType.REAL]
    random_pos = [det[4] for det in all_detections if det[3] == NoiseType.RANDOM_CLUTTER]
    mp_pos = [det[4] for det in all_detections if det[3] == NoiseType.MULTIPATH_GHOST]
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

    # === PLOT 4: Detection Counts (Bar Chart) ===
    ax4 = axes[1, 0]
    count_labels = ['True Pos (TP)', 'False Pos (FP)', 'False Neg (FN)', 'True Neg (TN)']
    count_values = [tp, total_fp, fn, total_tn]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E']
    bars = ax4.bar(count_labels, count_values, color=colors)
    ax4.bar_label(bars); ax4.set_title("4. Overall Detection Counts"); ax4.set_ylabel("Number of Points")

    # === PLOT 5: FP/TN Breakdown (Bar Chart) ===
    ax5 = axes[1, 1]
    noise_labels = ['Random', 'Multipath']
    fp_values = [fp_dict.get('random', 0), fp_dict.get('mp', 0)]
    tn_values = [tn_dict.get('random', 0), tn_dict.get('mp', 0)]
    bar_width = 0.35; index = np.arange(len(noise_labels))
    bars1 = ax5.bar(index - bar_width/2, fp_values, bar_width, label='False Pos (Clustered)', color='#F44336')
    bars2 = ax5.bar(index + bar_width/2, tn_values, bar_width, label='True Neg (Filtered)', color='#9E9E9E')
    ax5.bar_label(bars1); ax5.bar_label(bars2); ax5.set_title('5. Noise Filter Performance Breakdown')
    ax5.set_ylabel('Number of Points'); ax5.set_xticks(index); ax5.set_xticklabels(noise_labels)
    ax5.legend(fontsize=8)
    
    # === PLOT 6: Placeholder (empty in 2x3) ===
    # Axes[1, 2] is not used in the 2x3 layout since the 3D plots are added separately.
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
    COLOR_MULTIPATH = (0, 0, 255)  # Red
    COLOR_RANDOM = (0, 0, 0)

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
            cv2.circle(img_with_points, (int(u), int(v)), 2, color, -1)
            
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

def save_frame_histogram_by_object(
    frame_number: int,
    all_frame_detections: List[DetectionTuple],
    image_rgb: np.ndarray,                   
    T_Cam_from_Radar: np.ndarray,      # Extrinsics (T_C_R)
    K: np.ndarray,                     # Intrinsics (K)
    output_dir: str = "output/object_analysis"
):
    """
    Groups detections, calculates metrics, and saves a 2x3 statistical analysis plot
    and an image projection plot per object.
    """
    
    # 1. Group all detections by object_id
    grouped_detections: Dict[int, List[DetectionTuple]] = {}
    for det in all_frame_detections:
        object_id = det[7]
        if object_id not in grouped_detections:
            grouped_detections[object_id] = []
        grouped_detections[object_id].append(det)

    # get random noise detections
    random_noise_detections = [det for det in all_frame_detections if det[3] == NoiseType.RANDOM_CLUTTER]
    
    # 3. Iterate and process each object
    for object_id, detections in grouped_detections.items():
        
        
        # --- 2.1 Extract and Filter Data: Split Noise Types ---
        real_disp_errors = []
        multipath_disp_errors, random_disp_errors = [], []
        
        real_positions, multipath_positions, random_positions = [], [], []
        real_velocities, multipath_velocities, random_velocities = [], [], []
        real_pred_vel_vectors = []

        gt_vel_radar = detections[0][5]
        gt_vel_world = detections[0][6]

        for det in detections:
            vel_mag, vel_err, disp_err, noiseType, pos_3d, pred_vel, _, _ = det[:8]
            if noiseType == NoiseType.REAL:
                real_disp_errors.append(disp_err)
                real_positions.append(pos_3d); real_velocities.append(pred_vel)
                real_pred_vel_vectors.append(pred_vel)
            elif noiseType == NoiseType.MULTIPATH_GHOST:
                multipath_disp_errors.append(disp_err)
                multipath_positions.append(pos_3d); multipath_velocities.append(pred_vel)

        for det in random_noise_detections:
            _, _, disp_err, noiseType, pos_3d, pred_vel, _, _ = det[:8]
            random_disp_errors.append(disp_err)
            random_positions.append(pos_3d); random_velocities.append(pred_vel)


        
        # --- 2.2 Calculate Mean Velocities and Errors ---
        real_disp_np = np.array(real_disp_errors)
        multipath_disp_np = np.array(multipath_disp_errors)
        random_disp_np = np.array(random_disp_errors)
        
        # Calculate averages based on the split lists
        pred_vel_avg_radar = np.mean(np.array(real_pred_vel_vectors), axis=0) if real_pred_vel_vectors else np.array([0.0, 0.0, 0.0])
        pred_vel_avg_multipath = np.mean(np.array(multipath_velocities), axis=0) if multipath_velocities else np.array([0.0, 0.0, 0.0])
        pred_vel_avg_random = np.mean(np.array(random_velocities), axis=0) if random_velocities else np.array([0.0, 0.0, 0.0])
        
        # MAE calculation (Mean Absolute Component Error between Avg Real Vector and GT Vector)
        if real_pred_vel_vectors:
            error_vector_avg = pred_vel_avg_radar - gt_vel_radar
            mae = np.mean(np.abs(error_vector_avg))
        else:
            mae = 0.0

        # 1. Average Closest Distance within Real Positions (meters)
        dist_R_pos_internal = calculate_avg_intra_set_distance(real_positions)
        
        # 2. Average Closest Distance within Multipath Positions (meters)
        dist_MP_pos_internal = calculate_avg_intra_set_distance(multipath_positions)
        
        # 3. Average Closest Distance within Real Velocities (m/s)
        dist_R_vel_internal = calculate_avg_intra_set_distance(real_velocities)
        
        # 4. Average Closest Distance within Multipath Velocities (m/s)
        dist_MP_vel_internal = calculate_avg_intra_set_distance(multipath_velocities)

        # Calculate Cohesion Ratios (Scaling Factors) ---
        
        epsilon = 1e-6 # To avoid division by zero
        
        # Real Cohesion Ratio (Velocity / Position)
        if dist_R_pos_internal > epsilon:
            cohesion_ratio_R = dist_R_vel_internal / dist_R_pos_internal
        else:
            cohesion_ratio_R = 999.0 # Use a large sentinel value if points are perfectly stacked
            
        # Multipath Cohesion Ratio (Velocity / Position)
        if dist_MP_pos_internal > epsilon:
            cohesion_ratio_MP = dist_MP_vel_internal / dist_MP_pos_internal
        else:
            cohesion_ratio_MP = 999.0 # Use a large sentinel value
        
        # --- 3.2 Setup Directories and Header ---
        object_output_dir = os.path.join(output_dir, f"object_{object_id:04d}")
        os.makedirs(object_output_dir, exist_ok=True)
        
        gt_rad_str = f"({gt_vel_radar[0]:.2f}, {gt_vel_radar[1]:.2f}, {gt_vel_radar[2]:.2f})"
        gt_wld_str = f"({gt_vel_world[0]:.2f}, {gt_vel_world[1]:.2f}, {gt_vel_world[2]:.2f})"
        avg_pred_str = f"({pred_vel_avg_radar[0]:.2f}, {pred_vel_avg_radar[1]:.2f}, {pred_vel_avg_radar[2]:.2f})"

        header_text = (
            f"Real Points: {len(real_positions)} | Multipath: {len(multipath_positions)} | Random Noise: {len(random_positions)}\n"
            f"GT Velocity (RADAR): {gt_rad_str} | GT Velocity (WORLD): {gt_wld_str}\n"
            f"Avg PD Velocity (RADAR, Real Only): {avg_pred_str} | MAE (Vector Diff): {mae:.4f} m/s"
            f"Average Internal Cohesion (Nearest Neighbor Distances):\n"
            f"  Pos Cohesion: Real={dist_R_pos_internal:.3f}m | MP={dist_MP_pos_internal:.3f}m\n"
            f"  Vel Cohesion: Real={dist_R_vel_internal:.3f}m/s | MP={dist_MP_vel_internal:.3f}m/s"
            f"Cohesion Scaling (Vel/Pos Ratio):\n"
            f"  Real Ratio (V/P): {cohesion_ratio_R:.3f} | MP Ratio (V/P): {cohesion_ratio_MP:.3f}"
        )
        
        # --- 3.3 Generate Image Projection ---
        try:
            # Note: image_rgb is assumed to be RGB, but cv2 functions require BGR.
            image_bgr_to_draw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            all_detections = detections + random_noise_detections

            points_2d, _, full_object_mask = project_and_get_depths(
                all_detections, T_Cam_from_Radar, K
            )
            image_with_points = draw_points_by_noise(
                image_bgr_to_draw, all_detections, points_2d, full_object_mask
            )

            image_save_path = os.path.join(object_output_dir, f"frame_{frame_number:08d}_projection.png")
            cv2.imwrite(image_save_path, image_with_points)
            
        except Exception as e:
            # Placeholder/Error handling if image ops fail
            pass
        

        # --- 3.4 Plotting (Statistical Analysis: 2x3 Grid) ---

        plt.style.use('ggplot')
        fig, axes = plt.subplots(2, 3, figsize=(26, 18)) 
        
        fig.suptitle(f'Frame {frame_number} - Object ID {object_id} Analysis', fontsize=22, y=0.98) 
        fig.text(
            0.5, 0.92, header_text, 
            ha='center', fontsize=14, family='monospace', 
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.8')
        )
        
        # --- Shared Limits Setup ---
        all_disp_data_hist = np.concatenate((real_disp_np, multipath_disp_np, random_disp_np)) if real_disp_np.size + multipath_disp_np.size + random_disp_np.size > 0 else np.array([])
        
        def get_shared_limits(data_array: np.ndarray):
            if data_array.size == 0: return 0, 1
            data_min, data_max = np.min(data_array), np.max(data_array)
            if data_min == data_max: data_min -= 0.5; data_max += 0.5
            padding = (data_max - data_min) * 0.05
            return data_min - padding, data_max + padding
        
        disp_min, disp_max = get_shared_limits(all_disp_data_hist)
        disp_bins = np.linspace(disp_min, disp_max, 21) if all_disp_data_hist.size > 0 else 20
        
        # === ROW 1: VELOCITY, DISPLACEMENT HISTOGRAMS ===

        # --- Row 1, Col 1 (axes[0, 0]): GT vs. Predicted Velocity Vector (Bar Graph) ---
        ax = axes[0, 0]
        gt_vec = gt_vel_radar
        pred_vec_real = pred_vel_avg_radar 
        pred_vec_mp = pred_vel_avg_multipath 

        labels = [r'$V_x$', r'$V_y$', r'$V_z$']
        x = np.arange(len(labels)) 
        width = 0.25

        max_val = np.max(np.abs(np.concatenate([gt_vec, pred_vec_real, pred_vec_mp])))
        limit = max(1.0, max_val * 1.2) 

        # Create the bars (shifted for 3 groups)
        rects1 = ax.bar(x - width, gt_vec, width, label='GT Velocity', color='#1f77b4') 
        rects2 = ax.bar(x, pred_vec_real, width, label='PD Velocity (Avg Real)', color='#ff7f0e') 
        rects3 = ax.bar(x + width, pred_vec_mp, width, label='PD Velocity (Avg Multipath)', color='#2ca02c') 
        
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'GT vs. Predicted Velocity Vector (Radar Coords)')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--'); ax.set_ylim(-limit, limit) 

        # --- Row 1, Col 2 (axes[0, 1]): Real Displacement Error (Histogram) ---
        ax = axes[0, 1]
        if real_disp_np.size > 0:
            mean_val = np.mean(real_disp_np)
            ax.hist(real_disp_np, bins=disp_bins, color='blue', alpha=0.7, edgecolor='black')
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2)
            ax.set_title(f"Real Displacement Error (N={len(real_disp_np)})\nMean: {mean_val:.4f} pix")
            ax.set_xlabel("Displacement Error (pixels)")
            if all_disp_data_hist.size > 0: ax.set_xlim(disp_bins[0], disp_bins[-1])
        else: ax.set_title("Real Displacement Error\n(No Data)"); ax.set_ylabel("Count")

        # --- Row 1, Col 3 (axes[0, 2]): Multipath Displacement Error (Histogram) ---
        ax = axes[0, 2]
        if multipath_disp_np.size > 0:
            mean_val = np.mean(multipath_disp_np)
            ax.hist(multipath_disp_np, bins=disp_bins, color='purple', alpha=0.7, edgecolor='black') 
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2)
            ax.set_title(f"Multipath Displacement Error (N={len(multipath_disp_np)})\nMean: {mean_val:.4f} pix") 
            ax.set_xlabel("Displacement Error (pixels)")
            if all_disp_data_hist.size > 0: ax.set_xlim(disp_bins[0], disp_bins[-1])
        else: ax.set_title("Multipath Displacement Error\n(No Data)"); ax.set_ylabel("Count")

        # === ROW 2: RANDOM DISPLACEMENT, 3D POSITION, AND 3D VELOCITY ===

        # --- Row 2, Col 1 (axes[1, 0]): Random Noise Displacement Error (Histogram) ---
        ax = axes[1, 0]
        if random_disp_np.size > 0:
            mean_val = np.mean(random_disp_np)
            ax.hist(random_disp_np, bins=disp_bins, color='red', alpha=0.7, edgecolor='black')
            ax.axvline(mean_val, color='blue', linestyle='dashed', linewidth=2)
            ax.set_title(f"Random Noise Disp. Error (N={len(random_disp_np)})\nMean: {mean_val:.4f} pix") 
            ax.set_xlabel("Displacement Error (pixels)")
            if all_disp_data_hist.size > 0: ax.set_xlim(disp_bins[0], disp_bins[-1])
        else: ax.set_title("Random Noise Disp. Error\n(No Data)"); ax.set_ylabel("Count")

        # --- Row 2, Col 2 (axes[1, 1]): 3D Position (Radar Coords) ---
        axes[1, 1].remove(); ax_pos = fig.add_subplot(2, 3, 5, projection='3d')
        if real_positions:
            real_pos_np = np.array(real_positions)
            ax_pos.scatter(real_pos_np[:, 0], real_pos_np[:, 1], real_pos_np[:, 2], c='blue', s=10, label=f'Real (N={len(real_pos_np)})', alpha=0.5)
        if multipath_positions:
            multipath_pos_np = np.array(multipath_positions)
            ax_pos.scatter(multipath_pos_np[:, 0], multipath_pos_np[:, 1], multipath_pos_np[:, 2], c='red', s=5, label=f'Mulitpath (N={len(multipath_pos_np)})', alpha=0.4)
        ax_pos.set_title("3D Position (Radar Coordinates)"); ax_pos.set_xlabel("X (m)"); ax_pos.set_ylabel("Y (m)"); ax_pos.set_zlabel("Z (m)"); ax_pos.legend()

        # --- Row 2, Col 3 (axes[1, 2]): 3D Solved Velocity (Radar Coords) ---
        axes[1, 2].remove(); ax_vel = fig.add_subplot(2, 3, 6, projection='3d')
        if real_velocities:
            real_vel_np = np.array(real_velocities)
            ax_vel.scatter(real_vel_np[:, 0], real_vel_np[:, 1], real_vel_np[:, 2], c='blue', s=10, label=f'Real (N={len(real_vel_np)})', alpha=0.5)
        if multipath_velocities:
            multipath_vel_np = np.array(multipath_velocities)
            ax_vel.scatter(multipath_vel_np[:, 0], multipath_vel_np[:, 1], multipath_vel_np[:, 2], c='red', s=5, label=f'Multipath (N={len(multipath_vel_np)})', alpha=0.4)
        ax_vel.set_title("3D Solved Velocity (Radar Coords)"); ax_vel.set_xlabel("Vx (m/s)"); ax_vel.set_ylabel("Vy (m/s)"); ax_vel.set_zlabel("Vz (m/s)"); ax_vel.legend()
        limit = 5; ax_vel.set_xlim([-limit, limit]); ax_vel.set_ylim([-limit, limit]); ax_vel.set_zlim([-limit, limit])

        # --- 3.5 Save and Close ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.90]) 
        save_path = os.path.join(object_output_dir, f"frame_{frame_number:08d}_analysis.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)

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