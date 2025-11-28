import numpy as np
from sklearn.cluster import DBSCAN
from scipy import stats
from typing import List, Tuple, Union, Optional, Dict, DefaultDict
from .types import DetectionTuple, NoiseType

def filter_static_points(detections: List[DetectionTuple]) -> List[DetectionTuple]:
    # vel_3d_gt
    filtered_detections = [det for det in detections if np.linalg.norm(det[13]) > 0.05]

    return filtered_detections

def cluster_detections_perfect(detections: List[DetectionTuple]) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Groups detections into clusters based on ground truth object IDs.
    
    Args:
        detections: A list of DetectionTuples.
        
    Returns:
        clusters: A list of lists, where each inner list contains DetectionTuples 
                  belonging to a specific real object.
        noise: A flat list of DetectionTuples considered noise (Clutter, Ghosts, etc.).
    """
    
    # Constants for tuple indices
    IDX_NOISE = 3   # Based on the tuple definition comment
    IDX_OBJ_ID = 7  # Based on the user prompt
    
    # Use a dictionary to group real points: { object_id: [list_of_detections] }
    clusters_map: Dict[int, List[DetectionTuple]] = DefaultDict(list)
    noise_list: List[DetectionTuple] = []

    for det in detections:
        # Extract necessary fields
        n_type = det[IDX_NOISE]

        if n_type == NoiseType.RANDOM_CLUTTER:
            noise_list.append(det)
        else:
            obj_id = det[IDX_OBJ_ID]
            clusters_map[obj_id].append(det)

    # Convert dictionary values to the list of lists format
    clusters = list(clusters_map.values())

    return clusters, noise_list


def cluster_detections_6d(
    detections: List[DetectionTuple],
    eps: float,
    min_samples: int,
    velocity_weight: float
) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Filters detections using a single 6D DBSCAN on scaled position + velocity.

    Args:
        detections: List of detection tuples.
        eps: The max 6D distance to be considered a neighbor.
             This is now in "scaled meters".
        min_samples: The minimum number of points to form a cluster.
        velocity_weight: How much to scale velocity. A 1m/s difference
                         will be treated as a `velocity_weight` meter
                         difference in the distance calculation.
                         A good start is 1.0.

    Returns:
        (clusters, noise_points)
    """
    
    if not detections:
        return [], []

    # --- 1. Create the N x 6 data array ---
    data_6d = []
    for det in detections:
        pos = det[4] # pos_3d_radar
        vel = det[5] # vel_3d_radar
        data_6d.append(np.concatenate([pos, vel]))
    
    if not data_6d:
         return [], []

    X = np.array(data_6d)

    # --- 2. Apply Manual Scaling (The Critical Step) ---
    X_scaled = np.copy(X)
    X_scaled[:, 3:6] *= velocity_weight # Scale only the velocity components
    
    # --- 3. Run DBSCAN on the scaled 6D data ---
    # `eps` is now a 6D distance. If velocity_weight=1.0 and eps=1.0,
    # a point at 0.8m and 0.5m/s distance away would be a neighbor,
    # since sqrt(0.8^2 + 0.5^2) = sqrt(0.89) < 1.0.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_

    # --- 4. Sort results into clusters and noise ---
    final_clusters = []
    final_noise = []

    unique_labels = set(labels)
    for k in unique_labels:
        indices = np.where(labels == k)[0]
        
        if k == -1:
            # This point was labeled as "NOISE" by DBSCAN
            for i in indices:
                det = detections[i]
                final_noise.append(det)
                
        else:
            # This point was put into a CLUSTER
            cluster_group = []
            for i in indices:
                det = detections[i]
                cluster_group.append(det)
            
            final_clusters.append(cluster_group)
            
            
    return final_clusters, final_noise

def cluster_detections_polar(
    detections: List[DetectionTuple],
    eps: float,             # Base distance threshold
    min_samples: int,
    azimuth_weight: float = 20.0, # 1 radian diff = X meters diff
    velocity_weight: float = 1.0  # 1 m/s diff = X meters diff
) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Clusters detections using a Hybrid Polar-Cartesian feature space.
    
    Feature Vector: [Range, Azimuth_Scaled, Z, Vx_Scaled, Vy_Scaled, Vz_Scaled]
    
    Why this works better:
    Radar is very precise in Range, but blurry in Azimuth.
    This allows us to set a tight 'eps' for Range, but effectively 
    scale down the Azimuth distance so points in an arc are grouped.
    """
    
    if not detections:
        return [], []

    # --- 1. Extract and Transform Data ---
    # We need to build a matrix of shape (N, 6)
    # columns: [range, azimuth*scale, z, vx*scale, vy*scale, vz*scale]
    
    data_hybrid = []
    
    for det in detections:
        pos = det[4] # [x, y, z]
        vel = det[5] # [vx, vy, vz]
        
        x, y, z = pos
        
        # Coordinate Transform: Cartesian -> Polar
        r = np.sqrt(x**2 + y**2)
        azimuth = np.arctan2(y, x) # Result in radians (-pi to pi)
        
        # Build feature vector
        # We apply weights immediately here
        features = [
            r,                          # 0: Range (Keep as meters)
            azimuth * azimuth_weight,   # 1: Azimuth (Weighted to match meters)
            z,                          # 2: Height (Meters)
            vel[0] * velocity_weight,   # 3: Vx
            vel[1] * velocity_weight,   # 4: Vy
            vel[2] * velocity_weight    # 5: Vz
        ]
        data_hybrid.append(features)
    
    X = np.array(data_hybrid)
    
    # --- 2. Run DBSCAN ---
    # Note: Standard Euclidean DBSCAN on Polar coordinates 
    # creates a "Mahalanobis-like" effect in Cartesian space.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # --- 3. Group Results (Standard) ---
    final_clusters = []
    final_noise = []

    unique_labels = set(labels)
    for k in unique_labels:
        indices = np.where(labels == k)[0]
        
        if k == -1:
            for i in indices:
                final_noise.append(detections[i])
        else:
            cluster_group = [detections[i] for i in indices]
            final_clusters.append(cluster_group)
            
    return final_clusters, final_noise

def cluster_detections_anisotropic(
    detections: List[DetectionTuple],
    eps: float,
    min_samples: int,
    weight_vz: float = 2.0,    # High weight: Trust Doppler/Longitudinal
    weight_vxy: float = 0.0,   # Zero weight: Ignore Transverse during grouping
) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Step 1: Groups detections into clusters using a 6D anisotropic distance.
    Crucially, weight_vxy should be 0.0 to allow 'supersonic' ghosts to 
    cluster with the real object.
    
    Returns:
        (raw_clusters, dbscan_noise)
    """
    if not detections:
        return [], []

    # --- 1. Create Data Arrays ---
    data_6d = []
    for det in detections:
        pos = det[4] # pos_3d_radar (x, y, z)
        vel = det[5] # vel_3d_radar (vx, vy, vz)
        data_6d.append(np.concatenate([pos, vel]))
    
    X = np.array(data_6d)

    # --- 2. Anisotropic Scaling ---
    X_scaled = np.copy(X)
    
    # Transverse Velocity (Vx, Vy): Set to 0.0
    # This collapses the "velocity noise cloud" into a single point in Vxy space.
    X_scaled[:, 3] *= weight_vxy 
    X_scaled[:, 4] *= weight_vxy 
    
    # Longitudinal Velocity (Vz): Set high
    # This acts as the anchor. Points must match in Doppler to cluster.
    X_scaled[:, 5] *= weight_vz 
    
    # --- 3. Run DBSCAN ---
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_

    # --- 4. Organize Results ---
    raw_clusters = []
    dbscan_noise = []

    unique_labels = set(labels)
    for k in unique_labels:
        indices = np.where(labels == k)[0]
        
        if k == -1:
            # Points that didn't match anything Spatially or in Doppler
            for i in indices:
                dbscan_noise.append(detections[i])
        else:
            # Valid groups (potentially containing mixed real + ghost points)
            cluster_group = []
            for i in indices:
                cluster_group.append(detections[i])
            raw_clusters.append(cluster_group)
            
    return raw_clusters, dbscan_noise

# def filter_static_points(
#     detections: List[DetectionTuple], 
#     min_vel: float,
#     t_a_to_b: Optional[np.ndarray] = None, 
#     t_a_to_r: Optional[np.ndarray] = None,
#     delta_t: float = 0.05
# ) -> List[DetectionTuple]:
#     """
#     Filters out static points with corrected physics and coordinate rotation.
#     """
    
#     # Default v_ego (in Radar Frame)
#     v_ego_radar_frame = np.zeros(3)
    
#     if t_a_to_b is not None:
#         # 1. Extract Translation (Camera Frame Motion of POINTS)
#         trans_cam = t_a_to_b[:3, 3]
        
#         # 2. Invert to get Camera Velocity (Camera Frame Motion of CAMERA)
#         v_ego_cam_frame = -(trans_cam / delta_t)
        
#         # 3. Rotate into Radar Frame
#         if t_a_to_r is not None:
#             # Extract 3x3 Rotation: Camera -> Radar
#             R_cam_to_radar = t_a_to_r[:3, :3]
#             v_ego_radar_frame = R_cam_to_radar @ v_ego_cam_frame
#         else:
#             # Fallback (Only safe if sensors are perfectly aligned)
#             v_ego_radar_frame = v_ego_cam_frame

#     filtered = []
#     for det in detections:
#         # det[5] is predicted relative velocity (Radar Frame)
#         v_rel = det[5] 
        
#         # 4. Calculate Absolute World Velocity (All in Radar Frame now)
#         v_world = v_rel + v_ego_radar_frame
        
#         vel_mag = np.linalg.norm(v_world)
        
#         if vel_mag >= min_vel:
#             filtered.append(det)
            
#     return filtered

def filter_clusters_quantile(
    clusters: List[List[DetectionTuple]],
    keep_ratio: Union[float, Tuple[float, float, float]] = 0.90 
) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Refines clusters by keeping only the central percentage of points for each axis.
    
    Args:
        keep_ratio: 
            The ratio of data to KEEP around the median.
            0.90 = Keep middle 90% (Trim top 5% and bottom 5%).
            Can be a float (applied to x,y,z) or a tuple (ratio_x, ratio_y, ratio_z).
    """
    refined_clusters = []
    purge_noise = []
    
    # 1. Parse ratios into X, Y, Z components
    if isinstance(keep_ratio, (float, int)):
        r_x = r_y = r_z = float(keep_ratio)
    else:
        r_x, r_y, r_z = keep_ratio

    # Helper to calculate quantiles for a specific ratio
    # e.g., ratio 0.8 -> low=0.1, high=0.9
    def get_bounds(ratio):
        alpha = (1.0 - ratio) / 2.0
        return alpha, 1.0 - alpha

    lx, hx = get_bounds(r_x)
    ly, hy = get_bounds(r_y)
    lz, hz = get_bounds(r_z)

    for cluster in clusters:
        # hack, skip processing background that has no multipath noise
        num_static_points = sum(1 for det in cluster if det[7] == 0)
        total_points = len(cluster)
        if(num_static_points > 0.5 * total_points):
            refined_clusters.append(cluster)
            continue

        # Edge case: Cannot calculate meaningful quantiles on tiny clusters
        # We assume clusters < 3 points are either valid small objects or 
        # will be filtered by min_samples later. We keep them intact here.
        if len(cluster) < 3:
            refined_clusters.append(cluster)
            continue
            
        # Extract velocities: [N, 3]
        velocities = np.array([det[5] for det in cluster])
        
        # 2. Calculate Upper and Lower Bounds for each axis
        # shape of q_low/high is (3,) -> [x_val, y_val, z_val]
        q_lows = np.quantile(velocities, [lx, ly, lz], axis=0) # Note: this applies lx to all cols first
        q_highs = np.quantile(velocities, [hx, hy, hz], axis=0) 
        
        # We need specific bounds per axis because ratios might differ
        # (Slightly less efficient but clearer logic for different ratios)
        vx_bounds = np.quantile(velocities[:, 0], [lx, hx])
        vy_bounds = np.quantile(velocities[:, 1], [ly, hy])
        vz_bounds = np.quantile(velocities[:, 2], [lz, hz])
        
        current_refined_group = []
        
        for i, det in enumerate(cluster):
            vx, vy, vz = velocities[i]
            
            # --- THE CHECK ---
            # Point is outlier if it is OUTSIDE the bounds on ANY axis
            bad_x = (vx < vx_bounds[0]) or (vx > vx_bounds[1])
            bad_y = (vy < vy_bounds[0]) or (vy > vy_bounds[1])
            bad_z = (vz < vz_bounds[0]) or (vz > vz_bounds[1])
            
            if bad_x or bad_y or bad_z:
                purge_noise.append(det)
            else:
                current_refined_group.append(det)
        
        if len(current_refined_group) > 0:
            refined_clusters.append(current_refined_group)
            
    return refined_clusters, purge_noise

def filter_clusters_mad(
    clusters: List[List[DetectionTuple]],
    std_threshold: Union[float, Tuple[float, float, float]] = 3.0
) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Refines clusters using the Double Median Absolute Deviation (Double MAD) method.
    This is robust against skewed distributions (like exponential) where the tail 
    is longer on one side than the other.
    """
    refined_clusters = []
    purge_noise = []
    
    # Constant to convert MAD to Standard Deviation (consistent with Normal dist)
    MAD_SCALE_FACTOR = 1.4826
    EPSILON = 1e-6 # To prevent division by zero

    # 1. Parse thresholds into X, Y, Z components (Restored logic)
    if isinstance(std_threshold, (float, int)):
        t_x = t_y = t_z = float(std_threshold)
    else:
        t_x, t_y, t_z = std_threshold
        
    # Shape: (3,)
    thresholds_vec = np.array([t_x, t_y, t_z])

    for cluster in clusters:
        # --- Logic: Skip "static" background clusters ---
        # Checks if > 50% of points are static (det[7] == 0)
        num_static_points = sum(1 for det in cluster if det[7] == 0)
        total_points = len(cluster)
        if num_static_points > 0.5 * total_points:
            refined_clusters.append(cluster)
            continue

        # --- Logic: Skip tiny clusters ---
        if len(cluster) < 3:
            refined_clusters.append(cluster)
            continue
            
        # Extract velocities: Shape [N, 3]
        velocities = np.array([det[14] for det in cluster])
        n_points, n_dims = velocities.shape
        
        # 2. Calculate Median
        medians = np.median(velocities, axis=0) # Shape [3]
        
        # 3. Calculate Double MAD Statistics
        # We need to compute a modified Z-score for every point.
        # Unlike standard MAD, we use a different divisor depending on if 
        # the point is to the left or right of the median.
        
        modified_z_scores = np.zeros_like(velocities)
        
        for dim in range(n_dims):
            # Isolate data for this axis (X, Y, or Z)
            data_dim = velocities[:, dim]
            med_dim = medians[dim]
            
            # Deviations
            devs = data_dim - med_dim
            
            # Identify Left vs Right
            left_mask = devs < 0
            right_mask = devs >= 0
            
            # Calculate MAD for Left tail
            if np.any(left_mask):
                mad_left = np.median(np.abs(devs[left_mask]))
            else:
                mad_left = 0.0
                
            # Calculate MAD for Right tail
            if np.any(right_mask):
                mad_right = np.median(np.abs(devs[right_mask]))
            else:
                mad_right = 0.0
            
            # Estimate Sigma (Robust Standard Deviation) for each side
            sigma_left = mad_left * MAD_SCALE_FACTOR
            sigma_right = mad_right * MAD_SCALE_FACTOR
            
            # Prevent division by zero if all points are identical on one side
            sigma_left = max(sigma_left, EPSILON)
            sigma_right = max(sigma_right, EPSILON)
            
            # Compute Modified Z-Scores
            # If point is on left, divide by sigma_left. If right, sigma_right.
            modified_z_scores[left_mask, dim] = np.abs(devs[left_mask]) / sigma_left
            modified_z_scores[right_mask, dim] = np.abs(devs[right_mask]) / sigma_right

        # 4. Determine Outliers
        # Check against the specific threshold for each axis
        # shape: [N, 3] boolean
        is_outlier_axis = modified_z_scores > thresholds_vec
        
        # Collapse to [N] boolean: True if outlier in X OR Y OR Z
        is_outlier_point = np.any(is_outlier_axis, axis=1)
        
        # 5. Filter the list
        current_refined_group = []
        
        for i, is_bad in enumerate(is_outlier_point):
            if is_bad:
                purge_noise.append(cluster[i])
            else:
                current_refined_group.append(cluster[i])
        
        if len(current_refined_group) > 0:
            refined_clusters.append(current_refined_group)
            
    return refined_clusters, purge_noise