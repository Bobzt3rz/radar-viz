import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Any
from .types import DetectionTuple, NoiseType

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

def filter_clusters_median(
    clusters: List[List[DetectionTuple]],
    purge_threshold: float = 5.0 # Max allowed deviation from median (m/s)
) -> Tuple[List[List[DetectionTuple]], List[DetectionTuple]]:
    """
    Step 2: Refines clusters by identifying and removing 'supersonic' outliers.
    Calculates the component-wise median velocity, but REJECTS points based 
    ONLY on their Transverse Velocity (Vx, Vy) deviation.
    """
    refined_clusters = []
    purge_noise = []
    
    for cluster in clusters:
        if not cluster:
            continue
            
        # Extract velocities: [N, 3] array of (vx, vy, vz)
        velocities = np.array([det[5] for det in cluster])
        
        # 1. Calculate Median of ALL axes
        # We still want the median Vz to know the true speed of the object,
        # even if we don't filter based on it.
        median_vel = np.median(velocities, axis=0)
        
        current_refined_group = []
        
        for i, det in enumerate(cluster):
            pt_vel = velocities[i]
            
            # --- THE FIX ---
            # Compare only Transverse parts (Indices 0 and 1)
            # We ignore index 2 (Vz) because ghosts often have CORRECT Vz.
            transverse_vel = pt_vel[:2]       # [vx, vy]
            transverse_median = median_vel[:2] # [med_vx, med_vy]
            
            # Calculate deviation in the 2D Transverse Plane only
            deviation = np.linalg.norm(transverse_vel - transverse_median)
            
            if deviation > purge_threshold:
                # Point has 'Supersonic' transverse error -> Ghost
                purge_noise.append(det)
            else:
                current_refined_group.append(det)
        
        if len(current_refined_group) > 0:
            refined_clusters.append(current_refined_group)
            
    return refined_clusters, purge_noise