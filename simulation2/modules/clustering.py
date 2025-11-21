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