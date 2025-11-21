import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple
from .types import DetectionTuple, Matrix4x4

class ClusterTracker:
    def __init__(self, 
                 dist_threshold=3.0, 
                 max_history=3, 
                 hit_threshold=3,
                 ghost_angle_thresh=80.0, 
                 ghost_vel_thresh=5.0):
        
        self.tracks = [] 
        self.dist_threshold = dist_threshold
        self.max_misses = max_history
        self.hit_threshold = hit_threshold
        self.ghost_angle_thresh = ghost_angle_thresh
        self.ghost_vel_thresh = ghost_vel_thresh
        self.next_id = 1

    def _is_ghost(self, track: Dict, dt: float) -> bool:
        # RELAXATION: Don't check physics until we have a stable trajectory.
        # 3 frames (0.15s) is too short; centroid jitter dominates the calculation.
        if track['hits'] < 5: 
            return False

        # 1. Trajectory Velocity
        # Now valid because 'start_pos' is updated to current frame every step
        displacement = track['pos'] - track['start_pos']
        total_time = track['age'] * dt 
        if total_time <= 0: return False
        
        v_traj = displacement / total_time
        v_solved = track['avg_vel'] 
        
        # 3. Consistency Checks
        mag_traj = np.linalg.norm(v_traj)
        mag_solved = np.linalg.norm(v_solved)
        
        if mag_traj > 0.5 and mag_solved > 0.5:
            cosine_sim = np.dot(v_traj, v_solved) / (mag_traj * mag_solved)
            angle_deg = np.degrees(np.arccos(np.clip(cosine_sim, -1.0, 1.0)))
        else:
            angle_deg = 0.0

        magnitude_diff = abs(mag_traj - mag_solved)

        if angle_deg > self.ghost_angle_thresh: return True
        if magnitude_diff > self.ghost_vel_thresh: return True

        return False

    def update(self, 
               clusters: List[List[DetectionTuple]], 
               dt: float,
               T_parent_to_current: Matrix4x4 # <--- NEW ARGUMENT (Ego Motion)
               ) -> List[List[DetectionTuple]]:
        
        # --- 0. Compensate for Ego-Motion ---
        # We must transform all existing tracks into the CURRENT sensor frame
        # before we can match them to new detections.
        
        # Rotation part of the matrix (3x3)
        R = T_parent_to_current[0:3, 0:3]
        
        for track in self.tracks:
            # 1. Transform Position (Rigid Body Transform)
            # Convert to homogeneous, multiply, convert back
            pos_h = np.append(track['pos'], 1.0)
            new_pos = (T_parent_to_current @ pos_h)[:3]
            track['pos'] = new_pos
            
            # 2. Transform Start Position (Crucial for _is_ghost logic)
            # We must drag the 'start' point along with the car so 
            # displacement calculations remain valid in the current frame.
            start_h = np.append(track['start_pos'], 1.0)
            track['start_pos'] = (T_parent_to_current @ start_h)[:3]

            # 3. Transform Velocity Vectors (Rotation Only)
            # Vectors don't translate, they only rotate.
            track['vel'] = R @ track['vel']
            track['avg_vel'] = R @ track['avg_vel']

        # --- 1. Extract Centroids ---
        current_centroids = []
        current_velocities = []
        
        for cluster in clusters:
            positions = np.array([det[4] for det in cluster])
            velocities = np.array([det[5] for det in cluster])
            current_centroids.append(np.mean(positions, axis=0))
            current_velocities.append(np.mean(velocities, axis=0))
        current_centroids = np.array(current_centroids)
        
        # --- 2. Predict (Physics) ---
        # Now that we are in the correct frame, apply object motion
        for track in self.tracks:
            track['pos'] += track['vel'] * dt

        # --- 3. Associate ---
        matched_indices = [] 
        track_indices = []   
        
        if len(self.tracks) > 0 and len(current_centroids) > 0:
            track_positions = np.array([t['pos'] for t in self.tracks])
            dists = cdist(track_positions, current_centroids)
            
            for _ in range(len(self.tracks)):
                if dists.min() > self.dist_threshold: break
                t_idx, c_idx = np.unravel_index(dists.argmin(), dists.shape)
                matched_indices.append(c_idx)
                track_indices.append(t_idx)
                
                track = self.tracks[t_idx]
                track['hits'] += 1
                track['misses'] = 0
                track['age'] += 1
                
                # Update State (EMA)
                alpha = 0.6 
                track['pos'] = (1-alpha)*track['pos'] + alpha*current_centroids[c_idx]
                track['vel'] = (1-alpha)*track['vel'] + alpha*current_velocities[c_idx]
                
                curr_vel = current_velocities[c_idx]
                track['avg_vel'] += (curr_vel - track['avg_vel']) / track['hits']
                
                dists[t_idx, :] = np.inf; dists[:, c_idx] = np.inf

        # --- 4. Handle Misses ---
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in track_indices: track['misses'] += 1
        
        # --- 5. Create New Tracks ---
        for c_idx, centroid in enumerate(current_centroids):
            if c_idx not in matched_indices:
                self.tracks.append({
                    'id': self.next_id, 'pos': centroid, 'vel': current_velocities[c_idx],
                    'hits': 1, 'misses': 0, 
                    'start_pos': centroid.copy(), 'age': 1, 
                    'avg_vel': current_velocities[c_idx].copy()
                })
                self.next_id += 1

        # --- 6. Filter ---
        valid_clusters = []
        for c_idx in matched_indices:
            match_pos = matched_indices.index(c_idx)
            t_idx = track_indices[match_pos]
            track = self.tracks[t_idx]
            
            if track['hits'] >= self.hit_threshold:
                if not self._is_ghost(track, dt):
                    valid_clusters.append(clusters[c_idx])

        # --- 7. Prune ---
        self.tracks = [t for t in self.tracks if t['misses'] < self.max_misses]
        return valid_clusters