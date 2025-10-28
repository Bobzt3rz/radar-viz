import numpy as np
from typing import List, Tuple

from .entity import Entity
from .world import World
from .cube import Cube
from .types import Vector3, Matrix3x3

class Radar(Entity):
    """ Simulates a radar sensor detecting cube corners. """
    def __init__(self,
                 position: Vector3,
                 velocity: Vector3,
                 rotation: Matrix3x3 = np.eye(3),
                 fov_azimuth_deg: float = 60.0,
                 fov_elevation_deg: float = 30.0,
                 max_range: float = 100.0):
        super().__init__(position, velocity, rotation)
        self.fov_azimuth_rad = np.radians(fov_azimuth_deg)
        self.fov_elevation_rad = np.radians(fov_elevation_deg)
        self.max_range = max_range
        # Radar local axes (Z forward, Y down, X right)
        self.axis_forward = np.array([0, 0, 1], dtype=np.float32)
        # Assuming Y-down convention for radar matching camera/world
        self.axis_up = np.array([0, -1, 0], dtype=np.float32)
        self.axis_right = np.array([1, 0, 0], dtype=np.float32)

    def generate_point_cloud(self, world: World) -> List[Tuple[Vector3, float, Cube, int]]:
        """
        Generates radar detections by checking cube corners.
        Returns: [(point_pos_radar_coords, speed_radial, source_entity, corner_index), ...]
        """
        # <<<<<<<<<<<< MODIFIED OUTPUT LIST TYPE >>>>>>>>>>>>>>
        point_cloud: List[Tuple[Vector3, float, Cube, int]] = []

        radar_origin_world = self.position
        M_world_to_radar = self.get_pose_world_to_local()
        R_world_to_radar = M_world_to_radar[0:3, 0:3]

        potential_targets = [e for e in world.entities if isinstance(e, Cube) and e is not self and np.sum((e.position - radar_origin_world)**2) < (self.max_range + e.size * 1.74)**2]

        seen_directions = [] # Stores (direction_vector, distance) for occlusion

        for entity in potential_targets:
            # Type hint for clarity
            if not isinstance(entity, Cube): continue # Should already be filtered, but safe check

            M_cube_local_to_world = entity.get_pose_local_to_world()
            local_vertices = entity.get_local_vertices()
            cube_vel_world = entity.velocity

            # <<<<<<<<<<<< USE ENUMERATE TO GET INDEX >>>>>>>>>>>>>>
            for corner_idx, local_vert in enumerate(local_vertices):
                # --- 1. Get Corner Position in World ---
                target_point_world = (M_cube_local_to_world @ np.append(local_vert, 1.0))[0:3]

                # --- 2. Vector from Radar to Corner (World Coords) & Range Check ---
                radar_to_target_world = target_point_world - radar_origin_world
                distance = np.linalg.norm(radar_to_target_world)
                if distance < 1e-6 or distance > self.max_range: continue

                # --- 3. Transform Vector to Radar Coords & Check FoV ---
                radar_to_target_radar = R_world_to_radar @ radar_to_target_world
                if abs(np.arctan2(radar_to_target_radar[0], radar_to_target_radar[2])) > self.fov_azimuth_rad / 2.0: continue
                if abs(np.arctan2(radar_to_target_radar[1], radar_to_target_radar[2])) > self.fov_elevation_rad / 2.0: continue

                # --- 4. Basic Occlusion Check ---
                current_direction = radar_to_target_world / distance
                is_occluded = False
                temp_seen_indices_to_remove = []
                for i in range(len(seen_directions)):
                    seen_dir, seen_dist = seen_directions[i]
                    if np.dot(current_direction, seen_dir) > 0.995: # Similar direction tolerance
                        if distance > seen_dist + 1e-4: is_occluded = True; break
                        elif distance < seen_dist - 1e-4: temp_seen_indices_to_remove.append(i)
                if is_occluded: continue
                if temp_seen_indices_to_remove:
                    for idx in sorted(temp_seen_indices_to_remove, reverse=True): del seen_directions[idx]

                # --- 5. Calculate Radial Velocity ---
                speed_radial = np.dot(cube_vel_world, current_direction)

                # --- 6. Position in Radar Coordinates ---
                point_pos_radar = (M_world_to_radar @ np.append(target_point_world, 1.0))[0:3]

                # --- 7. Add to Point Cloud & Track Occlusion ---
                # <<<<<<<<<<<< INCLUDE ENTITY AND INDEX IN OUTPUT TUPLE >>>>>>>>>>>>>>
                point_cloud.append((point_pos_radar, speed_radial, entity, corner_idx))
                seen_directions.append((current_direction, distance))

        return point_cloud