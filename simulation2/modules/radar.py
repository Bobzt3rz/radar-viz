import numpy as np
import cv2
from typing import List, Tuple, Optional
import math
import os

from .entity import Entity
from .world import World
from .cube import Cube
from .types import Vector3, Matrix3x3

def ray_cube_intersection(ray_origin_world: Vector3,
                          ray_direction_world: Vector3,
                          cube: Cube) -> Optional[float]:
    """
    Checks for intersection between a ray and a cube using the Slab method.
    Transforms the ray into the cube's local coordinate system.

    Returns:
        The distance 't' along the ray to the intersection point,
        or None if no intersection occurs within the cube.
    """
    # Transform ray into cube's local coordinate system
    M_world_to_local = cube.get_pose_world_to_local()
    ray_origin_local_h = M_world_to_local @ np.append(ray_origin_world, 1.0)
    ray_origin_local = ray_origin_local_h[:3] / ray_origin_local_h[3] # Ensure proper perspective division if M includes perspective

    # Transform direction vector (only rotation part)
    R_world_to_local = M_world_to_local[0:3, 0:3]
    ray_direction_local = R_world_to_local @ ray_direction_world
    # Renormalize direction vector after rotation if scaling was involved (unlikely here)
    # ray_direction_local /= np.linalg.norm(ray_direction_local) # Usually not needed if R is pure rotation

    # Slab method for Axis-Aligned Bounding Box (AABB) intersection
    # Cube in local coords is centered at origin, extends -s to +s
    s = cube.size / 2.0
    min_bounds = np.array([-s, -s, -s], dtype=np.float32)
    max_bounds = np.array([ s,  s,  s], dtype=np.float32)

    t_min = 0.0 # Start of the ray segment
    t_max = np.inf # End of the ray segment (we check max_range later)

    for i in range(3): # Iterate over x, y, z axes
        if abs(ray_direction_local[i]) < 1e-6: # Ray parallel to slab i
            # If origin is outside the slab, no intersection
            if ray_origin_local[i] < min_bounds[i] or ray_origin_local[i] > max_bounds[i]:
                return None
        else:
            # Calculate intersection distances with slab planes
            t1 = (min_bounds[i] - ray_origin_local[i]) / ray_direction_local[i]
            t2 = (max_bounds[i] - ray_origin_local[i]) / ray_direction_local[i]

            # Ensure t1 is intersection with near plane, t2 with far plane
            if t1 > t2:
                t1, t2 = t2, t1

            # Update overall intersection interval [t_min, t_max]
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            # If interval becomes invalid, no intersection
            if t_min > t_max:
                return None

    # If t_min is positive and finite, there is an intersection at distance t_min
    if t_min >= 0 and t_min != np.inf:
         # Check if t_max is also valid (ray exits cube)
         if t_max >= t_min:
              return t_min # Return distance to entry point

    return None # No valid intersection

class Radar(Entity):
    """ Simulates a radar sensor detecting cube corners. """
    def __init__(self,
                 position: Vector3,
                 velocity: Vector3,
                 rotation: Matrix3x3 = np.eye(3),
                 fov_azimuth_deg: float = 60.0,
                 fov_elevation_deg: float = 30.0,
                 max_range: float = 40.0,
                 num_rays_azimuth: int = 64, # Number of horizontal rays
                 num_rays_elevation: int = 32, # Number of vertical rays,\
                 angular_velocity: Vector3 = np.zeros(3)
                 ):
        super().__init__(position, velocity, rotation, angular_velocity)
        self.fov_azimuth_rad = np.radians(fov_azimuth_deg)
        self.fov_elevation_rad = np.radians(fov_elevation_deg)
        self.max_range = max_range
        self.num_rays_azimuth = num_rays_azimuth
        self.num_rays_elevation = num_rays_elevation
        # Radar local axes (Z forward, Y down, X right)
        self.axis_forward = np.array([0, 0, 1], dtype=np.float32)
        # Assuming Y-down convention for radar matching camera/world
        self.axis_up = np.array([0, -1, 0], dtype=np.float32)
        self.axis_right = np.array([1, 0, 0], dtype=np.float32)

    def _generate_ray_directions_local(self) -> np.ndarray:
        """ Generates a grid of unit direction vectors in local radar coords. """
        azimuths = np.linspace(-self.fov_azimuth_rad / 2.0,
                               self.fov_azimuth_rad / 2.0,
                               self.num_rays_azimuth, dtype=np.float32)
        elevations = np.linspace(-self.fov_elevation_rad / 2.0,
                                 self.fov_elevation_rad / 2.0,
                                 self.num_rays_elevation, dtype=np.float32)

        directions = []
        # Create rays pointing generally along +Z (forward)
        for el in elevations:
            for az in azimuths:
                # Spherical to Cartesian conversion (assuming Z forward, Y down, X right)
                # Adjust if your local radar convention differs
                x = np.sin(az) * np.cos(el)
                y = np.sin(el) # Y is elevation angle from XZ plane
                z = np.cos(az) * np.cos(el) # Should be positive for forward
                
                # Check convention: Y = sin(el) assumes Y is up relative to horizon?
                # If Y is down: y = -sin(el)? Let's stick to standard Y-up spherical
                # then apply simulation's Y-down convention later if needed, OR adjust here.
                # Assuming standard math convention first: Z-fwd, Y-up, X-right for angles
                
                # Re-deriving for Z-fwd, Y-down, X-right:
                # Azimuth 'az' is angle from Z-axis in XZ plane (positive towards +X)
                # Elevation 'el' is angle from XZ plane (positive towards +Y, which is DOWN)
                x = np.sin(az) * np.cos(el)
                y = np.sin(el) # Positive elevation = positive Y = Down
                z = np.cos(az) * np.cos(el) # Forward
                
                direction = np.array([x, y, z], dtype=np.float32)
                # Normalize (should be close to 1 already)
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    directions.append(direction / norm)
                else: # Handle center case
                    directions.append(np.array([0,0,1], dtype=np.float32))


        return np.array(directions, dtype=np.float32) # Shape: (N_rays, 3)

    def generate_point_cloud(self, world: World) -> List[Tuple[Vector3, float, Optional[Entity], bool]]:
        """
        Generates radar detections using dense ray casting.
        Returns: [(point_pos_radar_coords, speed_radial, hit_entity, isNoise), ...]
        """
        point_cloud: List[Tuple[Vector3, float, Optional[Entity], bool]] = []

        radar_origin_world = self.position
        M_world_to_radar = self.get_pose_world_to_local()
        M_radar_to_world = self.get_pose_local_to_world()
        R_radar_to_world = M_radar_to_world[0:3, 0:3] # Rotation part only for directions

        local_ray_directions = self._generate_ray_directions_local()

        cubes_in_world = [e for e in world.entities if isinstance(e, Cube) and e is not self]

        for local_dir in local_ray_directions:
            # Transform ray direction to world coordinates
            ray_dir_world = R_radar_to_world @ local_dir

            closest_hit_distance = self.max_range
            closest_hit_entity = None

            # --- Test intersection with all cubes ---
            for cube in cubes_in_world:
                intersection_distance = ray_cube_intersection(radar_origin_world, ray_dir_world, cube)

                if intersection_distance is not None and intersection_distance < closest_hit_distance:
                     # Check distance again to be sure (intersection func returns local dist)
                     # Need world distance - calculate hit point
                     hit_point_world = radar_origin_world + ray_dir_world * intersection_distance
                     actual_distance = np.linalg.norm(hit_point_world - radar_origin_world)
                     
                     if actual_distance < closest_hit_distance: # Compare world distances
                         closest_hit_distance = actual_distance
                         closest_hit_entity = cube


            # --- If a hit was found within range ---
            if closest_hit_entity is not None:
                # Calculate the exact hit point in world coordinates
                hit_point_world = radar_origin_world + ray_dir_world * closest_hit_distance

                # Transform hit point to radar coordinates
                hit_point_world_h = np.append(hit_point_world, 1.0)
                hit_point_radar_h = M_world_to_radar @ hit_point_world_h
                hit_point_radar = hit_point_radar_h[:3]

                # Calculate radial velocity
                hit_entity_vel_world = closest_hit_entity.velocity
                los_unit_vec_world = ray_dir_world # Direction is already normalized unit vector
                speed_radial = np.dot(hit_entity_vel_world, los_unit_vec_world)

                point_cloud.append((hit_point_radar, speed_radial, closest_hit_entity, False))

        # --- 4. GENERATE CLUTTER (FALSE POSITIVES) ---
        # You can make this a random number, e.g., np.random.poisson(3)
        num_clutter_points = 200 
        
        azimuth_fov = self.fov_azimuth_rad
        elevation_fov = self.fov_elevation_rad
        
        for _ in range(num_clutter_points):
            # Generate a random point within the radar's FoV and range
            clutter_r = np.random.uniform(1.0, self.max_range) # 1.0 = min range
            clutter_az = np.random.uniform(-azimuth_fov / 2.0, azimuth_fov / 2.0)
            clutter_el = np.random.uniform(-elevation_fov / 2.0, elevation_fov / 2.0)
            
            clutter_point_radar = spherical_to_cartesian(clutter_r, clutter_az, clutter_el)
            
            # Give it a random radial velocity
            clutter_speed = np.random.uniform(-30.0, 30.0) # e.g., +/- 1 m/s
            
            point_cloud.append((clutter_point_radar, clutter_speed, None, True)) # isNoise = True

        return point_cloud
    
def visualize_radar_points(
    detections: List[Tuple[Vector3, float, Optional[Entity], bool]], # Assuming ray casting output
    radar_fov_az_rad: float,
    radar_fov_el_rad: float,
    output_width: int,
    output_height: int,
    color_map_range: Optional[Tuple[float, float]] = (-10.0, 10.0) # Speed range for color
) -> np.ndarray:
    """
    Projects radar points onto a 2D image plane based on radar FoV.

    Args:
        detections: List of tuples (point_radar_coords, speed_radial, entity).
        radar_fov_az_rad: Horizontal field of view in radians.
        radar_fov_el_rad: Vertical field of view in radians.
        output_width: Width of the output visualization image.
        output_height: Height of the output visualization image.
        color_map_range: Tuple (min_speed, max_speed) for color mapping radial velocity.
                         Set to None to draw all points in white.

    Returns:
        A NumPy array (HxWx3 BGR uint8) representing the visualization.
    """
    # Create a blank black image (BGR format for OpenCV)
    image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Calculate scaling factors based on FoV for perspective projection
    # tan(FoV/2) relates the half-dimension at distance z=1
    max_u_at_z1 = np.tan(radar_fov_az_rad / 2.0)
    max_v_at_z1 = np.tan(radar_fov_el_rad / 2.0)

    # Avoid division by zero if FoV is extremely small
    scale_x = (output_width / 2.0) / max_u_at_z1 if max_u_at_z1 > 1e-6 else output_width
    scale_y = (output_height / 2.0) / max_v_at_z1 if max_v_at_z1 > 1e-6 else output_height

    center_x = output_width / 2.0
    center_y = output_height / 2.0

    # Prepare colormap if needed
    cmap = cv2.COLORMAP_JET # Or COLORMAP_VIRIDIS, etc.
    min_speed, max_speed = 0, 1 # Default values
    use_color = color_map_range is not None
    if use_color and color_map_range:
        min_speed, max_speed = color_map_range
        speed_range = max_speed - min_speed
        if speed_range <= 0: speed_range = 1.0 # Avoid division by zero


    for point_rad, speed_rad, entity, isNoise in detections:
        x, y, z = point_rad

        # Ensure point is in front of the radar
        if z <= 1e-3: # Use a small epsilon
            continue

        # Perspective projection to normalized coords (relative to FoV at z=1)
        u_norm = x / z
        v_norm = y / z # Positive V means DOWN in radar space

        # Map normalized coords to pixel coords
        # Points at +/- max_u_at_z1 should map to 0 or width
        # Points at +/- max_v_at_z1 should map to 0 or height
        pixel_x = int(round(center_x + u_norm * scale_x))
        # Negate v_norm's contribution to map Y-down radar space to Y-up visualization effect
        pixel_y = int(round(center_y - v_norm * scale_y))


        # Check if point is within image bounds
        if 0 <= pixel_x < output_width and 0 <= pixel_y < output_height:
            # Determine color
            color = (255, 255, 255) # Default: White (BGR)
            if use_color:
                # Normalize speed to 0-1 range
                norm_speed = (speed_rad - min_speed) / speed_range # type: ignore
                norm_speed_clipped = np.clip(norm_speed, 0.0, 1.0)
                # Map to colormap (expects value 0-255)
                color_idx = int(norm_speed_clipped * 255)
                # Apply colormap (returns BGR)
                color = cv2.applyColorMap(np.array([[color_idx]], dtype=np.uint8), cmap)[0][0].tolist()

            # Draw a small circle for the point
            cv2.circle(image, (pixel_x, pixel_y), radius=2, color=color, thickness=-1) # Filled circle

    return image

def cartesian_to_spherical_radar(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Converts Cartesian radar coordinates (X-right, Y-down, Z-forward)
    to Spherical coordinates (range, azimuth, elevation).

    Args:
        x, y, z: Cartesian coordinates in radar frame.

    Returns:
        Tuple (range, azimuth_rad, elevation_rad).
        - range: Distance from origin.
        - azimuth: Angle from Z-axis in the XZ plane (positive towards +X). Radians.
        - elevation: Angle from the XZ plane (positive towards +Y, which is down). Radians.
    """
    range_dist = math.sqrt(x**2 + y**2 + z**2)
    if range_dist < 1e-6: # Avoid division by zero at origin
        return 0.0, 0.0, 0.0

    # Azimuth: angle in XZ plane from Z axis
    azimuth_rad = math.atan2(x, z)

    # Elevation: angle from the XZ plane
    # If z is near zero but x is not, atan2 handles it for azimuth.
    # For elevation, arcsin(y/range) is generally robust.
    elevation_rad = math.asin(y / range_dist) # asin domain is [-1, 1]

    return range_dist, azimuth_rad, elevation_rad

def spherical_to_cartesian(r: float, az: float, el: float) -> Vector3:
    """
    Converts Radar Spherical (range, azimuth, elevation) to 
    Radar Cartesian (X-right, Y-down, Z-forward).

    - az: Azimuth from Z-axis in XZ plane (positive to +X).
    - el: Elevation from XZ plane (positive to +Y, which is down).
    """
    # Calculate the components based on the correct definitions
    x = r * np.cos(el) * np.sin(az)
    y = r * np.sin(el)
    z = r * np.cos(el) * np.cos(az)
    
    return np.array([x, y, z])

def save_radar_point_cloud_ply(
    detections: List[Tuple[Vector3, float, Optional[Entity], bool]], # Ray casting output format
    file_path: str
) -> bool:
    """
    Saves radar detections to a PLY file with specified fields.

    Args:
        detections: List of tuples (point_radar_coords, speed_radial, entity).
        file_path: Full path for the output PLY file.

    Returns:
        True if successful, False otherwise.
    """
    if not isinstance(file_path, str) or not file_path:
        print("Error: Invalid file path for PLY.")
        return False
    if not detections:
        print(f"Warning: No radar detections to save to {file_path}.")
        # Optionally create an empty PLY file
        try:
            directory = os.path.dirname(file_path)
            if directory: os.makedirs(directory, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex 0\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property float azimuth\n")
                f.write("property float elevation\n")
                f.write("property float range\n")
                f.write("property float radial_velocity\n")
                f.write("end_header\n")
            return True
        except Exception as e:
            print(f"Error creating empty PLY file {file_path}: {e}")
            return False


    num_points = len(detections)

    # --- Prepare PLY Header ---
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float azimuth",    # In radians
        "property float elevation",   # In radians
        "property float range",
        "property float radial_velocity",
        "end_header"
    ]

    # --- Prepare Data Lines ---
    data_lines = []
    for point_rad, speed_rad, entity, isNoise in detections:
        x, y, z = point_rad
        range_val, az_rad, el_rad = cartesian_to_spherical_radar(x, y, z)
        # Format: x y z azimuth elevation range radial_velocity
        data_lines.append(f"{x:.6f} {y:.6f} {z:.6f} {az_rad:.6f} {el_rad:.6f} {range_val:.6f} {speed_rad:.6f}")

    # --- Write to File ---
    try:
        directory = os.path.dirname(file_path)
        if directory: os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write("\n".join(header) + "\n")
            f.write("\n".join(data_lines) + "\n")
        return True
    except Exception as e:
        print(f"Error writing PLY file {file_path}: {e}")
        return False