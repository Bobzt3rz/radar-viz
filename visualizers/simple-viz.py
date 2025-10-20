import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SimpleRadarVisualizer:
    def __init__(self, dataset_path, calib_path, scene_number):
        self.dataset_path = Path(dataset_path)
        self.calib_path = Path(calib_path)
        self.scene_path = self.dataset_path / f"Scene{scene_number}"
        # testing
        # self.radar_path = self.scene_path / "RadarCubes"
        self.radar_path = self.scene_path / "rosDS" / "radar_ososos"  
        self.image_path = self.scene_path / "rosDS" / "ueye_left_image_rect_color"
        self.lidar_path = self.scene_path / "rosDS" / "rslidar_points_clean"
        
        
        # Hardcoded calibration from the dataset
        self.azimuth_offset = 7  # degrees
        self.x_offset = 0  # cm
        self.y_offset = 0  # cm
        
    def get_sensor_transforms(self):
        calibration_file = self.calib_path
        with open(calibration_file, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)

        camera_projection_matrix, T_camera_lidar = intrinsic, extrinsic

        return camera_projection_matrix, T_camera_lidar

    
    # In class SimpleRadarVisualizer:
    def load_radar_points(self, timestamp_file):
        """Load pre-processed radar point cloud"""
        # These are already detected points from CFAR
        points = np.load(timestamp_file)
        
        # Return all data, including the 4th column (power/velocity)
        return points
    
    def find_matching_files(self, frame_num=10):
        """
        Finds the radar file for the given frame number and the image file 
        with the closest timestamp.
        """
        # 1. Get and sort all file paths
        radar_files = sorted(self.radar_path.glob("*.npy"))
        image_files = sorted(self.image_path.glob("*.jpg"))

        # 2. Basic checks to ensure there are files to process
        if not radar_files or not image_files:
            print("Warning: Radar or image folder is empty.")
            return None, None

        # Ensure the requested frame number is valid for the radar files
        if frame_num >= len(radar_files):
            print(f"Warning: frame_num {frame_num} is out of bounds. Using last available frame.")
            frame_num = len(radar_files) - 1

        # 3. Select the radar file and get its timestamp
        radar_file_to_match = radar_files[frame_num]
        # The timestamp is the filename without the extension
        radar_ts = float(radar_file_to_match.stem)

        # 4. Find the image file with the minimum time difference
        best_image_match = None
        min_diff = np.inf  # Start with an infinitely large difference

        for image_file in image_files:
            image_ts = float(image_file.stem)
            diff = abs(radar_ts - image_ts)
            
            # If this image's timestamp is closer, update our best match
            if diff < min_diff:
                min_diff = diff
                best_image_match = image_file

        print(f"   -> Match found with time difference: {min_diff:.4f} seconds")
        return radar_file_to_match, best_image_match
    
    def visualize_frame(self, frame_num=10):
        """Visualize radar overlay on image"""
        # Get matching files
        radar_file, image_file = self.find_matching_files(frame_num)
        
        print(f"Loading radar: {radar_file.name}")
        print(f"Loading image: {image_file.name}")
        
        # Load data
        all_radar_data = self.load_radar_points(radar_file)
        image = plt.imread(image_file)

         # --- NEW FILTERING STEP ---
        # Set a threshold for the intensity value (4th column, index 3)
        # You will need to experiment to find a good value for your data.
        intensity_threshold = 0.0  

        # Create a boolean mask of points that meet the threshold
        confident_mask = all_radar_data[:, 3] >= intensity_threshold
        
        # Apply the mask to get only the confident points.
        # Note: We only pass the x, y, z columns to the next functions.
        radar_points = all_radar_data[confident_mask, :3]

        # ADD negative height filter for "underground points"
        #radar_points = radar_points[radar_points[:, 2] > -1.0]
        #radar_points = radar_points[radar_points[:, 2] < 5.0]
        
        print(f"Original points: {len(all_radar_data)}, Confident points: {len(radar_points)}")
        # --- END OF NEW STEP ---

        pointcloud = transform_point_cloud(radar_points, [0, 0, -self.azimuth_offset],
                                                            [-self.x_offset / 100, -self.y_offset / 100,
                                                             0])
        
        camera_projection_matrix, T_camera_lidar = self.get_sensor_transforms()
        # Example usage
        #roll, pitch, yaw = -2.478, -83.131, 90.419  # Roll, pitch, yaw angles in degrees
        roll, pitch, yaw = 0, -85, 90  # Roll, pitch, yaw angles in degrees
        translation_vector = [0.195, 0.207, -0.482]  # T_camera_lidar[:3, 3]  # Translation vector

        # Create the transformation matrix
        transformation_matrix = create_transformation_matrix(roll, pitch, yaw, translation_vector)
        # transformation_matrix[:, 1] = first_column * -1

        uvs, point_depths, filtered_idx = project_pcl_to_image(pointcloud, transformation_matrix,
                                                               camera_projection_matrix, (1216, 1936))

        visualize_radar_dual_view(
            image, 
            pointcloud,
            uvs, 
            point_depths, 
            filtered_idx,
            point_size=20,
            bev_camera_fov_only=True
        )

def visualize_radar_dual_view(image, radar_points, uvs, point_depths, filtered_idx, 
                               point_size=20, bev_camera_fov_only=True):
    """
    Creates a dual visualization with BEV map and image projection side by side.
    Following RaDelft coordinate conventions:
    - X: longitudinal (forward/backward)
    - Y: lateral (left/right)
    - Z: height
    
    Args:
        image: The camera image (numpy array)
        radar_points: The transformed radar points (N, 3) or (N, 4)
        uvs: The (u, v) coordinates for ALL projected points
        point_depths: The depth values for ALL projected points
        filtered_idx: The indices of points within the image frame
        point_size: Size of plotted points on image
        bev_camera_fov_only: If True, only show radar points visible in camera FOV in BEV
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # === LEFT SUBPLOT: Image Projection ===
    # Filter visible points
    visible_points = uvs[filtered_idx]
    visible_depths = point_depths[filtered_idx]
    
    # Display image
    ax1.imshow(image)
    
    # Scatter plot with depth-based coloring
    scatter = ax1.scatter(
        visible_points[:, 0],  # u-coordinates (horizontal)
        visible_points[:, 1],  # v-coordinates (vertical)
        c=visible_depths,      # Color based on depth
        cmap='viridis',
        s=point_size,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8
    )
    
    # Add colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(scatter, cax=cax1)
    cbar1.set_label('Depth (m)', fontsize=12, weight='bold')
    
    ax1.set_title('Radar Points Projected on Camera Image', fontsize=14, weight='bold')
    ax1.axis('off')
    
    # === RIGHT SUBPLOT: BEV Map (Following RaDelft conventions) ===
    # Filter to only camera FOV points if requested
    if bev_camera_fov_only:
        bev_points = radar_points[filtered_idx]
        title_suffix = " (Camera FOV Only)"
    else:
        bev_points = radar_points
        title_suffix = ""
    
    # Plot BEV: -Y on x-axis (lateral), X on y-axis (longitudinal)
    # Colored by height (Z)
    scatter_bev = ax2.scatter(
        -bev_points[:, 1],  # -Y (lateral) on horizontal axis
        bev_points[:, 0],   # X (longitudinal) on vertical axis
        c=bev_points[:, 2], # Color by height (Z)
        cmap='viridis',
        s=5,
        alpha=0.8
    )
    
    # Set axis limits (matching RaDelft conventions)
    ax2.set_xlim(-25, 25)  # Lateral range
    ax2.set_ylim(0, 50)    # Longitudinal range (forward)
    
    # Add colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(scatter_bev, cax=cax2)
    cbar2.set_label('Height (m)', fontsize=12, weight='bold')
    
    # Add grid and labels
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('y [m]', fontsize=12, weight='bold')
    ax2.set_ylabel('x [m]', fontsize=12, weight='bold')
    ax2.set_title(f'Bird\'s Eye View (BEV){title_suffix}', fontsize=14, weight='bold')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
        

def transform_point_cloud(point_cloud, rotation_angles, translation):
    """
    Transform a 3D point cloud by rotating and translating it.

    :param point_cloud (numpy.ndarray): 3D point cloud as a NumPy array with shape (N, 3),
            where N is the number of points.
    :param rotation_angles (tuple or list): Angles for rotation around the X, Y, and Z axes
            in degrees. For example, (45, 30, 60) for 45 degrees around X, 30 degrees
            around Y, and 60 degrees around Z.
    :param translation (tuple or list): Translation values along the X, Y, and Z axes.
            For example, (1.0, 2.0, 3.0) for translation of (1.0, 2.0, 3.0).

    :return: numpy.ndarray: Transformed point cloud as a NumPy array.
    """
    # Extract rotation angles
    rx, ry, rz = map(np.radians, rotation_angles)

    # Create rotation matrices
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    # Apply rotations
    rotated_cloud = point_cloud[:, 0:3]
    rotated_cloud = rotated_cloud.dot(rot_x).dot(rot_y).dot(rot_z)

    # Apply translation
    translated_cloud = rotated_cloud + np.array(translation)
    if point_cloud.shape[1] > 3:
        translated_cloud = np.hstack((translated_cloud, np.expand_dims(point_cloud[:, 3], 1)))

    return translated_cloud

def create_transformation_matrix(roll, pitch, yaw, translation_vector):
    """
    Create a transformation matrix from roll, pitch, yaw angles (in degrees) and a translation vector.

    Args:
    roll (float): Roll angle in degrees.
    pitch (float): Pitch angle in degrees.
    yaw (float): Yaw angle in degrees.
    translation_vector (list): A list of three elements representing the translation vector.

    Returns:
    numpy.ndarray: A 4x4 transformation matrix.
    """
    # Convert angles from degrees to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Create individual rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # rotate around y axis again h degrees
    h = 1
    h_rad = np.radians(h)
    R_h = np.array([[np.cos(h_rad), 0, np.sin(h_rad)],
                    [0, 1, 0],
                    [-np.sin(h_rad), 0, np.cos(h_rad)]])

    R = np.dot(R_h, R)
    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    uvs = np.round(uvs).astype(int)

    return uvs


def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape):
    """
A helper function which projects a point clouds specific to the dataset to the camera image frame.
    :param point_cloud: Point cloud to be projected.
    :param t_camera_pcl: Transformation from the pcl frame to the camera frame.
    :param camera_projection_matrix: The 4x4 camera projection matrix.
    :param image_shape: Size of the camera image.
    :return: Projected points, and the depth of each point.
    """
    point_homo = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))

    radar_points_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=t_camera_pcl)

    point_depth = radar_points_camera_frame[:, 2]

    uvs = project_3d_to_2d(points=radar_points_camera_frame,
                           projection_matrix=camera_projection_matrix)

    filtered_idx = canvas_crop(points=uvs,
                               image_size=image_shape,
                               points_depth=point_depth)

    # uvs = uvs[filtered_idx]
    # point_depth = point_depth[filtered_idx]

    return uvs, point_depth, filtered_idx


def canvas_crop(points, image_size, points_depth=None):
    """
This function filters points that lie outside a given frame size.
    :param points: Input points to be filtered.
    :param image_size: Size of the frame.
    :param points_depth: Filters also depths smaller than 0.
    :return: Filtered points.
    """
    idx = points[:, 0] > 0
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    idx = np.logical_and(idx, points[:, 1] > 0)
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = np.logical_and(idx, points_depth > 0)

    return idx

# Usage
if __name__ == "__main__":
    viz = SimpleRadarVisualizer(
        dataset_path="/home/bobberman/programming/radar/radar-viz/data",
        calib_path="/home/bobberman/programming/radar/radar-viz/utils/calib.txt",
        scene_number=2
    )

    count = 0
    while(True):
        # Visualize frame 10
        viz.visualize_frame(count)
        count += 20