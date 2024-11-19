#------------------------------------------------------------------------------
# Experimental simultaneous RM Depth AHAT and RM Depth Long Throw.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import time
import multiprocessing as mp
import cv2
import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

# Settings --------------------------------------------------------------------

# HoloLens address
host = '169.254.174.24'

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Buffer size in seconds
buffer_size = 5

# IMU
imu_mode = hl2ss.StreamMode.MODE_1
imu_port = hl2ss.StreamPort.RM_IMU_ACCELEROMETER


#------------------------------------------------------------------------------

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

def synchronize_streams(sink_lt, client, tolerance=1e5):
    """
    Synchronize the data streams from two sinks and the IMU client.

    Args:
    - sink_ht: High-throughput data sink.
    - sink_lt: Low-throughput data sink.
    - client: IMU data client.
    - calibration_imu: IMU calibration matrix.
    - tolerance: Maximum allowed time delta for synchronization (default: 1e3).

    Returns:
    - data_ht: Synchronized high-throughput frame.
    - data_lt: Synchronized low-throughput frame.
    - si_data: Synchronized IMU packet.
    """

    _, data_lt = sink_lt.get_most_recent_frame()

    si_data = client.get_next_packet()
    time_delta = data_lt.timestamp - si_data.timestamp

    # Synchronize streams
    while abs(time_delta) > tolerance:
        if time_delta > 0:  # IMU is behind, get the next IMU packet
            si_data = client.get_next_packet()
        else:  # Low-throughput data is behind, get the next frame
            _, data_lt = sink_lt.get_most_recent_frame()
        time_delta = data_lt.timestamp - si_data.timestamp

    print(f"Synchronized time delta: {time_delta}")
    return data_lt, si_data

# Initialize Kalman filters for roll, pitch, yaw
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])  # Initial state (angle, angle_rate)
    kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
    kf.H = np.array([[1., 0.]])  # Measurement function
    kf.P *= 1000.  # Initial uncertainty
    kf.R = 5  # Measurement noise
    kf.Q = np.array([[0.01, 0], [0, 0.1]])  # Process noise
    return kf

# Separate filters for roll, pitch, yaw
angle_filters = [initialize_kalman_filter() for _ in range(3)]

def apply_kalman_filter(kf, measurement):
    kf.predict()
    kf.update(measurement)
    return kf.x[0, 0]  # Return the filtered angle

def stabilize_angles(pose_matrix):
    """
    Stabilize the angles (roll, pitch, yaw) in a 4x4 pose matrix using Kalman filtering.
    
    Args:
    - pose_matrix: np.array of shape (4, 4) representing the 4x4 pose matrix.

    Returns:
    - stabilized_pose: np.array of shape (4, 4) with stabilized rotation.
    """
    # Extract and convert rotation matrix to Euler angles
    rotation_matrix = pose_matrix[:3, :3]
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    # Apply Kalman filter to each Euler angle
    filtered_angles = [
        apply_kalman_filter(angle_filters[0], roll),
        apply_kalman_filter(angle_filters[1], pitch),
        apply_kalman_filter(angle_filters[2], yaw)
    ]

    # Convert filtered angles back to a rotation matrix
    filtered_rotation_matrix = R.from_euler('xyz', filtered_angles, degrees=False).as_matrix()

    # Reconstruct the stabilized pose matrix with the filtered rotation and original position
    stabilized_pose = np.eye(4)
    stabilized_pose[:3, :3] = filtered_rotation_matrix
    stabilized_pose[:3, 3] = pose_matrix[:3, 3]  # Keep original position

    return stabilized_pose


#------------------------------------------------------------------------------
def transform_points(points, pose_matrix):
    """
    Transforms a set of points using a 4x4 pose matrix.

    Args:
    - points: np.array of shape (N, 3) representing the point cloud.
    - rot_matrix: np.array of shape (4, 4) representing the pose matrix.

    Returns:
    - transformed_points: np.array of shape (N, 3) representing the transformed points.
    """
    
    # Apply the transformation
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (pose_matrix @ homogeneous_points.T).T
    
    # Remove the homogeneous coordinate
    return transformed_points[:,:3]

def downsample_point_cloud(point_cloud, voxel_size):
    """
    Downsample the input point cloud using a voxel grid.
    
    Args:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud to downsample.
    - voxel_size (float): The size of the voxel grid in the same units as the point cloud.

    Returns:
    - open3d.geometry.PointCloud: The downsampled point cloud.
    """
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    return downsampled_point_cloud

def find_plane_ransac_o3d(point_cloud, pose, max_iterations=100, distance_threshold=0.1, min_inliers=250, angle_threshold=30):
    """
    RANSAC algorithm to find the floor plane in a point cloud.
    """
    # Convert Open3D point cloud to NumPy array
    original_points = np.asarray(point_cloud.points)
    points = transform_points(original_points, pose)
    head_height = pose[1,3]

    up_vector = np.array([0,1,0])

    # Extract the floor normal vector from the IMU pose matrix

    # Initialize variables
    best_inliers = set()
    num_points = points.shape[0]

    if num_points < min_inliers:
        return list()

    def calculate_inliers(params):
        """
        Calculate inliers for a given plane defined by three random points.
        """

        max_retries = 100  # Maximum retries to find a suitable plane
        retries = 0
        plane_height = 0

        while retries < max_retries:
            # Randomly sample 3 points
            if len(points) > 3:
                sample_indices = random.sample(range(num_points), 3)
            p1, p2, p3 = points[sample_indices]

            # Define the plane equation ax + by + cz + d = 0
            vec1, vec2 = p2 - p1, p3 - p1
            plane_normal = np.cross(vec1, vec2)

            # Check for degeneracy
            norm = np.linalg.norm(plane_normal)
            if norm == 0:
                retries += 1
                continue
            plane_normal = plane_normal / norm  # Normalize

            # Check angle with IMU normal
            angle = np.degrees(np.arccos(np.clip(np.dot(up_vector, plane_normal), -1.0, 1.0)))
            if angle <= angle_threshold:  # Plane is within desired angle
                break
            retries += 1
        else:
            return set(), 0  # Return empty set if no valid plane is found within retries

        # Plane equation coefficients
        a, b, c = plane_normal
        d = -(a * p1[0] + b * p1[1] + c * p1[2])

        # Calculate inliers
        plane_norm = np.sqrt(a**2 + b**2 + c**2)
        inliers = set(
            i for i, point in enumerate(points)
            if abs(a * point[0] + b * point[1] + c * point[2] + d) / plane_norm < distance_threshold
        )
        if len(inliers) > 0:
            plane_height = np.mean(points[np.array(list(inliers)), 2])

        return inliers, plane_height

    with ThreadPoolExecutor() as executor:
        # Submit tasks for each RANSAC iteration
        height = 0
        futures = [executor.submit(calculate_inliers, _) for _ in range(max_iterations)]
        for future in futures:
            inliers, plane_height = future.result()
            if len(inliers) > len(best_inliers) and len(inliers) > min_inliers:
                best_inliers = inliers
                height = plane_height

    if plane_height > head_height + 0.25:
        return list()
    
    return list(best_inliers)

def find_floor_with_pose_ransac_o3d(point_cloud, pose_matrix, max_iterations=50, distance_threshold=0.01, min_inliers=1000, angle_threshold=10):
    """
    RANSAC algorithm to find the floor plane in a point cloud using IMU pose matrix for guidance.
    
    Args:
    - point_cloud: open3d.geometry.PointCloud object representing the point cloud.
    - pose_matrix: np.array of shape (4, 4) representing the 4x4 IMU pose matrix.
    - max_iterations: Maximum iterations to run RANSAC for each plane.
    - distance_threshold: Distance threshold to consider a point as inlier.
    - min_inliers: Minimum number of inliers to consider a detected plane valid.
    - angle_threshold: Maximum angle (in degrees) between plane normal and IMU normal to consider it as floor.

    Returns:
    - floor_plane_inliers: List of indices of the inliers of the floor plane in the original point cloud.
    - floor_plane_height: The height (z-coordinate) of the floor plane.
    """
    # Convert Open3D point cloud to NumPy array
    original_points = np.asarray(point_cloud.points)
    transformed_points = transform_points(original_points, pose_matrix)

    # Initialize variables
    points_remaining = transformed_points.copy()
    original_indices = np.arange(len(transformed_points))  # Track original indices
    floor_plane_inliers = []
    floor_plane_height = float('inf')

    # Extract the floor normal vector from the IMU pose matrix
    imu_normal = extract_floor_normal_from_pose(pose_matrix)
    print("IMU Normal:", imu_normal)

    if imu_normal[2] < 0:
        print("IMU is confused")
        return None, None

    while len(points_remaining) > min_inliers:
        best_inliers = set()
        best_plane = None

        # Perform RANSAC to detect the largest plane
        for _ in range(max_iterations):
            # Randomly sample 3 points
            sample_indices = random.sample(range(len(points_remaining)), 3)
            p1, p2, p3 = points_remaining[sample_indices]

            # Define the plane equation ax + by + cz + d = 0
            vec1, vec2 = p2 - p1, p3 - p1
            plane_normal = np.cross(vec1, vec2)
            plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize

            # Calculate angle between IMU normal and plane normal
            angle = np.degrees(np.arccos(np.clip(np.dot(imu_normal, plane_normal), -1.0, 1.0)))

            # Check if the plane normal aligns with the IMU normal
            # if angle > angle_threshold:
            #     continue  # Skip this plane if it doesn't match the expected floor orientation

            # Plane equation constant
            a, b, c = plane_normal
            d = -(a * p1[0] + b * p1[1] + c * p1[2])

            # Measure distances of all points to the plane
            inliers = set()
            for i, point in enumerate(points_remaining):
                dist = abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)
                if dist < distance_threshold:
                    inliers.add(i)

            # Update best inliers if current set is larger
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (a, b, c, d)
                # angle = np.degrees(np.arccos(np.clip(np.dot(imu_normal, plane_normal), -1.0, 1.0)))
                # print(angle)


        # If no valid plane was found, break
        if len(best_inliers) < min_inliers:
            break

        # Calculate the average height (z-coordinate) of the inlier points for this plane
        plane_points = points_remaining[list(best_inliers)]
        plane_height = np.mean(plane_points[:, 2])

        # Check if this is the lowest plane so far
        if plane_height < floor_plane_height:
            floor_plane_height = plane_height
            # Map inliers back to the original indices
            floor_plane_inliers = [original_indices[i] for i in best_inliers]

        # Remove inliers from remaining points
        mask = np.ones(len(points_remaining), dtype=bool)
        mask[list(best_inliers)] = False
        points_remaining = points_remaining[mask]
        original_indices = original_indices[mask]

    return floor_plane_inliers, floor_plane_height
#------------------------------------------------------------------------------
def dbscan_clustering(point_cloud, colors, non_floor_mask, eps=0.1, min_samples=10):
    """
    Apply DBSCAN clustering to a point cloud.

    Args:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud.
    - colors (numpy.ndarray): Full color array for all points.
    - non_floor_mask (numpy.ndarray): Boolean mask for non-floor points.
    - eps (float): Maximum distance between two samples for them to be considered neighbors.
    - min_samples (int): Minimum number of points to form a dense cluster.

    Returns:
    - labels (numpy.ndarray): Array of cluster labels for each point (-1 indicates noise).
    - filtered_colors (numpy.ndarray): Updated colors for non-floor points after clustering.
    """
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)
    filtered_points = points[non_floor_mask]
    filtered_colors = colors[non_floor_mask]

    # Check for edge case: no non-floor points
    if len(filtered_points) == 0:
        print("No non-floor points found for clustering.")
        return np.array([]), filtered_colors

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
    labels = db.labels_

    # Assign random colors to clusters
    unique_labels = np.unique(labels)
    cluster_colors = np.random.rand(len(unique_labels), 3)  # Generate random colors for all unique labels
    cluster_colors[unique_labels == -1] = [0, 1, 0]  # Green for noise

    # Map labels to their respective colors
    label_to_color_map = cluster_colors[labels - labels.min()]
    filtered_colors = label_to_color_map

    return labels, filtered_colors

#------------------------------------------------------------------------------

def extract_floor_normal_from_pose(pose_matrix):
    """
    Extract the floor normal vector from a 4x4 pose matrix.
    
    Args:
    - pose_matrix: np.array of shape (4, 4) representing the 4x4 IMU pose matrix.

    Returns:
    - floor_normal: np.array of shape (3,) representing the transformed floor normal.
    """
    # Extract the 3x3 rotation part of the pose matrix
    rotation_matrix = pose_matrix[:3, :3]
    
    # Default floor normal (assuming floor is in +Z direction in IMU coordinate system)
    default_floor_normal = np.array([0, 0, 1])

    # Calculate the transformed floor normal in the point cloud's frame
    floor_normal = rotation_matrix @ default_floor_normal  # Matrix-vector multiplication

    # Normalize the resulting vector
    return floor_normal / np.linalg.norm(floor_normal)

def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    Assumes the rotation order is XYZ (roll-pitch-yaw).
    
    Args:
    - rotation_matrix: np.array of shape (3, 3) representing the rotation matrix.

    Returns:
    - euler_angles: tuple of (roll, pitch, yaw) in radians.
    """
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def rectify_pose(pose):
    quat = R.from_matrix(pose[:3,:3]).as_quat()
    if quat[3] < 0:
        quat = quat * -1
    pose[:3,:3] = R.from_quat(quat).as_matrix()

def rotate_point_cloud_to_imu_frame(point_cloud):
    """
    Rotates the point cloud to align with the IMU frame where z is height.
    
    Args:
    - point_cloud: open3d.geometry.PointCloud object representing the point cloud.

    Returns:
    - rotated_point_cloud: open3d.geometry.PointCloud object in the IMU frame.
    """
    # Define the 90-degree rotation around the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # Rotate the point cloud
    rotated_point_cloud = point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    
    return rotated_point_cloud

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    # Create Open3D visualizer ------------------------------------------------
    o3d_lt_intrinsics = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT, calibration_lt.intrinsics[0, 0], calibration_lt.intrinsics[1, 1], calibration_lt.intrinsics[2, 0], calibration_lt.intrinsics[2, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    first_pcd = True

    # Start streams -----------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, buffer_size * hl2ss.Parameters_RM_DEPTH_AHAT.FPS)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, buffer_size * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)

    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    # Without this delay, the depth streams might crash and require rebooting 
    # the HoloLens
    time.sleep(5) 
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)    

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)
    sink_lt = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, None)

    sink_ht.get_attach_response()
    sink_lt.get_attach_response()

    # # IMU
    # calibration_imu = hl2ss_lnm.download_calibration_rm_imu(host, imu_port).extrinsics
    # client = hl2ss_lnm.rx_rm_imu(host, imu_port, mode=imu_mode)
    # client.open()
    # Spatial Input
    client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
    client.open()

    # Main loop ---------------------------------------------------------------
    while (enable):
        
        data_lt, si_data = synchronize_streams(sink_lt,client)

        si = hl2ss.unpack_si(si_data.payload)
        if (si.is_valid_head_pose()):
            head_pose = si.get_head_pose()
            up = head_pose.up
            forward = np.array(head_pose.forward)
            right = np.cross(up, -forward)

            rotation = np.column_stack((right, up, -forward))
            pose = np.eye(4) 
            pose[:3, :3] = rotation  
            pose[:3, 3] = head_pose.position

            #print(f'Head pose: Position={head_pose.position} Forward={head_pose.forward} Up={head_pose.up}')
            # right = cross(up, -forward)
            # up => y, forward => -z, right => x
        else:
            print('No head pose data')
            continue


        # if (data_ht is not None):
        #     cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT) + '-depth', data_ht.payload.depth * 64) # Scaled for visibility
        #     cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT) + '-ab', data_ht.payload.ab)

        if (data_lt is not None):
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) + '-depth', data_lt.payload.depth * 8) # Scaled for visibility
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) + '-ab', data_lt.payload.ab)

            # Preprocess frames ---------------------------------------------------
            depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
            depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)

            # Assuming you have the depth image and intrinsics setup
            depth_image = o3d.geometry.Image(depth)
            tmp_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic=o3d_lt_intrinsics, depth_scale=1)
            ds_pcd = downsample_point_cloud(tmp_pcd,voxel_size=0.05)
            # Create a copy of the point cloud to modify colors
            highlighted_pcd = o3d.geometry.PointCloud(ds_pcd)

            # Initialize colors: Set all points to black
            colors = np.full((len(ds_pcd.points), 3), [0, 0, 0], dtype=float)  # Base color: black

            # Find floor inliers using RANSAC
            floor_inliers = find_plane_ransac_o3d(ds_pcd, pose)
            
            if len(floor_inliers) > 0:
                colors[floor_inliers] = [1, 0, 0]  # Highlight floor inliers in red

                # Create a mask for non-floor points
                non_floor_mask = np.ones(len(ds_pcd.points), dtype=bool)
                non_floor_mask[floor_inliers] = False

                # Apply DBSCAN clustering to non-floor points
                cluster_labels, filtered_colors = dbscan_clustering(ds_pcd, colors, non_floor_mask)
                if cluster_labels.size > 0:
                    colors[non_floor_mask,:] = filtered_colors  # Update non-floor colors with clustering results

            # Update the colors in the copied point cloud
            highlighted_pcd.colors = o3d.utility.Vector3dVector(colors)

            # Display point cloud --------------------------------------------------
            pcd.points = highlighted_pcd.points
            pcd.colors = highlighted_pcd.colors

            if first_pcd:
                vis.add_geometry(pcd)
                first_pcd = False
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            #cv2.waitKey(1)


    # Stop streams ------------------------------------------------------------
    sink_ht.detach()
    sink_lt.detach()

    producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
