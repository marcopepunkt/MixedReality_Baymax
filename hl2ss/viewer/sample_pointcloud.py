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
host = '169.254.236.128'

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Buffer size in seconds
buffer_size = 5

# IMU
imu_mode = hl2ss.StreamMode.MODE_1
imu_port = hl2ss.StreamPort.RM_IMU_ACCELEROMETER


#------------------------------------------------------------------------------

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

def synchronize_streams(sink_lt, sink_ht, spi_client, tolerance=1e5):
    """
    Synchronize the data streams from two sinks and the IMU client.

    Args:
    - sink_lt: Low-throughput data sink.
    - sink_ht: High-throughput data sink.
    - spi_client: IMU data client.
    - tolerance: Maximum allowed time delta for synchronization (default: 1e5).

    Returns:
    - data_lt: Synchronized low-throughput frame.
    - data_ht: Synchronized high-throughput frame.
    - spi_data: Synchronized IMU packet.
    """
    _, data_lt = sink_lt.get_most_recent_frame()
    _, data_ht = sink_ht.get_most_recent_frame()
    spi_data = spi_client.get_next_packet()

    while True:
        # Calculate time deltas
        time_delta_lt_spi = data_lt.timestamp - spi_data.timestamp
        time_delta_ht_spi = data_ht.timestamp - spi_data.timestamp
        time_delta_ht_lt = data_ht.timestamp - data_lt.timestamp

        # Check if all streams are synchronized
        if (
            abs(time_delta_lt_spi) <= tolerance and
            abs(time_delta_ht_spi) <= tolerance and
            abs(time_delta_ht_lt) <= tolerance
        ):
            break

        # Adjust streams based on time differences
        if time_delta_lt_spi > tolerance:  # IMU is behind low-throughput
            spi_data = spi_client.get_next_packet()
        elif time_delta_lt_spi < -tolerance:  # Low-throughput is behind
            _, data_lt = sink_lt.get_most_recent_frame()

        if time_delta_ht_spi > tolerance:  # IMU is behind high-throughput
            spi_data = spi_client.get_next_packet()
        elif time_delta_ht_spi < -tolerance:  # High-throughput is behind
            _, data_ht = sink_ht.get_most_recent_frame()

        # Synchronize high-throughput with low-throughput
        if time_delta_ht_lt > tolerance:  # High-throughput is ahead
            _, data_ht = sink_ht.get_most_recent_frame()
        elif time_delta_ht_lt < -tolerance:  # Low-throughput is ahead
            _, data_lt = sink_lt.get_most_recent_frame()

    return data_lt, data_ht, spi_data
#------------------------------------------------------------------------------
def merge_depth_images_with_focal_lengths(
    depth_lt, depth_ht, focal_length_lt, focal_length_ht, lt_resolution, ht_resolution, smooth_kernel_size=15
):
    """
    Merges two depth images (Long Throw and AHAT) where the center is from the LT sensor
    and the outer areas are from the HT sensor, using the focal lengths to define the radius.

    Args:
    - depth_lt: Long Throw depth image (2D numpy array).
    - depth_ht: AHAT depth image (2D numpy array).
    - focal_length_lt: Focal length of the LT sensor in pixels.
    - focal_length_ht: Focal length of the HT sensor in pixels.
    - lt_resolution: Tuple of (width, height) of the LT image.
    - ht_resolution: Tuple of (width, height) of the HT image.
    - smooth_kernel_size: Size of the smoothing kernel for blending (default: 15).

    Returns:
    - merged_depth: Combined depth image.
    - blended_depth: Smoothed and blended depth image.
    """
    # Extract resolutions
    lt_width, lt_height = lt_resolution
    ht_width, ht_height = ht_resolution

    # Calculate the radius for the LT region based on the LT focal length
    radius_lt = int(focal_length_lt * lt_width / (2 * focal_length_lt))  # Adjust scaling if needed
    radius_ht = int((radius_lt / lt_width) * ht_width)  # Scale to HT resolution

    # Create a mask for the LT priority region
    mask = np.zeros((ht_height, ht_width), dtype=np.uint8)
    center = (ht_width // 2, ht_height // 2)
    cv2.circle(mask, center, radius_ht, 1, thickness=-1)

    # Merge the depth images using the mask
    merged_depth = np.where(mask, depth_lt, depth_ht)

    # Smooth the boundaries for blending
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
    smooth_mask = cv2.dilate(mask, kernel, iterations=1).astype(np.float32)
    blended_depth = smooth_mask * depth_lt + (1 - smooth_mask) * depth_ht

    return merged_depth, blended_depth

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
    #original_points = np.asarray(point_cloud.points)
    #points = transform_points(original_points, pose)
    points = np.asarray(point_cloud.points)
    head_height = pose[1,2]

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
            plane_height = np.mean(points[np.array(list(inliers)), 1])

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

    if height > head_height + 0.25:
        print("Found plane but too high: ", height," with head height: ", head_height)
        return list()

    return list(best_inliers)

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

def fit_bounding_boxes_with_threshold_and_order(point_cloud, labels, non_floor_mask, min_points=50, reference_point=(0, 0)):
    """
    Fit bounding boxes around clusters in the point cloud, filter by minimum number of points,
    and order by proximity to a reference point in the XZ plane.

    Args:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud.
    - labels (numpy.ndarray): Cluster labels for the points (-1 indicates noise).
    - non_floor_mask (numpy.ndarray): Mask for non-floor points.
    - min_points (int): Minimum number of points required to create a bounding box.
    - reference_point (tuple): Reference point (x, z) for sorting by proximity.

    Returns:
    - bounding_boxes (list): List of bounding boxes for clusters meeting the threshold, ordered by proximity.
    - centers_radii (list): List of (center_x, center_z, radius) tuples for each bounding box, ordered by proximity.
    """
    points = np.asarray(point_cloud.points)
    non_floor_points = points[non_floor_mask]
    bounding_boxes = []
    centers_radii = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            # Skip noise
            continue

        # Extract points belonging to the current cluster
        cluster_points = non_floor_points[labels == label]

        # Skip clusters with fewer points than the threshold
        if len(cluster_points) < min_points:
            continue

        # Create an Open3D point cloud for the cluster
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

        # Compute the axis-aligned bounding box (AABB)
        aabb = cluster_pcd.get_axis_aligned_bounding_box()
        bounding_boxes.append(aabb)

        # Compute the center and radius for the XZ plane
        bbox_points = np.asarray(aabb.get_box_points())
        xz_points = bbox_points[:, [0, 2]]  # Project onto XZ plane
        center_xz = xz_points.mean(axis=0)
        min_xz = xz_points.min(axis=0)
        max_xz = xz_points.max(axis=0)
        radius = np.linalg.norm(max_xz - min_xz) / 2

        if center_xz[1] > 0 :
            continue
        
        # Convert to planning frame
        centers_radii.append((center_xz[0], - center_xz[1], radius))

    # Compute distances from the reference point
    distances = [
        np.linalg.norm([cx - reference_point[0], cz - reference_point[1]])
        for cx, cz, _ in centers_radii
    ]

    # Sort bounding boxes and centers_radii by distance
    sorted_indices = np.argsort(distances)
    bounding_boxes = [bounding_boxes[i] for i in sorted_indices]
    centers_radii = [centers_radii[i] for i in sorted_indices]

    return bounding_boxes, centers_radii

def get_xz_centers_and_radii(bounding_boxes):
    """
    Compute the center and radius of bounding boxes projected onto the XZ plane.

    Args:
    - bounding_boxes (list): List of Open3D bounding boxes (Axis-Aligned or Oriented).

    Returns:
    - centers_radii (list): List of (center_x, center_z, radius) tuples for each bounding box.
    """
    centers_radii = []

    for bbox in bounding_boxes:
        # Get the 8 corner points of the bounding box
        bbox_points = np.asarray(bbox.get_box_points())

        # Project points onto the XZ plane
        xz_points = bbox_points[:, [0, 2]]

        # Compute the center in the XZ plane
        center_xz = xz_points.mean(axis=0)

        # Compute the radius as half the diagonal of the XZ projection
        min_xz = xz_points.min(axis=0)
        max_xz = xz_points.max(axis=0)
        diagonal_length = np.linalg.norm(max_xz - min_xz)
        radius = diagonal_length / 2

        # Store the result
        centers_radii.append((center_xz[0], center_xz[1], radius))

    return centers_radii

import matplotlib.pyplot as plt

def visualize_circles_2d_realtime(centers_radii):
    """
    Real-time visualization of bounding box centers and radii in 2D space.

    Args:
    - centers_radii (list): List of (center_x, center_z, radius) tuples for each bounding box.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    circles = []  # To store circle objects

    # Configure plot
    ax.set_xlim(-10, 10)  # Adjust limits based on your data
    ax.set_ylim(-10, 10)  # Adjust limits based on your data
    ax.set_aspect('equal', adjustable='box')
    ax.grid()

    while True:
        # Clear existing circles
        for circle in circles:
            circle.remove()
        circles.clear()

        # Plot updated circles
        for cx, cz, r in centers_radii:
            circle = plt.Circle((cx, cz), r, color="blue", fill=False, linewidth=2)
            ax.add_patch(circle)
            circles.append(circle)
# -------------------------------------------------------------------------------------------

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


def keep_rotations_xz(rotation_matrix):
    """
    Extract and keep only rotations around the X- and Z-axes from a 3D rotation matrix.

    Args:
    - rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
    - numpy.ndarray: Modified 3x3 rotation matrix with only X- and Z-rotation components.
    """
    # Decompose the rotation matrix into Euler angles
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    
    # Check for gimbal lock
    singular = sy < 1e-6

    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Rotation around X
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Rotation around Z
        #print("X rotation: ", x_angle, "Z_rotation: ", z_angle)
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])  # Rotation around X
        z_angle = 0  # Rotation around Z is undefined in this case

    # Reconstruct rotation matrix with only X and Z rotations
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    Rz = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])

    # Combine the X and Z rotations
    R_xz = Rz @ Rx

    return R_xz

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Get Depth calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)
    calibration_ht = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_AHAT, calibration_path)

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    _ , scale_lt = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_ht.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    _ , scale_ht = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_ht.scale)

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

    spi_client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
    spi_client.open()

    navigation = False
    object_detect = False

    # Main loop ---------------------------------------------------------------
    while (enable):
        
        data_lt, data_ht, si_data = synchronize_streams(sink_lt,sink_ht,spi_client)

        si = hl2ss.unpack_si(si_data.payload)
        target_up = np.array([0,1,0])
        if (si.is_valid_head_pose()):

            head_pose = si.get_head_pose()
            up = head_pose.up
            forward = np.array(head_pose.forward)

            x_flip_rot = np.eye(3)
            x_flip_rot[1,1] = -1
            x_flip_rot[2,2] = -1

            right = np.cross(up, -forward)

            full_rotation = np.column_stack((right, up, -forward))
            roll_yaw_rot = keep_rotations_xz(full_rotation)

            #rotation =  rotation_x * x_flip_rot
            #rotation = np.eye(3) * x_flip_rot
            pose = np.eye(4) 
            pose[:3, :3] = np.matmul(full_rotation, x_flip_rot)
            pose[:3, :3] = roll_yaw_rot @ x_flip_rot

            #pose[:3, 3] = head_pose.position
            pose[:3, 3] = [0,0,0]

            #print(pose)
            #print(f'Head pose: Position={head_pose.position} Forward={head_pose.forward} Up={head_pose.up}')
            # right = cross(up, -forward)
            # up => y, forward => -z, right => x
        else:
            print('No head pose data')
            continue


        # if (data_ht is not None):
        #     cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT) + '-depth', data_ht.payload.depth * 64) # Scaled for visibility
        #     cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT) + '-ab', data_ht.payload.ab)

        if (data_lt is not None and data_ht is not None):
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) + '-depth', data_lt.payload.depth * 8) # Scaled for visibility
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) + '-ab', data_lt.payload.ab)

            # Preprocess frames ---------------------------------------------------
            depth_lt = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
            depth_lt = hl2ss_3dcv.rm_depth_normalize(depth_lt, scale_lt)
            depth_ht = hl2ss_3dcv.rm_depth_undistort(data_ht.payload.depth, calibration_ht.undistort_map)
            depth_ht = hl2ss_3dcv.rm_depth_normalize(depth_ht, scale_ht)

            depth = depth_lt

            _ , blended_depth = merge_depth_images_with_focal_lengths(depth_lt,depth_ht,calibration_lt.focal_length)

            # Here merge two depth data

            # Assuming you have the depth image and intrinsics setup
            depth_image = o3d.geometry.Image(depth)
            tmp_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic=o3d_lt_intrinsics, depth_scale=1)
            ds_pcd = downsample_point_cloud(tmp_pcd,voxel_size=0.05)
            # points = np.asarray(ds_pcd.points)
            # print(points[:10,:])
            ds_pcd.transform(pose)


            # points = np.asarray(ds_pcd.points)
            # print(points[:10,:])
            # Create a copy of the point cloud to modify colors
            highlighted_pcd = o3d.geometry.PointCloud(ds_pcd)

            # Initialize colors: Set all points to black
            colors = np.full((len(ds_pcd.points), 3), [0, 0, 0], dtype=float)  # Base color: black

            # Find floor inliers using RANSAC
            floor_inliers = find_plane_ransac_o3d(ds_pcd, pose)
            bounding_boxes = []
            
            if len(floor_inliers) > 0:

                navigation = True
                colors[floor_inliers] = [1, 0, 0]  # Highlight floor inliers in red

                # Create a mask for non-floor points
                non_floor_mask = np.ones(len(ds_pcd.points), dtype=bool)
                non_floor_mask[floor_inliers] = False

                # Apply DBSCAN clustering to non-floor points
                cluster_labels, filtered_colors = dbscan_clustering(ds_pcd, colors, non_floor_mask)
                if cluster_labels.size > 0:
                    colors[non_floor_mask,:] = filtered_colors  # Update non-floor colors with clustering results
                
                    bounding_boxes, centers_and_radii = fit_bounding_boxes_with_threshold_and_order(ds_pcd,cluster_labels, non_floor_mask, reference_point=(pose[0,3],pose[2,3]))

                    #centers_and_radii = get_xz_centers_and_radii(bounding_boxes)
                    #visualize_circles_2d_realtime(centers_and_radii)
                    if isinstance(centers_and_radii, list) and len(centers_and_radii) > 0:
                        print(centers_and_radii[0])
                    else:
                        print("centers_and_radii is either not a list or is empty.")

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
