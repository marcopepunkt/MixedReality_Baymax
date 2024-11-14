#------------------------------------------------------------------------------
# Experimental simultaneous RM Depth AHAT and RM Depth Long Throw.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import time
import multiprocessing as mp
import cv2
import open3d as o3d
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

import numpy as np
import random

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
    - pose_matrix: np.array of shape (4, 4) representing the pose matrix.

    Returns:
    - transformed_points: np.array of shape (N, 3) representing the transformed points.
    """
    # Add a fourth homogeneous coordinate of 1 to each point
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply the transformation
    transformed_points = (pose_matrix @ homogeneous_points.T).T
    
    # Remove the homogeneous coordinate
    return transformed_points[:, :3]

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
            if angle > angle_threshold:
                continue  # Skip this plane if it doesn't match the expected floor orientation

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
                angle = np.degrees(np.arccos(np.clip(np.dot(imu_normal, plane_normal), -1.0, 1.0)))
                print(angle)


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

    # IMU
    calibration_imu = hl2ss_lnm.download_calibration_rm_imu(host, imu_port).extrinsics
    client = hl2ss_lnm.rx_rm_imu(host, imu_port, mode=imu_mode)
    client.open()

    # Main loop ---------------------------------------------------------------
    while (enable):
        _, data_ht = sink_ht.get_most_recent_frame()
        _, data_lt = sink_lt.get_most_recent_frame()

        imu_data = client.get_next_packet()
        pose = calibration_imu * imu_data.pose
        rectify_pose(pose)
        stabilized_pose = stabilize_angles(pose)

        if (data_ht is not None):
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT) + '-depth', data_ht.payload.depth * 64) # Scaled for visibility
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT) + '-ab', data_ht.payload.ab)

        if (data_lt is not None):
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) + '-depth', data_lt.payload.depth * 8) # Scaled for visibility
            cv2.imshow(hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) + '-ab', data_lt.payload.ab)

            # Preprocess frames ---------------------------------------------------
            depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
            depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)

            # Assuming you have the depth image and intrinsics setup
            depth_image = o3d.geometry.Image(depth)
            tmp_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic=o3d_lt_intrinsics, depth_scale=1)
            tmp_pcd = rotate_point_cloud_to_imu_frame(tmp_pcd)

            # Run RANSAC to find floor plane inliers
            floor_inliers, floor_height = find_floor_with_pose_ransac_o3d(tmp_pcd, stabilized_pose)

            # Create a copy of the point cloud to modify colors
            highlighted_pcd = o3d.geometry.PointCloud(tmp_pcd)

            # Assign colors: Set all points to a base color (e.g., white), then highlight inliers
            colors = np.full((len(tmp_pcd.points), 3), [0, 0, 0])  # Base color: white
            colors[floor_inliers] = [1, 0, 0]  # Highlight inliers in red

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
            cv2.waitKey(1)


    # Stop streams ------------------------------------------------------------
    sink_ht.detach()
    sink_lt.detach()

    producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
