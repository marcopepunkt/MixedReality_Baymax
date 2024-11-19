#------------------------------------------------------------------------------
# This script receives IMU samples from the HoloLens and prints them.
# Sensor details:
# Accelerometer:  93 samples per frame, sample rate ~1100 Hz effective ~12 Hz
# Gyroscope:     315 samples per frame, sample rate ~7500 Hz effective ~24 Hz
# Magnetometer:   11 samples per frame, sample rate   ~50 Hz effective  ~5 Hz
# The streams support three operating modes: 0) samples, 1) samples + rig pose,
# 2) query calibration (single transfer), except for the magnetometer stream
# which does not support mode 2.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import hl2ss
import hl2ss_lnm
import numpy as np
from scipy.spatial.transform import Rotation as R

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.174.24"

# Port
# Options:
# hl2ss.StreamPort.RM_IMU_ACCELEROMETER
# hl2ss.StreamPort.RM_IMU_GYROSCOPE
# hl2ss.StreamPort.RM_IMU_MAGNETOMETER
port = hl2ss.StreamPort.RM_IMU_ACCELEROMETER

# Operating mode
# 0: samples
# 1: samples + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

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
#------------------------------------------------------------------------------

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_rm_imu(host, port)
    print('Calibration data')
    print('Extrinsics')
    print(data.extrinsics)
    quit()

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_imu(host, port, mode=mode)
calibration_imu = hl2ss_lnm.download_calibration_rm_imu(host, port).extrinsics
client.open()


while (enable):
    data = client.get_next_packet()

    imu_data = hl2ss.unpack_rm_imu(data.payload)
    count = imu_data.get_count()
    sample = imu_data.get_frame(0)
    
    # print(f'Got {count} samples at time {data.timestamp}')
    # print(f'First sample: sensor_ticks={sample.vinyl_hup_ticks} soc_ticks={sample.soc_ticks} x={sample.x} y={sample.y} z={sample.z} temperature={sample.temperature}')
    # print(f'Pose')
    pose = calibration_imu* data.pose
    rectify_pose(pose)
    print(extract_floor_normal_from_pose(pose))

client.close()
listener.join()
