#------------------------------------------------------------------------------
# This script receives spatial input data from the HoloLens, which comprises:
# 1) Head pose, 2) Eye ray, 3) Hand tracking, and prints it. 30 Hz sample rate.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import hl2ss
import hl2ss_lnm

import numpy as np

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.174.24"

#------------------------------------------------------------------------------

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable
def get_finger_joints(hand):
    joints = []
    for i in range(5):
        joints.append(hand.get_joint_pose(hl2ss.SI_HandJointKind(i)))
    return joints

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client.open()

def get_relative_poses(hand, finger_links):
    for link in finger_links:
        link_position = hand.get_joint_pose(link).position
        link_orientation = hand.get_joint_pose(link).orientation


def calculate_finger_angles(joint_positions):
    """
    Calculate angles between finger joints using position data.
    
    Parameters:
    joint_positions: dict with keys ['wrist', 'metacarpal', 'proximal', 'intermediate', 'distal', 'tip']
    Each value should be a numpy array or list of 3 coordinates [x, y, z]
    
    Returns:
    dict: Contains angles (in degrees) between each segment of the finger
    """
    def normalize_vector(vector):
        return vector / np.linalg.norm(vector)
    
    def angle_between_vectors(v1, v2):
        v1_normalized = normalize_vector(v1)
        v2_normalized = normalize_vector(v2)
        dot_product = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)
    
    # Convert input positions to numpy arrays if they aren't already
    positions = {k: np.array(v) for k, v in joint_positions.items()}
    
    # Calculate vectors between joints
    wrist_to_metacarpal = positions['metacarpal'] - positions['wrist']
    metacarpal_to_proximal = positions['proximal'] - positions['metacarpal']
    proximal_to_intermediate = positions['intermediate'] - positions['proximal']
    intermediate_to_distal = positions['distal'] - positions['intermediate']
    distal_to_tip = positions['tip'] - positions['distal']
    
    # Calculate angles between segments
    angles = {
        'wrist_metacarpal_angle': angle_between_vectors(wrist_to_metacarpal, metacarpal_to_proximal),
        'metacarpal_proximal_angle': angle_between_vectors(metacarpal_to_proximal, proximal_to_intermediate),
        'proximal_intermediate_angle': angle_between_vectors(proximal_to_intermediate, intermediate_to_distal),
        'intermediate_distal_angle': angle_between_vectors(intermediate_to_distal, distal_to_tip)
    }
    
    # Calculate total flexion (sum of all joint angles)
    angles['total_flexion'] = sum(angles.values())
    
    return angles

while (enable):
    data = client.get_next_packet()
    si = hl2ss.unpack_si(data.payload)

    print(f'Tracking status at time {data.timestamp}')

    if (si.is_valid_hand_right()):
        hand_right = si.get_hand_right()
        wrist_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist)

        index_finger_links = [
            hl2ss.SI_HandJointKind.IndexMetacarpal,
            hl2ss.SI_HandJointKind.IndexProximal,
            hl2ss.SI_HandJointKind.IndexIntermediate,
            hl2ss.SI_HandJointKind.IndexDistal,
            hl2ss.SI_HandJointKind.IndexTip,
            ]
        
        index_finger_positions = {
            'wrist':  hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist).position,
            'metacarpal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal).position,
            'proximal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal).position,
            'intermediate': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexIntermediate).position,   
            'distal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position,
            'tip': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
        }

        index_finger_orientations = {
            'wrist':  hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist).orientation,
            'metacarpal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal).orientation,
            'proximal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal).orientation,
            'intermediate': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexIntermediate).orientation,   
            'distal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).orientation,
            'tip': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).orientation
        }

        angles = calculate_finger_angles(index_finger_orientations)

        print("----Single frame joint angles----")
        for joint, angle in angles.items():
            print(f"{joint}: {angle:.2f} degrees")



        # index_m_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal)
        # index_p_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal)
        # index_i_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexIntermediate)
        # index_d_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal)
        # index_t_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip)
        # print(f'Right wrist pose: Position={wrist_pose.position} Orientation={wrist_pose.orientation} Radius={wrist_pose.radius} Accuracy={wrist_pose.accuracy}')
        # print(f'Right index metacarpal pose: Position={index_m_pose.position} Orientation={index_m_pose.orientation} Radius={index_m_pose.radius} Accuracy={index_m_pose.accuracy}')
        # print(f'Right index proximal pose: Position={index_p_pose.position} Orientation={index_p_pose.orientation} Radius={index_p_pose.radius} Accuracy={index_p_pose.accuracy}')
        # print(f'Right index intermediate pose: Position={index_i_pose.position} Orientation={index_i_pose.orientation} Radius={index_i_pose.radius} Accuracy={index_i_pose.accuracy}')
        # print(f'Right index distal pose: Position={index_d_pose.position} Orientation={index_d_pose.orientation} Radius={index_d_pose.radius} Accuracy={index_d_pose.accuracy}')
        # print(f'Right index tip pose: Position={index_t_pose.position} Orientation={index_t_pose.orientation} Radius={index_t_pose.radius} Accuracy={index_t_pose.accuracy}')
    else:
        print('No right hand data')

client.close()
listener.join()
