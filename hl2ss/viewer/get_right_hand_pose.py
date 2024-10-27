#------------------------------------------------------------------------------
# This script receives spatial input data from the HoloLens, which comprises:
# 1) Head pose, 2) Eye ray, 3) Hand tracking, and prints it. 30 Hz sample rate.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import hl2ss
import hl2ss_lnm
import hl2ss_utilities

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

def calculate_finger_angles(joint_positions, palm_position):
    """
    Calculate joint angles of the a non-thumb finger.
    
    Args:
        index_joint_positions: Dictionary containing 3D positions of index finger joints
        palm_position: 3D position of the palm center
    
    Returns:
        Dictionary containing the calculated angles in degrees:
        - DIP (Distal Interphalangeal)
        - PIP (Proximal Interphalangeal)
        - MCP (Metacarpophalangeal) flexion
        - ADD (Adduction/Abduction)
    """
    # Calculate vectors between joints
   
    index_palm = index_joint_positions['proximal'] - index_joint_positions['metacarpal']  # Metacarpal to proximal
    index_pp = index_joint_positions['intermediate'] - index_joint_positions['proximal']  # Proximal to intermediate
    index_ip = index_joint_positions['distal'] - index_joint_positions['intermediate']  # Intermediate to distal
    index_dp = index_joint_positions['tip'] - index_joint_positions['distal']  # Distal to tip
    
    def normalize_vector(v):
        """Normalize a vector"""
        return v / np.linalg.norm(v)
    
    def angle_between_vectors(v1, v2):
        """Calculate angle between two vectors in degrees"""
        v1_norm = normalize_vector(v1)
        v2_norm = normalize_vector(v2)
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))
    
    def project_vector_to_plane(vector, plane_normal):
        """Project a vector onto a plane defined by its normal"""
        plane_normal = normalize_vector(plane_normal)
        projection = vector - np.dot(vector, plane_normal) * plane_normal
        return projection
    
    # Calculate DIP (angle between dp and ip)
    dip_angle = angle_between_vectors(index_dp, index_ip)
    # Calculate PIP (angle between ip and pp)
    pip_angle = angle_between_vectors(index_ip, index_pp)
    
    # Calculate palm plane normal (assuming palm plane is defined by metacarpal, carpal and palm center)
    palm_normal = np.cross(normalize_vector(index_palm), normalize_vector(index_joint_positions['metacarpal'] - palm_position))
    palm_normal = normalize_vector(palm_normal)
    
    # Calculate MCP flexion (angle between pp and palm plane)
    # Project index_pp onto the plane perpendicular to the palm plane normal
    pp_projection_flexion = project_vector_to_plane(index_pp, np.cross(palm_normal, index_palm))
    mcp_flexion = angle_between_vectors(pp_projection_flexion, index_palm)
    
    # Calculate adduction angle (projection onto palm plane)
    # Project index_pp onto the palm plane
    pp_projection_add = project_vector_to_plane(index_pp, palm_normal)
    adduction_angle = angle_between_vectors(pp_projection_add, index_palm)

    mcp_flexion = angle_between_vectors(index_pp, index_palm)
    
    return {
        'DIP': dip_angle,
        'PIP': pip_angle,
        'MCP_flexion': mcp_flexion,
        'ADD': adduction_angle
    }

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client.open()



while (enable):
    data = client.get_next_packet()
    si = hl2ss.unpack_si(data.payload)

    print(f'Tracking status at time {data.timestamp}')

    if (si.is_valid_hand_right()):
        hand_right = si.get_hand_right()
        
        palm_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Palm).position
        
        index_joint_positions = {
           
            'metacarpal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal).position,
            'proximal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal).position,
            'intermediate': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexIntermediate).position,   
            'distal': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position,
            'tip': hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
        }

        index_angles = calculate_finger_angles(index_joint_positions, palm_position)

        print(f'Index finger angles:', index_angles)





  



 
        
    else:
        print('No right hand data')

client.close()
listener.join()
