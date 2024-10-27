import numpy as np
from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_utilities

def normalize_vector(v):
    return v / (np.linalg.norm(v) + 1e-10)  # Added small epsilon to prevent division by zero

def calculate_finger_angles(joint_positions, palm_position, palm_normal_ref=None):
    """
    Calculate joint angles of a finger.
    
    Args:
        joint_positions: Dictionary containing 3D positions of finger joints
        palm_position: 3D position of the palm center
        palm_normal_ref: Reference palm normal vector for consistent sign convention
    
    Returns:
        Dictionary containing the calculated angles in degrees with anatomical sign convention:
        - DIP: positive for flexion, negative for extension
        - PIP: positive for flexion, negative for extension
        - MCP_flexion: positive for flexion, negative for extension
        - ADD: positive for abduction, negative for adduction
    """
    # Calculate vectors between joints
    metacarpal_vector = joint_positions['proximal'] - joint_positions['metacarpal']
    pp_vector = joint_positions['intermediate'] - joint_positions['proximal']
    ip_vector = joint_positions['distal'] - joint_positions['intermediate']
    dp_vector = joint_positions['tip'] - joint_positions['distal']
    
    def signed_angle_between_vectors(v1, v2, normal):
        """Calculate signed angle between vectors using the right-hand rule"""
        v1_norm = normalize_vector(v1)
        v2_norm = normalize_vector(v2)
        # Calculate unsigned angle
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        # Determine sign using cross product
        cross_product = np.cross(v1_norm, v2_norm)
        sign = np.sign(np.dot(cross_product, normal))
        return angle * sign
    
    def project_vector_to_plane(vector, plane_normal):
        plane_normal = normalize_vector(plane_normal)
        projection = vector - np.dot(vector, plane_normal) * plane_normal
        return projection

    # Calculate palm normal if not provided
    palm_to_metacarpal = joint_positions['metacarpal'] - palm_position
    if palm_normal_ref is None:
        palm_normal = normalize_vector(np.cross(metacarpal_vector, palm_to_metacarpal))
    else:
        palm_normal = palm_normal_ref

    # Calculate DIP angle (positive for flexion)
    dip_normal = normalize_vector(np.cross(dp_vector, ip_vector))
    dip_angle = signed_angle_between_vectors(dp_vector, ip_vector, dip_normal)
    
    # Calculate PIP angle (positive for flexion)
    pip_normal = normalize_vector(np.cross(ip_vector, pp_vector))
    pip_angle = signed_angle_between_vectors(ip_vector, pp_vector, pip_normal)
    
    # Calculate MCP flexion (positive for flexion)
    pp_projection_flexion = project_vector_to_plane(pp_vector, np.cross(palm_normal, metacarpal_vector))                                        
    mcp_normal = normalize_vector(np.cross(pp_projection_flexion, metacarpal_vector))
    mcp_flexion = signed_angle_between_vectors(pp_projection_flexion, metacarpal_vector, mcp_normal)
    
    # Calculate adduction angle (positive for abduction)
    pp_projection_abd = project_vector_to_plane(pp_vector, palm_normal)
    abd_normal = normalize_vector(np.cross(pp_projection_abd, metacarpal_vector))   
    adduction_angle = signed_angle_between_vectors(pp_projection_abd, metacarpal_vector, abd_normal)
    
    return {
        'DIP': dip_angle,
        'PIP': pip_angle,
        'MCP_flexion': mcp_flexion,
        'ADD': adduction_angle
    }

# Main script
host = "169.254.174.24"
enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def get_finger_joint_positions(hand, finger_name):
    """Get joint positions for a specific finger"""
    joint_kinds = {
        'thumb': [
            hl2ss.SI_HandJointKind.ThumbMetacarpal,
            hl2ss.SI_HandJointKind.ThumbProximal,
            hl2ss.SI_HandJointKind.ThumbDistal,
            hl2ss.SI_HandJointKind.ThumbTip
        ],
        'index': [
            hl2ss.SI_HandJointKind.IndexMetacarpal,
            hl2ss.SI_HandJointKind.IndexProximal,
            hl2ss.SI_HandJointKind.IndexIntermediate,
            hl2ss.SI_HandJointKind.IndexDistal,
            hl2ss.SI_HandJointKind.IndexTip
        ],
        'middle': [
            hl2ss.SI_HandJointKind.MiddleMetacarpal,
            hl2ss.SI_HandJointKind.MiddleProximal,
            hl2ss.SI_HandJointKind.MiddleIntermediate,
            hl2ss.SI_HandJointKind.MiddleDistal,
            hl2ss.SI_HandJointKind.MiddleTip
        ],
        'ring': [
            hl2ss.SI_HandJointKind.RingMetacarpal,
            hl2ss.SI_HandJointKind.RingProximal,
            hl2ss.SI_HandJointKind.RingIntermediate,
            hl2ss.SI_HandJointKind.RingDistal,
            hl2ss.SI_HandJointKind.RingTip
        ],
        'little': [
            hl2ss.SI_HandJointKind.LittleMetacarpal,
            hl2ss.SI_HandJointKind.LittleProximal,
            hl2ss.SI_HandJointKind.LittleIntermediate,
            hl2ss.SI_HandJointKind.LittleDistal,
            hl2ss.SI_HandJointKind.LittleTip
        ]
    }
    
    joint_positions = {}
    joints = joint_kinds[finger_name]
    
    if finger_name == 'thumb':
        # Special case for thumb since it has different structure
        joint_positions = {
            'metacarpal': hand.get_joint_pose(joints[0]).position,
            'proximal': hand.get_joint_pose(joints[1]).position,
            'intermediate': hand.get_joint_pose(joints[1]).position,  # Duplicate proximal for consistency
            'distal': hand.get_joint_pose(joints[2]).position,
            'tip': hand.get_joint_pose(joints[3]).position
        }
    else:
        joint_positions = {
            'metacarpal': hand.get_joint_pose(joints[0]).position,
            'proximal': hand.get_joint_pose(joints[1]).position,
            'intermediate': hand.get_joint_pose(joints[2]).position,
            'distal': hand.get_joint_pose(joints[3]).position,
            'tip': hand.get_joint_pose(joints[4]).position
        }
    
    return joint_positions

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client.open()

print("Starting hand tracking. Press ESC to stop.")
print("Angle conventions:")
print("- Flexion: Positive")
print("- Extension: Negative")
print("- Abduction: Positive")
print("- Adduction: Negative")

while (enable):
    data = client.get_next_packet()
    si = hl2ss.unpack_si(data.payload)

    if (si.is_valid_hand_right()):
        hand_right = si.get_hand_right()
        palm_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Palm).position
        
        # Calculate palm normal once for consistent reference
        wrist_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist).position
        middle_metacarpal = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleMetacarpal).position
        palm_vector = middle_metacarpal - wrist_position
        side_vector = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleMetacarpal).position - hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbMetacarpal).position
        palm_normal = normalize_vector(np.cross(palm_vector, side_vector))
        
        # Get angles for each finger
        fingers = ['thumb', 'index', 'middle', 'ring', 'little']
        print(f'\nHand joint angles at time {data.timestamp}:')
        
        for finger in fingers:
            joint_positions = get_finger_joint_positions(hand_right, finger)
            angles = calculate_finger_angles(joint_positions, palm_position, palm_normal)
            print(f'\n{finger.capitalize()} finger:')
            print(f'  DIP: {angles["DIP"]:6.2f}° {"(flexion)" if angles["DIP"] > 0 else "(extension)"}')
            print(f'  PIP: {angles["PIP"]:6.2f}° {"(flexion)" if angles["PIP"] > 0 else "(extension)"}')
            print(f'  MCP: {angles["MCP_flexion"]:6.2f}° {"(flexion)" if angles["MCP_flexion"] > 0 else "(extension)"}')
            print(f'  ADD: {angles["ADD"]:6.2f}° {"(abduction)" if angles["ADD"] > 0 else "(adduction)"}')
    else:
        print('No right hand data')

client.close()
listener.join()