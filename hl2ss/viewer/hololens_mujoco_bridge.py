
# ------------------------------------------
# Connect to HoloLens and initialize MuJoCo simulator
# Track hand movements and convert joint angles
# Send normalized joint angles to MuJoCo simulator
# ------------------------------------------

import multiprocessing
import numpy as np
from pynput import keyboard
import hl2ss
import hl2ss_lnm

# Import existing functions
from get_right_hand_pose import (
    calculate_finger_angles,
    get_finger_joint_positions,
    normalize_vector
)
from mujoco_hand_simulator import GripperSimulator

class HoloLensHandBridge:
    def __init__(self, host="10.1.0.143"):
        self.host = host
        self.enable = True
        self.command_queue = multiprocessing.Queue()
        
        # Initialize MuJoCo simulator
        self.simulator = GripperSimulator(self.command_queue)
        
        # Define joint mapping from HoloLens to MuJoCo
        # Format: MuJoCo joint name: (finger name, joint type)
        self.joint_mapping = {
            'root2thumb_base': ('thumb', 'ADD'),
            'thumb_base2pp': ('thumb', 'MCP_flexion'),
            'thumb_pp2mp': ('thumb', 'PIP'),
            'root2index_pp': ('index', 'MCP_flexion'),
            'index_pp2mp': ('index', 'PIP'),
            'root2middle_pp': ('middle', 'MCP_flexion'),
            'middle_pp2mp': ('middle', 'PIP'),
            'root2ring_pp': ('ring', 'MCP_flexion'),
            'ring_pp2mp': ('ring', 'PIP'),
            'root2pinky_pp': ('little', 'MCP_flexion'),
            'pinky_pp2mp': ('little', 'PIP')
        }

    def on_press(self, key):
        self.enable = key != keyboard.Key.esc
        return self.enable

    def normalize_joint_angles(self, angles, joint_name):
        """
        Normalize joint angles to match MuJoCo joint limits
        """
        # Joint limits from MuJoCo model
        joint_limits = {
            'root2thumb_base': [0, 2.27],
            'thumb_base2pp': [-1.66, 1.05],
            'thumb_pp2mp': [0, 0.96],
            'root2index_pp': [0, 0.83],
            'index_pp2mp': [0, 0.96],
            'root2middle_pp': [0, 0.83],
            'middle_pp2mp': [0, 0.96],
            'root2ring_pp': [0, 0.83],
            'ring_pp2mp': [0, 0.96],
            'root2pinky_pp': [0, 0.83],
            'pinky_pp2mp': [0, 0.96]
        }
        
        if joint_name in joint_limits:
            lower, upper = joint_limits[joint_name]
            # Convert degree to radian
            angle_rad = np.radians(angles)
            # Clamp the angle to the joint limits
            return np.clip(angle_rad, lower, upper)
        return 0.0

    def run(self):
        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        # Start MuJoCo simulation in separate process
        sim_process = multiprocessing.Process(target=self.simulator.simulation)
        sim_process.start()

        # Connect to HoloLens
        client = hl2ss_lnm.rx_si(self.host, hl2ss.StreamPort.SPATIAL_INPUT)
        client.open()

        print("Starting HoloLens hand tracking to MuJoCo bridge. Press ESC to stop.")

        while self.enable:
            try:
                data = client.get_next_packet()
                si = hl2ss.unpack_si(data.payload)

                if si.is_valid_hand_right():
                    hand_right = si.get_hand_right()
                    if not hand_right:
                        hand_right = si.get_hand_left() # Use left hand if right hand is not detected

                    palm_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Palm).position
                    wrist_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist).position
                    wrist_orientation = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist).orientation
                    
                    # Calculate palm normal
                    middle_metacarpal = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleMetacarpal).position
                    palm_vector = middle_metacarpal - palm_position
                    side_vector = (hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleMetacarpal).position - 
                                 hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbMetacarpal).position)
                    palm_normal = normalize_vector(np.cross(palm_vector, side_vector))

                    # Initialize joint angles array for MuJoCo
                    mujoco_joint_angles = []
                    
                    # Process each MuJoCo joint
                    for mujoco_joint in self.joint_mapping.keys():
                        finger_name, joint_type = self.joint_mapping[mujoco_joint]
                        
                        # Get finger joint positions using imported function
                        joint_positions = get_finger_joint_positions(hand_right, finger_name)
                        
                        # Calculate finger angles using imported function
                        angles = calculate_finger_angles(joint_positions, palm_position, palm_normal)
                        
                        # Get the specific angle we need and normalize it
                        angle = angles[joint_type]
                        normalized_angle = self.normalize_joint_angles(angle, mujoco_joint)
                        mujoco_joint_angles.append(normalized_angle)

                    # Send joint angles to MuJoCo simulator
                    self.command_queue.put((np.array(mujoco_joint_angles), wrist_position, wrist_orientation))    
                    
                    print(f"Sent joint angles to MuJoCo at time {data.timestamp}")
                else:
                    print('No right hand data')

            except Exception as e:
                print(f"Error: {str(e)}")
                continue

        # Cleanup
        client.close()
        listener.join()
        sim_process.terminate()
        sim_process.join()

if __name__ == "__main__":
    bridge = HoloLensHandBridge()
    bridge.run()