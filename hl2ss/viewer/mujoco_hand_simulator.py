
import multiprocessing
import time
import os
import numpy as np
import mujoco
from mujoco import viewer
import pyquaternion as pyq

from gripper_controller import GripperController

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def key_callback_data(key, data):
    """
    Callback for key presses but with data passed in
    :param key: Key pressed
    :param data:  MjData object
    :return: None
    """
    if key == 265:  # Up arrow
        data.mocap_pos[0, 2] += 0.01
    elif key == 264:  # Down arrow
        data.mocap_pos[0, 2] -= 0.01
    elif key == 263:  # Left arrow
        data.mocap_pos[0, 0] -= 0.01
    elif key == 262:  # Right arrow
        data.mocap_pos[0, 0] += 0.01
    elif key == 320:  # Numpad 0
        data.mocap_pos[0, 1] += 0.01
    elif key == 330:  # Numpad .
        data.mocap_pos[0, 1] -= 0.01
    elif key == 260:  # Insert
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [1, 0, 0], 10)
    elif key == 261:  # Home
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [1, 0, 0], -10)
    elif key == 268:  # Home
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 1, 0], 10)
    elif key == 269:  # End
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 1, 0], -10)
    elif key == 266:  # Page Up
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 0, 1], 10)
    elif key == 267:  # Page Down
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 0, 1], -10)
    else:
        print(key)

class GripperSimulator(GripperController):
    def __init__(self, command_queue,  sim_model="../model/faive_hand_p0/hand_IMU.xml"):
        self.command_queue = command_queue


        self.num_of_joints = 11
        self.joint_pos_command = np.zeros(self.num_of_joints)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), sim_model))
        self.data = mujoco.MjData(self.model)
        self.time_start = time.monotonic()
        self.time_elapsed = 0




    def simulation(self):
        def key_callback(key):
            key_callback_data(key, self.data)
        
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback = key_callback) as viewer: 
            self.time_start = time.monotonic()
            print("Simulation started at time: ", self.time_start)
            set_camera_view = False
            while viewer.is_running():  

                # Get joint angles command from the controller
                try:
                    joint_angles, wrist_position, wrist_orientation  = self.command_queue.get_nowait()  # Non-blocking call to get new data
                    
                    assert len(joint_angles) == self.num_of_joints
                    self.joint_pos_command = joint_angles
                    
                    self.time_elapsed = time.monotonic() - self.time_start

                    if not set_camera_view:
                        viewer.cam.lookat = wrist_position
                        set_camera_view = True
                
                    self.data.ctrl[-len(self.joint_pos_command):] = self.joint_pos_command
                    self.data.mocap_pos[0] = wrist_position
                    self.data.mocap_quat[0] = self.reorder_quaternion(wrist_orientation)

                    print(wrist_orientation)

                    #print("Joint angles received and applied to simulation:", self.joint_pos_array)
                except multiprocessing.queues.Empty:
                    pass

                # Step the simulation forward
                mujoco.mj_step(self.model, self.data) 
                with viewer.lock():           
                    viewer.sync()
    @staticmethod
    def reorder_quaternion(q):
        
        """
        Convert quaternion from XYZW to WXYZ order and correct the axes:
        - Negate X rotation (roll)
        - Swap Y and Z rotations
        Input q: [x, y, z, w]
        Output: [w, -x, z, y]  # WXZY with negated X
        """
        #return np.array([q[3], q[0], q[2], q[1]]) # only y inversed.
        # return np.array([q[3], -q[0], -q[1], q[2]])  #rotation around x (roll) will be in negative x direction. Rotation in y and z will swap

        return np.array([q[3], q[0], -q[2], q[1]])
   
        


    def controller(self, command_queue):
        
        # Joints_limit = self.model.jnt_range.copy() 

        # Controllable joint limits for p0 hand
        joint_limits = {'root2thumb_base': [0, 2.27],'thumb_base2pp':[-1.66,1.05],'thumb_pp2mp':[0, 0.96],
                        'root2index_pp':[0,0.83],'index_pp2mp':[0,0.96],
                        'root2middle_pp':[0,0.83],'middle_pp2mp':[0,0.96],
                        'root2ring_pp':[0,0.83],'ring_pp2mp':[0,0.96],
                        'root2pinky_pp':[0,0.83],'pinky_pp2mp':[0,0.96]}
        joint_names = list(joint_limits.keys())

        freq = 10
        print("Controller started with sinusoid command frequency: ", freq)
        i = 0
        while True:
            joint_angles = []
            for joint_name in joint_names:

                if i%200 > 100: # Generate a joint angle within the joint limit range using a sine wave
                    lower_limit, upper_limit = joint_limits[joint_name]
                    joint_angle = lower_limit + (upper_limit - lower_limit) * (np.sin(i * freq) + 1) / 2  
                else:
                    joint_angle = 0
                joint_angles.append(joint_angle)
           
            command_queue.put(np.array(joint_angles))
            i += 1



 




if __name__ == "__main__":
    command_queue = multiprocessing.Queue()


    #gripper_simulator = GripperSimulator(command_queue, data_queue, sim_model="../model/index_finger_prototype/index_finger_prototype_IMU.xml")
    gripper_simulator = GripperSimulator(command_queue, sim_model="../model/faive_hand_p0/hand_IMU.xml")

    
    simulation_process = multiprocessing.Process(target=gripper_simulator.simulation)
    controller_process = multiprocessing.Process(target=gripper_simulator.controller, args=(command_queue,))

    simulation_process.start()
    controller_process.start() 




    # Ensure processes are terminated when the application closes
    simulation_process.join()
    controller_process.join()
