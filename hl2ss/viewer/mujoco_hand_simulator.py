
import multiprocessing
import time
import os
import numpy as np
import mujoco
from mujoco import viewer

from gripper_controller import GripperController



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
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer: 
            self.time_start = time.monotonic()
            print("Simulation started at time: ", self.time_start)

            while viewer.is_running():  

                # Get joint angles command from the controller
                try:
                    joint_angles = self.command_queue.get_nowait()  # Non-blocking call to get new data
                    assert len(joint_angles) == self.num_of_joints
                    self.joint_pos_command = joint_angles
                    self.time_elapsed = time.monotonic() - self.time_start
                
                    self.data.ctrl[-len(self.joint_pos_command):] = self.joint_pos_command
                    #print("Joint angles received and applied to simulation:", self.joint_pos_array)
                except multiprocessing.queues.Empty:
                    pass

                # Step the simulation forward
                mujoco.mj_step(self.model, self.data) 
                with viewer.lock():           
                    viewer.sync()

        

   
        


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
