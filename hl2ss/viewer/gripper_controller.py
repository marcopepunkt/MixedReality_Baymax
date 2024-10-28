from re import L
#from dynamixel_client import *
import numpy as np
import time
from copy import deepcopy
from click import getchar
import yaml
import os
# import cvxopt
# from finger_kinematics import *
import sympy as sym
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from threading import Thread, RLock, Event
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


class MuscleGroup:
    """
    An isolated muscle group comprised of joints and tendons, which do not affect the joints and tendons not included in the group.
    """

    attributes = [
        "joint_ids",
        "joint_names",
        "tendon_ids",
        "motor_ids",
        "motor_map",
        "spool_rad",
        "joint_ranges",
        "motor_init_pos",
        "motor_lookup_ids",
    ]

    def __init__(self, name, muscle_group_json: dict):
        self.name = name
        for attr_name in MuscleGroup.attributes:
            setattr(self, attr_name, muscle_group_json[attr_name])
        print(
            f"Created muscle group {name} with joint ids {self.joint_ids}, tendon ids {self.tendon_ids}, motor ids {self.motor_ids}, lookup ids {self.motor_lookup_ids} spool_rad {self.spool_rad}"
        )

class EKF:
    '''
    A simple extended Kálmán filter implementation for tracking separate MuscleGroups.
    Takes in the motor position and velocity measurements and outputs the joint
    position and velocity estimates.
    Notation (follows the Wikipedia page for EKF):
        - n_joints: number of joints in the muscle group
        - n_motors: number of motors in the muscle group
        - x: state vector, shape (2*n_joints,1) (composed of joint positions and velocities)
        - z: measurement vector, shape (2*n_motors,1) (composed of motor positions and velocities)
        - A: state transition matrix, shape (2*n_joints,2*n_joints)
        - P: state covariance matrix, shape (2*n_joints,2*n_joints)
        - y: measurement residual, shape (2*n_motors,1) (difference between the predicted and actual measurements)
        - h: predicted measurement, shape (2*n_motors,1)
        - H: measurement Jacobian, shape (2*n_motors,2*n_joints)
        - Q: process noise covariance matrix, shape (2*n_joints,2*n_joints)
        - R: measurement noise covariance matrix, shape (2*n_motors,2*n_motors)
        - S: residual covariance matrix, shape (2*n_motors,2*n_motors)
        - K: Kálmán gain, shape (2*n_joints,2*n_motors)
        
        The dynamics are assumed to be based on a simple constant velocity model.
        That is, the state x is propagated as:
        x = A * x
        where A is an identity matrix with the upper right quadrant replaced by
        the identity times the time step.
    '''
    def __init__(self, muscle_group : MuscleGroup, gc, init_p = 0.01):
        # initialize the matrices
        self.n_joints = len(muscle_group.joint_ids)
        self.n_motors = len(muscle_group.motor_ids)
        # note the muscle group name (this is needed as the thumb needs separate treatment)
        self.muscle_group_name = muscle_group.name
        self.joint_ranges = np.deg2rad(muscle_group.joint_ranges)
        # define just the matrices that should keep their value across updates
        self.x = np.zeros((2 * self.n_joints))
        self.A = np.eye(2 * self.n_joints)
        self.P = init_p * np.eye(2 * self.n_joints)
        self.Q = np.deg2rad(2) ** 2 * np.eye(2 * self.n_joints)  # 2 degrees  TODO: check if this should be in radians
        self.R = 0.005 ** 2 * np.eye(2 * self.n_motors)  # 5 mm

        # initialize the symbolic measurement Jacobian
        sym_joints = []
        for joint_name in muscle_group.joint_names:
            sym_joints.append(sym.Symbol(joint_name))

        # computing p and the Jacobian takes time, as symbolic computation is slow.
        # therefore, try to use cached values if possible
        p_pickle_file = f'/tmp/p_{muscle_group.name}.pkl'
        J_pickle_file = f'/tmp/J_{muscle_group.name}.pkl'
        compute_p_and_J = True
        if os.path.exists(p_pickle_file) and os.path.exists(J_pickle_file):
            # here we must check if the pickle files are up to date.
            # compare timestamps of the pickle files and the source code
            # (there may be better ways to check if the pickle files are still valid)
            sources_to_check = ["finger_kinematics.py", "gripper_controller.py"]
            # get the oldest timestamp from the pickle files
            pickle_file_oldest_timestamp = min(os.path.getmtime(p_pickle_file), os.path.getmtime(J_pickle_file))
            # get the newest timestamp from the source code
            this_dir = os.path.dirname(os.path.realpath(__file__))
            source_newest_timestamp = max([os.path.getmtime(os.path.join(this_dir, source)) for source in sources_to_check])
            if source_newest_timestamp < pickle_file_oldest_timestamp:
                # the pickle files are up to date
                compute_p_and_J = False
                print(f"Using cached p and J for muscle group {muscle_group.name}")
                with open(p_pickle_file, 'rb') as f:
                    p = pickle.load(f)
                with open(J_pickle_file, 'rb') as f:
                    J = pickle.load(f)
        if compute_p_and_J:
            # initialize the symbolic measurement Jacobian
            print(f"Computing p and J for muscle group {muscle_group.name}")
            p = gc.pose2motors_sym(*sym_joints, muscle_group = muscle_group)
            q = sym_joints 
            J = gc.calculate_Jacobian(p,q)
            # save the computed values to pickle files
            with open(p_pickle_file, 'wb') as f:
                pickle.dump(p, f)
            with open(J_pickle_file, 'wb') as f:
                pickle.dump(J, f)
        self.J_func = sym.lambdify(sym_joints,J)
        self.pose2motors = sym.lambdify(sym_joints,p)

        self.last_t = time.time()

    def update(self, motor_pos : np.ndarray, motor_speed : np.ndarray):
        '''
        Steps the EKF forward by:
            - predicting the state x and the measurement h
            - computing the residual y and updating the state x and
                its covariance matrix P with the Kálmán gain K times
                the residual
        '''
        dt = time.time() - self.last_t
        self.A[:self.n_joints, self.n_joints:] = dt * np.eye(self.n_joints)
        self.last_t = time.time()

        z = np.concatenate((motor_pos, motor_speed))

        # predict
        self.x[:] = self.A @ self.x
        self.P[:] = self.P + self.Q

        # update the joint-muscle model based on prediction
        motor_pos_pred = self.pose2motors(*self.x[:self.n_joints]).reshape(-1)
        J_pred = self.J_func(*self.x[:self.n_joints])
        h = np.concatenate((motor_pos_pred, J_pred @ self.x[self.n_joints:]))
        H = np.block([[J_pred, np.zeros((self.n_motors, self.n_joints))],
                      [np.zeros((self.n_motors, self.n_joints)), J_pred]])

        # update
        y = z - h
        S = H @ self.P @ H.T + self.R

        # running np.linalg.inv on S is slow, so leverage the fact that S is sparse for more efficient computation
        S_sparse = sparse.csc_matrix(S)
        # print(f"{self.muscle_group_name} S: {S}")
        K = self.P @ H.T @ splinalg.inv(S_sparse)

        self.x[:] = self.x + K @ y
        self.P[:] = (np.eye(2 * self.n_joints) - K @ H) @ self.P

        # EKF can predict a little bit outside the joint limits but not beyond this value, to avoid wacky predictions
        # set it to zero for now since base joint of thumb is unstable even at margin of 10 degrees
        margin = np.deg2rad(0)
        self.x[:self.n_joints] = np.clip(self.x[:self.n_joints], self.joint_ranges[:, 0] - margin, self.joint_ranges[:, 1] + margin)

        return self.x[:self.n_joints], self.x[self.n_joints:]

class GripperController:
    """
    class specialized for the VGripper
    wraps DynamixelClient to make it easier to access hand-related functions, letting the user think with "tendons" instead of "motors"
    eventually, the functionality for joint-level control will also be integrated to this class

    ## about tendon direction
    Signs for the tendon length is modified before sending to the robot so for the user, it is always [positive] = [actual tendon length increases]
    The direction of each tendon is set by the sign of the `spool_rad` variable in each muscle group
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        config_yml: str = "gripper_defs.yaml",
        init_motor_pos_update_thread: bool = True,
        use_sim_dynamixels: bool = False,
        use_sim_joint_measurement : bool = False,
        compliant_test_mode: bool = False,
        max_motor_current: float = 300.0,  # max current when using position control
    ):
        """
        config_yml: path to the config file, relative to this source file
        """
        baudrate = 3000000

        self.motor_lock = RLock()
        self.motor_status_lock = RLock()
        self.joint_value_lock = RLock()
        self.operating_mode = -1
        # if compliant test mode is selected, torques will be disabled
        self.compliant_test_mode = compliant_test_mode
        self.use_sim_dynamixels = use_sim_dynamixels
        self.use_sim_joint_measurement = use_sim_joint_measurement
        self.max_motor_current = max_motor_current
        self._load_musclegroup_yaml(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), config_yml)
        )

        # initialize dynamixel client
        if self.use_sim_dynamixels:
            self._dxc = DummyDynamixelClient(self.motor_ids, port, baudrate)
        else:
            self._dxc = DynamixelClient(self.motor_ids, port, baudrate)
        self.connect_to_dynamixels()

        self.init_motor_pos_update_thread = init_motor_pos_update_thread
        # initialize motor positions before calibration
        self.motor_pos_array = np.zeros(len(self.motor_ids))
        self.motor_vel_array = np.zeros(len(self.motor_ids))
        self.motor_cur_array = np.zeros(len(self.motor_ids))
        self.motor_pos_norm = None
        self.motor_id2init_pos = None
        self.num_of_joints = 0
        for muscle_group in self.muscle_groups:
            self.num_of_joints += len(muscle_group.joint_ids)
        self.joint_pos_array = np.zeros((self.num_of_joints,1))
        self.joint_pos_array_ekf = np.zeros(self.num_of_joints)
        self.joint_vel_array_ekf = np.zeros(self.num_of_joints)
        self.ekfs = []
        for muscle_group in self.muscle_groups:
            self.ekfs.append(EKF(muscle_group, self))

    def _load_musclegroup_yaml(self, filename):
        """
        load muscle group definitions from a yaml file
        Assumed to only run once, i.e. muscle groups are not changed during runtime
        """
        with open(filename, "r") as f:
            print(f"reading muscle group definitions from {filename} ...")
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.muscle_groups = []
        for muscle_group_name, muscle_group_data in data["muscle_groups"].items():
            self.muscle_groups.append(MuscleGroup(muscle_group_name, muscle_group_data))

        # define some useful variables to make it easier to access tendon information
        attrs_to_get = [
            "joint_ids",
            "motor_ids",
            "tendon_ids",
            "spool_rad",
            "motor_init_pos",
        ]
        for attr in attrs_to_get:
            setattr(self, attr, [])
            for muscle_group in self.muscle_groups:
                getattr(self, attr).extend(getattr(muscle_group, attr))
        for attr in attrs_to_get:
            setattr(self, attr, np.array(getattr(self, attr)))

        # run some sanity checks
        for muscle_group in self.muscle_groups:
            assert len(muscle_group.tendon_ids) == len(
                muscle_group.spool_rad
            ), "spool_rad must be defined for all tendons"
            assert len(muscle_group.motor_map) == len(
                muscle_group.tendon_ids
            ), "motor_map must be defined for all tendons"
        assert len(self.motor_ids) == len(
            set(self.motor_ids)
        ), "duplicate tendon ids should not exist"

    def _load_joint_lookup_tables(self) -> None:
        """
        Loads the lookup tables for all MuscleGroups and stores them in a dict.
        """
        # init joint range arrays
        self.joint_values = {}
        self.joint_step = {}
        self.joint_lookup = {}
        self.joint_lookup["thumb"] = {}
        lookup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookup")
        self.joint_lookup["thumb"]["tcmc_flex"] = np.load(
                        os.path.join(
                            lookup_dir, f"tcmc_flex_thumb.npy"))
        self.lookup_resolution = self.joint_lookup["thumb"]["tcmc_flex"].shape[0]
        print("The resolution on the precomputed tables is ", self.lookup_resolution)
        for muscle_group in self.muscle_groups:
            self.joint_values[muscle_group.name] = []
            self.joint_step[muscle_group.name] = []
            for joint_range in muscle_group.joint_ranges:
                current_range = np.linspace(joint_range[0], joint_range[1], self.lookup_resolution)
                self.joint_values[muscle_group.name].append(current_range)
                self.joint_step[muscle_group.name].append(
                    current_range[1] - current_range[0]
                )

        self.joint_lookup = {}
        self.joint_value_subtables = {}
        for muscle_group in tqdm(self.muscle_groups):
            # print("Loading lookup tables for muscle group: ", muscle_group.name, " ...")
            if muscle_group.name == "thumb":
                self.joint_lookup[muscle_group.name] = {}
                self.joint_value_subtables[muscle_group.name] = {}
                for idx, joint_type in enumerate(["tcmc_flex", "tcmc_abd", "tmcp", "tip"]):
                    self.joint_lookup[muscle_group.name][joint_type + "_non_monotonic_idx"] = 0
                    self.joint_lookup[muscle_group.name][joint_type + "_monotonic_flipped_table"] = {}
                    self.joint_lookup[muscle_group.name][joint_type] = np.load(
                        os.path.join(
                            lookup_dir, f"{joint_type}_{muscle_group.name}.npy"
                        )
                    )
                    if joint_type == "tcmc_flex" or joint_type == "tcmc_abd": 
                        if not np.all(np.diff(self.joint_lookup[muscle_group.name][joint_type]) > 0):
                            # print(f"WARNING: Lookup table for group {muscle_group.name} joint {joint_type} is not monotonically increasing. Trying to flip...")
                            self.joint_lookup[muscle_group.name][joint_type] = np.flip(self.joint_lookup[muscle_group.name][joint_type])
                            if not np.all(np.diff(self.joint_lookup[muscle_group.name][joint_type]) > 0):
                                # print("WARNING: Lookup table could not be flipped")
                                pass
                            else:
                                # print("Lookup table flipped")
                                self.joint_values[muscle_group.name][idx] = np.flip(self.joint_values[muscle_group.name][idx])
                        # else:
                        #     print("Lookup table for group ", muscle_group.name, " joint ", joint_type, " is monotonically increasing.")
                    elif joint_type == "tmcp":
                        self.joint_value_subtables[muscle_group.name][idx] = {}
                        self.joint_lookup[muscle_group.name]["tmcp_short_motor_table"] = {}
                        self.joint_lookup[muscle_group.name]["tmcp_short_joint_table"] = {}
                        self.joint_lookup[muscle_group.name]["tmcp_non_monotonic_idx"] = {}
                        # print("fitting tmcp")
                        for tcmc_idx in range(self.lookup_resolution):
                            # flip the whole table
                            current_motor_vals = deepcopy(self.joint_lookup[muscle_group.name][joint_type][tcmc_idx,:])
                            current_joint_vals = deepcopy(self.joint_values[muscle_group.name][idx])
                            if not np.all(np.diff(current_motor_vals) > 0):
                                current_motor_vals = np.flip(current_motor_vals)
                                current_joint_vals = np.flip(current_joint_vals)
                            # discard the first 10 values as they are non monotonous
                            current_motor_vals = current_motor_vals[10:]
                            current_joint_vals = current_joint_vals[10:]
                            # assign the flipped tables to the joints
                            self.joint_lookup[muscle_group.name]["tmcp_short_motor_table"][tcmc_idx] = current_motor_vals
                            self.joint_lookup[muscle_group.name]["tmcp_short_joint_table"][tcmc_idx] = current_joint_vals
            else:
                self.joint_lookup[muscle_group.name] = {}
                self.joint_value_subtables[muscle_group.name] = {}
                for idx, joint_type in enumerate(["pp", "mp"]):
                    self.joint_value_subtables[muscle_group.name][idx] = {}
                    self.joint_lookup[muscle_group.name][joint_type] = np.load(
                        os.path.join(
                            lookup_dir, f"{joint_type}_{muscle_group.name}.npy"
                        )
                    )
                    if joint_type == "pp": 
                        if not np.all(np.diff(self.joint_lookup[muscle_group.name][joint_type]) > 0):
                            self.joint_lookup[muscle_group.name][joint_type] = np.flip(self.joint_lookup[muscle_group.name][joint_type])
                            if not np.all(np.diff(self.joint_lookup[muscle_group.name][joint_type]) > 0):
                                pass
                            else:
                                self.joint_values[muscle_group.name][idx] = np.flip(self.joint_values[muscle_group.name][idx])

                    elif joint_type == "mp":
                        self.joint_lookup[muscle_group.name]["mp_monotonic_flipped_table"] = {}
                        self.joint_lookup[muscle_group.name]["mp_non_monotonic_motor_val"] = {}
                        self.joint_lookup[muscle_group.name]["mp_non_monotonic_idx"] = {}
                        for mp_idx in range(self.lookup_resolution):
                            self.joint_lookup[muscle_group.name]["mp_non_monotonic_motor_val"][mp_idx] = float('inf')
                            if not np.all(np.diff(self.joint_lookup[muscle_group.name][joint_type][mp_idx,:]) > 0):
                                self.joint_lookup[muscle_group.name][joint_type][mp_idx,:] = np.flip(self.joint_lookup[muscle_group.name][joint_type][mp_idx,:])
                                if not np.all(np.diff(self.joint_lookup[muscle_group.name][joint_type][mp_idx,:]) > 0):
                                    self.joint_lookup[muscle_group.name][joint_type][mp_idx,:] = np.flip(self.joint_lookup[muscle_group.name][joint_type][mp_idx,:])
                                    start_flip_idx = np.argmax(np.diff(self.joint_lookup[muscle_group.name][joint_type][mp_idx,:]) < 0)
                                    self.joint_lookup[muscle_group.name]["mp_non_monotonic_motor_val"][mp_idx] = self.joint_lookup[muscle_group.name][joint_type][mp_idx,start_flip_idx-1]
                                    sub_lookup_flipped = np.flip(self.joint_lookup[muscle_group.name][joint_type][mp_idx,start_flip_idx:].copy())
                                    self.joint_lookup[muscle_group.name]["mp_monotonic_flipped_table"][mp_idx] = sub_lookup_flipped
                                    sub_joint_vals_flipped = np.flip(self.joint_values[muscle_group.name][1][start_flip_idx:].copy())
                                    self.joint_value_subtables[muscle_group.name][idx][mp_idx] = sub_joint_vals_flipped
                                    # store the starting index of the flipped part
                                    self.joint_lookup[muscle_group.name]["mp_non_monotonic_idx"][mp_idx] = start_flip_idx
                                else:
                                    self.joint_values[muscle_group.name][idx] = np.flip(self.joint_values[muscle_group.name][idx])


    def tendon_pos2motor_pos_sym(self, tendon_lengths, muscle_group):
        """Input: desired tendon lengths
        Output: desired motor positions"""
        motor_pos = sym.matrices.zeros(1, len(muscle_group.motor_ids))
        muscle_groups = [muscle_group]

        for m_i, m_id in enumerate(muscle_group.motor_ids):
            m_id = muscle_group.motor_ids[m_i]
            t_i = muscle_group.motor_map.index(m_id)
            motor_pos[m_i] = (
                tendon_lengths[t_i] / muscle_group.spool_rad[t_i]
            )

        return motor_pos

    def motor_pos2tendon_pos(self, motor_pos):
        """Input: motor positions
        Output: tendon lengths"""
        tendon_lengths = np.zeros(len(self.tendon_ids))
        m_idx = 0
        t_idx = 0
        for muscle_group in self.muscle_groups:
            m_nr = len(muscle_group.motor_ids)
            t_nr = len(muscle_group.tendon_ids)
            for m_i in range(m_nr):
                m_id = muscle_group.motor_ids[m_i]
                t_i = np.where(np.array(muscle_group.motor_map) == m_id)[0]
                for i in t_i:
                    tendon_lengths[t_idx + i] = (
                        motor_pos[m_idx + m_i] * muscle_group.spool_rad[i]
                    )
            m_idx += m_nr
            t_idx += t_nr
        return tendon_lengths

    def write_desired_motor_pos(self, motor_positions_rad):
        """
        send position command to the motors
        unit is rad, angle of the motor connected to tendon
        """
        with self.motor_lock:
            self._dxc.write_desired_pos(self.motor_ids, motor_positions_rad)


    def write_desired_motor_current(self, motor_currents_mA):
        """
        send current command to the motors
        unit is mA (positive = pull the tendon)
        """
        m_nr = len(motor_currents_mA)
        m_idx = 0
        directions = np.zeros(m_nr)
        for muscle_group in self.muscle_groups:
            for m_id in muscle_group.motor_ids:
                idx = muscle_group.motor_map.index(m_id)
                directions[m_idx] = np.sign(muscle_group.spool_rad[idx])
                m_idx += 1
        with self.motor_lock:
            self._dxc.write_desired_current(
                self.motor_ids, -motor_currents_mA * directions
            )

    def connect_to_dynamixels(self):
        with self.motor_lock:
            self._dxc.connect()

    def disconnect_from_dynamixels(self):
        with self.motor_lock:
            self._dxc.disconnect()

    def set_operating_mode(self, mode):
        """
        see dynamixel_client.py for the meaning of the mode
        """
        with self.motor_lock:
            self._dxc.set_operating_mode(self.motor_ids, mode)

    def get_motor_pos(self):
        with self.motor_lock:
            return self._dxc.read_pos_vel_cur()[0]

    def get_motor_cur(self):
        with self.motor_lock:
            return self._dxc.read_pos_vel_cur()[2]

    def get_motor_vel(self):
        with self.motor_lock:
            return self._dxc.read_pos_vel_cur()[1]

    def wait_for_motion(self):
        reached_pos = False
        while not reached_pos:
            if all(self._dxc.read_status_is_done_moving()):
                reached_pos = True

    def enable_torque(self, motor_ids=None):
        if motor_ids is None:
            motor_ids = self.motor_ids
        with self.motor_lock:
            self._dxc.set_torque_enabled(motor_ids, True)

    def disable_torque(self, motor_ids=None):
        if motor_ids is None:
            motor_ids = self.motor_ids
        with self.motor_lock:
            self._dxc.set_torque_enabled(motor_ids, False)

    def init_tendons(self):
        """
        set the tendon lengths based on the current motor positions
        """
        self.initial_tendon_lengths = self.motor_pos2tendon_pos(self.get_motor_pos())

    def pose2motors(self, joint_angles):
        """Input: joint angles
        Output: motor positions"""
        motor_positions = np.zeros(len(self.motor_ids))
        motor_pos_begin_idx = 0
        joint_pos_begin_idx = 0
        for ekf in self.ekfs:
            # the EKF has lambdified functions (which can be computed really fast) of pose2motors for each muscle group
            # so use that for fast computation
            motor_pos_end_idx = motor_pos_begin_idx + ekf.n_motors
            joint_pos_end_idx = joint_pos_begin_idx + ekf.n_joints
            motor_positions[motor_pos_begin_idx:motor_pos_end_idx] = ekf.pose2motors(
                *joint_angles[joint_pos_begin_idx:joint_pos_end_idx]
            )
            motor_pos_begin_idx = motor_pos_end_idx
            joint_pos_begin_idx = joint_pos_end_idx
        return motor_positions

    def pose2motors_sym(self, joint1, joint2, joint3 = None, muscle_group = None):
        """Input: joint angles
        Output: motor positions
        """
        if joint3 is not None:
            # assume that it's the thumb
            tendon_lengths = pose2tendon_thumb(joint1, joint2, joint3)
        else:
            tendon_lengths = pose2tendon(joint1, joint2)
        return self.tendon_pos2motor_pos_sym(tendon_lengths, muscle_group=muscle_group)

    def init_joints(self, calibrate: bool = False):
        """
        Set the offsets based on the current (initial) motor positions
        :param calibrate: if True, perform calibration and set the offsets else move to the initial position
        """

        cal_yaml_fname = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cal.yaml"
        )
        cal_exists = os.path.isfile(cal_yaml_fname)

        if not calibrate and cal_exists:
            # Load the calibration file
            with open(cal_yaml_fname, "r") as cal_file:
                cal_data = yaml.load(cal_file, Loader=yaml.FullLoader)
            self.motor_id2init_pos = np.array(cal_data["motor_init_pos"])
        else:
            # Disable torque to allow the motors to move freely
            self.disable_torque()
            input("Move fingers to init posiiton and press Enter to continue...")
            time.sleep(1)  # give some time to hold fingers
            self.enable_torque()

            # Set to current control mode and pull on all tendons (while the user holds the fingers in the init position)
            self.set_operating_mode(0)
            self.write_desired_motor_current(60 * np.ones(len(self.motor_ids)))
            time.sleep(2)
            print("Reading in current motor positions...")
            self.motor_id2init_pos = self.get_motor_pos()
            print("Current motor positions are...", self.motor_id2init_pos)

            # Set the offsets based on the current motor positions
            print(f"Motor positions after calibration: {self.motor_id2init_pos}")

            # Rearrange the offsets to match the order of the motor_ids
            # self.motor_init_pos = self.motor_id2init_pos[self.motor_ids-1]
            # print(f"Initial motor positions (rearranged to match motor IDs): \n{self.motor_ids}\n{self.format_cal_array_for_print(self.motor_init_pos)}")

            # Save the offsets to a YAML file
            cal_data = {"motor_init_pos": self.motor_id2init_pos.tolist()}
            with open(cal_yaml_fname, "w") as cal_file:
                yaml.dump(cal_data, cal_file, default_flow_style=False)

        # read lookup tables from file
        self._load_joint_lookup_tables()
        if self.init_motor_pos_update_thread:
            # start pos update thread
            self.motor_pos_update_thread = Thread(target=self._update_motor_status_loop)
            self.motor_pos_update_thread.start()
        
        if self.compliant_test_mode:
            print("Setting compliant test mode! This disables position control.")
            self.set_operating_mode(0)
            self.write_desired_motor_current(30 * np.ones(len(self.motor_ids)))
        else:
            # start position control
            self.set_operating_mode(5)
            self.write_desired_motor_current(self.max_motor_current * np.ones(len(self.motor_ids)))
            self.write_desired_motor_pos(self.motor_id2init_pos)

        self.motor_pos_norm = self.pose2motors(np.zeros(len(self.joint_ids)))
        print("Motor positions of zero joint measurements: ", self.motor_pos_norm)
        print("Calib motor positions: ", self.motor_id2init_pos)

    def command_joint_angles(self, joint_angles: np.array):
        """
        Command joint angles
        :param: joint_angles: [joint 1 angle, joint 2 angle, ...]
        """
        motor_pos_des = self.pose2motors(np.deg2rad(joint_angles))
        # print("Desired motor positions: ", motor_pos_des)
        motor_pos_des = motor_pos_des - self.motor_pos_norm + self.motor_id2init_pos
        if self.use_sim_joint_measurement:
            with self.joint_value_lock:
                self.joint_pos_array = joint_angles.copy()
        self.write_desired_motor_pos(motor_pos_des)

    def _update_motor_status_loop(self):
        while True:
            # Rate: ~100 Hz
            time.sleep(0.005)
            self.update_motor_status()

    def update_motor_status(self):
        """
        Update the motor angles and joint angle estimates.
        """
        motor_pos_array, motor_vel_array, motor_cur_array = self.get_motor_pos_vel_cur()
        with self.motor_status_lock:
            if (
                self.motor_pos_array is None
                or self.motor_pos_norm is None
                or self.motor_id2init_pos is None
            ):
                return
            else:
                self.motor_pos_array = (
                    motor_pos_array.copy()
                    + self.motor_pos_norm
                    - self.motor_id2init_pos
                )
                if self.use_sim_joint_measurement:
                    # print("Using sim joint measurements!")
                    pass
                else:
                    # if we're not usingstore motor velocities and currents too
                    self.motor_vel_array = motor_vel_array
                    self.motor_cur_array = motor_cur_array
                    motor_pos_begin_idx = 0
                    joint_pos_begin_idx = 0
                    for ekf in self.ekfs:
                        # this assumes that the joints and motors go in order 0, 1, ... in the gripper_def
                        motor_pos_end_idx = motor_pos_begin_idx + ekf.n_motors
                        joint_pos_end_idx = joint_pos_begin_idx + ekf.n_joints
                        # update pos and vel
                        self.joint_pos_array_ekf[joint_pos_begin_idx:joint_pos_end_idx], \
                            self.joint_vel_array_ekf[joint_pos_begin_idx:joint_pos_end_idx] = \
                                ekf.update(self.motor_pos_array[motor_pos_begin_idx:motor_pos_end_idx],
                                           motor_vel_array[motor_pos_begin_idx:motor_pos_end_idx],
                                          )
                        motor_pos_begin_idx = motor_pos_end_idx
                        joint_pos_begin_idx = joint_pos_end_idx
                    joint_pos_array = self.motors_pos2joint_pos()
                    with self.joint_value_lock:
                        self.joint_pos_array = joint_pos_array.copy()


    def _get_motor_pos(self):
        with self.motor_lock:
            motor_pos = self._dxc.read_pos()
            return motor_pos

    def _get_motor_cur(self):
        with self.motor_lock:
            return self._dxc.read_pos_vel_cur()[2]

    def _get_motor_vel(self):
        with self.motor_lock:
            return self._dxc.read_pos_vel_cur()[1]

    def get_motor_pos_vel_cur(self):
        with self.motor_lock:
            pos, vel, cur = self._dxc.read_pos_vel_cur()
            return pos, vel, cur

    def motors_pos2joint_pos(self) -> np.ndarray:
        """
        Looks up the joint positions from the motor positions
        Args:
            self (GripperController): gripper controller object
        Returns:
            joint_states (numpy.ndarray): positions for the joints in
                degrees, as a numpy array of shape (16,1).
        """
        # apply calibration
        calibrated_motor_post = self.motor_pos_array
        # iterate through muscle groups
        joint_values = np.zeros((16,))
        for m_idx, muscle_group in enumerate(self.muscle_groups):
            if muscle_group.name == "thumb":
                theta_tcmc_flex = self._lookup_tcmc_flex(
                    muscle_group, calibrated_motor_post
                )
                theta_tcmc_abd = self._lookup_tcmc_abd(
                    muscle_group, theta_tcmc_flex, calibrated_motor_post
                )
                theta_tmcp = self._lookup_tmcp(
                    muscle_group, theta_tcmc_flex, theta_tcmc_abd, calibrated_motor_post
                )
                theta_ttip = 0.0  # self._lookup_ttip(muscle_group, theta_tcmc_flex, theta_tcmc_abd, theta_tmcp)
                joint_values[:4] = [
                    theta_tcmc_flex,
                    theta_tcmc_abd,
                    theta_tmcp,
                    theta_ttip,
                ]
            else:
                # iterate through mcp and pip joints and look up their values from the motors
                theta_mcp = self._lookup_mcp(muscle_group, calibrated_motor_post)
                theta_pip = self._lookup_pip(
                    muscle_group, theta_mcp, calibrated_motor_post
                )
                if muscle_group.name == "index":
                    theta_pip_rad = np.deg2rad(theta_pip)
                theta_tip = eqn_coupl(theta_pip_rad)
                joint_start = 4 + (m_idx - 1) * 3
                joint_values[joint_start : joint_start + 3] = [
                    theta_mcp,
                    theta_pip,
                    np.rad2deg(theta_tip),
                ]
        return joint_values.reshape(-1,)
         
    def _lookup_mcp(self, muscle_group, calibrated_motor_pos) -> float:
        """
        Looks up the mcp joint value in radians for a motor
        """ 
        assert muscle_group.name != "thumb", "Use the thumb-specific lookup function!"
        motor_table = self.joint_lookup[muscle_group.name]["pp"].flatten()
        joint_table = self.joint_values[muscle_group.name][0]
        motor_angle = calibrated_motor_pos[muscle_group.motor_lookup_ids[0]]
        mcp_angle = np.interp(motor_angle, motor_table, joint_table)

        return mcp_angle

    def _lookup_pip(self, muscle_group, theta_mcp, calibrated_motor_pos) -> float:
        """
        Looks up the mcp joint value in radians for a motor
        """
        assert muscle_group.name != "thumb", "Use the thumb-specific lookup function!"
        mcp_idx = min(int(round(
            (theta_mcp - muscle_group.joint_ranges[0][0])
            / self.joint_step[muscle_group.name][0]
            )
        ), self.lookup_resolution - 1)
        # see if we need to load a flipped subtable
        motor_value = calibrated_motor_pos[muscle_group.motor_lookup_ids[1]] 
        if  motor_value > self.joint_lookup[muscle_group.name]["mp_non_monotonic_motor_val"][mcp_idx]:
            motor_table = self.joint_lookup[muscle_group.name]["mp_monotonic_flipped_table"][mcp_idx]
            joint_table = self.joint_value_subtables[muscle_group.name][1][mcp_idx]

        else:
            motor_table = self.joint_lookup[muscle_group.name]["mp"][mcp_idx, :]
            joint_table = self.joint_values[muscle_group.name][1]
            if mcp_idx in self.joint_lookup[muscle_group.name]["mp_non_monotonic_idx"].keys():
                # clip the motor table to be monotonous
                clip_idx = self.joint_lookup[muscle_group.name]["mp_non_monotonic_idx"][mcp_idx]
                joint_table = joint_table[:clip_idx]
                motor_table = motor_table[:clip_idx]
            
        joint_value = np.interp(
            motor_value, motor_table, joint_table
        )
        return joint_value

    def _lookup_tcmc_flex(self, muscle_group, calibrated_motor_pos) -> float:
        """
        Looks up tcmc flexor joint angle
        """
        assert (
            muscle_group.name == "thumb"
        ), "Use the non thumb-specific lookup function!"
        motor_table = self.joint_lookup[muscle_group.name]["tcmc_flex"].flatten()
        joint_table = self.joint_values[muscle_group.name][0]
        theta_tcmc_flex = np.interp(
            calibrated_motor_pos[0], motor_table, joint_table
        )
        return theta_tcmc_flex

    def _lookup_tcmc_abd(
        self, muscle_group, theta_tcmc_flex, calibrated_motor_pos
    ) -> float:
        """
        Looks up the tcmc abductor joint angle
        """
        assert (
            muscle_group.name == "thumb"
        ), "Use the non thumb-specific lookup function!"
        motor_table = self.joint_lookup[muscle_group.name]["tcmc_abd"]
        joint_table = self.joint_values[muscle_group.name][1]
        motor_pos = calibrated_motor_pos[muscle_group.motor_lookup_ids[1]]
        return np.interp(
            motor_pos, motor_table, joint_table
        )

    def _lookup_tmcp(
        self, muscle_group, theta_tcmc_flex, theta_tcmc_abd, calibrated_motor_pos
    ) -> float:
        """
        Looks up the tmcp joint angle
        """
        assert (
            muscle_group.name == "thumb"
        ), "Use the non thumb-specific lookup function!"
        tcmc_abd_idx = min(int(round(
            (theta_tcmc_abd - muscle_group.joint_ranges[1][0])
            / self.joint_step[muscle_group.name][1]
            )
        ), self.lookup_resolution - 1)
        motor_value = - calibrated_motor_pos[muscle_group.motor_lookup_ids[2]]
        motor_table = self.joint_lookup[muscle_group.name]["tmcp_short_motor_table"][tcmc_abd_idx]
        joint_table = self.joint_lookup[muscle_group.name]["tmcp_short_joint_table"][tcmc_abd_idx]
        
        interpolated = np.interp(motor_value, motor_table, joint_table)
        
        return interpolated

    def _lookup_ttip(
        self,
        muscle_group,
        theta_tcmc_flex,
        theta_tcmc_abd,
        theta_tmcp,
        calibrated_motor_pos,
    ) -> float:
        """
        Looks up the ttip abductor joint angle
        """
        # TODO fix the ttip table
        assert (
            muscle_group.name == "thumb"
        ), "Use the non thumb-specific lookup function!"
        tcmc_flex_idx = int(
            (theta_tcmc_flex - muscle_group.joint_ranges[0][0])
            // self.joint_step[muscle_group.name][0]
        )
        tmcp_idx = int(
            (theta_tmcp - muscle_group.joint_ranges[2][0])
            // self.joint_step[muscle_group.name][2]
        )
        motor_table = self.joint_lookup[muscle_group.name]["tmcp"][
            tcmc_flex_idx, tmcp_idx, :
        ]
        joint_table = self.joint_values[muscle_group.name][1]
        return np.interp(
            calibrated_motor_pos[muscle_group.motor_ids[0]], motor_table, joint_table
        )


    def get_joint_pos(self, use_ekf=True):
        with self.joint_value_lock:
            if use_ekf:
                return self.joint_pos_array_ekf.copy()
            else:
                return self.joint_pos_array.copy()

    def get_joint_vel(self):
        with self.joint_value_lock:
            return self.joint_vel_array_ekf.copy()

    def calculate_Jacobian(self, p, q):
        """Input: array of expressions of motor positions, array of controllable joint angles
        Output: jacobian"""
        dim_p = len(p)
        dim_q = len(q)
        J = sym.matrices.zeros(dim_p, dim_q)
        for i in range(dim_p):
            for j in range(dim_q):
                J[i, j] = sym.diff(p[i], q[j])
        return sym.sympify(J)

    def pose2jacobian(self, theta_MCP, theta_PIP, J_func):
        """Input: controllable joint angles
        Output: functionalized jacobian for given joint angles"""
        return J_func(theta_MCP, theta_PIP)

    def compute_mot_torque(self, pose, tau, J_func):
        """Input:
        pose: 2D vector [theta_MCP, theta_PIP]
        tau: 2D vector of generalized forces (torques)
        solves QP to compute muscle force from generalized forces
        QP is formulated to
        - minimize squared sum of muscle tension
        - each muscle tension is negative
        - achieve generalized force
        Output: motor_torques"""
        num_motors = 3
        min_torque = 20  # mA. Minimum torque of motors. Use this to avoid slack wires.
        max_torque = 60
        # define matrices that are constant in the QP formulation
        P = cvxopt.matrix(np.identity(num_motors)) * 1.0
        p = cvxopt.matrix(np.zeros(num_motors)) * 1.0
        G = cvxopt.matrix(np.identity(num_motors)) * -1.0
        # TODO: Adjust G matrix for motor directions!
        h = -min_torque * cvxopt.matrix(np.ones(num_motors)) * 1.0
        cvxopt.solvers.options["show_progress"] = False  # suppress output
        assert len(pose) == 2
        assert len(tau) == 2
        Jacobian = self.pose2jacobian(pose[0], pose[1], J_func)
        # define matrices for the equality constraint
        A = cvxopt.matrix(np.transpose(Jacobian)) * 1.0
        b = cvxopt.matrix(tau) * 1.0
        # solve QP
        sol = cvxopt.solvers.qp(P, p, G, h, A, b)
        for i in range(len(sol["x"])):
            if sol["x"][i] > max_torque:
                sol["x"][i] = max_torque
        return sol["x"]

    # T = compute_tension([np.pi/6, np.pi/6],[2, 2])
    # print(T)


if __name__ == "__main__":
    gc = GripperController("/dev/ttyUSB0")

    gc.init_joints(calibrate=True)

    time.sleep(3.0)