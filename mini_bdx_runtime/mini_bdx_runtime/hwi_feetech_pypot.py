import time
from typing import List

import numpy as np

from pypot.feetech import FeetechSTS3215IO


class HWI:
    def __init__(self, usb_port="/dev/ttyACM0", baudrate=1_000_000):
        self.dxl_io = FeetechSTS3215IO(
            usb_port,
            baudrate=baudrate,
            use_sync_read=True,
        )
        self.joints = {
            "left_hip_yaw": 8,
            "left_hip_roll": 9,
            "left_hip_pitch": 6,
            "left_knee": 4,
            "left_ankle": 2,
            "neck_pitch": 1,
            "head_pitch": 12,
            "head_yaw": 13,
            "head_roll": 14,
            # "left_antenna": None,
            # "right_antenna": None,
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 7,
            "right_knee": 5,
            "right_ankle": 3,
        }

        self.zero_pos = {
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_hip_pitch": 0,
            "left_knee": 0,
            "left_ankle": 0,
            "neck_pitch": 0,
            "head_pitch": 0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna":0,
            # "right_antenna":0,
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_hip_pitch": 0,
            "right_knee": 0,
            "right_ankle": 0,
        }

        self.init_pos = {
            "left_hip_yaw": 0.002,
            "left_hip_roll": 0.053,
            "left_hip_pitch": -0.63,
            "left_knee": 1.368,
            "left_ankle": -0.784,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            # "left_antenna": 0.0
            # "right_antenna": 0.0
            "right_hip_yaw": -0.003,
            "right_hip_roll": -0.065,
            "right_hip_pitch": 0.635,
            "right_knee": 1.379,
            "right_ankle": -0.796,
        }

        self.joints_offsets = {
            "left_hip_yaw" : -0.107,
            "left_hip_roll" : 0.0010000000000000009,
            "left_hip_pitch" : -0.118,
            "left_knee" : 0.087,
            "left_ankle" : -0.0010000000000000009,
            "neck_pitch" : 0.567,
            "head_pitch" : 0.0,
            "head_yaw" : 0.154,
            "head_roll" : 0,
            "right_hip_yaw" : 0.041,
            "right_hip_roll" : 0.086,
            "right_hip_pitch" : -0.04100000000000001,
            "right_knee" : 0.11699999999999999,
            "right_ankle" : 0.0,
        }

        self.set_kds(32)
        self.set_kps(32)
        self.set_kis(0)
        self.get_pid_all()

    def set_pid(self, pid, joint_name):
        # TODO
        pass

    def set_kps(self, kp):
        self.dxl_io.set_P_coefficient({id: kp for id in self.joints.values()})
        
    def set_kds(self, kd):
        self.dxl_io.set_D_coefficient({id: kd for id in self.joints.values()})

    def set_kis(self, ki):
        self.dxl_io.set_I_coefficient({id: 0 for id in self.joints.values()})
    
    def get_pid_all(self):
        Ps = self.dxl_io.get_P_coefficient(self.joints.values())
        Is = self.dxl_io.get_I_coefficient(self.joints.values())
        Ds = self.dxl_io.get_D_coefficient(self.joints.values())
        print("Ps", Ps)
        print("Is", Is)
        print("Ds", Ds)

        return Ps, Is, Ds

    def turn_on(self):
        self.dxl_io.enable_torque(self.joints.values())
        self.set_position_all(self.init_pos)
        time.sleep(0.5)

    def turn_off(self):
        self.dxl_io.disable_torque(self.joints.values())

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: np.rad2deg(position + self.joints_offsets[joint])
            for joint, position in joints_positions.items()
        }

        self.dxl_io.set_goal_position(ids_positions)

    def set_position(self, joint_name, position):
        self.dxl_io.set_goal_position({self.joints[joint_name]: np.rad2deg(position)})

    def get_present_positions(self):
        """
        Returns the present positions in radians
        """
        present_positions = np.deg2rad(
            self.dxl_io.get_present_position(self.joints.values())
        )
        present_positions = [
            pos - self.joints_offsets[joint]
            for joint, pos in zip(self.joints.keys(), present_positions)
        ]
        return np.array(present_positions) 

    def get_present_velocities(self):
        """
        Returns the present velocities in rad/s
        """
        # rev/min
        present_velocities = np.array(
            self.dxl_io.get_present_speed(self.joints.values())
        )
        return np.deg2rad(present_velocities)
