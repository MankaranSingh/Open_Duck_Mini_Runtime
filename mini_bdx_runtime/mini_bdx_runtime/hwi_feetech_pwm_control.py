import time
from typing import List

import numpy as np

# from pypot.feetech import FeetechSTS3215IO
from mini_bdx_runtime.feetech_pwm_control import FeetechPWMControl


class HWI:
    def __init__(self, usb_port="/dev/ttyACM0"):

        # Order matters here
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
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.0,
            "left_knee": 0.0,
            "left_ankle": 0.0,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            # "left_antenna": 0.0
            # "right_antenna": 0.0
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.0,
            "right_knee": 0.0,
            "right_ankle": 0.0,
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

        init_pos_with_offsets = {
            joint: pos + self.joints_offsets[joint]
            for joint, pos in self.init_pos.items()
        }

        self.control = FeetechPWMControl(
            ids=list(self.joints.values()),
            init_pos_rad=list(init_pos_with_offsets.values()),
            usb_port=usb_port,
        )

        self.kps = np.ones(len(self.joints)) * 32  # default kp
        self.low_torque_kps = np.ones(len(self.joints)) * 8  # default kp

    def set_kps(self, kps):
        self.kps = kps
        self.control.set_kps(self.kps)

    def turn_on(self):
        self.control.set_kps(self.low_torque_kps)
        print("turn on : low KPS set")
        time.sleep(1)

        self.set_position_all(self.init_pos)
        print("turn on : init pos set")

        time.sleep(1)

        self.control.set_kps(self.kps)
        print("turn on : high kps")

    def turn_off(self):
        self.control.disable_torque()

    def freeze(self):
        self.control.freeze()

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: np.rad2deg(position + self.joints_offsets[joint])
            for joint, position in joints_positions.items()
        }

        self.control.goal_positions = list(ids_positions.values())

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """

        # present_positions = np.deg2rad(
        #     self.control.io.get_present_position(self.joints.values())
        # )

        present_positions = np.deg2rad(self.control.present_positions)
        present_positions = [
            pos - self.joints_offsets[joint]
            for joint, pos in zip(self.joints.keys(), present_positions)
            if joint not in ignore
        ]
        return np.array(np.around(present_positions, 3))

    def get_present_velocities(self, rad_s=True, ignore=[]):
        """
        Returns the present velocities in rad/s (default) or rev/min
        """
        # deg/s
        present_velocities = np.array(self.control.present_speeds)
        present_velocities = [
            vel for joint, vel in zip(self.joints.keys(), present_velocities) if joint not in ignore
        ]
        # present_velocities = np.array(
        #     self.control.io.get_present_speed(self.joints.values())
        # )
        if rad_s:
            present_velocities = np.deg2rad(present_velocities)  # rad/s
        return np.array(np.around(present_velocities, 3))

    def get_present_voltages(self):
        return np.array(self.control.io.get_present_voltage(self.joints.values())) * 0.1

    # def get_present_velocities(self, rad_s=True):
    #     """
    #     Returns the present velocities in rad/s (default) or rev/min
    #     """
    #     # rev/min
    #     present_velocities = np.array(
    #         self.control.io.get_present_speed(self.joints.values())
    #     )
    #     if rad_s:
    #         present_velocities = (2 * np.pi * present_velocities) / 60  # rad/s
    #     return np.array(np.around(present_velocities, 3))
