import time
from typing import List

import numpy as np
import rustypot


class HWI:
    def __init__(self, usb_port="/dev/ttyACM0"):

        # Order matters here
        self.joints = {
            "neck_pitch": 3,
            "head_pitch": 9,
            "head_yaw": 11,
            "tail": 4,
            "right_hip_yaw": 1,
            "right_hip_roll": 12,
            "right_hip_pitch": 10,
            "right_knee": 6,
            "right_ankle": 7,
            "left_hip_yaw": 0,
            "left_hip_roll": 13,
            "left_hip_pitch": 2,
            "left_knee": 5,
            "left_ankle": 8,
        }

        self.init_pos = {
            "neck_pitch": 0,
            "head_pitch": 0,
            "head_yaw": 0,
            "tail": 0,
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_hip_pitch": 0,
            "right_knee": 0,
            "right_ankle": 0,
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_hip_pitch": 0,
            "left_knee": 0,
            "left_ankle": 0,
        }

        self.joints_offsets = {
            "neck_pitch": 1.506,
            "head_pitch": 1.767,
            "head_yaw": -0.002,
            "tail": -0.061,
            "right_hip_yaw": 0.072,
            "right_hip_roll": -0.043,
            "right_hip_pitch": -1.006,
            "right_knee": -0.365,
            "right_ankle": 0.439,
            "left_hip_yaw": -0.025,
            "left_hip_roll": 0.072,
            "left_hip_pitch": 1.049,
            "left_knee": 0.522,
            "left_ankle": -0.425,

        }

        init_pos_with_offsets = {
            joint: np.rad2deg(pos + self.joints_offsets[joint])
            for joint, pos in self.init_pos.items()
        }

        self.kps = np.ones(len(self.joints)) * 22  # default kp
        self.kds = np.ones(len(self.joints)) * 0  # default kd
        self.low_torque_kps = np.ones(len(self.joints)) * 2

        # self.control = rustypot.FeetechController(
        #     usb_port, 1000000, 100, list(self.joints.values()), list(self.kps), list(init_pos_with_offsets.values())
        # )
        self.io = rustypot.feetech(usb_port, 1000000)

    def set_kps(self, kps):
        self.kps = kps
        self.io.set_kps(list(self.joints.values()), self.kps)
        # self.control.set_new_kps(self.kps)

    def set_kds(self, kds):
        self.kds = kds
        self.io.set_kds(list(self.joints.values()), self.kds)

    def set_kp(self, id, kp):
        # self.kps[id] = kp
        self.io.set_kps([id], [kp])

    def turn_on(self):
        self.io.set_kps(list(self.joints.values()), self.low_torque_kps)
        # self.control.set_new_kps(self.low_torque_kps)
        print("turn on : low KPS set")
        time.sleep(1)

        self.set_position_all(self.init_pos)
        print("turn on : init pos set")

        time.sleep(1)

        self.io.set_kps(list(self.joints.values()), self.kps)
        print("turn on : high kps")

    def turn_off(self):
        self.io.disable_torque(list(self.joints.values()))
        # self.control.disable_torque()

    # def freeze(self):
    #     self.control.freeze()

    def set_position(self, joint_name, pos):
        """
        pos is in radians
        """
        id = self.joints[joint_name]
        pos = pos + self.joints_offsets[joint_name]
        self.io.write_goal_position([id], [pos])
        # self.control.set_new_target([pos])

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: position + self.joints_offsets[joint]
            for joint, position in joints_positions.items()
        }

        self.io.write_goal_position(
            list(self.joints.values()), list(ids_positions.values())
        )
        # self.control.set_new_target(list(ids_positions.values()))
        # self.control.goal_positions = list(ids_positions.values())

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """

        # present_positions = np.deg2rad(
        #     self.control.io.get_present_position(self.joints.values())
        # )

        present_positions = self.io.read_present_position(list(self.joints.values()))
        # present_positions = np.deg2rad(self.control.get_present_position())
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
        present_velocities = self.io.read_present_velocity(list(self.joints.values()))
        # present_velocities = np.array(self.control.get_current_speed())
        present_velocities = [
            vel
            for joint, vel in zip(self.joints.keys(), present_velocities)
            if joint not in ignore
        ]
        # present_velocities = np.array(
        #     self.control.io.get_present_speed(self.joints.values())
        # )
        # if rad_s:
        #     present_velocities = np.deg2rad(present_velocities)  # rad/s
        return np.array(np.around(present_velocities, 3))

    def get_present_voltages(self):
        return np.array(self.control.io.get_present_voltage(self.joints.values())) * 0.1

if __name__ == "__main__":
    hwi = HWI()
    hwi.set_kps(np.ones(len(hwi.joints)) * 1)

    hwi.set_position_all(hwi.init_pos)
