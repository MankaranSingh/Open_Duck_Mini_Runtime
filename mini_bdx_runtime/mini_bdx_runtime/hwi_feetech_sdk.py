from mini_bdx_runtime.feetech import (
    FeetechMotorsBus,
    convert_radians_to_steps,
    convert_steps_to_radians,
    configure,
)
import time
import numpy as np

# Everything is in radians here


class HWI:
    def __init__(self, usb_port="/dev/ttyACM0"):
        self.motors = {
            "left_hip_yaw": (8, "sts3215"),
            "left_hip_roll": (9, "sts3215"),
            "left_hip_pitch": (6, "sts3215"),
            "left_knee": (4, "sts3215"),
            "left_ankle": (2, "sts3215"),
            "neck_pitch": (1, "sts3215"),
            "head_pitch": (12, "sts3215"),
            "head_yaw": (13, "sts3215"),
            "head_roll": (14, "sts3215"),
            # "left_antenna": (None, "sts3215"),
            # "right_antenna": (None, "sts3215"),
            "right_hip_yaw": (10, "sts3215"),
            "right_hip_roll": (11, "sts3215"),
            "right_hip_pitch": (7, "sts3215"),
            "right_knee": (5, "sts3215"),
            "right_ankle": (3, "sts3215"),
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

        self.joints_sign = {
            "left_hip_yaw" : 1,
            "left_hip_roll" : 1,
            "left_hip_pitch" : 1,
            "left_knee" : 1,
            "left_ankle" : 1,
            "neck_pitch" : 1,
            "head_pitch" : 1,
            "head_yaw" : 1,
            "head_roll" : 1,
            "right_hip_yaw" : 1,
            "right_hip_roll" : 1,
            "right_hip_pitch" : 1,
            "right_knee" : 1,
            "right_ankle" : 1,
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
        self.baudrate = 1000000

        self.motors_bus = FeetechMotorsBus(port=usb_port, motors=self.motors)
        self.motors_bus.connect()
        configure(self.motors_bus)

        # We add 180 to all dofs. They were mounted in a way that 180 degrees is the neutral position.
        self.global_offset = np.deg2rad(180)

    def set_kps(self, kp=32):
        self.motors_bus.write("P_Coefficient", kp)
    
    def set_kds(self, kd=32):
        self.motors_bus.write("D_Coefficient", kd)
        
    def enable_torque(self):
        self.motors_bus.write("Torque_Enable", 1)

    def disable_torque(self):
        self.motors_bus.write("Torque_Enable", 0)

    def turn_on(self):
        self.enable_torque()
        self.set_position_all(self.init_pos)
        # self.set_position_all(self.zero_pos)
        time.sleep(1)

    def set_position_all(self, joints_positions: dict):
        positions = np.array(list(joints_positions.values()))
        signs = np.array(list(self.joints_sign.values()))
        rads = positions * signs + self.global_offset
        rads = list(rads)

        steps = convert_radians_to_steps(rads, ["sts3215"])
        self.motors_bus.write("Goal_Position", steps)

    def get_present_velocities(self):

        steps = self.motors_bus.read("Present_Position", self.motors)
        rads = np.array(convert_steps_to_radians(steps, ["sts3215"]))
        return rads - self.global_offset
        # return rads * np.array(list(self.joints_sign.values()))# - self.global_offset


if __name__ == "__main__":
    hwi = HWI()
    hwi.turn_on()

    # hwi.disable_torque()
    # time.sleep(1)
    # while True:
    #     print(hwi.get_position_all())
    #     time.sleep(0.1)
