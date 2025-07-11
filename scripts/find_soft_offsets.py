"""
Find the offsets to set in self.joints_offsets in hwi_feetech_pwm_control.py
"""

# from mini_bdx_runtime.hwi_feetech_pwm_control import HWI
from mini_bdx_runtime.rustypot_position_hwi import HWI
import time

hwi = HWI()
hwi.joints_offsets = {
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

current_pos = hwi.get_present_positions()
print("Current positions : ")
for joint_name, pos in zip(hwi.joints.keys(), current_pos):
    print(f"{joint_name}: {pos},")
