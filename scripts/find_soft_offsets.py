"""
Find the offsets to set in self.joints_offsets in hwi_feetech_pwm_control.py
"""

# from mini_bdx_runtime.hwi_feetech_pwm_control import HWI
from mini_bdx_runtime.rustypot_position_hwi import HWI
import time

input(
    "Press any key to start. The robot will move to its zero position. Make sure it is safe to do so. At any time, press ctrl+c to stop, the motors will be turned off."
)
hwi = HWI()
hwi.joints_offsets = {
    "left_hip_yaw": 0,
    "left_hip_roll": 0,
    "left_hip_pitch": 0,
    "left_knee": 0,
    "left_ankle": 0,
    "neck_pitch": 0,
    "head_pitch": 0,
    "head_yaw": 0,
    "head_roll": 0,
    "right_hip_yaw": 0,
    "right_hip_roll": 0,
    "right_hip_pitch": 0,
    "right_knee": 0,
    "right_ankle": 0,
}

hwi.init_pos = hwi.zero_pos
hwi.set_kds([0] * len(hwi.joints))
hwi.turn_on()
print("")
print("")
print("")
print("")
hwi.set_position_all(hwi.zero_pos)
time.sleep(1)
try:
    for i, joint_name in enumerate(hwi.joints.keys()):
        joint_id = hwi.joints[joint_name]
        ok = False
        while not ok:
            res = input(f" === Setting up {joint_name} === (Y/(s)kip : ").lower()
            if res == "s":
                break
            hwi.set_position_all(hwi.zero_pos)
            time.sleep(0.5)
            current_pos = hwi.get_present_positions()[i]
            # hwi.control.kps[i] = 0
            hwi.io.disable_torque([joint_id])
            input(
                f"{joint_name} is now turned off. Move it to the desired zero position and press any key to confirm the offset"
            )
            new_pos = hwi.get_present_positions()[i]
            offset = new_pos - current_pos
            print(f" ---> Offset is {offset}")
            hwi.joints_offsets[joint_name] = offset
            input(
                "Press any key to move the motor to its zero position with offset taken into account"
            )
            hwi.set_position_all(hwi.zero_pos)
            time.sleep(0.5)
            hwi.io.enable_torque([joint_id])
            # hwi.control.kps[i] = 32
            res = input("Is that ok ? (Y/n)").lower()
            if res == "Y" or res == "":
                print("Ok, setting offset")
                hwi.joints_offsets[joint_name] = offset
                ok = True
                print("------")
                print("Current offsets : ")
                for k, v in hwi.joints_offsets.items():
                    print(f"{k} : {v}")
                print("------")
                print("")
            else:
                print("Ok, let's try again")
                hwi.joints_offsets[joint_name] = 0

            print("===")

    print("Done ! ")
    print("Now you can copy the offsets in the hwi_feetech_pwm_control.py file")


except KeyboardInterrupt:
    hwi.turn_off()
