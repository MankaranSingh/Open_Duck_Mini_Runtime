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
            "neck_pitch": 0.13805827090970757,
            "head_pitch": -0.012271846303085088,
            "head_yaw": 0.09357282806102418,
            "tail": -0.038349519697141066,
            "right_hip_yaw": -0.035281558121369905,
            "right_hip_roll": -0.004601942363656963,
            "right_hip_pitch": 0.04295146206079803,
            "right_knee": 2.82252464970958,
            "right_ankle": -0.47246608266877743,
            "left_hip_yaw": -0.15033011721279266,
            "left_hip_roll": -0.019941750242513212,
            "left_hip_pitch": -1.5723303075827821,
            "left_knee": 0.07056311624273937,
            "left_ankle": 0.24697090684958845,
        }

        init_pos_with_offsets = {
            joint: np.rad2deg(pos + self.joints_offsets[joint])
            for joint, pos in self.init_pos.items()
        }

      
        self.io = rustypot.feetech(usb_port, 1000000)

    def set_kps(self, kps, joints=None):
        joints = joints or list(self.joints.keys())
        joint_ids = [self.joints[joint] for joint in joints]
        self.io.set_kps(joint_ids, kps)

    def set_kds(self, kds, joints=None):
        joints = joints or list(self.joints.keys())
        joint_ids = [self.joints[joint] for joint in joints]
        self.io.set_kds(joint_ids, kds)

    def set_kp(self, id, kp):
        self.io.set_kps([id], [kp])

    def disable_torque(self, joints=None):
        joints = joints or list(self.joints.keys())
        joint_ids = [self.joints[joint] for joint in joints]
        self.io.disable_torque(joint_ids)

    def set_positions(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: position + self.joints_offsets[joint]
            for joint, position in joints_positions.items()
        }

        self.io.write_goal_position(
            list(ids_positions.keys()), list(ids_positions.values())
        )

    def get_present_positions(self, joints=None):
        """
        Returns the present positions in radians
        """
        joints = joints or list(self.joints.keys())
        joint_ids = [self.joints[joint] for joint in joints]
        
        present_positions = self.io.read_present_position(joint_ids)
        present_positions = [
            pos - self.joints_offsets[joint]
            for joint, pos in zip(joints, present_positions)
        ]
        return np.array(present_positions)

    def get_present_velocities(self, joints=None):
        """
        Returns the present velocities in rad/s
        """
        joints = joints or list(self.joints.keys())
        joint_ids = [self.joints[joint] for joint in joints]
        
        present_velocities = self.io.read_present_velocity(joint_ids)
        return np.array(present_velocities)

    def get_present_voltages(self, joints=None):
        joints = joints or list(self.joints.keys())
        joint_ids = [self.joints[joint] for joint in joints]
        
        return np.array(self.io.get_present_voltage(joint_ids)) * 0.1

if __name__ == "__main__":
    hwi = HWI()
    hwi.set_kps(np.ones(len(hwi.joints)) * 1)

    hwi.set_position_all(hwi.init_pos)
