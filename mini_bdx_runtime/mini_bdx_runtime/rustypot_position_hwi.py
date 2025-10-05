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
            "neck_pitch": 0.17794177139473444,
            "head_pitch": 0.05982525072754008,
            "head_yaw": 0.08897088569736722,
            "tail": 0.1012427320004523,
            "right_hip_yaw": -0.019941750242513212,
            "right_hip_roll": -0.006135923151542766,
            "right_hip_pitch": 0.09203884727313838,
            "right_knee": 2.908427573831176,
            "right_ankle": -0.411106851153352,
            "left_hip_yaw": -0.16260196351587775,
            "left_hip_roll": -0.01380582709097089,
            "left_hip_pitch": -1.5661943844312396,
            "left_knee": 0.07516505860639633,
            "left_ankle": 0.18714565612204836,
        }

        init_pos_with_offsets = {
            joint: np.rad2deg(pos + self.joints_offsets[joint])
            for joint, pos in self.init_pos.items()
        }

      
        self.io = rustypot.feetech(usb_port, 1000000)

    def set_kps(self, kps, joint_ids=None):
        joint_ids = joint_ids or list(self.joints.values())
        self.io.set_kps(joint_ids, kps)

    def set_kds(self, kds, joint_ids=None):
        joint_ids = joint_ids or list(self.joints.values())
        self.io.set_kds(joint_ids, kds)

    def set_kp(self, id, kp):
        self.io.set_kps([id], [kp])

    def disable_torque(self, joint_ids=None):
        joint_ids = joint_ids or list(self.joints.values())
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

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """
        present_positions = self.io.read_present_position(list(self.joints.values()))
        present_positions = [
            pos - self.joints_offsets[joint]
            for joint, pos in zip(self.joints.keys(), present_positions)
            if joint not in ignore
        ]
        return np.array(present_positions)

    def get_present_velocities(self, ignore=[]):
        """
        Returns the present velocities in rad/s
        """
        present_velocities = self.io.read_present_velocity(list(self.joints.values()))
        present_velocities = [
            vel
            for joint, vel in zip(self.joints.keys(), present_velocities)
            if joint not in ignore
        ]
        return np.array(present_velocities)

    def get_present_voltages(self):
        return np.array(self.control.io.get_present_voltage(self.joints.values())) * 0.1

if __name__ == "__main__":
    hwi = HWI()
    hwi.set_kps(np.ones(len(hwi.joints)) * 1)

    hwi.set_position_all(hwi.init_pos)
