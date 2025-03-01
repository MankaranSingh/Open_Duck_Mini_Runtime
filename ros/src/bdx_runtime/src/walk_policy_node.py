import rospy
import onnxruntime as ort
import numpy as np
import time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32
from mini_bdx_runtime.rl_utils import quat_rotate_inverse

class WalkPolicyNode:
    def __init__(self):
        rospy.init_node('walk_policy_node')
        
        # Load the ONNX model
        self.model = ort.InferenceSession('../assets/policy.onnx')
        self.rate = rospy.Rate(50)  # 50 Hz
        
        # Initialize subscribers
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.feet_contact_sub = rospy.Subscriber('/feet_switch', Int32, self.feet_contact_callback)
        self.joint_states_sub = rospy.Subscriber('/current_joint_states', JointState, self.joint_states_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        
        # Initialize publisher
        self.target_joint_states_pub = rospy.Publisher('/target_joint_states', JointState, queue_size=10)
        
        # Initialize variables to store inputs
        self.cmd_vel = None
        self.feet_contact = None
        self.joint_positions = None
        self.joint_velocities = None
        self.projected_gravity = None
        self.angular_velocity = None

        self.joint_pos_scale = 1.0
        self.joint_vel_scale = 1.0
        self.angular_vel_scale = 0.25
        self.obs_history_length = 2
        self.action_history_length = 2
        self.power_scale = 1.5

        self.joint_names = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
                            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
                            "head_yaw", "head_roll", "left_antenna", "right_antenna",
                            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",]
        
        self.mask_joints = ['neck_pitch', 'head_pitch', 'head_yaw', "head_roll", "left_antenna", "right_antenna"]
        self.mask_joint_idx = np.array([self.joint_names.index(joint) for joint in self.mask_joints])
        
        self.obs_history = np.zeros((self.obs_history_length, 40))
        self.action_history = np.zeros((self.action_history_length, len(self.joint_names)))
    
    def imu_callback(self, msg):
        quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.projected_gravity = quat_rotate_inverse(quat, [0, 0, -1.0])
        self.angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    def cmd_vel_callback(self, msg):
        self.cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z*self.angular_vel_scale])

    def feet_contact_callback(self, msg):
        left_foot = (msg.data & 1)  # Extract left foot status (bit 0)
        right_foot = (msg.data >> 1) & 1  # Extract right foot status (bit 1)
        self.feet_contact = np.array([left_foot, right_foot])

    def joint_states_callback(self, msg):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

    def run_policy(self):

        while any(x is None for x in [self.cmd_vel, self.feet_contact, self.joint_positions, 
                              self.joint_velocities, self.projected_gravity, self.angular_velocity]):
            rospy.loginfo("Waiting for all policy inputs to be available..")
            time.sleep(0.5)

        while not rospy.is_shutdown():
            # Prepare the input for the model
            obs = np.concatenate([self.projected_gravity, 
                                         self.joint_positions*self.joint_pos_scale, 
                                         self.joint_velocities*self.joint_vel_scale, 
                                         self.angular_velocity, 
                                         self.feet_contact])

            self.obs_history[1:, :] = self.obs_history[:-1, :].copy()
            self.obs_history[0, :] = obs 

            input = np.concatenate([self.obs_history.flatten(), self.action_history.flatten(), self.cmd_vel]).reshape(1, -1)
                        
            # Run the model
            outputs = self.model.run(None, {'obs': input.astype(np.float32)})  # Run the model
            
            # Extract the target joint states
            actions = outputs[0].flatten()
            actions[self.mask_joint_idx] = 0.0

            self.action_history[1:, :] = self.action_history[:-1, :].copy()
            self.action_history[0, :] = actions 
            
            # Publish the target joint states
            joint_state_msg = JointState()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = (actions * self.power_scale).tolist()
            self.target_joint_states_pub.publish(joint_state_msg)

            self.rate.sleep()

if __name__ == '__main__':
    node = WalkPolicyNode()
    node.run_policy()
