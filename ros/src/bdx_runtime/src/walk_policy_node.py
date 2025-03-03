import rospy
import onnxruntime as ort
import numpy np
import time
from collections import deque
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32
from mini_bdx_runtime.rl_utils import quat_rotate_inverse
import os

class WalkPolicyNode:
    def __init__(self):
        rospy.init_node('walk_policy_node')

        # Load the ONNX model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the model path relative to the current file
        model_path = os.path.join(current_dir, '../assets/policy.onnx')
        self.model = ort.InferenceSession(model_path)
        self.rate = rospy.Rate(50)  # 50 Hz

        # Define the expected publishing rates (in Hz) for each topic.
        self.expected_rates = {
            'cmd_vel': 50,       # cmd_vel is expected at 50 Hz
            'feet_contact': 50, # Other topics are expected at 50 Hz
            'joint_states': 50,
            'imu': 100
        }
        # Acceptable rate threshold as a fraction of the expected rate.
        self.acceptable_fraction = 0.8

        window_size = 10  # Number of recent messages to average over
        self.msg_times = {
            'cmd_vel': deque(maxlen=window_size),
            'feet_contact': deque(maxlen=window_size),
            'joint_states': deque(maxlen=window_size),
            'imu': deque(maxlen=window_size)
        }

        # Initialize subscribers
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.feet_contact_sub = rospy.Subscriber('/feet_switch', Int32, self.feet_contact_callback)
        self.joint_states_sub = rospy.Subscriber('/current_joint_states', JointState, self.joint_states_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Initialize publisher
        self.target_joint_states_pub = rospy.Publisher('/target_joint_states', JointState, queue_size=1)

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
        self.lin_vel_x_range = [-0.3, 0.3]
        self.lin_vel_y_range = [-0.3, 0.3]
        self.yaw_range = [-0.3, 0.3]

        self.joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
            "head_yaw", "head_roll", "left_antenna", "right_antenna",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
        ]
        
        self.mask_joints = ['neck_pitch', 'head_pitch', 'head_yaw', "head_roll", "left_antenna", "right_antenna"]
        self.mask_joint_idx = np.array([self.joint_names.index(joint) for joint in self.mask_joints])
        
        self.obs_history = np.zeros((self.obs_history_length, 40))
        self.action_history = np.zeros((self.action_history_length, len(self.joint_names)))        

        # For rate logging control (log once per second)
        self.last_rate_log_time = rospy.Time.now().to_sec()

    def compute_avg_rate(self, timestamps):
        """Compute average rate from a deque of timestamps."""
        if len(timestamps) < 2:
            return None
        dt = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / dt if dt > 0 else float('inf')

    def check_topic_rates(self):
        """
        Check if all topics are publishing at an acceptable rate.
        Returns a tuple: (all_good: bool, low_topics: list)
        """
        low_topics = []
        for topic, times in self.msg_times.items():
            avg_rate = self.compute_avg_rate(times)
            if avg_rate is None:
                low_topics.append(topic)
            else:
                min_rate = self.expected_rates[topic] * self.acceptable_fraction
                if avg_rate < min_rate:
                    low_topics.append(topic)
        if low_topics:
            return False, low_topics
        return True, []

    def imu_callback(self, msg):
        current_time = rospy.Time.now().to_sec()
        self.msg_times['imu'].append(current_time)
        quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.projected_gravity = quat_rotate_inverse(quat, [0, 0, -1.0])
        self.angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    def scale_cmd_vel(self, linear_x, linear_y, angular_z):
        """Scale cmd_vel values to their respective ranges."""
        lin_vel_x = np.clip(linear_x, -1, 1) * (self.lin_vel_x_range[1] - self.lin_vel_x_range[0]) / 2 + (self.lin_vel_x_range[1] + self.lin_vel_x_range[0]) / 2
        lin_vel_y = np.clip(linear_y, -1, 1) * (self.lin_vel_y_range[1] - self.lin_vel_y_range[0]) / 2 + (self.lin_vel_y_range[1] + self.lin_vel_y_range[0]) / 2
        yaw = np.clip(angular_z, -1, 1) * (self.yaw_range[1] - self.yaw_range[0]) / 2 + (self.yaw_range[1] + self.yaw_range[0]) / 2
        return np.array([lin_vel_x, lin_vel_y, yaw * self.angular_vel_scale])

    def cmd_vel_callback(self, msg):
        current_time = rospy.Time.now().to_sec()
        self.msg_times['cmd_vel'].append(current_time)
        self.cmd_vel = self.scale_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)

    def feet_contact_callback(self, msg):
        current_time = rospy.Time.now().to_sec()
        self.msg_times['feet_contact'].append(current_time)
        left_foot = (msg.data & 1)  # Extract left foot status (bit 0)
        right_foot = (msg.data >> 1) & 1  # Extract right foot status (bit 1)
        self.feet_contact = np.array([left_foot, right_foot])

    def joint_states_callback(self, msg):
        current_time = rospy.Time.now().to_sec()
        self.msg_times['joint_states'].append(current_time)
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

    def run_policy(self):
        # Wait until all required data is available
        while any(x is None for x in [self.cmd_vel, self.feet_contact, self.joint_positions, 
                                       self.joint_velocities, self.projected_gravity, self.angular_velocity]):
            rospy.loginfo("Waiting for all policy inputs to be available..")
            time.sleep(0.5)

        while not rospy.is_shutdown():
            ok, low_topics = self.check_topic_rates()
            if not ok:
                current_time = rospy.Time.now().to_sec()
                # Log which topics are below threshold once per second
                if current_time - self.last_rate_log_time >= 1.0:
                    rospy.logwarn("Pausing policy execution due to low topic rates: " + ", ".join(low_topics))
                    self.last_rate_log_time = current_time
                self.rate.sleep()
                continue

            # Prepare the input for the model
            obs = np.concatenate([self.projected_gravity,
                                  self.joint_positions * self.joint_pos_scale,
                                  self.joint_velocities * self.joint_vel_scale,
                                  #self.angular_velocity,
                                  #self.feet_contact,
                                  ])
            
            # Update observation history (shift and insert new observation)
            self.obs_history[1:, :] = self.obs_history[:-1, :].copy()
            self.obs_history[0, :] = obs

            input_data = np.concatenate([self.obs_history.flatten(), self.action_history.flatten(), self.cmd_vel]).reshape(1, -1)

            # Run the model
            outputs = self.model.run(None, {'obs': input_data.astype(np.float32)})
            
            # Extract the target joint states and mask specified joints
            actions = outputs[0].flatten()
            actions[self.mask_joint_idx] = 0.0
            
            # Update action history
            self.action_history[1:, :] = self.action_history[:-1, :].copy()
            self.action_history[0, :] = actions
            
            # Publish the target joint states
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = rospy.Time.now()  # Add timestamp
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = (actions * self.power_scale).tolist()
            self.target_joint_states_pub.publish(joint_state_msg)

            self.rate.sleep()


if __name__ == '__main__':
    node = WalkPolicyNode()
    node.run_policy()
