#!/usr/bin/env python3

import rospy
import onnxruntime as ort
import numpy as np
import time
import os
from collections import deque

# ROS message types
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

def quat_rotate_inverse(q, v):
    q = np.array(q)
    v = np.array(v)

    q_w = q[-1]
    q_vec = q[:3]

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0

    return a - b + c

class WalkPolicyInferenceNode:
    """Walk policy inference node that subscribes to sensor data and publishes target joint commands."""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('walk_policy_inference_node')
        
        # Load the ONNX model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '../../assets/policy.onnx')
        self.model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Setup joint names and policy parameters
        self.setup_policy_params()
        
        # Initialize message rate tracking
        self.setup_rate_tracking()
        
        # Initialize publishers
        self.target_joint_states_pub = rospy.Publisher('/target_joint_states', JointState, queue_size=1)
        
        # Initialize subscribers
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.feet_contact_sub = rospy.Subscriber('/feet_switch', Int32, self.feet_contact_callback)
        self.joint_states_sub = rospy.Subscriber('/current_joint_states', JointState, self.joint_states_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        
        # Initialize sensor data variables
        self.cmd_vel = None
        self.feet_contact = None
        self.joint_positions = None
        self.joint_velocities = None
        self.projected_gravity = None
        self.angular_velocity = None
        
        # Initialize history arrays
        self.obs_history = np.zeros((self.obs_history_length, 35))  # Adjust size as needed
        self.action_history = np.zeros((self.action_history_length, len(self.joint_names)))
        
        # For rate logging
        self.last_rate_log_time = rospy.Time.now().to_sec()
        self.init_pos = np.array([
                0.002, 0.053, -0.63, 1.368, -0.784, 
                0.0, 0, 0, 0, 0, 0, 
                -0.003, -0.065, 0.635, 1.379, -0.796,
            ])
        
        rospy.loginfo("Walk Policy Inference Node initialized.")
    
    def setup_policy_params(self):
        """Setup policy parameters and joint information."""
        # Policy parameters
        self.joint_pos_scale = 1.0
        self.joint_vel_scale = 1.0
        self.angular_vel_scale_obs = 1.0
        self.angular_vel_scale = 0.25
        self.obs_history_length = 2
        self.action_history_length = 2

        self.action_clip = [-5.0, 5.0]
        self.obs_clip = [-5.0, 5.0]
        
        self.power_scale = 1.5
        self.lin_vel_x_range = [-0.3, 0.3]
        self.lin_vel_y_range = [-0.3, 0.3]
        self.yaw_range = [-0.3, 0.3]
        
        # Joint names and masking
        self.joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
            "head_yaw", "head_roll", "left_antenna", "right_antenna",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
        ]
        
        self.mask_joints = ['neck_pitch', 'head_pitch', 'head_yaw', "head_roll", "left_antenna", "right_antenna"]
        self.mask_joint_idx = np.array([self.joint_names.index(joint) for joint in self.mask_joints])
    
    def setup_rate_tracking(self):
        """Setup message rate tracking."""
        # Define expected publishing rates (in Hz) for each topic
        self.expected_rates = {
            'cmd_vel': 50,
            'feet_contact': 50,
            'joint_states': 50,
            'imu': 50
        }
        # Acceptable rate threshold as a fraction of the expected rate
        self.acceptable_fraction = 0.8
        
        window_size = 5  # Number of recent messages to average over
        self.msg_times = {
            'cmd_vel': deque(maxlen=window_size),
            'feet_contact': deque(maxlen=window_size),
            'joint_states': deque(maxlen=window_size),
            'imu': deque(maxlen=window_size)
        }
    
    def compute_avg_rate(self, timestamps):
        """Compute average rate from a deque of timestamps."""
        if len(timestamps) < 2:
            return None
        dt = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / dt if dt > 0 else float('inf')
    
    def check_topic_rates(self):
        """Check if all topics are publishing at an acceptable rate."""
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
        """Process incoming IMU data."""
        current_time = rospy.Time.now().to_sec()
        self.msg_times['imu'].append(current_time)
        
        quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.projected_gravity = quat_rotate_inverse(quat, [0, 0, -1.0])
        self.angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    
    def scale_cmd_vel(self, linear_x, linear_y, angular_z):
        """Scale command velocities to their respective ranges."""
        lin_vel_x = np.clip(linear_x, -1, 1) * (self.lin_vel_x_range[1] - self.lin_vel_x_range[0]) / 2 + (self.lin_vel_x_range[1] + self.lin_vel_x_range[0]) / 2
        lin_vel_y = np.clip(linear_y, -1, 1) * (self.lin_vel_y_range[1] - self.lin_vel_y_range[0]) / 2 + (self.lin_vel_y_range[1] + self.lin_vel_y_range[0]) / 2
        yaw = np.clip(angular_z, -1, 1) * (self.yaw_range[1] - self.yaw_range[0]) / 2 + (self.yaw_range[1] + self.yaw_range[0]) / 2
        return np.array([lin_vel_x, lin_vel_y, yaw * self.angular_vel_scale])
    
    def cmd_vel_callback(self, msg):
        """Process incoming velocity commands."""
        current_time = rospy.Time.now().to_sec()
        self.msg_times['cmd_vel'].append(current_time)
        self.cmd_vel = self.scale_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)
    
    def feet_contact_callback(self, msg):
        """Process incoming feet contact data."""
        current_time = rospy.Time.now().to_sec()
        self.msg_times['feet_contact'].append(current_time)
        left_foot = (msg.data & 1)  # Extract left foot status (bit 0)
        right_foot = (msg.data >> 1) & 1  # Extract right foot status (bit 1)
        self.feet_contact = np.array([left_foot, right_foot])
    
    def joint_states_callback(self, msg):
        """Process incoming joint states."""
        current_time = rospy.Time.now().to_sec()
        self.msg_times['joint_states'].append(current_time)
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)
    
    def run_policy(self):
        """Run the inference loop."""
        rate = rospy.Rate(50)  # 50 Hz
        
        # Wait until all required data is available
        while any(x is None for x in [self.cmd_vel, self.feet_contact, self.joint_positions, 
                                     self.joint_velocities, self.projected_gravity, self.angular_velocity]):
            rospy.loginfo_throttle(1.0, "Waiting for all policy inputs to be available...")
            if rospy.is_shutdown():
                return
            rate.sleep()
        
        rospy.loginfo("All sensor data received. Starting policy execution.")
        
        while not rospy.is_shutdown():
            # Check if all topics are publishing at expected rates
            ok, low_topics = self.check_topic_rates()
            if not ok:
                current_time = rospy.Time.now().to_sec()
                if current_time - self.last_rate_log_time >= 1.0:
                    rospy.logwarn("Pausing policy execution due to low topic rates: " + ", ".join(low_topics))
                    self.last_rate_log_time = current_time
                rate.sleep()
                continue
            
            # Prepare the input for the model
            obs = np.concatenate([
                self.projected_gravity,
                self.joint_positions * self.joint_pos_scale,
                self.joint_velocities * self.joint_vel_scale,
                # self.angular_velocity * self.angular_vel_scale_obs,
                # self.feet_contact
            ])
            
            # Update observation history
            self.obs_history[1:, :] = self.obs_history[:-1, :].copy()
            self.obs_history[0, :len(obs)] = obs
            
            # Combine with action history
            input_data = np.concatenate([self.obs_history.flatten(), self.action_history.flatten(), self.cmd_vel]).reshape(1, -1)

            input_data = np.clip(input_data, self.obs_clip[0], self.obs_clip[1])
            
            # Run the model
            outputs = self.model.run(None, {'obs': input_data.astype(np.float32)})
            
            # Extract and process actions
            actions = outputs[0].flatten()
            actions[self.mask_joint_idx] = 0.0

            actions = np.clip(actions, self.action_clip[0], self.action_clip[1])
            
            # Update action history
            self.action_history[1:, :] = self.action_history[:-1, :].copy()
            self.action_history[0, :] = actions
            
            # Publish target joint states
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = (actions*self.power_scale+self.init_pos).tolist()
            self.target_joint_states_pub.publish(joint_state_msg)
            
            # Sleep to maintain 50Hz
            rate.sleep()


if __name__ == '__main__':
    try:
        node = WalkPolicyInferenceNode()
        node.run_policy()
    except rospy.ROSInterruptException:
        pass
