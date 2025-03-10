#!/usr/bin/env python3

import rospy
import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import os

# ROS message types
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from mini_bdx.utils.mujoco_utils import check_contact

class MujocoSimNode:
    """MuJoCo simulation node that publishes sensor data and consumes target joint positions."""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('mujoco_sim_node')
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False,
                           help="Run without visualization")
        parser.add_argument("--control_mode", type=str, default="torque", choices=["position", "torque"],
                           help="Control mode for the simulation")
        args, unknown = parser.parse_known_args()

        if args.control_mode == "torque":
            self.model_path = "/home/mankaran/Desktop/rl/Open_Duck_Mini/mini_bdx/robots/open_duck_mini_v2/scene.xml"
        else:
            self.model_path = "/home/mankaran/Desktop/rl/Open_Duck_Mini/mini_bdx/robots/open_duck_mini_v2/scene_position.xml"
        
        self.headless = args.headless
        self.position_control = args.control_mode == "position"
        
        # Initialize MuJoCo model
        self.setup_mujoco()
        
        # Initialize joint state mappings
        self.setup_joint_mappings()
        
        # Initialize ROS publishers
        self.joint_pub = rospy.Publisher('/current_joint_states', JointState, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=1)
        self.feet_pub = rospy.Publisher('/feet_switch', Int32, queue_size=1)
        
        # Initialize ROS subscriber for target joint positions
        self.target_sub = rospy.Subscriber('/target_joint_states', JointState, self.target_callback)
        
        # Control variables
        self.target_positions = np.copy(self.init_pos)
        self.last_positions = np.copy(self.init_pos)
        
        # PD control parameters
        self.kps = np.array([6.55] * 16)
        self.kds = np.array([0.6] * 16)
        
        # Low-pass filter parameters
        self.dt = self.model.opt.timestep
        self.cutoff_freq = 37.5  # Hz
        self.rc = 1.0 / (2.0 * np.pi * self.cutoff_freq)
        self.alpha = self.dt / (self.dt + self.rc)
        self.filtered_positions = np.zeros(16)  # Initialize with zeros
        rospy.loginfo(f"Low-pass filter configured: cutoff freq={self.cutoff_freq}Hz, alpha={self.alpha:.4f}")
        
        # Simulation control
        self.control_decimation = 2  # Run control at slower rate than physics
        self.counter = 0
        
        # Set up the viewer
        self.viewer = None if self.headless else mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )
        
        # Identify the body to which the IMU is attached
        self.imu_body_name = rospy.get_param('~imu_body_name', 'torso')
        self.imu_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.imu_body_name)
        if self.imu_body_id < 0:
            # Try some common alternatives
            for name in ['base', 'body', 'chassis', 'root']:
                self.imu_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.imu_body_id >= 0:
                    self.imu_body_name = name
                    break
        
        if self.imu_body_id < 0:
            rospy.logwarn(f"Could not find the IMU body '{self.imu_body_name}'. Using body ID 1 (typically the root body).")
            self.imu_body_id = 1  # Typically the root body in MuJoCo models
        
        rospy.loginfo("MuJoCo simulation node initialized. Model: %s", self.model_path)
    
    def setup_mujoco(self):
        """Setup the MuJoCo model and data."""
        rospy.loginfo(f"Loading MuJoCo model from: {self.model_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.model.opt.timestep = 0.01  # 100 Hz
            self.data = mujoco.MjData(self.model)
            
            # Initial pose
            self.init_pos = np.array([
                0.002, 0.053, -0.63, 1.368, -0.784, 
                0.0, 0, 0, 0, 0, 0, 
                -0.003, -0.065, 0.635, 1.379, -0.796,
            ])
            
            # Set initial state
            self.data.qpos[3:7] = [1, 0, 0.0, 0]  # Unit quaternion for orientation
            self.data.qpos[7:7+16] = self.init_pos
            
            mujoco.mj_forward(self.model, self.data)
            rospy.loginfo("MuJoCo model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load MuJoCo model: {e}")
            raise
    
    def setup_joint_mappings(self):
        """Setup joint name mappings."""
        self.joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
            "head_yaw", "head_roll", "left_antenna", "right_antenna",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
        ]
    
    def target_callback(self, msg):
        """Handle incoming target joint positions."""
        for i, name in enumerate(msg.name):
            if i < len(self.target_positions):  # Ensure we don't go out of bounds
                self.target_positions[i] = msg.position[i]
    
    def compute_control(self):
        """PD controller for joint positions with low-pass filter."""
        # Calculate raw PD control signal
        self.filtered_positions = self.alpha * self.target_positions + (1.0 - self.alpha) * self.last_positions
        self.last_positions[:] = self.target_positions

        if not self.position_control:
            torques = (self.filtered_positions - self.data.qpos[7:23]) * self.kps
            torques -= self.data.qvel[6:22] * self.kds
            return torques
        else:
            return self.filtered_positions 
    
    def get_feet_contact(self):
        """Check foot contact with the floor."""
        left_contact = check_contact(self.data, self.model, "foot_assembly", "floor")
        right_contact = check_contact(self.data, self.model, "foot_assembly_2", "floor")
        return np.array([left_contact, right_contact]).astype(np.int32)
    
    def quat_rotate_inverse(self, q, v):
        """Rotate a vector using a quaternion inverse."""
        q = np.array(q)
        v = np.array(v)

        q_w = q[-1]
        q_vec = q[:3]

        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * (np.dot(q_vec, v)) * 2.0

        return a - b + c
    
    def publish_joint_states(self):
        """Publish joint states from the simulation."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        
        # Copy joint positions and velocities
        msg.position = self.data.qpos[7:23].tolist()
        msg.velocity = self.data.qvel[6:22].tolist()
        
        self.joint_pub.publish(msg)
    
    def publish_imu_data(self):
        """Publish IMU data from the simulation."""
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "imu_link"
        
        # Extract quaternion from model state (wxyz -> xyzw)
        noise_orientation = np.random.normal(0, 0.02, 3)
        quat = self.data.qpos[3:7].copy()  # [w, x, y, z]
        msg.orientation.x = quat[1] + noise_orientation[0]
        msg.orientation.y = quat[2] + noise_orientation[1]
        msg.orientation.z = quat[3] + noise_orientation[2]
        msg.orientation.w = quat[0]
        
        # Copy angular velocity with added noise
        noise_angular_velocity = np.random.normal(0, 0.05, 3)  # Mean 0, std 0.01
        msg.angular_velocity.x = self.data.qvel[3] + noise_angular_velocity[0]
        msg.angular_velocity.y = self.data.qvel[4] + noise_angular_velocity[1]
        msg.angular_velocity.z = self.data.qvel[5] + noise_angular_velocity[2]
        
        # Get acceleration data for the robot body
        accel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectAcceleration(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.imu_body_id, accel, 0)
        
        # Linear acceleration in world frame is the first 3 values
        lin_acc_world = accel[:3]
        
        # Transform to robot frame using quaternion inverse
        lin_acc_local = self.quat_rotate_inverse([quat[1], quat[2], quat[3], quat[0]], lin_acc_world)
        
        # Add gravity effect (in robot frame) - gravity in MuJoCo is [0, 0, -9.81]
        gravity_local = self.quat_rotate_inverse([quat[1], quat[2], quat[3], quat[0]], [0, 0, -9.81])
        lin_acc_local += gravity_local
        
        # Add noise to linear acceleration
        noise_linear_acceleration = np.random.normal(0, 0.5, 3)
        
        # Set linear acceleration in message
        msg.linear_acceleration.x = lin_acc_local[0] + noise_linear_acceleration[0]
        msg.linear_acceleration.y = lin_acc_local[1] + noise_linear_acceleration[1]
        msg.linear_acceleration.z = lin_acc_local[2] + noise_linear_acceleration[2]
        
        self.imu_pub.publish(msg)
    
    def publish_feet_contact(self):
        """Publish feet contact data."""
        contacts = self.get_feet_contact()
        # Convert binary array [left, right] to integer
        feet_state = (1 if contacts[0] else 0) | ((1 if contacts[1] else 0) << 1)
        self.feet_pub.publish(Int32(feet_state))
    
    def run(self):
        """Main simulation loop."""
        rate = rospy.Rate(100)  # Match MuJoCo physics rate (250 Hz, was 200 Hz)
        last_report_time = time.time()
        iterations = 0
        
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                
                # Apply control
                control = self.compute_control()
                self.data.ctrl[:] = control                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                self.counter += 1
                iterations += 1
                
                # Publish sensor data
                self.publish_joint_states()
                self.publish_imu_data()
                self.publish_feet_contact()
                
                # Update viewer if available
                if self.viewer:
                    self.viewer.sync()
                
                # Performance reporting
                if time.time() - last_report_time >= 5.0:
                    fps = iterations / (time.time() - last_report_time)
                    rospy.loginfo(f"Simulation running at {fps:.2f} FPS")
                    iterations = 0
                    last_report_time = time.time()
                
                # Control timing to maintain simulation rate
                elapsed = time.time() - start_time
                if elapsed < self.model.opt.timestep:
                    remaining = self.model.opt.timestep - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                else:
                    rospy.logwarn_throttle(1.0, f"Simulation running slower than real-time: {1.0/elapsed:.2f} Hz")
                
                rate.sleep()
                
        finally:
            if self.viewer:
                self.viewer.close()
            rospy.loginfo("MuJoCo simulation stopped")


if __name__ == '__main__':
    try:
        node = MujocoSimNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
