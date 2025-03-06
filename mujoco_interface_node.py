#!/usr/bin/env python3

import rospy
import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import os
from threading import Thread

# ROS message types
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

class MujocoInterfaceNode:
    """MuJoCo simulation interface for walk_policy_node."""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('mujoco_interface_node')
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default="ros/src/bdx_runtime/robots/open_duck_mini_v2/scene.xml",
                           help="Path to the MuJoCo XML model")
        parser.add_argument("--zero_head", action="store_true", default=False,
                           help="Set head joints to zero")
        parser.add_argument("--headless", action="store_true", default=False,
                           help="Run without visualization")
        args, unknown = parser.parse_known_args()
        
        self.model_path = os.path.join(os.getcwd(), args.model_path)
        self.zero_head = args.zero_head
        self.headless = args.headless
        
        # Initialize MuJoCo model
        self.setup_mujoco()
        
        # Initialize joint state mappings
        self.setup_joint_mappings()
        
        # Initialize ROS publishers
        self.joint_pub = rospy.Publisher('/current_joint_states', JointState, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=1)
        self.feet_pub = rospy.Publisher('/feet_switch', Int32, queue_size=1)
        
        # Initialize ROS subscribers
        self.target_sub = rospy.Subscriber('/target_joint_states', JointState, self.target_callback)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # Control variables
        self.target_positions = self.init_pos.copy()
        self.cmd_vel = np.zeros(3)  # [lin_vel_x, lin_vel_y, ang_vel]
        
        # PD control parameters
        self.kps = np.array([6.55] * 16)
        self.kds = np.array([0.65] * 16)
        self.action_scale = 1.0
        
        # Simulation control
        self.control_decimation = 4  # Run control at slower rate than physics
        self.counter = 0
        self.running = False
        self.sim_thread = None
        self.viewer = None
        
        rospy.loginfo("MuJoCo interface initialized. Model: %s", self.model_path)
    
    def setup_mujoco(self):
        """Setup the MuJoCo model and data."""
        rospy.loginfo(f"Loading MuJoCo model from: {self.model_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.model.opt.timestep = 0.005  # 200 Hz physics
            self.data = mujoco.MjData(self.model)
            
            # Initial pose
            self.init_pos = np.array([
                0.002, 0.053, -0.63, 1.368, -0.784, 
                0.0, 0, 0, 0, 0, 0, 
                -0.003, -0.065, 0.635, 1.379, -0.796,
            ])
            
            # Set initial state
            self.data.qpos[3:7] = [1, 0, 0.0, 0]  # Unit quaternion for orientation
            self.data.qpos[7:23] = self.init_pos
            
            mujoco.mj_forward(self.model, self.data)
            rospy.loginfo("MuJoCo model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load MuJoCo model: {e}")
            raise
    
    def setup_joint_mappings(self):
        """Setup joint name mappings to match the walk_policy_node."""
        self.joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
            "head_yaw", "head_roll", "left_antenna", "right_antenna",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
        ]
        
        self.mask_joints = ['neck_pitch', 'head_pitch', 'head_yaw', "head_roll", "left_antenna", "right_antenna"]
        self.mask_joint_idx = np.array([self.joint_names.index(joint) for joint in self.mask_joints])
    
    def target_callback(self, msg):
        """Handle incoming target joint positions from walk_policy_node."""
        for i, name in enumerate(msg.name):
            if i < len(self.target_positions):  # Ensure we don't go out of bounds
                self.target_positions[i] = msg.position[i]
    
    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands."""
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z
    
    def pd_control(self):
        """PD controller for joint positions."""
        tau = (self.target_positions - self.data.qpos[7:23]) * self.kps
        tau -= self.data.qvel[6:22] * self.kds
        
        return tau
    
    def check_contact(self, geom1, geom2):
        """Check if two geoms are in contact."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (self.model.geom_id2name(contact.geom1) == geom1 and 
                self.model.geom_id2name(contact.geom2) == geom2) or \
               (self.model.geom_id2name(contact.geom2) == geom1 and 
                self.model.geom_id2name(contact.geom1) == geom2):
                return True
        return False
    
    def get_feet_contact(self):
        """Check foot contact with the floor."""
        left_contact = self.check_contact("foot_assembly", "floor")
        right_contact = self.check_contact("foot_assembly_2", "floor")
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
        quat = self.data.qpos[3:7].copy()  # [w, x, y, z]
        msg.orientation.x = quat[1]
        msg.orientation.y = quat[2]
        msg.orientation.z = quat[3]
        msg.orientation.w = quat[0]
        
        # Copy angular velocity
        msg.angular_velocity.x = self.data.qvel[3]
        msg.angular_velocity.y = self.data.qvel[4]
        msg.angular_velocity.z = self.data.qvel[5]
        
        # Set acceleration (in this simulation, gravity is -z)
        # Transform from global to robot frame using the quaternion
        gravity = self.quat_rotate_inverse([quat[1], quat[2], quat[3], quat[0]], [0, 0, -9.81])
        msg.linear_acceleration.x = gravity[0]
        msg.linear_acceleration.y = gravity[1]
        msg.linear_acceleration.z = gravity[2]
        
        self.imu_pub.publish(msg)
    
    def publish_feet_contact(self):
        """Publish feet contact data."""
        contacts = self.get_feet_contact()
        # Convert binary array [left, right] to integer
        feet_state = (1 if contacts[0] else 0) | ((1 if contacts[1] else 0) << 1)
        self.feet_pub.publish(Int32(feet_state))
    
    def run_simulation(self):
        """Main simulation loop."""
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=True
            )
        
        self.running = True
        last_time = time.time()
        
        try:
            while self.running and not rospy.is_shutdown():
                # Measure loop time
                start_time = time.time()
                
                # Apply control
                tau = self.pd_control()
                self.data.ctrl[:] = tau
                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                self.counter += 1
                
                # Publish state at a rate determined by control_decimation
                if self.counter % self.control_decimation == 0:
                    self.publish_joint_states()
                    self.publish_imu_data()
                    self.publish_feet_contact()
                
                # Update viewer if available
                if not self.headless and self.viewer:
                    self.viewer.sync()
                
                # Control timing to maintain simulation rate
                elapsed = time.time() - start_time
                sleep_time = max(0, self.model.opt.timestep - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Log performance occasionally
                if time.time() - last_time > 5.0:
                    fps = self.counter / (time.time() - last_time)
                    rospy.loginfo(f"Simulation running at {fps:.2f} FPS")
                    self.counter = 0
                    last_time = time.time()
                
        finally:
            if not self.headless and self.viewer:
                self.viewer.close()
            rospy.loginfo("MuJoCo simulation stopped")
    
    def start(self):
        """Start the simulation thread."""
        self.sim_thread = Thread(target=self.run_simulation)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        
        rospy.spin()  # Keep the ROS node alive
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)
        rospy.loginfo("MuJoCo interface shutdown complete")


if __name__ == '__main__':
    try:
        node = MujocoInterfaceNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Ensure cleanup on exit
        if hasattr(node, 'stop'):
            node.stop()
