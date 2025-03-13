#!/usr/bin/env python3
import time
import numpy as np
import argparse
import os
import bluetooth
import RPi.GPIO as GPIO
import onnxruntime as ort
import select
from typing import Dict, List, Any, Optional

# Import hardware interfaces
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.rl_utils import quat_rotate_inverse

from tf.transformations import (
    quaternion_multiply, 
    quaternion_inverse, 
    quaternion_from_euler
)

class WalkPolicy:
    def __init__(self):
        # Joint names and masking (hardcoded)
        self.joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
            "head_yaw", "head_roll", "left_antenna", "right_antenna",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
        ]
        
        self.mask_joints = ['neck_pitch', 'head_pitch', 'head_yaw', "head_roll", "left_antenna", "right_antenna"]
        self.mask_joint_idx = np.array([self.joint_names.index(joint) for joint in self.mask_joints])
        self.enabled_joint_idx = np.array([i for i in range(len(self.joint_names)) if i not in self.mask_joint_idx])
        
        # Initialize policy parameters
        self.init_policy_params()
        
        # Setup sensor interfaces
        self.setup_imu()
        self.setup_feet_sensors()
        self.setup_joystick()
        self.setup_hardware_interface()
        
        # Load the ONNX model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../assets/policy_low_vel12.onnx")
        print(f"Loading model from: {model_path}")
        self.model = ort.InferenceSession(model_path)
        
        # Initialize state
        self.cmd_vel = np.zeros(3)  # [linear_x, linear_y, angular_z]
        self.feet_contact = np.zeros(2)  # [left_foot, right_foot]
        self.joint_positions = None
        self.joint_velocities = None
        self.projected_gravity = np.array([0, 0, -1.0])
        self.angular_velocity = np.zeros(3)
        
        # Initialize histories
        self.obs_history = np.zeros((self.obs_history_length, self.obs_size))
        self.action_history = np.zeros((self.action_history_length, len(self.enabled_joint_idx)))
        
        # Setup control loop timing
        self.control_rate_hz = 50
        self.control_period = 1.0 / self.control_rate_hz
        
        # Bluetooth buffer and state
        self.bt_buffer = ""
        self.running = False
        
        # Debug settings
        self.log_interval = 5.0
        self.verbose = True
        self.enable_motors = True
        
        if self.verbose:
            print("WalkPolicy initialized with hardcoded values")

    def init_policy_params(self):
        """Initialize policy parameters with hardcoded values"""
        # Policy scaling factors
        self.joint_pos_scale = 1.0
        self.joint_vel_scale = 1.0
        self.angular_vel_scale_obs = 1.0
        self.angular_vel_scale = 1.0
        self.power_scale = 1.0
        
        # History lengths
        self.obs_history_length = 3
        self.action_history_length = 3
        self.obs_size = 40
        
        # Command velocity limits
        self.lin_vel_x_range = [-0.3, 0.4]
        self.lin_vel_y_range = [-0.3, 0.3]
        self.yaw_range = [-0.7, 0.7]
        
        # Clipping params
        self.action_clip = [-1.5, 1.5]
        self.obs_clip = [-5.0, 5.0]
        
        # Initialize default position
        self.init_pos = np.array([
            0.002, 0.053, -0.63, 1.368, -0.784, 
            0.0, 0, 0, 0, 0, 0, 
            -0.003, -0.065, 0.635, 1.379, -0.796,
        ])
        self.target_joint_states = np.zeros(len(self.joint_names))

    def setup_hardware_interface(self):
        """Initialize hardware interface for motor control"""
        print("Initializing hardware interface...")
        usb_port = "/dev/ttyACM0"
        self.hwi = HWI(usb_port=usb_port)
        
        self.hwi.turn_on()
        
        # Hardcoded KP and KD values
        kp_value = 32.0
        kd_value = 0.0
                        
        print(f"Motors initialized with KP={kp_value}, KD={kd_value}")            
        # Initialize target positions with hardware's init positions
        self.target_positions = self.hwi.init_pos.copy()

    def setup_imu(self):
        """Initialize IMU sensor"""
        print("Initializing IMU sensor...")
        
        try:
            # Hardcoded I2C bus
            i2c_bus = 3
            self.i2c = I2C(i2c_bus)
            self.imu_sensor = adafruit_bno055.BNO055_I2C(self.i2c)
            self.imu_sensor.mode = adafruit_bno055.CONFIG_MODE
            time.sleep(0.2)

            # Set axis remap
            self.imu_sensor.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,        # X (Forward) now maps to physical Y
                adafruit_bno055.AXIS_REMAP_X,        # Y (Right) now maps to physical X
                adafruit_bno055.AXIS_REMAP_Z,        # Z (Up) remains Z
                adafruit_bno055.AXIS_REMAP_POSITIVE, # X (new) keeps positive
                adafruit_bno055.AXIS_REMAP_NEGATIVE, # Y (new) must be inverted
                adafruit_bno055.AXIS_REMAP_POSITIVE  # Z (new) keeps positive
            )
            time.sleep(0.2)

            self.imu_sensor.mode = adafruit_bno055.IMUPLUS_MODE
            time.sleep(0.2)
            
            print(f"IMU initialized successfully on I2C bus {i2c_bus}")
        except Exception as e:
            print(f"Error initializing IMU: {e}")
            self.imu_sensor = None

    def setup_feet_sensors(self):
        """Initialize foot contact sensors"""
        print("Initializing foot sensors...")
        
        try:
            # Hardcoded GPIO pins
            self.LEFT_FOOT_PIN = 6
            self.RIGHT_FOOT_PIN = 5
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.LEFT_FOOT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.RIGHT_FOOT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print(f"Foot sensors initialized: Left pin={self.LEFT_FOOT_PIN}, Right pin={self.RIGHT_FOOT_PIN}")
        except Exception as e:
            print(f"Error initializing foot sensors: {e}")

    def setup_joystick(self):
        """Initialize Bluetooth joystick connection"""
        print("Initializing Bluetooth joystick...")
        self.bt_client_sock = None
        self.bt_server_sock = None
        
        # Hardcoded Bluetooth setting (enabled=True)
        bluetooth_enabled = True
        
        if bluetooth_enabled:
            try:
                self.bt_server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                port = 1  # Hardcoded RFCOMM port
                self.bt_server_sock.bind(("", port))
                self.bt_server_sock.listen(1)
                print(f"Waiting for Bluetooth controller connection on port {port}...")
            except Exception as e:
                print(f"Error initializing Bluetooth: {e}")
                self.bt_server_sock = None
        else:
            print("Bluetooth disabled")

    def wait_for_joystick_connection(self):
        """Wait for joystick connection (blocking)"""
        if self.bt_server_sock and not self.bt_client_sock:
            print("Waiting for Bluetooth controller to connect... (Press Ctrl+C to skip)")
            try:
                # Make this a blocking call with a timeout
                self.bt_server_sock.settimeout(None)  # Block indefinitely
                self.bt_client_sock, address = self.bt_server_sock.accept()
                print(f"Accepted Bluetooth connection from {address}")
                # Initialize cmd_vel to ensure it's not zero
                self.cmd_vel = np.zeros(3)
                return True
            except KeyboardInterrupt:
                print("Skipping Bluetooth connection wait")
                return False
            except Exception as e:
                print(f"Error accepting Bluetooth connection: {e}")
                return False
        return False

    def read_imu_data(self):
        """Read data from IMU sensor"""
        # Get sensor quaternion (w, x, y, z format)
        qw, qx, qy, qz = self.imu_sensor.quaternion
        # Convert to (x, y, z, w) format
        q_final = [qx, qy, qz, qw]
        
        # Extract gravity vector using the quaternion
        self.projected_gravity = quat_rotate_inverse(q_final, [0, 0, -1.0])            
        self.angular_velocity = np.array(self.imu_sensor.gyro)

    def read_feet_contact(self):
        """Read data from foot contact sensors"""
        left_state = GPIO.input(self.LEFT_FOOT_PIN) == GPIO.LOW  # True if pressed
        right_state = GPIO.input(self.RIGHT_FOOT_PIN) == GPIO.LOW  # True if pressed
        self.feet_contact = np.array([left_state, right_state])

    def check_joystick_connection(self):
        """Check for new Bluetooth connection (non-blocking)"""
        if self.bt_server_sock and not self.bt_client_sock:
            ready, _, _ = select.select([self.bt_server_sock], [], [], 0)
            if ready:
                try:
                    self.bt_client_sock, address = self.bt_server_sock.accept()
                    print(f"Accepted Bluetooth connection from {address}")
                except Exception as e:
                    print(f"Error accepting Bluetooth connection: {e}")

    def read_joystick_data(self):
        """Read data from Bluetooth joystick (non-blocking)"""
        if not self.bt_client_sock:
            return
        try:
            ready, _, _ = select.select([self.bt_client_sock], [], [], 0)
            if ready:
                data = self.bt_client_sock.recv(1024).decode("utf-8")
                if not data:
                    # Connection closed
                    print("Bluetooth connection closed")
                    self.bt_client_sock.close()
                    self.bt_client_sock = None
                    return
                
                self.bt_buffer += data  # Append new data to buffer
                
                while "\n" in self.bt_buffer:
                    line, self.bt_buffer = self.bt_buffer.split("\n", 1)
                    line = line.strip()
                    
                    if line:
                        try:
                            x, y, yaw = map(float, line.split(","))
                            self.cmd_vel = self.scale_cmd_vel(x, y, yaw)
                        except ValueError as e:
                            print(f"Invalid joystick data format: {line} - {e}")
                        except Exception as e:
                            print(f"Error processing joystick data: {e}")
        except Exception as e:
            if "Resource temporarily unavailable" not in str(e):  # Ignore EAGAIN errors
                print(f"Error reading joystick data: {e}")

    def scale_cmd_vel(self, linear_x, linear_y, angular_z):
        """Scale command velocities to their respective ranges"""
        lin_vel_x = np.clip(linear_x, -1, 1) * (self.lin_vel_x_range[1] - self.lin_vel_x_range[0]) / 2 + (self.lin_vel_x_range[1] + self.lin_vel_x_range[0]) / 2
        lin_vel_y = np.clip(linear_y, -1, 1) * (self.lin_vel_y_range[1] - self.lin_vel_y_range[0]) / 2 + (self.lin_vel_y_range[1] + self.lin_vel_y_range[0]) / 2
        yaw = np.clip(angular_z, -1, 1) * (self.yaw_range[1] - self.yaw_range[0]) / 2 + (self.yaw_range[1] + self.yaw_range[0]) / 2
        return np.array([lin_vel_x, lin_vel_y, yaw * self.angular_vel_scale])

    def read_joint_states(self):
        """Read current joint positions and velocities from hardware"""
        try:
            positions = self.hwi.get_present_positions()
            velocities = self.hwi.get_present_velocities()
            
            # Add dummy joints
            positions_list = positions.tolist() if hasattr(positions, "tolist") else list(positions)
            velocities_list = velocities.tolist() if hasattr(velocities, "tolist") else list(velocities)
            
            # Insert dummy joint values at the appropriate position
            dummy_joint_insert_idx = 9
            dummy_joint_values = [0.0, 0.0]
            positions_list[dummy_joint_insert_idx:dummy_joint_insert_idx] = dummy_joint_values
            velocities_list[dummy_joint_insert_idx:dummy_joint_insert_idx] = dummy_joint_values
            
            self.joint_positions = np.array(positions_list)
            self.joint_velocities = np.array(velocities_list)
            return True
        except Exception as e:
            print(f"Error reading joint states: {e}")
            return False

    def run_policy(self):
        """Run the policy model and compute joint commands"""
        if any(x is None for x in [self.joint_positions, self.joint_velocities]):
            return None
            
        # Prepare the input for the model
        obs = np.concatenate([
                self.projected_gravity,
                (self.joint_positions-self.init_pos) * self.joint_pos_scale,
                self.joint_velocities * self.joint_vel_scale,
                self.angular_velocity * self.angular_vel_scale_obs,
                self.feet_contact
            ])
        
        # Update observation history
        self.obs_history[1:, :] = self.obs_history[:-1, :].copy()
        self.obs_history[0, :len(obs)] = obs
        
        # Combine with action history
        input_data = np.concatenate([self.obs_history.flatten(), self.action_history.flatten(), self.cmd_vel]).reshape(1, -1)
        
        # Clip observations
        input_data = np.clip(input_data, self.obs_clip[0], self.obs_clip[1])
        
        # Run the model
        outputs = self.model.run(None, {'obs': input_data.astype(np.float32)})
        
        # Extract and process actions
        actions = outputs[0].flatten()
        
        # Clip actions
        actions = np.clip(actions, self.action_clip[0], self.action_clip[1])
        
        # Update action history
        self.action_history[1:, :] = self.action_history[:-1, :].copy()
        self.action_history[0, :] = actions.copy()
        
        self.target_joint_states[:] = 0.0
        self.target_joint_states[self.enabled_joint_idx] = actions*self.power_scale
        self.target_joint_states += self.init_pos   

        return self.target_joint_states
    
    def run(self):
        """Main control loop running at 50Hz"""
        print("Starting walk policy main loop")
        self.running = True
        
        # Wait until initial joint states are read
        while not self.read_joint_states() and self.running:
            print("Waiting for joint states...")
            time.sleep(0.5)
        
        # Wait for joystick connection (blocking)
        if self.bt_server_sock:
            self.wait_for_joystick_connection()

        input("Press Enter to start the control loop...")
        
        last_time = time.time()
        log_time = time.time()
    
        while self.running:
            loop_start_time = time.time()
            
            # 1. Read all sensor data
            self.read_imu_data()
            self.read_feet_contact()        
            self.read_joystick_data()
            self.read_joint_states()
            
            # 2. Run policy computation
            joint_commands = self.run_policy()
            
            # 3. Send commands to hardware
            if joint_commands is not None:
                if self.enable_motors:
                    # Send commands directly as positions, no need for additional conversion
                    joint_dict = {}
                    for i, name in enumerate(self.joint_names):
                        if name in ["left_antenna", "right_antenna"]:
                                continue
                        joint_dict[name] = joint_commands[i]
                    
                    self.hwi.set_position_all(joint_dict)
            
            # 4. Print diagnostics occasionally
            current_time = time.time()
            if current_time - log_time >= self.log_interval:
                loop_rate = 1.0 / (current_time - last_time) if current_time > last_time else 0
                print(f"Control loop running at {loop_rate:.2f} Hz")
                
                if self.verbose:
                    print(f"Projected gravity: {self.projected_gravity}")
                    print(f"Angular velocity: {self.angular_velocity}")
                    print(f"cmd_vel: {self.cmd_vel}")
                    print(f"Feet contact: {self.feet_contact}")
                
                log_time = current_time
            
            # 5. Sleep to maintain control rate
            last_time = current_time
            elapsed = time.time() - loop_start_time
            sleep_time = self.control_period - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print("Control loop lagging by {:.2f} ms".format(-sleep_time*1000))
            
        self.stop()

    def start(self):
        """Start the control loop"""
        self.run()
    
    def stop(self):
        """Stop and clean up"""
        self.running = False
        
        # Close connections
        if self.bt_client_sock:
            self.bt_client_sock.close()
        if self.bt_server_sock:
            self.bt_server_sock.close()
        
        # Turn off motors
        if hasattr(self, 'hwi'):
            # Hardcoded motor enabling (True)
            enable_motors = True
            if enable_motors:
                self.hwi.turn_off()
        
        # Cleanup GPIO
        GPIO.cleanup()
        print("Walk policy stopped")

def main():
    # Create and initialize walk policy with hardcoded values
    policy = WalkPolicy()
    
    # Start policy (this will run the main loop)
    policy.start()

if __name__ == "__main__":
    main()
