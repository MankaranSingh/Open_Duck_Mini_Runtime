#!/usr/bin/env python3
import time
import numpy as np
import argparse
import os
import yaml
import bluetooth
import RPi.GPIO as GPIO
import onnxruntime as ort
import select
from typing import Dict, List, Any, Optional

# Import hardware interfaces
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
from mini_bdx_runtime.hwi_feetech_pypot import HWI
from mini_bdx_runtime.rl_utils import quat_rotate_inverse

from tf.transformations import (
    quaternion_multiply, 
    quaternion_inverse, 
    quaternion_from_euler
)

class WalkPolicy:
    def __init__(self, config_path=None):
        # Load configuration from file
        self.config = self.load_config(config_path)

        # Joint names and masking (from config)
        self.joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
            "left_knee", "left_ankle", "neck_pitch", "head_pitch", 
            "head_yaw", "head_roll", "left_antenna", "right_antenna",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
        ]
        
        self.mask_joints = self.config.get("mask_joints", [
            'neck_pitch', 'head_pitch', 'head_yaw', "head_roll", "left_antenna", "right_antenna"
        ])
        self.mask_joint_idx = np.array([self.joint_names.index(joint) for joint in self.mask_joints])
        self.enabled_joint_idx = np.array([i for i in range(len(self.joint_names)) if i not in self.mask_joint_idx])
        
        # Initialize policy parameters from config
        self.init_policy_params()
        
        # Setup sensor interfaces
        self.setup_imu()
        self.setup_feet_sensors()
        self.setup_joystick()
        self.setup_hardware_interface()
        
        # Load the ONNX model from config path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, self.config.get("model", {}).get("path", "../assets/policy_low_vel10.onnx"))
        print(f"Loading model from: {model_path}")
        self.model = ort.InferenceSession(model_path)
        
        # Initialize state
        self.cmd_vel = np.zeros(3)  # [linear_x, linear_y, angular_z]
        self.feet_contact = np.zeros(2)  # [left_foot, right_foot]
        self.joint_positions = None
        self.joint_velocities = None
        self.projected_gravity = np.array([0, 0, -1.0])
        self.angular_velocity = np.zeros(3)
        
        # Initialize histories from config
        self.obs_history = np.zeros((self.obs_history_length, self.config["policy_params"]["obs_dim"]))
        self.action_history = np.zeros((self.action_history_length, len(self.enabled_joint_idx)))
        
        # Setup control loop timing
        self.control_rate_hz = self.config.get("control_rate_hz", 50)
        self.control_period = 1.0 / self.control_rate_hz
        
        # Bluetooth buffer and state
        self.bt_buffer = ""
        self.running = False
        
        # Debug settings
        self.log_interval = self.config.get("debug", {}).get("log_interval", 5.0)
        self.verbose = self.config.get("debug", {}).get("verbose", False)
        
        if self.verbose:
            print("Configuration loaded:")
            print(yaml.dump(self.config, default_flow_style=False))

    def init_policy_params(self):
        """Initialize policy parameters from config"""
        policy_params = self.config.get("policy_params", {})
        
        # Policy scaling factors
        self.joint_pos_scale = policy_params.get("joint_pos_scale", 1.0)
        self.joint_vel_scale = policy_params.get("joint_vel_scale", 1.0)
        self.angular_vel_scale_obs = policy_params.get("angular_vel_scale_obs", 1.0)
        self.angular_vel_scale = policy_params.get("angular_vel_scale", 0.25)
        self.power_scale = policy_params.get("power_scale", 1.5)
        
        # History lengths
        self.obs_history_length = policy_params.get("obs_history_length", 3)
        self.action_history_length = policy_params.get("action_history_length", 3)
        self.obs_size = policy_params.get("obs_dim", 40)
        
        # Command velocity limits
        cmd_vel_limits = self.config.get("cmd_vel_limits", {})
        self.lin_vel_x_range = cmd_vel_limits.get("linear_x", [-0.3, 0.4])
        self.lin_vel_y_range = cmd_vel_limits.get("linear_y", [-0.3, 0.3])
        self.yaw_range = cmd_vel_limits.get("angular_z", [-0.5, 0.5])
        
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

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            "control_rate_hz": 50,
            "usb_port": "/dev/ttyACM0",
            "bluetooth": {
                "enabled": True
            },
            "hardware": {
                "enable_motors": True,
                "kp": 32.0,
                "kd": 0.0
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config

    def setup_hardware_interface(self):
        """Initialize hardware interface for motor control"""
        print("Initializing hardware interface...")
        usb_port = self.config.get("usb_port", "/dev/ttyACM0")
        self.hwi = HWI(usb_port=usb_port)
        
        hardware_config = self.config.get("hardware", {})
        self.hwi.turn_on()
        
        # Set KP and KD values from config
        kp_value = int(hardware_config.get("kp", 32.0))
        kd_value = int(hardware_config.get("kd", 0.0))
                
        self.hwi.set_kps(kp_value)
        self.hwi.set_kds(kd_value)
        
        print(f"Motors initialized with KP={kp_value}, KD={kd_value}")            
        # Initialize target positions with hardware's init positions
        self.target_positions = self.hwi.init_pos.copy()

    def setup_imu(self):
        """Initialize IMU sensor"""
        print("Initializing IMU sensor...")
        imu_config = self.config.get("sensors", {}).get("imu", {})
        
        try:
            # Set up I2C and sensor using config
            i2c_bus = imu_config.get("i2c_bus", 3)
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
        feet_config = self.config.get("sensors", {}).get("feet", {})
        
        try:
            # GPIO Pins from config
            self.LEFT_FOOT_PIN = feet_config.get("left_pin", 6)
            self.RIGHT_FOOT_PIN = feet_config.get("right_pin", 5)
            
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
        
        bt_config = self.config.get("bluetooth", {})
        if bt_config.get("enabled", True):
            try:
                self.bt_server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                port = bt_config.get("port", bluetooth.PORT_ANY)
                self.bt_server_sock.bind(("", port))
                self.bt_server_sock.listen(1)
                print(f"Waiting for Bluetooth controller connection on port {port}...")
            except Exception as e:
                print(f"Error initializing Bluetooth: {e}")
                self.bt_server_sock = None
        else:
            print("Bluetooth disabled in config")

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
                        except ValueError:
                            print(f"Invalid joystick data: {line}")
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
        
        self.check_joystick_connection()

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
            if joint_commands is not None and self.config.get("hardware", {}).get("enable_motors", True):
                # Send commands directly as positions, no need for additional conversion
                joint_dict = {}
                for i, name in enumerate(self.joint_names):
                    if name not in self.mask_joints:
                        joint_dict[name] = joint_commands[i]
                
                self.hwi.set_position_all(joint_dict)
            
            # 4. Print diagnostics occasionally based on config log_interval
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
        if hasattr(self, 'hwi') and self.config["hardware"]["enable_motors"]:
            self.hwi.turn_off()
        
        # Cleanup GPIO
        GPIO.cleanup()
        print("Walk policy stopped")

def main():
    parser = argparse.ArgumentParser(description='Run walk policy without ROS')
    parser.add_argument('--config', type=str, help='Path to config YAML file', default="config.yaml")
    args = parser.parse_args()
    
    # Create and initialize walk policy
    policy = WalkPolicy(config_path=args.config)
    
    # Start policy (this will run the main loop)
    policy.start()

if __name__ == "__main__":
    main()
