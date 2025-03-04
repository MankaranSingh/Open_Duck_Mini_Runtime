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
        
        # Initialize policy parameters from config
        self.init_policy_params()
        
        # Setup sensor interfaces
        self.setup_imu()
        self.setup_feet_sensors()
        self.setup_joystick()
        self.setup_hardware_interface()
        
        # Load the ONNX model from config path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, self.config.get("model", {}).get("path", "../assets/policy.onnx"))
        print(f"Loading model from: {model_path}")
        self.model = ort.InferenceSession(model_path)
        
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
        
        # Initialize state
        self.cmd_vel = np.zeros(3)  # [linear_x, linear_y, angular_z]
        self.feet_contact = np.zeros(2)  # [left_foot, right_foot]
        self.joint_positions = None
        self.joint_velocities = None
        self.projected_gravity = np.array([0, 0, -1.0])
        self.angular_velocity = np.zeros(3)
        
        # Initialize histories from config
        self.obs_history = np.zeros((self.obs_history_length, self.config["policy_params"]["obs_dim"]))
        self.action_history = np.zeros((self.action_history_length, len(self.joint_names)))
        
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
        self.angular_vel_scale = policy_params.get("angular_vel_scale", 0.25)
        self.power_scale = policy_params.get("power_scale", 1.5)
        
        # History lengths
        self.obs_history_length = policy_params.get("obs_history_length", 2)
        self.action_history_length = policy_params.get("action_history_length", 2)
        
        # Command velocity limits
        cmd_vel_limits = self.config.get("cmd_vel_limits", {})
        self.lin_vel_x_range = cmd_vel_limits.get("linear_x", [-0.3, 0.3])
        self.lin_vel_y_range = cmd_vel_limits.get("linear_y", [-0.3, 0.3])
        self.yaw_range = cmd_vel_limits.get("angular_z", [-0.3, 0.3])

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
        kp_value = hardware_config.get("kp", 32.0)
        kd_value = hardware_config.get("kd", 0.0)
        
        kps_list = [kp_value] * len(self.hwi.joints)
        kds_list = [kd_value] * len(self.hwi.joints)
        
        # Uncomment if your hardware supports setting these values
        # self.hwi.set_kps(kps_list)
        # self.hwi.set_kds(kds_list)
        
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
            self.imu_sensor.mode = adafruit_bno055.IMUPLUS_MODE
            
            # Define fixed rotation for axis re-mapping: -90° about z
            self.q_fixed = [0.0, 0.0, -0.7071, 0.7071]
            self.R_z = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            
            # Hard-coded pitch offset correction from config
            self.pitch_offset = imu_config.get("pitch_offset", 0.13)
            self.q_pitch_corr = quaternion_from_euler(0, -self.pitch_offset, 0)
            self.R_pitch = np.array([
                [ np.cos(-self.pitch_offset), 0, np.sin(-self.pitch_offset)],
                [ 0,                         1,                           0],
                [-np.sin(-self.pitch_offset), 0, np.cos(-self.pitch_offset)]
            ])
            
            # Combine the rotation matrices
            self.R_total = self.R_pitch @ self.R_z
            print(f"IMU initialized successfully on I2C bus {i2c_bus} with pitch offset {self.pitch_offset}")
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
        try:
            if self.imu_sensor:
                # Get sensor quaternion (w, x, y, z format)
                qw, qx, qy, qz = self.imu_sensor.quaternion
                # Convert to (x, y, z, w) format
                q_sensor_raw = [qx, qy, qz, qw]
                
                # Apply fixed rotation
                q_corr = quaternion_multiply(self.q_fixed, q_sensor_raw)
                q_corr = quaternion_multiply(q_corr, quaternion_inverse(self.q_fixed))
                
                # Apply pitch correction
                q_final = quaternion_multiply(self.q_pitch_corr, q_corr)
                
                # Extract gravity vector using the quaternion
                self.projected_gravity = quat_rotate_inverse(q_final, [0, 0, -1.0])
                
                # Get gyro data and apply rotation
                gyro = np.dot(self.R_total, self.imu_sensor.gyro)
                self.angular_velocity = gyro
        except Exception as e:
            print(f"Error reading IMU: {e}")

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
                    self.bt_client_sock.setblocking(0)  # Make socket non-blocking
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
            self.joint_positions * self.joint_pos_scale,
            self.joint_velocities * self.joint_vel_scale,
        ])
        
        # Update observation history
        self.obs_history[1:, :] = self.obs_history[:-1, :].copy()
        self.obs_history[0, :len(obs)] = obs
        
        # Combine observation and action histories
        combined = np.concatenate([self.obs_history, self.action_history], axis=-1)
        
        # Add command velocity
        if self.cmd_vel is None:
            self.cmd_vel = np.zeros(3)
        input_data = np.concatenate([combined.flatten(), self.cmd_vel]).reshape(1, -1)
        
        # Run the model
        outputs = self.model.run(None, {'obs': input_data.astype(np.float32)})
        # Extract actions
        actions = outputs[0].flatten()
        # Mask specified joints
        actions[self.mask_joint_idx] = 0.0
        
        # Update action history
        self.action_history[1:, :] = self.action_history[:-1, :].copy()
        self.action_history[0, :] = actions
        
        # Scale and return actions
        return actions * self.power_scale
    

    def run(self):
        """Main control loop running at 50Hz"""
        print("Starting walk policy main loop")
        self.running = True
        
        # Wait until initial joint states are read
        while not self.read_joint_states() and self.running:
            print("Waiting for joint states...")
            time.sleep(0.5)
        
        last_time = time.time()
        log_time = time.time()
    
        while self.running:
            loop_start_time = time.time()
            
            # 1. Read all sensor data
            self.read_imu_data()
            self.read_feet_contact()
            self.check_joystick_connection()
            self.read_joystick_data()
            self.read_joint_states()
            
            # 2. Run policy computation
            joint_commands = self.run_policy()
            
            # 3. Send commands to hardware
            if joint_commands is not None and self.config.get("hardware", {}).get("enable_motors", True):
                # Extract commands for actual joints
                command_dict = {}
                for i, name in enumerate(self.joint_names):
                    if name not in self.mask_joints and i < len(joint_commands):
                        command_dict[name] = joint_commands[i]
                
                # Send commands to hardware
                self.hwi.set_position_all(command_dict)
            
            # 4. Print diagnostics occasionally based on config log_interval
            current_time = time.time()
            if current_time - log_time >= self.log_interval:
                loop_rate = 1.0 / (current_time - last_time) if current_time > last_time else 0
                print(f"Control loop running at {loop_rate:.2f} Hz")
                print(f"Command velocity: {self.cmd_vel}")
                print(f"Feet contact: {self.feet_contact}")
                
                if self.verbose:
                    print(f"Projected gravity: {self.projected_gravity}")
                    print(f"Angular velocity: {self.angular_velocity}")
                
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
