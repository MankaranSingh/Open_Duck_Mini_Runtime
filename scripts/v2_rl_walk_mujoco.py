import os
import time
import numpy as np

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
#from mini_bdx_runtime.eyes import Eyes
#from mini_bdx_runtime.sounds import Sounds
#from mini_bdx_runtime.antennas import Antennas
#from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.common import mini2_constants, dino_constants
from mini_bdx_runtime.common.policies import JoystickPolicy, StandingPolicy, EpisodicPolicy


class RLWalk:
    def __init__(
        self,
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[20, 0, 0],
        robot="dino",
        initial_policy_type="standing"
    ):

        # Control
        self.control_freq = control_freq
        self.pid = pid
        self.constants = eval(f"{robot}_constants")

        self.hwi = HWI(serial_port)
        self.imu = Imu(sampling_freq=int(self.control_freq),)
        self.feet_contacts = FeetContacts()

        # get current script path
        DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../mini_bdx_runtime/data")

        # Define hardcoded paths for models
        self.model_paths = {
            "episodic": f"{DATA_PATH}/{robot}/models/{robot}_checkpoint_episodic_happy_dance.onnx",
            "joystick": f"{DATA_PATH}/{robot}/models/{robot}_checkpoint_joystick.onnx",
            "standing": f"{DATA_PATH}/{robot}/models/{robot}_checkpoint_standing.onnx",
        }
        
        self.reference_paths = {
            "episodic": f"{DATA_PATH}/{robot}/happy_dance.json",
        }
       
        # Initialize all policies 
        print("Loading all policies...")
        self.policies = {
            "joystick": JoystickPolicy(self.constants, self.model_paths["joystick"]),
            "standing": StandingPolicy(self.constants, self.model_paths["standing"]),
            "episodic": EpisodicPolicy(self.constants, self.model_paths["episodic"], self.reference_paths["episodic"])
        }
        
        # Set initial active policy
        self.active_policy_type = initial_policy_type
        self.policy = self.policies[self.active_policy_type]
        print(f"Initial active policy: {self.active_policy_type}")
        
        # Policy switching variables
        self.switch_pending = False
        self.target_policy_type = None
        self.switch_start_time = 0
        self.original_commands = None
        
        # Set decimation for all policies (how often to update)
        self.decimation = 1  # Default to 1 for real robot (no need for decimation)
        for policy in self.policies.values():
            if hasattr(policy, "decimation"):
                policy.decimation = self.decimation

        # Initialize commands
        self.commands = self.policy.get_default_commands()
        self.saved_obs = []

        self.start()
        
        # Expression package
        #self.sounds = Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets/")
        #self.antennas = Antennas()
        #self.eyes = Eyes()
        #self.projector = Projector()

        self.xbox_controller = XBoxController()
        if not self.xbox_controller.wait_for_connection(timeout=60):
            print("Warning: Starting without Bluetooth controller")
            self.use_controller = False
        else:
            print("Bluetooth controller connected, proceeding with initialization")
            self.use_controller = True

    def request_policy_switch(self, new_policy_type):
        """Request a policy switch with specific requirements for each policy"""
        if new_policy_type == self.active_policy_type or self.switch_pending:
            return
            
        print(f"Requesting switch from {self.active_policy_type} to {new_policy_type} policy")
        self.target_policy_type = new_policy_type
        self.switch_pending = True
        
        # Handle standing policy switch specially - need to set commands to 0 and wait
        if self.active_policy_type == "standing":
            print("Standing policy: setting commands to 0 and waiting 0.5s")
            self.original_commands = self.commands.copy()
            self.commands = np.zeros_like(self.commands)
            self.switch_start_time = time.time()
        
    def check_switch_conditions(self):
        """Check if conditions are met to complete the policy switch"""
        if not self.switch_pending:
            return
            
        # For standing policy, we just need to wait 0.5 seconds
        if self.active_policy_type == "standing":
            if time.time() - self.switch_start_time >= 0.5:
                self.complete_policy_switch()
                return
                
        # For joystick policy, we need to check if phase is at 0
        elif self.active_policy_type == "joystick":
            if self.policy.imitation_i == 0:  # Allow small tolerance
                self.complete_policy_switch()
                return
                
        # For episodic policy, we need to check if imitation_i is at 0
        elif self.active_policy_type == "episodic":
            if self.policy.imitation_i == 0:  # Allow small tolerance
                self.complete_policy_switch()
                return
       
    def complete_policy_switch(self):
        """Complete the policy switch once conditions are met"""
        print(f"Switching from {self.active_policy_type} to {self.target_policy_type} policy")
        self.active_policy_type = self.target_policy_type
        self.policy = self.policies[self.target_policy_type]
        self.policy.reset()
        self.commands = self.policy.get_default_commands()
        
        # Reset switching state
        self.switch_pending = False
        self.target_policy_type = None
        self.switch_start_time = 0
        self.original_commands = None

    def get_sensors(self):
        joint_angles = self.hwi.get_present_positions()
        joint_vel = self.hwi.get_present_velocities()  # rad/s
        imu_data = self.imu.get_data()
        accelerometer = imu_data["accel"]
        gyro = imu_data["gyro"]
        contacts = self.feet_contacts.get()

        return joint_angles, joint_vel, accelerometer, gyro, contacts
    
    def process_controller_input(self):
        """Process controller input to update commands"""
        if not self.use_controller or self.switch_pending:
            return
            
        # Get controller stick values
        stick_vals = self.xbox_controller.get_sticks()
        buttons = self.xbox_controller.get_buttons()
        
        # Check for policy switch buttons
        if buttons[0]: 
            self.request_policy_switch("joystick")
        elif buttons[1]: 
            self.request_policy_switch("standing")
        elif buttons[2]:  
            self.request_policy_switch("episodic")
            
        self.commands = self.policy.joystick_to_commands(stick_vals)

    def start(self):
        kps = [self.pid[0]] * len(self.constants.JOINTS_ORDER)
        kds = [self.pid[2]] * len(self.constants.JOINTS_ORDER)
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()
        time.sleep(2)

    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                t = time.time()
                
                # Process controller input
                self.process_controller_input()
                
                # Check if we need to complete a policy switch
                if self.switch_pending:
                    self.check_switch_conditions()
                
                # Get sensor data
                joint_angles, joint_vel, accelerometer, gyro, contacts = self.get_sensors()
                
                # Use the policy to get motor commands
                motor_targets = self.policy.infer(
                    joint_angles, 
                    joint_vel, 
                    accelerometer, 
                    gyro, 
                    contacts,
                    self.commands
                )
                
                # Create joint dictionary for hardware interface
                joint_names = self.constants.JOINTS_ORDER
                action_dict = {joint_names[i]: motor_targets[i] for i in range(len(motor_targets))}
                
                # Send commands to hardware
                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                if (1 / self.control_freq - took) < 0:
                    print("Policy control budget exceeded by", np.around(took - 1 / self.control_freq, 3),)
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=22)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument(
        "--robot",
        type=str,
        default="mini2",
        choices=["mini2", "dino"],
        help="Robot type to use for the simulation.",
    )
    parser.add_argument(
        "--policy_type", 
        type=str, 
        default="standing", 
        choices=["episodic", "joystick", "standing"],
        help="Initial policy to use (episodic, joystick, standing)"
    )

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        control_freq=args.control_freq,
        pid=pid,
        robot=args.robot,
        initial_policy_type=args.policy_type
    )
    rl_walk.run()
