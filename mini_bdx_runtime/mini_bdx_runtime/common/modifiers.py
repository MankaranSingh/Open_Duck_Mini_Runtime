import time
import random
import math

# Debug flag to control print statements
DEBUG = False

class EpisodicPolicyModifier:
    """Modifier for Episodic Policy - passes through original motor commands"""
    
    def __init__(self, constants):
        self.constants = constants
    
    def modify(self, motor_commands, joint_pos=None, joint_vel=None, accel=None, 
               gyro=None, contacts=None, commands=None, gravity=None, timestamp=None):
        """Simply pass through the original motor commands"""
        return motor_commands


class JoystickPolicyModifier:
    """Modifier for Joystick Policy - passes through original motor commands"""
    
    def __init__(self, constants):
        self.constants = constants
    
    def modify(self, motor_commands, joint_pos=None, joint_vel=None, accel=None, 
               gyro=None, contacts=None, commands=None, gravity=None, timestamp=None):
        """Simply pass through the original motor commands"""
        return motor_commands


class StandingPolicyModifier:
    """Modifier for Standing Policy - adds tail wiggling and head pose changes"""
    
    def __init__(self, constants):
        self.constants = constants
        
        # Find indices of tail joints
        self.tail_joint_indices = []
        for i, joint_name in enumerate(constants.JOINTS_ORDER):
            if 'tail' in joint_name.lower():
                self.tail_joint_indices.append(i)
        
        # Find indices of head joints (neck, head_yaw, head_pitch)
        self.head_joint_indices = []
        for i, joint_name in enumerate(constants.JOINTS_ORDER):
            if any(head_part in joint_name.lower() for head_part in ['neck', 'head_yaw', 'head_pitch']):
                self.head_joint_indices.append(i)
        
        # Find indices of leg/foot joints
        self.leg_joint_indices = []
        for i, joint_name in enumerate(constants.JOINTS_ORDER):
            if any(leg_part in joint_name.lower() for leg_part in ['hip', 'knee', 'ankle']):
                self.leg_joint_indices.append(i)
        
        # Timing variables
        self.last_tail_wiggle_time = time.time()
        self.next_tail_wiggle_interval = self._random_interval(5, 10)
        
        self.last_head_pose_time = time.time()
        self.next_head_pose_interval = self._random_interval(10, 12)
        
        # Contact tracking
        self.feet_off_ground_time = None
        self.emergency_mode = False
        self.emergency_intensity = 2.5  # Multiplier for movement intensity
        
        # Animation state - these will be randomized each wiggle
        self.wiggle_duration = 2.0  # seconds
        self.wiggle_start_time = None
        self.wiggle_amplitude = 0.2
        self.wiggle_freq = 2.0  # Hz
        
        # Current head pose - start with neutral
        self.current_head_pose = self._generate_neutral_head_pose()
        
        # Head jitter parameters (for emergency mode)
        self.head_jitter_start_time = None
        self.head_jitter_duration = 2.0
        self.head_jitter_amplitude = 0.15
        self.head_jitter_freq = 2.5  # Hz
        
    def _random_interval(self, min_sec, max_sec):
        """Generate random time interval in seconds"""
        return random.uniform(min_sec, max_sec)
    
    def _random_tail_pose(self):
        """Generate random tail wiggle pattern"""
        return [random.uniform(-self.wiggle_amplitude, self.wiggle_amplitude) 
                for _ in range(len(self.tail_joint_indices))]
    
    def _random_head_pose(self):
        """Generate random head pose sampling around the default position at 0.0"""
        random_pose = []
        
        for idx in self.head_joint_indices:
            joint_name = self.constants.JOINTS_ORDER[idx]
            min_limit, max_limit = self.constants.JOINT_LIMITS[joint_name]
            limit_range = max_limit - min_limit
            
            # Middle is at 0.0, not the average of joint limits
            middle = 0.0
            
            # Apply different percentage based on joint type
            if 'yaw' in joint_name:
                # 10% of the range for head_yaw, centered at 0.0
                scaled_min = middle - 0.05 * limit_range
                scaled_max = middle + 0.05 * limit_range
            else:
                # 4% of the range for neck_pitch and head_pitch, centered at 0.0
                scaled_min = middle - 0.02 * limit_range
                scaled_max = middle + 0.02 * limit_range
            
            # Sample within the scaled limits
            random_pose.append(random.uniform(scaled_min, scaled_max))
        return random_pose
    
    def _generate_neutral_head_pose(self):
        """Generate a neutral pose at 0.0 with small jitter"""
        neutral_pose = []
        for _ in self.head_joint_indices:
            # Add very small jitter around 0.0
            jitter = random.uniform(-0.02, 0.02)
            neutral_pose.append(jitter)
        return neutral_pose
    
    def _randomize_wiggle_params(self):
        """Randomize wiggle parameters for more natural movement"""
        self.wiggle_duration = random.uniform(1.0, 3.0)  # 1.5-3.0 seconds
        self.wiggle_amplitude = random.uniform(0.1, 0.3)  # 0.1-0.3 radians
        self.wiggle_freq = random.uniform(1.0, 2.0)  # 1.5-3.0 Hz
    
    def _randomize_jitter_params(self):
        """Randomize jitter parameters for more natural head movement in emergency mode"""
        self.head_jitter_duration = random.uniform(1.0, 2.0)  # 1-2 seconds
        self.head_jitter_amplitude = random.uniform(0.1, 0.15)  # 0.1-0.2 radians
        self.head_jitter_freq = random.uniform(2.0, 3.0)  # 2-3 Hz
    
    def modify(self, motor_commands, joint_pos=None, joint_vel=None, accel=None, 
               gyro=None, contacts=None, commands=None, gravity=None, timestamp=None):
        """Modify motor commands to add tail wiggling and head pose changes"""
        current_time = time.time() if timestamp is None else timestamp
        modified_commands = motor_commands.copy()
        
        # Check for feet contact status first
        if contacts is not None:
            # Assuming the first two contact sensors are for the feet
            feet_contacts = contacts[:2] if len(contacts) >= 2 else []
            both_feet_off_ground = all(contact == 0 for contact in feet_contacts)
            
            if both_feet_off_ground:
                # If this is the first time both feet are off ground, record the time
                if self.feet_off_ground_time is None:
                    self.feet_off_ground_time = current_time
                    
                # Check if feet have been off ground for more than 0.5 seconds
                elif current_time - self.feet_off_ground_time > 0.5 and not self.emergency_mode:
                    self.emergency_mode = True
                    # Force immediate tail wiggle and head jitter
                    self.wiggle_start_time = current_time
                    self.head_jitter_start_time = current_time
                    self._randomize_jitter_params()  # Initialize jitter parameters
                    self.last_tail_wiggle_time = current_time - 100  # Ensure it starts immediately
                    if DEBUG:
                        print("No ground contact detected - starting emergency head and tail movements")
            else:
                # Reset tracking when feet make contact
                if self.emergency_mode:
                    if DEBUG:
                        print("Ground contact reestablished - returning to normal movements")
                    self.emergency_mode = False
                    self.head_jitter_start_time = None
                self.feet_off_ground_time = None
        
        # Apply intensity modifier based on emergency mode
        movement_intensity = self.emergency_intensity if self.emergency_mode else 1.0
        
        # Check if it's time to start a new tail wiggle
        if (current_time - self.last_tail_wiggle_time > (self.next_tail_wiggle_interval / movement_intensity) and 
            self.wiggle_start_time is None):
            self.wiggle_start_time = current_time
            self.last_tail_wiggle_time = current_time
            self.next_tail_wiggle_interval = self._random_interval(3, 8) / movement_intensity
            # Randomize wiggle parameters for this new wiggle
            self._randomize_wiggle_params()
            wiggle_msg = "(EMERGENCY)" if self.emergency_mode else ""
            if DEBUG:
                print(f"Starting tail wiggle {wiggle_msg} (duration: {self.wiggle_duration:.2f}s, amplitude: {self.wiggle_amplitude:.2f}, freq: {self.wiggle_freq:.2f}Hz)")
        
        # Animate tail wiggle if active
        if self.wiggle_start_time is not None:
            wiggle_elapsed = current_time - self.wiggle_start_time
            actual_wiggle_duration = self.wiggle_duration / movement_intensity
            
            if wiggle_elapsed <= actual_wiggle_duration:
                # Apply intensity modifier to amplitude and frequency
                actual_amplitude = self.wiggle_amplitude * movement_intensity
                actual_freq = self.wiggle_freq * movement_intensity
                
                # Generate sine wave for tail with increased amplitude/frequency in emergency mode
                for i, idx in enumerate(self.tail_joint_indices):
                    phase_offset = i * math.pi / len(self.tail_joint_indices)
                    modified_commands[idx] += actual_amplitude * math.sin(
                        2 * math.pi * actual_freq * wiggle_elapsed + phase_offset
                    )
            else:
                # Wiggle completed
                self.wiggle_start_time = None
                
        # Handle head movement differently based on mode
        if self.emergency_mode:
            # In emergency mode, jitter the head like the tail
            
            # Start new head jitter when needed
            if self.head_jitter_start_time is None or (current_time - self.head_jitter_start_time > self.head_jitter_duration):
                self.head_jitter_start_time = current_time
                self._randomize_jitter_params()
                if DEBUG:
                    print(f"Starting head jitter (EMERGENCY) (amplitude: {self.head_jitter_amplitude:.2f}, freq: {self.head_jitter_freq:.2f}Hz)")
            
            # Apply jittering to head joints
            jitter_elapsed = current_time - self.head_jitter_start_time
            if jitter_elapsed <= self.head_jitter_duration:
                # Generate sine wave for head
                for i, idx in enumerate(self.head_joint_indices):
                    # Different phase for each joint to create complex motion
                    phase_offset = i * math.pi / (len(self.head_joint_indices) + 0.001)  # Avoid division by zero
                    modified_commands[idx] = self.head_jitter_amplitude * math.sin(
                        2 * math.pi * self.head_jitter_freq * jitter_elapsed + phase_offset
                    )
            
        else:
            # Normal operation - change head pose periodically
            if current_time - self.last_head_pose_time > self.next_head_pose_interval:
                self.last_head_pose_time = current_time
                self.next_head_pose_interval = self._random_interval(3, 8)
                
                # Generate and directly apply new random head pose
                self.current_head_pose = self._random_head_pose()
                if DEBUG:
                    print("Changing head pose (normal)")
                
            # Apply current head pose directly to motor commands
            for i, idx in enumerate(self.head_joint_indices):
                modified_commands[idx] = self.current_head_pose[i]
        
        return modified_commands


class EpisodicOpenLoopModifier:
    """Modifier for Episodic Open Loop Policy - passes through original motor commands"""
    
    def __init__(self, constants):
        self.constants = constants
    
    def modify(self, motor_commands, joint_pos=None, joint_vel=None, accel=None, 
               gyro=None, contacts=None, commands=None, gravity=None, timestamp=None):
        """Simply pass through the original motor commands"""
        return motor_commands
