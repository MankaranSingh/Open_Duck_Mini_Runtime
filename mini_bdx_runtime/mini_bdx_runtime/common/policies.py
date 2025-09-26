import numpy as np
from mini_bdx_runtime.common.onnx_infer import OnnxInfer
from mini_bdx_runtime.common.episodic_loader import EpisodicLoader
from mini_bdx_runtime.common.gait_blending import gait_sample_data, gait_sample_data_med_only, vel_to_step, blend_gait_parameters

DECIMATION = 1

class JoystickPolicy:
    """Policy that uses joystick input to control the robot"""
    
    def __init__(self, constants, onnx_model_path=None):
        super().__init__()
        
        self.decimation = DECIMATION

        # Control ranges - kept for reference to guide command generation
        self.COMMANDS_RANGE_X = [-0.1, 0.2]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]
        
        # Policy state - commands will be set from outside
        self.action_size = len(constants.JOINTS_ORDER) - len(constants.NON_LEG_JOINTS)
        self.full_action = np.zeros(len(constants.JOINTS_ORDER))
        self.last_action = np.zeros(self.action_size)
        self.active_idx = np.array([idx for idx, joint in enumerate(constants.JOINTS_ORDER) if joint not in constants.NON_LEG_JOINTS])
        self.default_actuator = constants.DEFAULT_ACTUATOR_POS.copy()

        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.proprioceptive_history_length = 4
        self.proprioceptive_obs_size = 2*len(constants.JOINTS_ORDER) + self.action_size

        self.proprioceptive_history = np.zeros(self.proprioceptive_history_length*self.proprioceptive_obs_size)
        
        # Initialize phaserate sampler
        self.gait_sample_data = gait_sample_data_med_only
        
        # Phase tracking
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_active = True
        self.waiting_for_zero = False
        
        self.prev_motor_targets = self.default_actuator.copy()
        self.max_motor_velocity = 5.0  # rad/s

        self.model = OnnxInfer(onnx_model_path, awd=True)
        print(f"Model loaded from {onnx_model_path}")
    
    def reset(self):
        """Reset the policy state"""
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_active = True
        self.waiting_for_zero = False
        self.last_action.fill(0)
        self.proprioceptive_history.fill(0)
        self.prev_motor_targets = self.default_actuator.copy()
        
    def update_phase(self, commands):
        """Update the imitation phase based on current commands"""
        velocity_commands = commands[:3]  # Only use velocity components
        
        command_norm = np.linalg.norm(velocity_commands)
        if 0: #command_norm < 0.01:
            if self.phase_active:
                self.phase_active = False
                self.waiting_for_zero = True
                print("Phase advancement paused - waiting for zero")
        else:
            if not self.phase_active and not self.waiting_for_zero:
                self.phase_active = True
                print("Phase advancement enabled")
        
        # Update phase only if active or waiting to reach zero
        if self.phase_active or self.waiting_for_zero:
            # G_output = blend_gait_parameters(self.gait_sample_data, velocity_commands[0], velocity_commands[1], velocity_commands[2], 
            #                                  0.15, 0.15, 0.5)
            # x_step, y_step, theta_step, period = vel_to_step(velocity_commands[0], velocity_commands[1], velocity_commands[2], G_output)
            self.nb_steps_in_period = 30 # int(period*50)

            self.imitation_i += 1.0 * self.phase_frequency_factor
            self.imitation_i = int(self.imitation_i) % int(self.nb_steps_in_period)

            # If we're waiting for zero and we've reached it, stop phase advancement
            if self.waiting_for_zero and self.imitation_i < 1.0:
                self.imitation_i = 0
                self.waiting_for_zero = False
                print("Phase advancement stopped at zero")
            
        self.imitation_phase = np.array(
                [
                    np.cos(self.imitation_i / self.nb_steps_in_period * 2 * np.pi),
                    np.sin(self.imitation_i / self.nb_steps_in_period * 2 * np.pi),
                ]
            )
    
    def get_default_commands(self):
        return np.array([0, 0, 0])
        
    def joystick_to_commands(self, joystick_values):
        j1_v, j1_h, j2_v, j2_h = joystick_values
        
        lin_vel_x_input = -j1_v  # Invert to match expected orientation
        lin_vel_y_input = -j1_h
        ang_vel_input = -j2_h

        # Scale inputs to command ranges
        lin_vel_x = np.interp(lin_vel_x_input, [-1, 1], self.COMMANDS_RANGE_X)
        lin_vel_y = np.interp(lin_vel_y_input, [-1, 1], self.COMMANDS_RANGE_Y)
        ang_vel = np.interp(ang_vel_input, [-1, 1], self.COMMANDS_RANGE_THETA)

        # Apply a small deadzone to prevent drift when joystick is near center
        deadzone = 0.05
        lin_vel_x = 0 if abs(lin_vel_x_input) < deadzone else lin_vel_x
        lin_vel_y = 0 if abs(lin_vel_y_input) < deadzone else lin_vel_y
        ang_vel = 0 if abs(ang_vel_input) < deadzone else ang_vel

        return np.array([lin_vel_x, lin_vel_y, ang_vel])
    
    def key_to_commands(self, keycode):                    
        if keycode == 72:  # h
            self.head_control_mode = not self.head_control_mode

        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]

        return np.array([lin_vel_x, lin_vel_y, ang_vel,]) 

    def infer(self, joint_pos, joint_vel, accel, gyro, contacts, commands, gravity=None):
        """Process observations to get actions"""
        # Update phase based on current commands
        self.update_phase(commands)

        proprioceptive_obs = np.concatenate([ 
            joint_pos - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
            ]
        )

        self.proprioceptive_history = np.roll(self.proprioceptive_history, self.proprioceptive_obs_size)
        self.proprioceptive_history[:self.proprioceptive_obs_size] = proprioceptive_obs

        obs = np.concatenate(
            [
                self.proprioceptive_history,
                gyro,
                accel,
                # gravity,
                commands,
                contacts,
                self.imitation_phase,
            ]
        )

        action = self.model.infer(obs)        
        self.full_action[self.active_idx] = action

        self.last_action = action.copy()

        motor_targets = (self.default_actuator + self.full_action * self.action_scale)
        return motor_targets

class StandingPolicy:
    """Policy for controlling the robot in standing mode"""
    
    def __init__(self, constants, onnx_model_path):
        self.decimation = DECIMATION
        # Control ranges
        self.HEIGHT_DELTA_RANGE = [-0.012, 0.012]
        self.ROLL_DELTA_RANGE = [-np.radians(5), np.radians(5)]
        self.PITCH_DELTA_RANGE = [-np.radians(5), np.radians(5)]
        self.YAW_DELTA_RANGE = [-np.radians(5), np.radians(5)]
        
        self.action_size = len(constants.JOINTS_ORDER) - len(constants.NON_LEG_JOINTS)
        self.default_actuator = constants.DEFAULT_ACTUATOR_POS.copy()
        
        # Parameters
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.5
        self.action_scale = 0.5
        self.proprioceptive_history_length = 3
        self.proprioceptive_obs_size = 2*len(constants.JOINTS_ORDER) + self.action_size
        self.obs_factor = 1
        
        self.proprioceptive_history = np.zeros(self.proprioceptive_history_length*self.proprioceptive_obs_size)
        
        self.last_action = np.zeros(self.action_size)
        self.full_action = np.zeros(len(constants.JOINTS_ORDER))
        self.active_idx = np.array([idx for idx, joint in enumerate(constants.JOINTS_ORDER) if joint not in constants.NON_LEG_JOINTS])
        
        self.prev_motor_targets = self.default_actuator.copy()
        self.max_motor_velocity = 5.24  # rad/s
        
        # Initialize ONNX model
        self.model = OnnxInfer(onnx_model_path, awd=True)
        print(f"Standing policy loaded from {onnx_model_path}")
    
    def reset(self):
        """Reset the policy state"""
        self.proprioceptive_history.fill(0)
        self.last_action.fill(0)
        self.prev_motor_targets = self.default_actuator.copy()
    
    def get_default_commands(self):
        return np.array([0, 0, 0, 0])
    
    def joystick_to_commands(self, joystick_values):
        j1_v, j1_h, j2_v, j2_h = joystick_values
        
        # Scale the raw joystick values to the appropriate ranges
        height_input = -j1_v  # Invert to match expected orientation
        roll_input = j1_h
        pitch_input = -j2_v
        yaw_input = -j2_h

        # Scale inputs to command ranges
        height_delta = np.interp(height_input, [-1, 1], self.HEIGHT_DELTA_RANGE)
        roll_delta = np.interp(roll_input, [-1, 1], self.ROLL_DELTA_RANGE)
        pitch_delta = np.interp(pitch_input, [-1, 1], self.PITCH_DELTA_RANGE)
        yaw_delta = np.interp(yaw_input, [-1, 1], self.YAW_DELTA_RANGE)

        return np.array([height_delta, roll_delta, pitch_delta, yaw_delta])

    def key_to_commands(self, keycode):
        height_delta = 0
        roll_delta = 0
        pitch_delta = 0
        yaw_delta = 0

        # Up/down arrows for height
        if keycode == 265:  # arrow up
            height_delta = self.HEIGHT_DELTA_RANGE[1]
        if keycode == 264:  # arrow down
            height_delta = self.HEIGHT_DELTA_RANGE[0]

        # Left/right arrows for roll
        if keycode == 263:  # arrow left
            roll_delta = self.ROLL_DELTA_RANGE[1]
        if keycode == 262:  # arrow right
            roll_delta = self.ROLL_DELTA_RANGE[0]

        # W/S for pitch
        if keycode == 65:  # w
            pitch_delta = self.PITCH_DELTA_RANGE[1]
        if keycode == 83:  # s
            pitch_delta = self.PITCH_DELTA_RANGE[0]

        # Q/E for yaw
        if keycode == 81:  # q
            yaw_delta = self.YAW_DELTA_RANGE[1]
        if keycode == 69:  # e
            yaw_delta = self.YAW_DELTA_RANGE[0]

        return np.array([height_delta, roll_delta, pitch_delta, yaw_delta])
    
    def infer(self, joint_pos, joint_vel, accel, gyro, contacts, commands, gravity=None):
        """Process observations to get actions"""
        proprioceptive_obs = np.concatenate([
            joint_pos - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
        ])
        
        self.proprioceptive_history = np.roll(self.proprioceptive_history, self.proprioceptive_obs_size)
        self.proprioceptive_history[:self.proprioceptive_obs_size] = proprioceptive_obs

        # frame_data = np.array(self.controller.query_lookup_table_jax(commands[1], commands[2], commands[3], commands[0]))
        
        # Normalize command to range [-1, 1] using defined ranges
        height_max = max(abs(self.HEIGHT_DELTA_RANGE[0]), abs(self.HEIGHT_DELTA_RANGE[1]))
        roll_max = max(abs(self.ROLL_DELTA_RANGE[0]), abs(self.ROLL_DELTA_RANGE[1]))
        pitch_max = max(abs(self.PITCH_DELTA_RANGE[0]), abs(self.PITCH_DELTA_RANGE[1]))
        yaw_max = max(abs(self.YAW_DELTA_RANGE[0]), abs(self.YAW_DELTA_RANGE[1]))
        
        normalized_commands = np.array([
            commands[0] / height_max,  # height_delta normalized
            commands[1] / roll_max,    # roll_delta normalized
            commands[2] / pitch_max,   # pitch_delta normalized
            commands[3] / yaw_max,     # yaw_delta normalized
        ])
        
        obs = np.concatenate([
            self.proprioceptive_history,
            gyro,
            accel,
            normalized_commands * self.obs_factor,
            contacts,
        ])
        
        action = self.model.infer(obs)
        self.last_action = action.copy()
        
        # self.full_action.fill(0)
        self.full_action[self.active_idx] = action
        
        motor_targets = self.default_actuator + self.full_action * self.action_scale
        
        self.prev_motor_targets = motor_targets.copy()
        return motor_targets

class EpisodicPolicy:
    """Policy for controlling the robot using episodic reference motion"""
    
    def __init__(self, constants, onnx_model_path, reference_data_path="playground/open_duck_mini_v2/data/happy_dance.json"):
        # Parameters
        self.action_size = len(constants.JOINTS_ORDER)
        self.active_action_idx = np.arange(self.action_size)
        self.constants = constants
        
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.5
        self.action_scale = 0.5
        self.proprioceptive_history_length = 3
        self.proprioceptive_obs_size = 2*len(constants.JOINTS_ORDER) + self.action_size
        
        self.proprioceptive_history = np.zeros(self.proprioceptive_history_length*self.proprioceptive_obs_size)
        
        # Initialize episodic loader
        self.EM = EpisodicLoader(reference_data_path)
        self.imitation_i = 0
        self.n_frames = self.EM.n_frames
        
        # State tracking
        self.last_action = np.zeros(self.action_size)
        self.prev_motor_targets = constants.DEFAULT_ACTUATOR_POS.copy()
        self.default_actuator = constants.DEFAULT_ACTUATOR_POS.copy()

        self.max_motor_velocity = 5.24  # rad/s
        self.decimation = DECIMATION
        
        # Initialize ONNX model
        self.model = OnnxInfer(onnx_model_path, awd=True)
        print(f"Episodic policy loaded from {onnx_model_path}")
        print(f"Reference data loaded from {reference_data_path} with {self.n_frames} frames")
    
    def reset(self):
        """Reset the policy state"""
        self.imitation_i = 0
        self.proprioceptive_history.fill(0)
        self.last_action.fill(0)
        self.prev_motor_targets = self.default_actuator.copy()
    
    def get_default_commands(self):
        """Get default commands for the episodic policy"""
        return [1]
    
    def joystick_to_commands(self, joystick_values):        
        return [1]

    def key_to_commands(self, keycode):
        return [1]
    
    def advance_frame(self, rate=1):
        """Advance to the next frame in the reference motion"""
        self.imitation_i += rate
        self.imitation_i %= self.n_frames
        return self.imitation_i
        
    def infer(self, joint_pos, joint_vel, accel, gyro, contacts, commands=[1], gravity=None):
        """Process observations to get actions, commands is phase rate"""
        # Get current reference motion
        current_reference_motion = self.EM.get_reference_motion(self.imitation_i)
        self.advance_frame(commands[0])
        
        proprioceptive_obs = np.concatenate([
            joint_pos - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
        ])
        
        self.proprioceptive_history = np.roll(self.proprioceptive_history, self.proprioceptive_obs_size)
        self.proprioceptive_history[:self.proprioceptive_obs_size] = proprioceptive_obs
        
        obs = np.concatenate([
            self.proprioceptive_history,
            gyro,
            accel,
            contacts,
            [self.imitation_i / self.n_frames],  # Normalized phase
            current_reference_motion[0:len(self.constants.JOINTS_ORDER)][self.constants.ISAAC_TO_MUJOCO] - self.default_actuator,
        ])
        
        action = self.model.infer(obs)
        self.last_action = action.copy()
        
        motor_targets = self.default_actuator + action * self.action_scale
                
        self.prev_motor_targets = motor_targets.copy()
        return motor_targets


class EpisodicOpenLoopPolicy:
    """Policy for controlling the robot using episodic reference motion directly (open loop)"""
    
    def __init__(self, constants, reference_data_path="playground/open_duck_mini_v2/data/happy_dance.json"):
        # Parameters
        self.action_size = len(constants.JOINTS_ORDER)
        self.constants = constants
        
        # Initialize episodic loader
        self.EM = EpisodicLoader(reference_data_path)
        self.imitation_i = 0
        self.n_frames = self.EM.n_frames
        
        # State tracking
        self.prev_motor_targets = constants.DEFAULT_ACTUATOR_POS.copy()
        self.default_actuator = constants.DEFAULT_ACTUATOR_POS.copy()

        self.max_motor_velocity = 5.24  # rad/s
        self.decimation = DECIMATION
        
        print(f"Episodic open loop policy loaded")
        print(f"Reference data loaded from {reference_data_path} with {self.n_frames} frames")
    
    def reset(self):
        """Reset the policy state"""
        self.imitation_i = 0
        self.prev_motor_targets = self.default_actuator.copy()
    
    def get_default_commands(self):
        """Get default commands for the episodic policy"""
        return [1]
    
    def joystick_to_commands(self, joystick_values):        
        return [1]

    def key_to_commands(self, keycode):
        return [1]
    
    def advance_frame(self, rate=1):
        """Advance to the next frame in the reference motion"""
        self.imitation_i += rate
        self.imitation_i = min(self.imitation_i, self.n_frames-1)
        return self.imitation_i
        
    def infer(self, joint_pos, joint_vel, accel, gyro, contacts, commands=[1], gravity=None):
        """Process observations to get actions, commands is phase rate"""
        # Get current reference motion
        current_reference_motion = self.EM.get_reference_motion(self.imitation_i)
        self.advance_frame(commands[0])
        
        # Directly use reference motion as motor targets
        motor_targets = current_reference_motion[self.EM.slices["joints_pos"]][self.constants.ISAAC_TO_MUJOCO]
        
        self.prev_motor_targets = motor_targets.copy()
        return motor_targets
