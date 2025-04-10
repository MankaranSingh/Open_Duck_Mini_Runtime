import time
import pickle

import numpy as np

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer

from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
#from mini_bdx_runtime.eyes import Eyes
#from mini_bdx_runtime.sounds import Sounds
#from mini_bdx_runtime.antennas import Antennas
#from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter

joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    # "left_antenna",
    # "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[32, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        replay_obs=None,
        standing=False,
        cutoff_frequency=None,
        bt_port=1,
        wait_for_bt=True,
        bt_timeout=60
    ):
        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 4.8  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.standing = standing

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(serial_port)
        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=False,
        )

        #self.eyes = Eyes()
        #self.projector = Projector()

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.proprioceptive_history_len = 3

        self.proprioceptive_history = np.zeros((self.proprioceptive_history_len * 28))

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = [
            0.002,
            0.053,
            -0.63,
            1.368,
            -0.784,
            0,
            0,
            0,
            0,
            -0.003,
            -0.065,
            0.635,
            1.379,
            -0.796,
        ]

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = False

        #self.sounds = Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets/")
        #self.antennas = Antennas()

        self.command_freq = 20  # hz
        if self.commands:
            # Initialize controller with Bluetooth support
            self.xbox_controller = XBoxController(
                self.command_freq, 
                self.standing, 
                use_bluetooth=True, 
                bt_port=1
            )
            
            # Wait for Bluetooth connection if requested
            if 1:
                if not self.xbox_controller.wait_for_connection(timeout=60):
                    print("Warning: Starting without Bluetooth controller")
                else:
                    print("Bluetooth controller connected, proceeding with initialization")

        if not self.standing:
            self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
            self.imitation_i = 0
            self.imitation_phase = np.array([0, 0])
            self.phase_active = True
            self.waiting_for_zero = False

    def add_fake_head(self, pos):
        # add just the antennas now
        assert len(pos) == self.num_dofs
        pos_with_head = np.insert(pos, 9, [0, 0])
        return np.array(pos_with_head)

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        # projected_gravity = quat_rotate_inverse(orientation_quat, [0, 0, -1])
        # projected_gravity = np.array(imu_mat).reshape((3, 3)).T @ np.array([0, 0, -1])

        cmds = self.last_commands

        feet_contacts = self.feet_contacts.get()

        # if not self.standing:
        #     ref = self.PRM.get_reference_motion(*cmds[:3], self.imitation_i)
        # else:
        #     ref = np.array([])

        proprioceptive_obs = np.concatenate(
            [
                dof_pos - self.init_pos,
                dof_vel * 0.05,
            ]
        )

        self.proprioceptive_history = np.roll(self.proprioceptive_history, 28)
        self.proprioceptive_history[:28] = proprioceptive_obs

        obs = np.concatenate(
            [
                self.proprioceptive_history,
                imu_data["gyro"],
                imu_data["accelero"],
                # projected_gravity,
                cmds,
                self.last_action,
                # self.last_last_action,
                # self.last_last_last_action,
                # self.motor_targets,
                feet_contacts,
                # ref,
                # [self.imitation_i],
                self.imitation_phase,
            ]
        )

        return obs

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

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
                A_pressed = False
                X_pressed = False
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    (
                        self.last_commands,
                        A_pressed,
                        X_pressed,
                        left_trigger,
                        right_trigger,
                    ) = self.xbox_controller.get_last_command()

                # Handle X button for phase stopping
                if X_pressed and not self.standing:
                    if self.phase_active:
                        self.phase_active = False
                        self.waiting_for_zero = True
                        print("Phase advancement will stop at zero")
                    else:
                        self.phase_active = True
                        self.waiting_for_zero = False
                        print("Phase advancement enabled")

                # Command-based phase control (similar to mujoco_infer.py)
                if not self.standing and self.commands:
                    velocity_commands = self.last_commands[:3]  # Only use velocity components
                    command_norm = np.linalg.norm(velocity_commands)
                    
                    # Activate/deactivate phase based on command norm
                    if command_norm < 0.01:
                        if self.phase_active:
                            self.phase_active = False
                            self.waiting_for_zero = True
                            print("Phase advancement paused - waiting for zero")
                    else:
                        if not self.phase_active and not self.waiting_for_zero:
                            self.phase_active = True
                            print("Phase advancement enabled")

                #self.antennas.set_position_left(right_trigger)
                #self.antennas.set_position_right(left_trigger)

                if A_pressed and not self.paused:
                    self.paused = True
                    print("PAUSE")
                elif A_pressed and self.paused:
                    self.paused = False
                    print("UNPAUSE")

                if self.paused:
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue

                if not self.standing:
                    # Update phase only if active or waiting to reach zero
                    if self.phase_active or self.waiting_for_zero:
                        self.imitation_i += 1
                        self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                        
                        # If we're waiting for zero and we've reached it, stop phase advancement
                        if self.waiting_for_zero and self.imitation_i < 1:
                            self.imitation_i = 0
                            self.waiting_for_zero = False
                            print("Phase advancement stopped at zero")
                            
                    self.imitation_phase = np.array(
                        [
                            np.cos(
                                self.imitation_i
                                / self.PRM.nb_steps_in_period
                                * 2
                                * np.pi
                            ),
                            np.sin(
                                self.imitation_i
                                / self.PRM.nb_steps_in_period
                                * 2
                                * np.pi
                            ),
                        ]
                    )

                self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                # obs = np.clip(obs, -100, 100)

                action = self.policy.infer(obs)

                # action = np.clip(action, -1, 1)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # action = np.zeros(10)

                # robot_action = self.init_pos + action * self.action_scale
                self.motor_targets = self.init_pos + action * self.action_scale

                self.motor_targets = np.clip(
                    self.motor_targets,
                    self.prev_motor_targets
                    - self.max_motor_velocity * (1 / self.control_freq),  # control dt
                    self.prev_motor_targets
                    + self.max_motor_velocity * (1 / self.control_freq),  # control dt
                )

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                self.prev_motor_targets = self.motor_targets.copy()
                # self.motor_targets[5:9] = self.last_commands[3:]

                action_dict = make_action_dict(self.motor_targets, joints_order)

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            pass

        pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=32)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=False,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument("--replay_obs", type=str, required=False, default=None)
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--bt_port", type=int, default=1, help="Bluetooth port for controller connection")
    parser.add_argument("--wait_for_bt", action="store_true", default=True, 
                        help="Wait for Bluetooth connection before starting")
    parser.add_argument("--bt_timeout", type=int, default=60, 
                        help="Timeout in seconds for waiting for Bluetooth connection")
    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        args.onnx_model_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        replay_obs=args.replay_obs,
        standing=args.standing,
        cutoff_frequency=args.cutoff_frequency,
        bt_port=args.bt_port,
        wait_for_bt=args.wait_for_bt,
        bt_timeout=args.bt_timeout,
    )
    print("Done instantiating RLWalk")
    # rl_walk.start()
    rl_walk.run()
