
from pypot.feetech import FeetechSTS3215IO
import time
import numpy as np
from threading import Thread


class FeetechPWMControl:
    def __init__(self, ids, init_pos_rad, usb_port="/dev/ttyACM0"):
        self.io = FeetechSTS3215IO(
            usb_port,
            baudrate=1000000,
            use_sync_read=True,
        )
        self.ids = ids
        # self.bench = []

        # TODO remove this ?
        # self.io.enable_torque(self.ids)
        # self.io.set_mode({id: 0 for id in self.ids})
        # self.io.set_goal_position({id: 0 for id in self.ids})
        # time.sleep(1)
        # self.io.disable_torque({id: 0 for id in self.ids})
        # exit()

        self.io.set_mode({id: 2 for id in self.ids})
        self.kps = np.ones(len(self.ids)) * 32  # default kp

        self.init_pos_rad = init_pos_rad
        self.init_pos_deg = np.rad2deg(self.init_pos_rad)

        self.control_freq = 100  # Hz
        self.goal_positions = [0] * len(self.ids)
        self.goal_positions = self.init_pos_deg
        self.present_positions = [0] * len(self.ids)
        self.last_positions = [0] * len(self.ids)
        self.present_speeds = [0] * len(self.ids)
        # self.speed_decimation = 2
        # self.speed_decimation_index = 0
        self.last_t = time.time()

        Thread(target=self.update, daemon=True).start()

    def set_kps(self, kps):
        self.kps = np.array(kps)

    def disable_torque(self):
        self.io.set_mode({id: 0 for id in self.ids})
        self.io.disable_torque(self.ids)

    #def enable_torque(self):
    #    self.io.enable_torque(self.ids)
    #    self.io.set_mode({id: 2 for id in self.ids})

    def freeze(self):
        present_position = list(self.io.get_present_position(self.ids))
        print(present_position)
        self.io.set_goal_position(
            {id: present_position[i] for i, id in enumerate(self.ids)}
        )
        self.io.set_mode({id: 0 for id in self.ids})
        self.io.enable_torque(self.ids)

    def update(self):
        while True:
            s = time.time()
            try:
                self.present_positions = self.io.get_present_position(self.ids)
                self.present_speeds = self.io.get_present_speed(self.ids)
            except Exception as e:
                print(e)
                continue

            if len(self.present_positions) != len(self.ids):
                print("ERROR : present_positions and ids do not have the same length")
                time.sleep(1 / self.control_freq)
                continue
            errors = np.array(self.goal_positions) - np.array(self.present_positions)

            pwms = self.kps * errors
            pwms = np.int16(pwms)
            pwms = np.clip(pwms, -1000, 1000)

            pwm_magnitudes = abs(pwms)

            direction_bits = [1 if pwm >= 0 else 0 for pwm in pwms]

            goal_times = [
                (direction_bits[i] << 10) | pwm_magnitudes[i]
                for i in range(len(pwm_magnitudes))
            ]
            try:
                self.io.set_goal_time({id: goal_times[i] for i, id in enumerate(self.ids)})
            except Exception as e:
                print(e)
                continue

            # if self.speed_decimation_index % self.speed_decimation == 0:
            #     self.present_speeds = (
            #         np.array(self.present_positions) - np.array(self.last_positions)
            #     ) / (time.time() - self.last_t)
            #     self.last_positions = np.array(self.present_positions).copy()
            #     self.last_t = time.time()

            took = time.time() - s
            # self.bench.append(took)
            # self.bench = self.bench[-100:]
            # print("mean took : ", np.mean(self.bench))
            # print("Took : ", np.around(took, 3), ". Budget : ", np.around(1/self.control_freq, 3), "diff : ", ((1/self.control_freq - took)))
            # if (1 / self.control_freq - took) < 0:
            #     print(
            #         "Low level control budget exceded by ",
            #         np.around(took - 1 / self.control_freq, 3),
            #     )
            time.sleep(max(0, (1 / self.control_freq - took)))
            # self.speed_decimation_index += 1


if __name__ == "__main__":
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    pwm_control = FeetechPWMControl(ids, [0]*len(ids))
    #pwm_control.enable_torque()
    #pwm_control.set_kps([32])

    init = pwm_control.present_positions
    try:
        while True:
            print(pwm_control.present_positions)
            #pwm_control.goal_positions = [init]
            time.sleep(0.5)
            #print(pwm_control.present_positions)
            #pwm_control.goal_positions = [init+5]
            #time.sleep(0.5)
    except:
        pwm_control.disable_torque()
