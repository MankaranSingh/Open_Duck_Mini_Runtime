import board
import busio
import adafruit_mpu6050
import numpy as np
from queue import Queue
from threading import Thread
import time


class Imu:
    def __init__(self, sampling_freq):
        self.sampling_freq = sampling_freq
        self.x_offset = 0

        # Initialize I2C and MPU6050
        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_mpu6050.MPU6050(i2c)

        self.last_imu_data = {
            "gyro": [0, 0, 0],
            "accel": [0, 0, 0],
        }

    def tare_x(self):
        print("Taring x ...")
        x_values = []
        num_values = 100
        ok = False
        while not ok:
            x_values.append(np.array(self.imu.acceleration)[0])
            x_values = x_values[-num_values:]

            if len(x_values) == num_values:
                mean = np.mean(x_values)
                std = np.std(x_values)
                if std < 0.05:
                    ok = True
                    self.x_offset = mean
                    print("Tare x done")
                else:
                    print(std)

            time.sleep(0.01)

    def get_data(self):
        gyro = np.array(self.imu.gyro)
        accelero = np.array(self.imu.acceleration)
        self.last_imu_data = {
                "gyro": gyro,
                "accel": accelero,
            }
        return self.last_imu_data


if __name__ == "__main__":
    imu = Imu(50)
    while True:
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3))
        print("accel", np.around(data["accel"], 3))
        print("---")
        time.sleep(1 / 25)
