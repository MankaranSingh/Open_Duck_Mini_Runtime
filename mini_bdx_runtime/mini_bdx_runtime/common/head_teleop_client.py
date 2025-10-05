import socket
import threading
import numpy as np

class HeadTelop:
    def __init__(self, host="10.144.135.208", port=5000):
        self.host = host
        self.port = port
        self._calibration_offset = None
        self.rpy_offset = (0.0, 0.0, 0.0)  # (yaw, pitch, roll)
        self._running = False
        self.roll_range = [-10, 10]
        self.pitch_range = [-10, 10]
        self.yaw_range = [-40, 40]

        self._thread = None

    def start(self):
        """Start the gyro reader in a background thread."""
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the gyro reader."""
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def get_rpy_offset(self):
        """Get the latest relative RPY values."""
        return self.rpy_offset

    def _run(self):
        """Internal method to connect to socket and read gyro data."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            print("Connected to server")

            while self._running:
                data = s.recv(1024)
                if not data:
                    break

                try:
                    yaw, pitch, roll = map(float, data.decode().strip().split(","))
                except ValueError:
                    continue  # skip malformed lines

                if self._calibration_offset is None:
                    self._calibration_offset = (yaw, pitch, roll)
                    print("Calibration set:", self._calibration_offset)
                    continue

                # Calculate relative values
                rel_yaw = -(self._calibration_offset[0] - yaw)
                rel_pitch = self._calibration_offset[1] - pitch
                rel_roll = self._calibration_offset[2] - roll 

                self.rpy_offset = np.deg2rad((np.clip(rel_roll, *self.roll_range), 
                             np.clip(rel_pitch, *self.pitch_range), 
                             np.clip(rel_yaw, *self.yaw_range)
                            ))
