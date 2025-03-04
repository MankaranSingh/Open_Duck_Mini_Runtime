import bluetooth
from bluetooth import Protocols
import pygame
import time
import math

J1_HORIZONTAL, J1_VERTICAL = 0, 1
J2_HORIZONTAL = 3  # Yaw rate

# Filtering parameters
RC = 0.1  # Response time constant (lower = more responsive, higher = smoother)
DT = 1 / 50  # Loop time (50Hz update rate)

class FirstOrderFilter:
    """ First-order low-pass filter for smooth joystick control """
    def __init__(self, x0, rc, dt, initialized=True):
        self.x = x0
        self.dt = dt
        self.update_alpha(rc)
        self.initialized = initialized

    def update_alpha(self, rc):
        self.alpha = self.dt / (rc + self.dt)

    def update(self, x):
        if self.initialized:
            self.x = (1. - self.alpha) * self.x + self.alpha * x
        else:
            self.initialized = True
            self.x = x
        return self.x

def init_joystick():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected.")
        exit(-1)

    handle = pygame.joystick.Joystick(0)
    handle.init()
    
    print(f"Joystick Name: {handle.get_name()}")
    return handle

def send_controller_data(target_mac):
    sock = bluetooth.BluetoothSocket(Protocols.RFCOMM)
    joystick = init_joystick()

    # Create filters for each axis
    filter_x = FirstOrderFilter(0.0, RC, DT)
    filter_y = FirstOrderFilter(0.0, RC, DT)
    filter_yaw = FirstOrderFilter(0.0, RC, DT)

    try:
        sock.connect((target_mac, 1))
        print(f"Connected to {target_mac}")

        while True:
            pygame.event.pump()  # Update joystick states
            
            raw_x = joystick.get_axis(J1_HORIZONTAL)
            raw_y = joystick.get_axis(J1_VERTICAL)
            raw_yaw = joystick.get_axis(J2_HORIZONTAL)

            # Apply first-order filter for smoother transitions
            x = round(filter_x.update(raw_x), 2)
            y = round(filter_y.update(raw_y), 2)
            yaw = round(filter_yaw.update(raw_yaw), 2)

            message = f"{x},{y},{yaw}\n"
            sock.send(message.encode())   

            time.sleep(DT)  # Maintain 50 Hz

    except bluetooth.btcommon.BluetoothError as e:
        print(f"Bluetooth error: {e}")
    finally:
        sock.close()
        pygame.quit()

if __name__ == "__main__":
    rpi_mac_address = "B8:27:EB:16:C4:AE"  # Replace with your Raspberry Pi's MAC address
    send_controller_data(rpi_mac_address)
