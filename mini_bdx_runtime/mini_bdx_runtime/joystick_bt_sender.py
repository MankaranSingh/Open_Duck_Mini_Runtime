import bluetooth
import pygame
import time
from bluetooth import Protocols
import sys
import argparse

BT = True
J1_HORIZONTAL, J1_VERTICAL = 0, 1
J2_HORIZONTAL, J2_VERTICAL = 3, 2  # Yaw rate

# Button IDs - adjust these values based on your specific controller
BTN_1 = 0
BTN_2 = 1
BTN_3 = 2
BTN_4 = 3
BTN_L1 = 4
BTN_L2 = 6
BTN_R1 = 5
BTN_R2 = 7

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

def list_available_devices():
    """List all available Bluetooth devices to help find the correct MAC address"""
    print("Scanning for nearby Bluetooth devices...")
    devices = bluetooth.discover_devices(duration=8, lookup_names=True)
    if devices:
        print("Found the following Bluetooth devices:")
        for addr, name in devices:
            print(f"  {addr} - {name}")
    else:
        print("No Bluetooth devices found.")
    return devices

def send_controller_data(target_mac, bt_port=1, retry_delay=5, max_retries=10):
    joystick = init_joystick()

    # Create filters for each axis
    filter_lv = FirstOrderFilter(0.0, RC, DT)
    filter_lh = FirstOrderFilter(0.0, RC, DT)
    filter_rv = FirstOrderFilter(0.0, RC, DT)
    filter_rh = FirstOrderFilter(0.0, RC, DT)

    retries = 0
    connected = False
    sock = None

    while not connected and retries < max_retries:
        try:
            print(f"Attempt {retries+1}/{max_retries}: Connecting to {target_mac} on port {bt_port}...")
            sock = bluetooth.BluetoothSocket(Protocols.RFCOMM)
            sock.connect((target_mac, bt_port))
            connected = True
            print(f"✓ Connected to {target_mac}")
        except bluetooth.btcommon.BluetoothError as e:
            print(f"× Connection failed: {e}")
            if sock:
                sock.close()
            
            if retries < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            retries += 1

    if not connected:
        print("Failed to connect after multiple attempts.")
        print("Possible issues:")
        print("1. The Bluetooth server may not be running on the target device")
        print("2. The MAC address might be incorrect")
        print("3. The target device might have Bluetooth disabled")
        return

    try:
        print("Sending controller data. Press Ctrl+C to exit.")
        while True:
            pygame.event.pump()  # Update joystick states
            
            lv = joystick.get_axis(J1_VERTICAL)
            lh = joystick.get_axis(J1_HORIZONTAL)
            rv = joystick.get_axis(J2_VERTICAL)
            rh = joystick.get_axis(J2_HORIZONTAL)

            # Apply first-order filter for smoother transitions
            lv = round(filter_lv.update(lv), 2)
            lh = round(filter_lh.update(lh), 2)
            rv = round(filter_rv.update(rv), 2)
            rh = round(filter_rh.update(rh), 2)

            # Read button states (1 for pressed, 0 for not pressed)
            btn1 = int(joystick.get_button(BTN_1))
            btn2 = int(joystick.get_button(BTN_2))
            btn3 = int(joystick.get_button(BTN_3))
            btn4 = int(joystick.get_button(BTN_4))
            btn_l1 = int(joystick.get_button(BTN_L1))
            btn_l2 = int(joystick.get_button(BTN_L2))
            btn_r1 = int(joystick.get_button(BTN_R1))
            btn_r2 = int(joystick.get_button(BTN_R2))

            # Include button states in the message
            message = f"{lv},{lh},{rv},{rh},{btn1},{btn2},{btn3},{btn4},{btn_l1},{btn_r1},{btn_l2},{btn_r2}\n"
            
            if connected:
                sock.send(message.encode()) 
            else:
                print(f"Would send: {message.strip()}")  

            time.sleep(DT)  # Maintain 50 Hz

    except KeyboardInterrupt:
        print("Exiting...")
    except bluetooth.btcommon.BluetoothError as e:
        print(f"Bluetooth error: {e}")
    finally:
        if sock:
            sock.close()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joystick to Bluetooth sender')
    parser.add_argument('--mac', type=str, help='MAC address of the target device')
    parser.add_argument('--port', type=int, default=1, help='Bluetooth port (default: 1)')
    parser.add_argument('--scan', action='store_true', help='Scan for available Bluetooth devices')
    
    args = parser.parse_args()
    
    if args.scan:
        list_available_devices()
        sys.exit(0)
        
    if not args.mac:
        rpi_mac_address = "B8:27:EB:BE:1D:BB"  # Default MAC address
        print(f"No MAC address specified, using default: {rpi_mac_address}")
        print("To specify a MAC address, use: --mac XX:XX:XX:XX:XX:XX")
        print("To scan for available devices, use: --scan")
    else:
        rpi_mac_address = args.mac
        
    send_controller_data(rpi_mac_address, args.port)
