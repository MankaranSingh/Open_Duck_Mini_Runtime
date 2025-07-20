import numpy as np
import bluetooth
from threading import Thread, Event
import time


class XBoxController:
    def __init__(self, command_freq=50, bt_port=1):
        self.command_freq = command_freq
        self.bt_port = bt_port

        # Initialize variables to store joystick data from BT
        self.bt_lv = 0.0
        self.bt_lh = 0.0
        self.bt_rv = 0.0
        self.bt_rh = 0.0
        
        # Store basic button states for reference
        self.bt_btn1 = 0  # A
        self.bt_btn2 = 0  # B
        self.bt_btn3 = 0  # X
        self.bt_btn4 = 0  # Y

        # Store last commands
        self.last_commands = [0.0, 0.0, 0.0, 0.0]
        
        # Add connection event for blocking until connected
        self.connection_established = Event()
        
        # Setup Bluetooth connection
        self._setup_bluetooth()
                    
    def _setup_bluetooth(self):
        """Set up Bluetooth server"""
        local_address = bluetooth.read_local_bdaddr()[0]
        print(f"Local Bluetooth address: {local_address}")
        print(f"Connect your joystick to this address using port {self.bt_port}")
        
        self.bt_server = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.bt_server.bind(("", self.bt_port))
        self.bt_server.listen(1)
        print(f"Bluetooth server started on port {self.bt_port}")
        print(f"Waiting for joystick connection...")
        Thread(target=self.bluetooth_receiver, daemon=True).start()

    def wait_for_connection(self, timeout=None):
        """Block until Bluetooth connection is established or timeout occurs"""
        print("Waiting for Bluetooth controller connection...")
        result = self.connection_established.wait(timeout)
        if result:
            print("Bluetooth controller connected!")
        else:
            print(f"Bluetooth connection timeout after {timeout} seconds")
        return result

    def bluetooth_receiver(self):
        """Handle Bluetooth connections and data reception"""
        while True:
            print(f"Waiting for Bluetooth connection on port {self.bt_port}...")
            client_sock, client_info = self.bt_server.accept()
            
            print(f"✓ Accepted connection from {client_info}")
            self.connection_established.set()
            
            while True:
                data = client_sock.recv(1024).decode('utf-8').strip()
                if not data:
                    continue
                    
                # Parse the data format
                values = data.split(',')
                self.bt_lv = float(values[0])
                self.bt_lh = float(values[1])
                self.bt_rv = float(values[2])
                self.bt_rh = float(values[3])
                self.bt_btn1 = int(values[4])
                self.bt_btn2 = int(values[5])
                self.bt_btn3 = int(values[6])
                self.bt_btn4 = int(values[7])

    def get_buttons(self):
        return self.bt_btn1, self.bt_btn2, self.bt_btn3, self.bt_btn4
    
    def get_sticks(self):
        return np.around([self.bt_lv, self.bt_lh, self.bt_rv, self.bt_rh], 4)

if __name__ == "__main__":
    controller = XBoxController(50, bt_port=1)
    
    print("Waiting for controller connection...")
    controller.wait_for_connection(timeout=60)
    
    print("Starting main loop...")
    while True:
        cmd = controller.get_sticks()
        btns = controller.get_buttons()
        status = "Connected" if controller.connection_established.is_set() else "Disconnected"
        print(f"Status: {status}, Commands: {cmd}, Buttons: {btns}")
        time.sleep(0.5)
