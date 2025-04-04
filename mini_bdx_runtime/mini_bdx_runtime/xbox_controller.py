import pygame
from threading import Thread, Event
from queue import Queue
import time
import numpy as np
import bluetooth
import socket

X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]

# rads
NECK_PITCH_RANGE = [-0.34, 1.1]
HEAD_PITCH_RANGE = [-0.78, 0.78]
HEAD_YAW_RANGE = [-1.7, 1.7]
HEAD_ROLL_RANGE = [-0.5, 0.5]


class XBoxController:
    def __init__(self, command_freq, standing=False, use_bluetooth=True, bt_port=1):
        self.command_freq = command_freq
        self.standing = standing
        self.head_control_mode = self.standing
        self.use_bluetooth = use_bluetooth
        self.bt_port = bt_port

        # Initialize variables to store joystick data from BT
        self.bt_l_x = 0.0
        self.bt_l_y = 0.0
        self.bt_r_x = 0.0
        self.bt_btn1 = 0  # A
        self.bt_btn2 = 0  # B
        self.bt_btn3 = 0  # X
        self.bt_btn4 = 0  # Y
        self.bt_l1 = 0    # L1
        self.bt_r1 = 0    # R1
        self.bt_l2 = 0    # L2
        self.bt_r2 = 0    # R2

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0
        
        # Add connection event for blocking until connected
        self.connection_established = Event()
        
        if self.use_bluetooth:
            try:
                # Get local device information 
                local_address = self.get_local_bt_address()
                if local_address:
                    print(f"Local Bluetooth address: {local_address}")
                    print(f"Connect your joystick to this address using port {self.bt_port}")
                
                # Start Bluetooth server
                self.bt_server = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.bt_server.bind(("", self.bt_port))
                self.bt_server.listen(1)
                print(f"Bluetooth server started on port {self.bt_port}")
                print(f"Waiting for joystick connection...")
                Thread(target=self.bluetooth_receiver, daemon=True).start()
            except Exception as e:
                print(f"Failed to initialize Bluetooth: {e}")
                print("Make sure Bluetooth is enabled on this device.")
                print("Falling back to local joystick mode.")
                self.use_bluetooth = False
                self.init_local_joystick()
                self.connection_established.set()
        else:
            self.init_local_joystick()
            self.connection_established.set()
            
        self.cmd_queue = Queue(maxsize=1)
        Thread(target=self.commands_worker, daemon=True).start()

    def get_local_bt_address(self):
        """Get the local Bluetooth adapter address"""
        try:
            return bluetooth.read_local_bdaddr()[0]
        except Exception as e:
            print(f"Could not get local Bluetooth address: {e}")
            return None
            
    def init_local_joystick(self):
        """Initialize local joystick"""
        pygame.init()
        try:
            if pygame.joystick.get_count() > 0:
                self.p1 = pygame.joystick.Joystick(0)
                self.p1.init()
                print(f"Loaded local joystick: {self.p1.get_name()} with {self.p1.get_numaxes()} axes.")
            else:
                print("No local joystick found.")
                self.p1 = None
        except Exception as e:
            print(f"Error initializing joystick: {e}")
            self.p1 = None

    def wait_for_connection(self, timeout=None):
        """Block until Bluetooth connection is established or timeout occurs"""
        if self.use_bluetooth:
            print("Waiting for Bluetooth controller connection...")
            result = self.connection_established.wait(timeout)
            if result:
                print("Bluetooth controller connected!")
                return True
            else:
                print(f"Bluetooth connection timeout after {timeout} seconds")
                return False
        return True  # If not using Bluetooth, always return True

    def bluetooth_receiver(self):
        connection_attempts = 0
        max_silent_failures = 3
        retry_delay = 3
        
        while True:
            try:
                print(f"Waiting for Bluetooth connection on port {self.bt_port}...")
                client_sock, client_info = self.bt_server.accept()
                connection_attempts = 0
                
                print(f"✓ Accepted connection from {client_info}")
                # Signal that connection is established
                self.connection_established.set()
                
                while True:
                    try:
                        data = client_sock.recv(1024).decode('utf-8').strip()
                        if not data:
                            continue
                            
                        # Parse the data format: x,y,yaw,btn1,btn2,btn3,btn4,btn_l1,btn_r1,btn_l2,btn_r2
                        values = data.split(',')
                        if len(values) == 11:
                            self.bt_l_y = float(values[0])      # x in bt_sender is mapped to l_y
                            self.bt_l_x = float(values[1])      # y in bt_sender is mapped to l_x
                            self.bt_r_x = float(values[2])      # yaw in bt_sender is mapped to r_x
                            self.bt_btn1 = int(values[3])       # A button
                            self.bt_btn2 = int(values[4])       # B button
                            self.bt_btn3 = int(values[5])       # X button
                            self.bt_btn4 = int(values[6])       # Y button
                            self.bt_l1 = int(values[7])         # L1 button
                            self.bt_r1 = int(values[8])         # R1 button
                            self.bt_l2 = int(values[9])         # L2 button
                            self.bt_r2 = int(values[10])        # R2 button
                    except Exception as e:
                        print(f"Error receiving data: {e}")
                        break
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                connection_attempts += 1
                if connection_attempts <= max_silent_failures:
                    print(f"Bluetooth connection attempt {connection_attempts} failed: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to establish Bluetooth connection after {connection_attempts} attempts")
                    print("Make sure the joystick_bt_sender is running and connecting to the correct address")
                    time.sleep(30)  # Wait longer between batches of attempts
                    connection_attempts = 0  # Reset counter for next batch
                
            finally:
                if 'client_sock' in locals():
                    client_sock.close()
                # Reset connection flag if disconnected
                self.connection_established.clear()
                print("Bluetooth connection closed. Waiting for new connection...")

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        A_pressed = False
        X_pressed = False
        last_commands = self.last_commands
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger

        if self.use_bluetooth:
            l_x = self.bt_l_x
            l_y = self.bt_l_y
            r_x = self.bt_r_x
            
            # Set triggers based on L2/R2 buttons
            right_trigger = 1.0 if self.bt_r2 else 0.0
            left_trigger = 1.0 if self.bt_l2 else 0.0
            
            # Check button presses
            A_pressed = self.bt_btn1 == 1
            X_pressed = self.bt_btn3 == 1
            
            # Toggle head control mode on Y button press
            if self.bt_btn4 == 1:
                self.head_control_mode = not self.head_control_mode
        else:
            l_x = -1 * self.p1.get_axis(0)
            l_y = -1 * self.p1.get_axis(1)
            r_x = -1 * self.p1.get_axis(2)
            r_y = -1 * self.p1.get_axis(3)

            right_trigger = np.around((self.p1.get_axis(4) + 1) / 2, 3)
            left_trigger = np.around((self.p1.get_axis(5) + 1) / 2, 3)

            if left_trigger < 0.1:
                left_trigger = 0
            if right_trigger < 0.1:
                right_trigger = 0
                
            for event in pygame.event.get():
                if self.p1.get_button(0):  # A button
                    A_pressed = True

                if self.p1.get_button(3):  # X button
                    X_pressed = True
                    
                if self.p1.get_button(4):  # Y button
                    self.head_control_mode = not self.head_control_mode

            pygame.event.pump()  # process event queue

        if not self.head_control_mode:
            lin_vel_y = l_x
            lin_vel_x = l_y
            ang_vel = r_x
            if lin_vel_x >= 0:
                lin_vel_x *= np.abs(X_RANGE[1])
            else:
                lin_vel_x *= np.abs(X_RANGE[0])

            if lin_vel_y >= 0:
                lin_vel_y *= np.abs(Y_RANGE[1])
            else:
                lin_vel_y *= np.abs(Y_RANGE[0])

            if ang_vel >= 0:
                ang_vel *= np.abs(YAW_RANGE[1])
            else:
                ang_vel *= np.abs(YAW_RANGE[0])

            last_commands[0] = lin_vel_x
            last_commands[1] = lin_vel_y
            last_commands[2] = ang_vel
        else:
            last_commands[0] = 0.0
            last_commands[1] = 0.0
            last_commands[2] = 0.0
            last_commands[3] = 0.0 # neck pitch 0 for now

            head_yaw = l_x
            head_pitch = l_y
            head_roll = r_x

            if head_yaw >= 0:
                head_yaw *= np.abs(HEAD_YAW_RANGE[0])
            else:
                head_yaw *= np.abs(HEAD_YAW_RANGE[1])

            if head_pitch >= 0:
                head_pitch *= np.abs(HEAD_PITCH_RANGE[0])
            else:
                head_pitch *= np.abs(HEAD_PITCH_RANGE[1])

            if head_roll >= 0:
                head_roll *= np.abs(HEAD_ROLL_RANGE[0])
            else:
                head_roll *= np.abs(HEAD_ROLL_RANGE[1])

            last_commands[4] = head_pitch
            last_commands[5] = head_yaw
            last_commands[6] = head_roll

        return np.around(last_commands, 3), A_pressed, X_pressed, left_trigger, right_trigger

    def get_last_command(self):
        A_pressed = False
        X_pressed = False

        try:
            self.last_commands, A_pressed, X_pressed, self.last_left_trigger, self.last_right_trigger = self.cmd_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_commands, A_pressed, X_pressed, self.last_left_trigger, self.last_right_trigger


if __name__ == "__main__":
    controller = XBoxController(50, use_bluetooth=True)
    
    print("Waiting for controller connection...")
    controller.wait_for_connection(timeout=60)
    
    print("Starting main loop...")
    while True:
        cmd, a, x, lt, rt = controller.get_last_command()
        status = "Connected" if controller.connection_established.is_set() else "Disconnected"
