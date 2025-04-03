import pygame
from threading import Thread
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
        
        if self.use_bluetooth:
            # Start Bluetooth server
            self.bt_server = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.bt_server.bind(("", self.bt_port))
            self.bt_server.listen(1)
            print(f"Waiting for Bluetooth connection on port {self.bt_port}...")
            Thread(target=self.bluetooth_receiver, daemon=True).start()
        else:
            pygame.init()
            self.p1 = pygame.joystick.Joystick(0)
            self.p1.init()
            print(f"Loaded joystick with {self.p1.get_numaxes()} axes.")
            
        self.cmd_queue = Queue(maxsize=1)
        Thread(target=self.commands_worker, daemon=True).start()

    def bluetooth_receiver(self):
        try:
            client_sock, client_info = self.bt_server.accept()
            print(f"Accepted connection from {client_info}")
            
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
                    
        except Exception as e:
            print(f"Bluetooth server error: {e}")
        finally:
            if 'client_sock' in locals():
                client_sock.close()
            self.bt_server.close()
            print("Bluetooth connection closed")

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

    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
