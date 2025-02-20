import bluetooth
import pygame
import time

# Joystick axis mapping
J1_HORIZONTAL, J1_VERTICAL = 0, 1
J2_HORIZONTAL = 3  # Yaw rate

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
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    joystick = init_joystick()

    try:
        sock.connect((target_mac, 1))
        print(f"Connected to {target_mac}")

        while True:
            pygame.event.pump()  # Update joystick states
            
            x = round(joystick.get_axis(J1_HORIZONTAL), 2)
            y = round(joystick.get_axis(J1_VERTICAL), 2)
            yaw = round(joystick.get_axis(J2_HORIZONTAL), 2)

            message = f"{x},{y},{yaw}"
            sock.send(message)
            print(f"Sent: {message}")

            time.sleep(1 / 50)  # Maintain 50 Hz

    except bluetooth.btcommon.BluetoothError as e:
        print(f"Bluetooth error: {e}")
    finally:
        sock.close()
        pygame.quit()

if __name__ == "__main__":
    rpi_mac_address = "B8:27:EB:16:C4:AE"  # Replace with your Raspberry Pi's MAC address
    send_controller_data(rpi_mac_address)
