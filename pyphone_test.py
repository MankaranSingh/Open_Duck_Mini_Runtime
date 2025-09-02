#!/usr/bin/env python3
import socket

HOST = "192.168.77.239"   # replace with your phone/server IP
PORT = 5000

def gyro_reader():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to server")

        baseline = None  # will hold first reading

        while True:
            data = s.recv(1024)
            if not data:
                break

            try:
                yaw, pitch, roll = map(float, data.decode().strip().split(","))
            except ValueError:
                continue  # skip malformed lines

            if baseline is None:
                baseline = (yaw, pitch, roll)
                print("Calibration set:", baseline)
                continue

            # Subtract baseline
            rel_yaw = yaw - baseline[0]
            rel_pitch = pitch - baseline[1]
            rel_roll = roll - baseline[2]

            print(f"Relative -> yaw: {rel_yaw:.3f}, pitch: {rel_pitch:.3f}, roll: {rel_roll:.3f}")