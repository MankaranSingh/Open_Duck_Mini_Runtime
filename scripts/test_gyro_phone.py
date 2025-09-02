#!/usr/bin/env python3
"""
Phyphox Gyroscope Data Reader
Reads gyroscope data from Phyphox streaming interface
Supports both HTTP GET requests and UDP socket listening
"""

import socket
import json
import time
import requests
import threading
import math
from datetime import datetime

class PhyphoxReader:
    def __init__(self, host="192.168.77.239", port=8080):
        self.host = host
        self.port = port
        self.running = False

        # Joystick-style angular velocity thresholds (rad/s)
        self.trigger_thresholds = {"X": 3.0, "Y": 4.0, "Z": 3.4}
        self.reset_thresholds   = {"X": 2.0, "Y": 3.4, "Z": 3.2}
                
        # Flag states (6 flags: X_CW, X_CCW, Y_CW, Y_CCW, Z_CW, Z_CCW)
        self.flags = {
            "X_CW": False,   # X-axis clockwise
            "X_CCW": False,  # X-axis counter-clockwise
            "Y_CW": False,   # Y-axis clockwise  
            "Y_CCW": False,  # Y-axis counter-clockwise
            "Z_CW": False,   # Z-axis clockwise
            "Z_CCW": False   # Z-axis counter-clockwise
        }
        
    def reset_flags(self):
        """Reset all flags to False"""
        for key in self.flags:
            self.flags[key] = False
    
    def update_flags(self, gyr_x, gyr_y, gyr_z):
        """Update flags based on current angular velocities"""

        axis_values = {"X": gyr_x, "Y": gyr_y, "Z": gyr_z}

        for axis, value in axis_values.items():
            trigger = self.trigger_thresholds[axis]
            reset   = self.reset_thresholds[axis]

            cw_flag  = f"{axis}_CW"
            ccw_flag = f"{axis}_CCW"

            # Trigger logic
            if abs(value) > trigger:
                if not self.flags[cw_flag] and not self.flags[ccw_flag]:
                    if value > 0:
                        self.flags[cw_flag] = True
                    else:
                        self.flags[ccw_flag] = True

            # Reset logic
            if self.flags[cw_flag] and value < -reset:
                self.flags[cw_flag] = False
            elif self.flags[ccw_flag] and value > reset:
                self.flags[ccw_flag] = False

    def format_flags_display(self):
        """Format flags for single line display"""
        flag_status = ""
        
        for flag in ["X_CW", "X_CCW", "Y_CW", "Y_CCW", "Z_CW", "Z_CCW"]:
            if self.flags[flag]:
                flag_status += f"[{flag}] "
            else:
                flag_status += f" {flag}  "
        
        return flag_status.strip()
        
    def read_http_data(self):
        """Read data via HTTP GET requests (polling method)"""
        url = f"http://{self.host}:{self.port}/get?gyrX&gyrY&gyrZ"
        
        print(f"Reading gyroscope data from: {url}")
        print("Commands: 'r' = reset flags, 't' = adjust thresholds, Ctrl+C = quit")
        print("Current thresholds per axis:")
        for axis in ["X", "Y", "Z"]:
            print(f"  {axis}: Trigger={self.trigger_thresholds[axis]:.1f}, Reset={self.reset_thresholds[axis]:.1f}")

        print("🕹️  Joystick mode: Flags trigger on angular velocity, reset on opposite movement\n")
        
        # Start input thread for commands
        def input_thread():
            while True:
                try:
                    cmd = input().strip().lower()
                    if cmd == 'r':
                        self.reset_flags()
                        print("🔄 All flags reset\n")
                    elif cmd == 't':
                        try:
                            axis = input("Choose axis (X/Y/Z): ").upper()
                            if axis in ["X", "Y", "Z"]:
                                trigger = float(input(f"Enter trigger threshold for {axis} (rad/s): "))
                                reset   = float(input(f"Enter reset threshold for {axis} (rad/s): "))
                                self.trigger_thresholds[axis] = trigger
                                self.reset_thresholds[axis]   = reset
                                print(f"✓ Thresholds updated for {axis}: Trigger={trigger:.1f}, Reset={reset:.1f}\n")
                            else:
                                print("❌ Invalid axis, must be X, Y, or Z\n")
                        except ValueError:
                            print("❌ Invalid threshold values\n")
                except:
                    break
        
        threading.Thread(target=input_thread, daemon=True).start()
        
        try:
            while True:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract gyroscope values
                        gyr_x = data.get('buffer', {}).get('gyrX', {}).get('buffer', [0])[-1] if data.get('buffer', {}).get('gyrX', {}).get('buffer') else 0
                        gyr_y = data.get('buffer', {}).get('gyrY', {}).get('buffer', [0])[-1] if data.get('buffer', {}).get('gyrY', {}).get('buffer') else 0
                        gyr_z = data.get('buffer', {}).get('gyrZ', {}).get('buffer', [0])[-1] if data.get('buffer', {}).get('gyrZ', {}).get('buffer') else 0
                        
                        # Update flags based on current angular velocities
                        self.update_flags(gyr_x, gyr_y, gyr_z)
                        
                        # Display data and flags
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        flags_display = self.format_flags_display()
                        
                        print(f"[{timestamp}] Gyro: X:{gyr_x:6.2f} Y:{gyr_y:6.2f} Z:{gyr_z:6.2f} rad/s | "
                              f"Flags: {flags_display}")
                              
                    else:
                        print(f"HTTP Error: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"Connection error: {e}")
                    time.sleep(1)
                    
                time.sleep(0.02)  # 20Hz sampling rate for better joystick responsiveness
                
        except KeyboardInterrupt:
            print("\nStopped reading data")
    


def main():
    reader = PhyphoxReader()
    
    print("Phyphox Gyroscope Data Reader with 6-Flag Detection")
    print("===================================================")
    print("Choose option:")
    print("1. Start reading gyroscope data")
    print("2. Test connection")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        reader.read_http_data()
    elif choice == "2":
        # Test connection
        print("Testing connection...")
        try:
            response = requests.get(f"http://{reader.host}:{reader.port}/", timeout=5)
            print(f"✓ Connection successful! Status: {response.status_code}")
            print(f"Response preview: {response.text[:200]}...")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Alternative: Direct HTTP reading
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "--direct":
        reader = PhyphoxReader()
        print("Direct HTTP reading mode")
        reader.read_http_data()
    else:
        main()
