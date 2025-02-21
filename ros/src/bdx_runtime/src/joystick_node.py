#!/usr/bin/env python

import rospy
import bluetooth
from geometry_msgs.msg import Twist

def listen_for_controller_data():
    rospy.init_node("bluetooth_joystick_receiver", anonymous=True)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rate = rospy.Rate(50)  # 50 Hz

    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    try:
        server_sock.bind(("", bluetooth.PORT_ANY))
        server_sock.listen(1)
        print("Waiting for connection...")

        client_sock, address = server_sock.accept()
        print(f"Accepted connection from {address}")

        while not rospy.is_shutdown():
            data = client_sock.recv(1024).decode("utf-8").strip()
            if not data:
                continue
            
            try:
                x, y, yaw = map(float, data.split(","))
                print(f"Received - X: {x}, Y: {y}, Yaw: {yaw}")

                # Publish as a ROS Twist message
                twist_msg = Twist()
                twist_msg.linear.x = x
                twist_msg.linear.y = y
                twist_msg.angular.z = yaw
                pub.publish(twist_msg)

            except ValueError:
                print(f"Invalid data received: {data}")

            rate.sleep()

    except bluetooth.btcommon.BluetoothError as e:
        print(f"Bluetooth error: {e}")
    finally:
        client_sock.close()
        server_sock.close()

if __name__ == "__main__":
    try:
        listen_for_controller_data()
    except rospy.ROSInterruptException:
        pass
