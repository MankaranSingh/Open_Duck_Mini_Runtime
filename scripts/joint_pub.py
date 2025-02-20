#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import numpy as np

from mini_bdx_runtime.hwi_feetech_pwm_control import HWI 

def main():
    rospy.init_node("real_joint_state_publisher")
    pub = rospy.Publisher("/current_joint_states", JointState, queue_size=10)
    rate = rospy.Rate(100)  # 10 Hz update rate

    # Initialize the hardware interface
    hwi = HWI(usb_port="/dev/ttyACM0")
    hwi.turn_off()

    # Ensure that the hardware is safely turned off on shutdown
    rospy.on_shutdown(hwi.turn_off)

    # Get the joint names from the HWI instance (antenna joints are not present)
    joint_names = list(hwi.joints.keys())

    while not rospy.is_shutdown():
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names

        # Get the present joint positions, velocities, and voltages (in radians, rad/s, and volts)
        positions = hwi.get_present_positions()
        velocities = hwi.get_present_velocities()
        voltages = hwi.get_present_voltages()

        # Convert to lists (in case the returned type is a numpy array)
        msg.position = positions.tolist() if hasattr(positions, "tolist") else list(positions)
        msg.velocity = velocities.tolist() if hasattr(velocities, "tolist") else list(velocities)
        msg.effort = voltages.tolist() if hasattr(voltages, "tolist") else list(voltages)

        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
