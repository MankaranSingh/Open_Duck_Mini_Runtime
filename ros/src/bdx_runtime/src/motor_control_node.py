#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import numpy as np
from std_srvs.srv import Trigger, TriggerResponse
from mini_bdx_runtime.hwi_feetech_pypot import HWI 


class JointController:
    def __init__(self):
        rospy.init_node("joint_controller")

        # Initialize hardware interface and set starting target positions
        self.hwi = HWI(usb_port="/dev/ttyACM0")
        self.hwi.turn_on()
        self.target_positions = self.hwi.init_pos.copy()  # start with the hardware's init positions

        self.dummy_joints = ["left_antenna", "right_antenna"] # Dummy joints for urdf viz. Not controlled by hardware.
        self.dummy_joint_values = [0.0, 0.0]
        self.dummy_joint_insert_idx = 9
        self.joint_names = list(self.hwi.joints.keys()).insert(self.dummy_joint_insert_idx, self.dummy_joints)

        # Publisher for current joint states
        self.pub = rospy.Publisher("/current_joint_states", JointState, queue_size=10)
        # Subscriber for target joint states
        rospy.Subscriber("/target_joint_states", JointState, self.target_callback)

        # Service servers using std_srvs/Trigger:
        # - turn_on: activates the hardware interface.
        # - turn_off: disables the hardware.
        # - set_kps: sets KP values for all joints using the parameter "~kps_value".
        self.s_turn_on = rospy.Service("turn_on", Trigger, self.handle_turn_on)
        self.s_turn_off = rospy.Service("turn_off", Trigger, self.handle_turn_off)
        self.s_set_kps = rospy.Service("set_kps", Trigger, self.handle_set_kps)
        self.s_set_kds = rospy.Service("set_kds", Trigger, self.handle_set_kds)

        self.rate = rospy.Rate(100)  # 10 Hz update rate
        rospy.loginfo("Joint Controller initialized.")

    def target_callback(self, msg):
        """
        Update target positions for joints. Only those joints mentioned in the incoming message are updated.
        """
        for name, pos in zip(msg.name, msg.position):
            if name in self.dummy_joints:
                continue
            if name in self.target_positions:
                self.target_positions[name] = pos
            else:
                rospy.logwarn("Received target for unknown joint: %s", name)

    def handle_turn_on(self, req):
        """
        Service callback to turn on the hardware.
        """
        try:
            self.hwi.turn_on()
            return TriggerResponse(success=True, message="Hardware turned on successfully.")
        except Exception as e:
            rospy.logerr("Error in turn_on: %s", str(e))
            return TriggerResponse(success=False, message="Failed to turn on hardware: " + str(e))

    def handle_turn_off(self, req):
        """
        Service callback to turn off the hardware.
        """
        try:
            self.hwi.turn_off()
            return TriggerResponse(success=True, message="Hardware turned off successfully.")
        except Exception as e:
            rospy.logerr("Error in turn_off: %s", str(e))
            return TriggerResponse(success=False, message="Failed to turn off hardware: " + str(e))

    def handle_set_kps(self, req):
        """
        Service callback to set KP values.
        It reads a single float from the parameter "~kps_value" and applies it to all joints.
        """
        try:
            # Retrieve a single float value from the ROS parameter server (default: 32.0)
            kps_value = rospy.get_param("kp", 32.0)
            # Create a list with the same value for each joint
            kps_list = [kps_value] * len(self.hwi.joints)
            self.hwi.set_kps(kps_list)
            message = "KP values set to {} for all joints.".format(kps_value)
            rospy.loginfo(message)
            return TriggerResponse(success=True, message=message)
        except Exception as e:
            rospy.logerr("Error in set_kps: %s", str(e))
            return TriggerResponse(success=False, message="Failed to set KP values: " + str(e))
    
    def handle_set_kds(self, req):
        """
        Service callback to set KP values.
        It reads a single float from the parameter "~kds_value" and applies it to all joints.
        """
        try:
            # Retrieve a single float value from the ROS parameter server (default: 32.0)
            kds_value = rospy.get_param("kd", 0.00)
            # Create a list with the same value for each joint
            kds_list = [kds_value] * len(self.hwi.joints)
            self.hwi.set_kds(kds_list)
            message = "KD values set to {} for all joints.".format(kds_value)
            rospy.loginfo(message)
            return TriggerResponse(success=True, message=message)
        except Exception as e:
            rospy.logerr("Error in set_kds: %s", str(e))
            return TriggerResponse(success=False, message="Failed to set KD values: " + str(e))

    def run(self):
        # Ensure hardware is safely turned off on shutdown.
        rospy.on_shutdown(self.hwi.turn_off)

        while not rospy.is_shutdown():
            # Apply target positions to the hardware
            self.hwi.set_position_all(self.target_positions)

            # Prepare and publish the JointState message with current data.
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = self.joint_names

            positions = self.hwi.get_present_positions()
            velocities = self.hwi.get_present_velocities()
            #voltages = self.hwi.get_present_voltages()

            msg.position = positions.tolist() if hasattr(positions, "tolist") else list(positions)
            msg.velocity = velocities.tolist() if hasattr(velocities, "tolist") else list(velocities)
            #msg.effort   = voltages.tolist()   if hasattr(voltages, "tolist")   else list(voltages)

            msg.position.insert(self.dummy_joint_insert_idx, self.dummy_joint_values)
            msg.velocity.insert(self.dummy_joint_insert_idx, self.dummy_joint_values)
            #msg.effort.insert(self.dummy_joint_insert_idx, self.dummy_joint_values)

            self.pub.publish(msg)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        controller = JointController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
