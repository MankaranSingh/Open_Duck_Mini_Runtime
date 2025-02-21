import rospy
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import (
    quaternion_multiply, 
    quaternion_inverse, 
    quaternion_from_euler, 
    euler_from_quaternion
)

# Set up I2C and sensor
i2c = I2C(3)
sensor = adafruit_bno055.BNO055_I2C(i2c)
sensor.mode = adafruit_bno055.IMUPLUS_MODE

# Initialize ROS node and publisher
rospy.init_node("bno055_imu_publisher")
imu_pub = rospy.Publisher("/imu/data", Imu, queue_size=1)
rate = rospy.Rate(100)

# Define fixed rotation for axis re-mapping: -90° about z
# (Re-maps sensor frame: x=right, y=forward, z=up to desired frame: x=forward, y=left, z=up)
q_fixed = [0.0, 0.0, -0.7071, 0.7071]
R_z = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])

# Hard-coded pitch offset correction (in radians)
# If your sensor's reading shows an unwanted pitch offset of 0.06488 radians,
# then use a correction of -0.06488 radians (rotation about y-axis).
pitch_offset = 0.13
q_pitch_corr = quaternion_from_euler(0, -pitch_offset, 0)
R_pitch = np.array([
    [ np.cos(-pitch_offset), 0, np.sin(-pitch_offset)],
    [ 0,                    1,                  0],
    [-np.sin(-pitch_offset), 0, np.cos(-pitch_offset)]
])

print("IMU node running ..")
while not rospy.is_shutdown():
    imu_msg = Imu()
    
    # Get sensor quaternion (sensor.quaternion returns (w, x, y, z))
    qw, qx, qy, qz = sensor.quaternion
    # Convert sensor quaternion to (x, y, z, w) format used by tf.transformations:
    q_sensor_raw = [qx, qy, qz, qw]
    
    # First, re-map the sensor frame using the fixed rotation:
    q_corr = quaternion_multiply(q_fixed, q_sensor_raw)
    q_corr = quaternion_multiply(q_corr, quaternion_inverse(q_fixed))
    
    # Then, apply the hard-coded pitch correction:
    q_final = quaternion_multiply(q_pitch_corr, q_corr)
    
    # For logging, convert the final quaternion to Euler angles
    #roll, pitch, yaw = euler_from_quaternion(q_final)
    #rospy.loginfo("Roll: {:.2f}, Pitch: {:.2f}, Yaw: {:.2f}".format(roll, pitch, yaw))
    
    # Fill in the IMU message with the corrected orientation
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "imu_link"
    imu_msg.orientation.x = q_final[0]
    imu_msg.orientation.y = q_final[1]
    imu_msg.orientation.z = q_final[2]
    imu_msg.orientation.w = q_final[3]
    
    # Combine the rotation matrices for vector correction: first fixed re-mapping, then pitch correction
    R_total = R_pitch @ R_z
    gyro = np.dot(R_total, sensor.gyro)
    lin_acc = np.dot(R_total, sensor.linear_acceleration)
    
    imu_msg.angular_velocity.x = gyro[0]
    imu_msg.angular_velocity.y = gyro[1]
    imu_msg.angular_velocity.z = gyro[2]
    
    imu_msg.linear_acceleration.x = lin_acc[0]
    imu_msg.linear_acceleration.y = lin_acc[1]
    imu_msg.linear_acceleration.z = lin_acc[2]
    
    imu_pub.publish(imu_msg)
