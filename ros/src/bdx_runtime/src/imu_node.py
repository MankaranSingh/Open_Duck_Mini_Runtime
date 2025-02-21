import rospy
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_multiply, quaternion_inverse

# Set up I2C and sensor
i2c = I2C(3)
sensor = adafruit_bno055.BNO055_I2C(i2c)
sensor.mode = adafruit_bno055.IMUPLUS_MODE

# Initialize ROS node and publisher
rospy.init_node("bno055_imu_publisher")
imu_pub = rospy.Publisher("/imu/data", Imu, queue_size=1)
rate = rospy.Rate(100)

# Define the fixed rotation as a quaternion (for a -90° rotation about z)
# Here the quaternion is given in (x, y, z, w) format.
q_fixed = [0.0, 0.0, -0.7071, 0.7071]

# Define the corresponding rotation matrix for vectors
R = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 1]])

print("IMU node running ..")
while not rospy.is_shutdown():
    imu_msg = Imu()
    
    # Get sensor quaternion (sensor.quaternion returns (w, x, y, z))
    qw, qx, qy, qz = sensor.quaternion
    # Convert sensor quaternion to (x, y, z, w) for tf.transformations:
    q_sensor = [qx, qy, qz, qw]
    
    # Correct the quaternion by conjugation:
    # q_corrected = q_fixed * q_sensor * q_fixed_inverse
    q_corr = quaternion_multiply(q_fixed, q_sensor)
    q_corr = quaternion_multiply(q_corr, quaternion_inverse(q_fixed))
    # q_corr is in (x, y, z, w) order
    
    # For the IMU message, assign orientation with q_corr:
    imu_msg.orientation.x = q_corr[0]
    imu_msg.orientation.y = q_corr[1]
    imu_msg.orientation.z = q_corr[2]
    imu_msg.orientation.w = q_corr[3]
    
    # Rotate gyro and linear acceleration vectors using the rotation matrix R
    gyro = np.dot(R, sensor.gyro)
    lin_acc = np.dot(R, sensor.linear_acceleration)
    
    imu_msg.angular_velocity.x = gyro[0]
    imu_msg.angular_velocity.y = gyro[1]
    imu_msg.angular_velocity.z = gyro[2]
    
    imu_msg.linear_acceleration.x = lin_acc[0]
    imu_msg.linear_acceleration.y = lin_acc[1]
    imu_msg.linear_acceleration.z = lin_acc[2]
    
    # Set header and publish
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "imu_link"
    imu_pub.publish(imu_msg)
    rate.sleep()
