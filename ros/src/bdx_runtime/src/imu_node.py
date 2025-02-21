import rospy
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
import math
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_from_euler


i2c = I2C(3)
sensor = adafruit_bno055.BNO055_I2C(i2c)
sensor.mode = adafruit_bno055.IMUPLUS_MODE

# Initialize ROS node
rospy.init_node("bno055_imu_publisher")

# Create a publisher
imu_pub = rospy.Publisher("/imu/data", Imu, queue_size=1)

# Set the loop rate (100 Hz)
rate = rospy.Rate(100)

print("IMU node running ..")
while not rospy.is_shutdown():
    imu_msg = Imu()
    
    qw, qx, qy, qz = sensor.quaternion
    gyro = sensor.gyro
    lin_acc = sensor.linear_acceleration
        
    # Fill IMU message
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "imu_link"
    
    imu_msg.orientation.x = qx
    imu_msg.orientation.y = qy
    imu_msg.orientation.z = qz
    imu_msg.orientation.w = qw
    
    imu_msg.angular_velocity.x = gyro[0]
    imu_msg.angular_velocity.y = gyro[1]
    imu_msg.angular_velocity.z = gyro[2]
    
    imu_msg.linear_acceleration.x = lin_acc[0]
    imu_msg.linear_acceleration.y = lin_acc[1]
    imu_msg.linear_acceleration.z = lin_acc[2]
    
    imu_pub.publish(imu_msg)
    rate.sleep()
