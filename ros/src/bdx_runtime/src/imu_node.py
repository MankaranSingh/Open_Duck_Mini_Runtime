import rospy
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
import time
from sensor_msgs.msg import Imu

# Set up I2C and sensor
i2c = I2C(3)
sensor = adafruit_bno055.BNO055_I2C(i2c)
sensor.mode = adafruit_bno055.CONFIG_MODE
time.sleep(0.2)

# Set axis remap
sensor.axis_remap = (
    adafruit_bno055.AXIS_REMAP_Y,        # X (Forward) now maps to physical Y
    adafruit_bno055.AXIS_REMAP_X,        # Y (Right) now maps to physical X
    adafruit_bno055.AXIS_REMAP_Z,        # Z (Up) remains Z
    adafruit_bno055.AXIS_REMAP_POSITIVE, # X (new) keeps positive
    adafruit_bno055.AXIS_REMAP_NEGATIVE, # Y (new) must be inverted
    adafruit_bno055.AXIS_REMAP_POSITIVE  # Z (new) keeps positive
)
time.sleep(0.2)

sensor.mode = adafruit_bno055.IMUPLUS_MODE
time.sleep(0.2)

# Initialize ROS node and publisher
rospy.init_node("bno055_imu_publisher")
imu_pub = rospy.Publisher("/imu/data", Imu, queue_size=1)
rate = rospy.Rate(50)

print("IMU node running ..")
while not rospy.is_shutdown():
    imu_msg = Imu()
    
    # Get sensor quaternion (sensor.quaternion returns (w, x, y, z))
    qw, qx, qy, qz = sensor.quaternion
    gyro = sensor.gyro
    
    # Fill in the IMU message with the corrected orientation
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "imu_link"
    imu_msg.orientation.x = qx
    imu_msg.orientation.y = qy
    imu_msg.orientation.z = qz
    imu_msg.orientation.w = qw
    
    imu_msg.angular_velocity.x = gyro[0]
    imu_msg.angular_velocity.y = gyro[1]
    imu_msg.angular_velocity.z = gyro[2]
    
    imu_pub.publish(imu_msg)
