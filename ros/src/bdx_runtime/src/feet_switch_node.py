import rospy
import RPi.GPIO as GPIO
from std_msgs.msg import Bool
from std_msgs.msg import Int32

# GPIO Pins
LEFT_FOOT_PIN = 6
RIGHT_FOOT_PIN = 5

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LEFT_FOOT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(RIGHT_FOOT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def feet_switch_publisher():
    rospy.init_node('feet_switch_node', anonymous=True)
    feet_pub = rospy.Publisher('/feet_switch', Int32, queue_size=10)
    rate = rospy.Rate(50)  # 100 Hz
    
    while not rospy.is_shutdown():
        left_state = GPIO.input(LEFT_FOOT_PIN) == GPIO.LOW  # True if pressed
        right_state = GPIO.input(RIGHT_FOOT_PIN) == GPIO.LOW  # True if pressed

        feet_state = (1 if left_state else 0) | ((1 if right_state else 0) << 1)
        feet_pub.publish(feet_state)

        rate.sleep()
    
if __name__ == '__main__':
    try:
        setup_gpio()
        feet_switch_publisher()
    except rospy.ROSInterruptException:
        pass
    finally:
        GPIO.cleanup()
