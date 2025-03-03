import rospy
import math
import argparse
from sensor_msgs.msg import JointState

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Publish wave signals to target joints.")
parser.add_argument("--wave_type", type=str, choices=["sine", "square"], default="sine", help="Type of wave to generate (sine or square)")
parser.add_argument("--frequency", type=float, default=1.0, help="Frequency of the wave in Hz")
parser.add_argument("--target_joints", nargs='+', default=["left_knee"], help="List of target joints")
args = parser.parse_args()

# Initialize ROS node
rospy.init_node("manual_joint_publisher")
pub = rospy.Publisher("/target_joint_states", JointState, queue_size=1)

rate = rospy.Rate(100)  # 100 Hz update rate

# Define joint names (Replace with actual joint names from your URDF)
joint_names = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", 
               "head_pitch", "head_yaw", "head_roll", "right_hip_yaw", 
               "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]  

start_time = rospy.Time.now().to_sec()

while not rospy.is_shutdown():
    current_time = rospy.Time.now().to_sec()
    elapsed_time = current_time - start_time  # Time since start

    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = joint_names
    msg.position = [0.0] * len(joint_names)  # Initialize all positions to 0.0
    
    # Apply selected wave type to target joints
    for joint in args.target_joints:
        if joint in joint_names:
            index = joint_names.index(joint)
            if args.wave_type == "sine":
                msg.position[index] = 0.5 * math.sin(2 * math.pi * args.frequency * elapsed_time)
            elif args.wave_type == "square":
                msg.position[index] = 0.5 if (math.sin(2 * math.pi * args.frequency * elapsed_time) >= 0) else -0.5
    
    msg.velocity = []
    msg.effort = []

    pub.publish(msg)
    rate.sleep()
