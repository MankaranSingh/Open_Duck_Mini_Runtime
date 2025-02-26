import rospy
import numpy as np
import subprocess
import time
import signal
from sensor_msgs.msg import JointState


# Parameters
N_WAVES = 5  # Number of composite waves to generate
TARGET_JOINTS = ["left_knee"]  # List of joints to control
PUBLISH_RATE = 100  # Hz
TOTAL_DURATION = 20  # Duration of each composite wave in seconds
DT = 1/PUBLISH_RATE  # Time step
RESET_DURATION = 1.5  # Time to hold joints at zero before new wave

def composite_wave(total_duration=20, dt=0.005, lower=-45, upper=45, degrees=True, randomize_offsets=False):
    """
    Generates a composite wave with fully randomized segments.
    Each segment is randomly chosen as a sine, square, or Gaussian noise waveform.
    Each segment’s frequency (or update interval), amplitude, and vertical offset are randomized.
    
    Parameters:
      total_duration: Overall duration of the composite wave (seconds)
      dt: Time step (seconds)
      lower, upper: Lower and upper bounds for the signal
      degrees: If True, the provided bounds are in degrees and are converted to radians
      
    Returns:
      t: Time array for the composite signal
      y: Composite signal (values clipped within [lower, upper])
    """
    # Convert bounds if working in degrees.
    if degrees:
        lower_bound = lower * np.pi / 180
        upper_bound = upper * np.pi / 180
    else:
        lower_bound, upper_bound = lower, upper

    # Choose a random number of segments (e.g., 2 or 3)
    num_segments = np.random.randint(1, 5)
    seg_duration = total_duration / num_segments
    segments_t, segments_y = [], []
    current_time = 0

    for _ in range(num_segments):
        t_seg = np.arange(0, seg_duration, dt)
        wave_type = np.random.choice(['sine', 'square', 'gaussian'])

        if wave_type in ['sine', 'square']:
            # Random frequency between 0.2 and 2.0 Hz.
            freq = np.random.uniform(0.5, 2.0)
            # Default amplitude is half the full range; choose a random fraction of that.
            default_amp = (upper_bound - lower_bound) / 3.0
            amp = np.random.uniform(0.1, 1.0) * default_amp
            offset = 0.0
            if randomize_offsets:
                # Random vertical offset: must be chosen so that [offset-amp, offset+amp] is within bounds.
                min_offset = lower_bound + amp
                max_offset = upper_bound - amp
                if min_offset > max_offset:
                    offset = (lower_bound + upper_bound) / 2.0
                else:
                    offset = np.random.uniform(min_offset, max_offset)
            phase = np.random.uniform(0, 2*np.pi)
            if wave_type == 'sine':
                y_seg = offset + amp * np.sin(2 * np.pi * freq * t_seg + phase)
            else:  # square wave
                y_seg = offset + amp * np.sign(np.sin(2 * np.pi * freq * t_seg + phase))
        else:  # Gaussian noise segment
            # Random update interval between 0.2 and 1.0 seconds.
            update_interval = np.random.uniform(0.1, 0.8)
            # Default standard deviation is one-sixth the range; randomize it.
            default_std = (upper_bound - lower_bound) / 2.0
            std_dev = np.random.uniform(0.1, 1.0) * default_std
            # Random mean offset within the allowed range.
            mean = 0.0 #np.random.uniform(lower_bound, upper_bound)
            y_seg = np.zeros_like(t_seg)
            last_update_time = -update_interval
            noise_val = np.clip(np.random.normal(mean, std_dev), lower_bound, upper_bound)
            for i, t_val in enumerate(t_seg):
                if t_val - last_update_time >= update_interval:
                    noise_val = np.clip(np.random.normal(mean, std_dev), lower_bound, upper_bound)
                    last_update_time = t_val
                y_seg[i] = noise_val

        # Ensure segment remains within bounds.
        y_seg = np.clip(y_seg, lower_bound, upper_bound)
        segments_t.append(t_seg + current_time)
        segments_y.append(y_seg)
        current_time += seg_duration

    t = np.concatenate(segments_t)
    y = np.concatenate(segments_y)
    return t, y

# Initialize ROS node
rospy.init_node("uan_data_collector")
pub = rospy.Publisher("/target_joint_states", JointState, queue_size=10)
rate = rospy.Rate(PUBLISH_RATE)

for wave_count in range(N_WAVES):
    rospy.loginfo(f"Starting wave {wave_count + 1} of {N_WAVES}")
    
    # Continuously publish reset messages for RESET_DURATION seconds
    start_reset_time = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - start_reset_time < RESET_DURATION and not rospy.is_shutdown():
        reset_msg = JointState()
        reset_msg.header.stamp = rospy.Time.now()
        reset_msg.name = TARGET_JOINTS
        reset_msg.position = [0.0] * len(TARGET_JOINTS)
        pub.publish(reset_msg)
        rate.sleep()
    
    # Start rosbag recording (recording all topics) using a terminal command
    record_cmd = ["rosbag", "record", "-a", "-O", f"wave_{wave_count + 1}.bag"]
    rospy.loginfo("Starting rosbag recording: " + " ".join(record_cmd))
    bag_process = subprocess.Popen(record_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Generate composite waves for each joint
    wave_data = {}
    for joint in TARGET_JOINTS:
        t, y = composite_wave(total_duration=TOTAL_DURATION, dt=DT)
        wave_data[joint] = (t, y)

    time.sleep(0.5)  # Wait for rosbag to start recording
    index = 0
    while not rospy.is_shutdown():
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = TARGET_JOINTS
        msg.position = []
    
        wave_complete = True
        for joint in TARGET_JOINTS:
            t, y = wave_data[joint]
            if index < len(y):
                msg.position.append(y[index])
                wave_complete = False
            else:
                msg.position.append(y[-1])
    
        pub.publish(msg)
        rate.sleep()
        index += 1
    
        if wave_complete:
            break
    
    # Stop rosbag recording after the wave completes
    rospy.loginfo("Stopping rosbag recording.")
    bag_process.send_signal(signal.SIGINT)
    bag_process.wait()
