import rospy
import numpy as np
import subprocess
import time
import signal
from sensor_msgs.msg import JointState

np.random.seed(42)

# Parameters
N_WAVES = 5  # Number of composite waves to generate
TARGET_JOINTS = ["left_knee"]  # List of joints to control
PUBLISH_RATE = 100  # Hz
TOTAL_DURATION = 10  # Duration of each composite wave in seconds
DT = 1/PUBLISH_RATE  # Time step
RESET_DURATION = 3  # Time to hold joints at zero before new wave

def composite_wave(total_duration=10, dt=0.01, lower=-45, upper=45, degrees=True, randomize_offsets=False):
    """
    Generates a composite wave with fully randomized segments.
    Each segment is randomly chosen as one of several waveform types:
      'sine', 'square', 'gaussian', 'sine_squared', 'compound', 'cubic_triangle'.
    Each segment's frequency (or update interval), amplitude, and vertical offset are randomized.
    
    For the 'sine_squared' type, the effective time (from the start of the segment) is clipped to 4 seconds.
    This ensures that the frequency (which is determined by (t*freq)**2) increases with time
    only until t = 4 (local to the segment) and then remains constant.
    
    Parameters:
      total_duration: Overall duration of the composite wave (seconds)
      dt: Time step (seconds)
      lower, upper: Lower and upper bounds for the signal
      degrees: If True, the provided bounds are in degrees and are converted to radians
      randomize_offsets: If True, random vertical offsets are used (ensuring the waveform remains within bounds)
      
    Returns:
      t: Time array for the composite signal
      y: Composite signal (values clipped within [lower_bound, upper_bound])
    """
    # Convert bounds if working in degrees.
    if degrees:
        lower_bound = lower * np.pi / 180
        upper_bound = upper * np.pi / 180
    else:
        lower_bound, upper_bound = lower, upper

    # Choose a random number of segments (between 1 and 4)
    num_segments = np.random.randint(1, 3)
    seg_duration = total_duration / num_segments
    segments_t, segments_y = [], []
    current_time = 0

    for _ in range(num_segments):
        t_seg = np.arange(0, seg_duration, dt)
        # Choose from all available wave types.
        wave_type = np.random.choice(['sine', 'square', 'gaussian', 'sine_squared', 'compound', 'cubic_triangle'])

        if wave_type in ['sine', 'square', 'sine_squared', 'compound']:
            # Random frequency between 0.5 and 2.0 Hz.
            freq = np.random.uniform(0.5, 2.0)
            # Default amplitude is a fraction of the full range.
            default_amp = (upper_bound - lower_bound) / 6.0
            amp = np.random.uniform(0.1, 1.0) * default_amp
            # Default offset is 0; if randomize_offsets, choose an offset that keeps [offset-amp, offset+amp] in bounds.
            offset = 0.0
            if randomize_offsets:
                min_offset = lower_bound + amp
                max_offset = upper_bound - amp
                if min_offset > max_offset:
                    offset = (lower_bound + upper_bound) / 2.0
                else:
                    offset = np.random.uniform(min_offset, max_offset)
            phase = np.random.uniform(0, 2*np.pi)

            if wave_type == 'sine':
                y_seg = offset + amp * np.sin(2 * np.pi * freq * t_seg + phase)
            elif wave_type == 'square':
                y_seg = offset + amp * np.sign(np.sin(2 * np.pi * freq * t_seg + phase))
            elif wave_type == 'sine_squared':
                # Use the segment's local time and clip it at 4 seconds.
                y_seg = offset + amp * np.sin((t_seg * freq)**2 + phase)
            elif wave_type == 'compound':
                # A combination of sine components.
                y_seg = offset + amp * (np.sin(t_seg) * (np.pi/2) + np.sin(5.0 * t_seg) * 0.5 * np.sin(2.0 * t_seg))
        
        elif wave_type == 'gaussian':
            # Gaussian noise segment.
            update_interval = np.random.uniform(0.1, 0.4)
            default_std = (upper_bound - lower_bound) / 3.0
            std_dev = np.random.uniform(0.1, 1.0) * default_std
            mean = 0 #np.random.uniform(lower_bound, upper_bound)
            y_seg = np.zeros_like(t_seg)
            last_update_time = -update_interval
            noise_val = np.clip(np.random.normal(mean, std_dev), lower_bound, upper_bound)
            for i, t_val in enumerate(t_seg):
                if t_val - last_update_time >= update_interval:
                    noise_val = np.clip(np.random.normal(mean, std_dev), lower_bound, upper_bound)
                    last_update_time = t_val
                y_seg[i] = noise_val

        elif wave_type == 'cubic_triangle':
            # For cubic_triangle, choose amplitude and offset similarly.
            default_amp = (upper_bound - lower_bound) / 6.0
            amp = np.random.uniform(0.1, 1.0) * default_amp
            offset = 0.0
            if randomize_offsets:
                min_offset = lower_bound + amp
                max_offset = upper_bound - amp
                if min_offset > max_offset:
                    offset = (lower_bound + upper_bound) / 2.0
                else:
                    offset = np.random.uniform(min_offset, max_offset)
            # Choose a random ratio for rising duration (faster rise and slower fall, for example).
            alpha = np.random.uniform(0.3, 0.5)
            n = len(t_seg)
            n_rise = int(np.ceil(n * alpha))
            n_fall = n - n_rise if n > n_rise else 0
            if n_fall > 0:
                # Rising: cubic interpolation from -1 to 1.
                rising = -1 + 2 * (np.linspace(0, 1, n_rise)**3)
                # Falling: cubic interpolation from 1 back to -1.
                falling = 1 - 2 * (np.linspace(0, 1, n_fall)**3)
                base_wave = np.concatenate([rising, falling])
            else:
                base_wave = -1 + 2 * (np.linspace(0, 1, n)**3)
            y_seg = amp * base_wave

        # Ensure the segment remains within bounds.
        y_seg = np.clip(y_seg, lower_bound, upper_bound)
        segments_t.append(t_seg + current_time)
        segments_y.append(y_seg)
        current_time += seg_duration

    t = np.concatenate(segments_t)
    y = np.concatenate(segments_y)
    return t, y

# Initialize ROS node
rospy.init_node("uan_data_collector")
pub = rospy.Publisher("/target_joint_states", JointState, queue_size=1)
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

    time.sleep(5)  # Wait for rosbag to start recording
    index = 0

    # send a bit of zero messages for init
    start_reset_time = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - start_reset_time < 0.5 and not rospy.is_shutdown():
        reset_msg = JointState()
        reset_msg.header.stamp = rospy.Time.now()
        reset_msg.name = TARGET_JOINTS
        reset_msg.position = [0.0] * len(TARGET_JOINTS)
        pub.publish(reset_msg)
        rate.sleep()

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
