import rosbag
import numpy as np
import argparse
from scipy.interpolate import interp1d

def extract_and_interpolate(bag_file, dt):
    target_times = []
    target_positions = []
    current_times = []
    current_positions = []
    current_velocities = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == "/target_joint_states":
                target_times.append(msg.header.stamp.to_sec())
                target_positions.append(msg.position)
            elif topic == "/current_joint_states":
                current_times.append(msg.header.stamp.to_sec())
                current_positions.append(msg.position)
                current_velocities.append(msg.velocity)
    
    target_times = np.array(target_times)
    target_positions = np.array(target_positions)
    current_times = np.array(current_times)
    current_positions = np.array(current_positions)
    current_velocities = np.array(current_velocities)
    
    start_time = max(target_times[0], current_times[0])
    end_time = min(target_times[-1], current_times[-1])
    time_interp = np.arange(start_time, end_time, dt)
    
    target_interp = interp1d(target_times, target_positions, axis=0, kind='linear', fill_value='extrapolate')(time_interp)
    current_interp = interp1d(current_times, current_positions, axis=0, kind='linear', fill_value='extrapolate')(time_interp)
    velocity_interp = interp1d(current_times, current_velocities, axis=0, kind='linear', fill_value='extrapolate')(time_interp)
    
    data = {
        "position_targets": np.array(target_interp),
        "actual_positions": np.array(current_interp),
        "actual_velocities": np.array(velocity_interp)
    }
    np.save(args.bag_file.replace(".bag", ".npy"), data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_file", type=str, help="Path to the ROS1 bag file")
    args = parser.parse_args()
    
    data = extract_and_interpolate(args.bag_file, dt=0.01)
