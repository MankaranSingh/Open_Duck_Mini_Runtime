import rosbag
import numpy as np
import argparse
import os
from scipy.interpolate import interp1d

def extract_and_interpolate(bag_file, target_joint, dt):
    target_times = []
    target_positions = []
    current_times = []
    current_positions = []
    current_velocities = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == "/target_joint_states":
                target_times.append(msg.header.stamp.to_sec())
                target_positions.append(msg.position[0])
            elif topic == "/current_joint_states":
                current_times.append(msg.header.stamp.to_sec())
                current_positions.append(msg.position[target_joint])
                current_velocities.append(msg.velocity[target_joint])
    
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
    
    return data

def process_bag_files(directory, target_joint, dt=0.004):
    npy_dir = os.path.join(directory, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    
    for file in os.listdir(directory):
        if file.endswith(".bag"):
            bag_path = os.path.join(directory, file)
            data = extract_and_interpolate(bag_path, target_joint, dt)
            npy_path = os.path.join(npy_dir, file.replace(".bag", ".npy"))
            np.save(npy_path, data)
            print(f"Saved {npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to the directory containing ROS1 bag files")
    parser.add_argument("target_joint", type=int, help="target joint for which data was collected")
    args = parser.parse_args()
    
    process_bag_files(args.directory, args.target_joint, dt=0.004)
