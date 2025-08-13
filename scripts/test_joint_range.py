import time
import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI

def test_joint_range():
    # Initialize the hardware interface
    print("Initializing HWI...")
    hwi = HWI()
    
    # Use lower stiffness for safety
    print("Setting lower stiffness...")
    hwi.set_kps(np.ones(len(hwi.joints)) * 5)
    
    # Define joint test ranges in radians (min, max, step)
    joint_ranges = {
        "neck_pitch": (-0.5, 0.5, 0.1),
        "head_pitch": (-0.5, 0.5, 0.1),
        "head_yaw": (-0.5, 0.5, 0.1),
        "tail": (-0.8, 0.8, 0.2),
        "right_hip_yaw": (-0.5, 0.5, 0.1),
        "right_hip_roll": (-0.3, 0.3, 0.1),
        "right_hip_pitch": (-0.5, 0.5, 0.1),
        "right_knee": (-0.5, 0.5, 0.1),
        "right_ankle": (-0.5, 0.5, 0.1),
        "left_hip_yaw": (-0.5, 0.5, 0.1),
        "left_hip_roll": (-0.3, 0.3, 0.1),
        "left_hip_pitch": (-0.5, 0.5, 0.1),
        "left_knee": (-0.5, 0.5, 0.1),
        "left_ankle": (-0.5, 0.5, 0.1),
    }
    
    try:
        # First move to initial position
        print("Moving to initial position...")
        hwi.set_position_all(hwi.init_pos)
        time.sleep(1)
        
        # Test each joint one by one
        for joint_name in hwi.joints:
            print(f"\nTesting joint: {joint_name}")
            input("Press Enter to start testing this joint...")
            
            min_angle, max_angle, step = joint_ranges[joint_name]
            
            # Test range from min to max
            for angle in np.arange(min_angle, max_angle + step, step):
                angle = round(angle, 2)
                print(f"Setting {joint_name} to {angle} radians")
                
                # Create a position dictionary with only the current joint moving
                positions = hwi.init_pos.copy()
                positions[joint_name] = angle
                
                # Set the position
                hwi.set_position_all(positions)
                
                # Wait for confirmation before next step
                input("Press Enter for next position...")
            
            # Return to neutral position
            print(f"Returning {joint_name} to initial position...")
            positions = hwi.init_pos.copy()
            hwi.set_position_all(positions)
            time.sleep(0.1)
            
            input("Press Enter to test the next joint...")
        
        print("Joint range testing complete!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        # Return to initial position
        print("Returning to initial position...")
        hwi.set_position_all(hwi.init_pos)
        time.sleep(1)
        
        # Turn off torque
        print("Turning off torque...")
        hwi.turn_off()

if __name__ == "__main__":
    test_joint_range()
