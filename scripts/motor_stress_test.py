from pypot.feetech import FeetechSTS3215IO
import time

io = FeetechSTS3215IO("/dev/ttyAMA0")
servo_ids = list(range(0, 11))

# Setup servos for position control
for servo_id in servo_ids:
    try:
        io.set_mode({servo_id: 0})
        io.set_acceleration({servo_id: 0})  # Maximum speed
        io.set_maximum_acceleration({servo_id: 0})  # Maximum speed
    except:
        pass

# Continuous fast rotation
try:
    while True:
        
 # Move to 180 degrees
        io.set_goal_position({servo_id: 180 for servo_id in servo_ids})
        time.sleep(1)
    
        # Move to -180 degrees
        io.set_goal_position({servo_id: -180 for servo_id in servo_ids})
        time.sleep(1)
        
except KeyboardInterrupt:
    # Stop at center
    io.set_goal_position({servo_id: 0 for servo_id in servo_ids})
