#!/usr/bin/env python3

import rospy
import curses
import threading
import signal
import sys
from geometry_msgs.msg import Twist

class KeyboardTeleopNode:
    """Node for controlling robot via keyboard input and publishing to cmd_vel."""
    
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('keyboard_teleop_node')
        
        # Initialize the publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Set up velocities
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        
        # Movement speeds
        self.speed_linear = 0.5  # Default linear speed
        self.speed_angular = 0.5  # Default angular speed
        self.speed_increment = 0.1  # Speed increment
        
        # Setup curses for keyboard input
        self.stdscr = None
        self.running = False
        self.key_thread = None
        
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Message formatting
        self.msg = """
        Control Your Robot with Keyboard
        ---------------------------
        Moving around:
            w: forward
            s: backward
            a: strafe left
            d: strafe right
            q: rotate left
            e: rotate right
            
        Speed Control:
            up/down arrows: increase/decrease linear speed
            left/right arrows: increase/decrease angular speed
            
        Other:
            space: stop
            Ctrl+C: quit
        """
        
        rospy.loginfo("Keyboard teleop node initialized")
    
    def process_key(self, key):
        """Process the pressed key and update velocities."""
        if key == ord('w'):
            self.linear_x = self.speed_linear
        elif key == ord('s'):
            self.linear_x = -self.speed_linear
        elif key == ord('a'):
            self.linear_y = self.speed_linear
        elif key == ord('d'):
            self.linear_y = -self.speed_linear
        elif key == ord('q'):
            self.angular_z = self.speed_angular
        elif key == ord('e'):
            self.angular_z = -self.speed_angular
        elif key == ord(' '):  # Space bar
            self.linear_x = 0.0
            self.linear_y = 0.0
            self.angular_z = 0.0
        elif key == curses.KEY_UP:
            self.speed_linear += self.speed_increment
            self.stdscr.addstr(10, 0, f"Linear Speed: {self.speed_linear:.1f}  ")
        elif key == curses.KEY_DOWN:
            self.speed_linear = max(0.1, self.speed_linear - self.speed_increment)
            self.stdscr.addstr(10, 0, f"Linear Speed: {self.speed_linear:.1f}  ")
        elif key == curses.KEY_LEFT:
            self.speed_angular += self.speed_increment
            self.stdscr.addstr(11, 0, f"Angular Speed: {self.speed_angular:.1f}  ")
        elif key == curses.KEY_RIGHT:
            self.speed_angular = max(0.1, self.speed_angular - self.speed_increment)
            self.stdscr.addstr(11, 0, f"Angular Speed: {self.speed_angular:.1f}  ")

    def keyboard_loop(self):
        """Loop to capture keyboard input."""
        # Initialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)
        
        try:
            # Display instructions
            self.stdscr.addstr(0, 0, self.msg)
            self.stdscr.addstr(10, 0, f"Linear Speed: {self.speed_linear:.1f}  ")
            self.stdscr.addstr(11, 0, f"Angular Speed: {self.speed_angular:.1f}  ")
            self.stdscr.addstr(13, 0, "Current command:")
            
            while self.running:
                # Get key
                try:
                    key = self.stdscr.getch()
                    if key != -1:  # -1 means no key is pressed
                        self.process_key(key)
                except Exception as e:
                    self.stdscr.addstr(15, 0, f"Error: {e}  ")
                
                # Update command display
                self.stdscr.addstr(14, 0, f"lin_x: {self.linear_x:.2f}, lin_y: {self.linear_y:.2f}, ang_z: {self.angular_z:.2f}  ")
                self.stdscr.refresh()
                
                # Short sleep to avoid consuming too much CPU
                rospy.sleep(0.01)
                
        finally:
            # Clean up curses environment
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
    
    def publish_cmd_vel(self):
        """Publish velocity commands at regular intervals."""
        rate = rospy.Rate(50)  # 10Hz
        
        while self.running and not rospy.is_shutdown():
            # Create and publish the message
            twist = Twist()
            twist.linear.x = self.linear_x
            twist.linear.y = self.linear_y
            twist.angular.z = self.angular_z
            
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
    
    def signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C)."""
        self.running = False
        if self.key_thread:
            self.key_thread.join(1.0)
        sys.exit(0)
    
    def start(self):
        """Start the keyboard teleop node."""
        self.running = True
        
        # Start keyboard input thread
        self.key_thread = threading.Thread(target=self.keyboard_loop)
        self.key_thread.daemon = True
        self.key_thread.start()
        
        # Start publishing loop in main thread
        self.publish_cmd_vel()


if __name__ == '__main__':
    try:
        node = KeyboardTeleopNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        # Make sure to clean up curses in case of unexpected errors
        curses.nocbreak()
        if hasattr(node, 'stdscr') and node.stdscr:
            node.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        print(f"Error: {e}")
        sys.exit(1)
