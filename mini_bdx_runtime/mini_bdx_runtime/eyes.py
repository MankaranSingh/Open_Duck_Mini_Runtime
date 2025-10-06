import pigpio
import numpy as np
import time

# RGB LED pins
RED_PIN = 19
GREEN_PIN = 26
BLUE_PIN = 13


class Eyes:
    def __init__(self):
        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Could not connect to pigpio daemon. Run 'sudo pigpiod' first.")
        
        # PWM range for pigpio is 0-255
        self._pwm_range = 255

    def set_color(self, r, g, b):
        """Set LED color with floats 0.0–1.0"""
        # Convert 0-1 range to 0-255 for pigpio
        r_val = int(r * self._pwm_range)
        g_val = int(g * self._pwm_range)
        b_val = int(b * self._pwm_range)
        
        self.pi.set_PWM_dutycycle(RED_PIN, r_val)
        self.pi.set_PWM_dutycycle(GREEN_PIN, g_val)
        self.pi.set_PWM_dutycycle(BLUE_PIN, b_val)
        
    def set_color_rgb(self, rgb_tuple):
        """Set LED color with RGB tuple of floats 0.0–1.0"""
        self.set_color(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])

    def cleanup(self):
        # Turn off all LEDs
        self.pi.set_PWM_dutycycle(RED_PIN, 0)
        self.pi.set_PWM_dutycycle(GREEN_PIN, 0)
        self.pi.set_PWM_dutycycle(BLUE_PIN, 0)
        
        # Stop connection to pigpio daemon
        self.pi.stop()


if __name__ == "__main__":
    from common.expressions import BlinkingEyes
    
    eyes = Eyes()
    blinking_eyes = BlinkingEyes()
    
    try:
        while True:
            start_time = time.time()
            
            # Update blinking logic and get color
            color = blinking_eyes.update().eyes_rgb
            eyes.set_color_rgb(color)
            
            # Maintain 50Hz timing
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.02 - elapsed)  # 50Hz = 20ms
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        eyes.cleanup()
