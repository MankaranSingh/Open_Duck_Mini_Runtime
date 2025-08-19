import RPi.GPIO as GPIO
import numpy as np
import time
import random
from threading import Thread

# RGB LED pins
RED_PIN = 19
GREEN_PIN = 26
BLUE_PIN = 13

# Default eye color (dark blue)
EYE_COLOR = np.array([8, 29, 54])/255


class Eyes:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup RGB pins
        GPIO.setup(RED_PIN, GPIO.OUT)
        GPIO.setup(GREEN_PIN, GPIO.OUT)
        GPIO.setup(BLUE_PIN, GPIO.OUT)

        # PWM setup
        self.red_pwm = GPIO.PWM(RED_PIN, 1000)
        self.green_pwm = GPIO.PWM(GREEN_PIN, 1000)
        self.blue_pwm = GPIO.PWM(BLUE_PIN, 1000)

        self.red_pwm.start(0)
        self.green_pwm.start(0)
        self.blue_pwm.start(0)

        Thread(target=self.run, daemon=True).start()

    def set_color(self, r, g, b):
        """Set LED color with floats 0.0–1.0"""
        self.red_pwm.ChangeDutyCycle(r * 100)
        self.green_pwm.ChangeDutyCycle(g * 100)
        self.blue_pwm.ChangeDutyCycle(b * 100)

    def blink(self, r=1.0, g=0.0, b=0.0):
        """Solid blink (no fade, just off/on)"""
        # Eye closes
        self.set_color(0, 0, 0)
        time.sleep(random.uniform(0.08, 0.18))  # blink duration
        # Eye opens
        self.set_color(r, g, b)

    def run(self):
        while True:
            # Eye stays on
            self.set_color(*EYE_COLOR)  # set to eye color
            
            # Wait random time before blink
            time.sleep(random.uniform(3, 8))

            # Normal blink
            self.blink(*EYE_COLOR)

            # ~10% chance of a quick second blink
            if random.random() < 0.1:
                time.sleep(random.uniform(0.1, 0.3))  # short pause
                self.blink(*EYE_COLOR)

    def cleanup(self):
        self.red_pwm.stop()
        self.green_pwm.stop()
        self.blue_pwm.stop()


if __name__ == "__main__":
    e = Eyes()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        e.cleanup()
        GPIO.cleanup()
