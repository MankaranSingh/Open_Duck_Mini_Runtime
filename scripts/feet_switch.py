import RPi.GPIO as GPIO
import time

SWITCH_PIN = 27  # Connect NO to GPIO 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Internal pull-up

while True:
    if GPIO.input(SWITCH_PIN) == GPIO.LOW:  # Switch closed (ON)
        print("Switch ON")
    else:
        print("Switch OFF")
    time.sleep(0.5)
