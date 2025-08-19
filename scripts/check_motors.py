from pypot.feetech import FeetechSTS3215IO
import argparse
import time

DEFAULT_ID = 1  # A brand new motor should have id 1

io = FeetechSTS3215IO("/dev/ttyACM0")


def scan():
    id = None
    for i in range(15):

        print(f"scanning for id {i} ...")
        try:
            io.get_present_position([i])
            id = i
            print(f"Found motor with id {id}")
        except Exception as e:
            print(f"Motor with id {i} not found: {e}")


scan()
