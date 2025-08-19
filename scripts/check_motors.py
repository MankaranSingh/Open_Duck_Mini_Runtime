from pypot.feetech import FeetechSTS3215IO
from mini_bdx_runtime.rustypot_position_hwi import HWI
import argparse
import time


NUM_SERVOS = 14  # Number of servos to scan
io = FeetechSTS3215IO("/dev/ttyACM0")

servo_dict = {value:key for key, value in HWI.joints.items()}

def scan():
    id = None
    for i in range(NUM_SERVOS):

        print(f"scanning for id {i} ...")
        try:
            io.get_present_position([i])
            id = i
            print(f"Found motor {servo_dict[id]} with id {id}")
        except Exception as e:
            print(f"Motor {servo_dict[id]} with id {i} not found: {e}")


scan()
