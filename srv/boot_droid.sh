#!/bin/bash
# Source bashrc to load envs/virtualenv
source /home/bdx/.bashrc
sudo /usr/bin/pigpiod

export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

# Run your Python script
# /home/bdx/.virtualenvs/open-duck-mini-runtime/bin/python  /home/bdx/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py --robot dino -p 18
