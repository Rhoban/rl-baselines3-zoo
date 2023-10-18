#!/bin/bash

env="footsteps-planning-right"
killall -9 python

python enjoy.py --env $env-v0 --algo td3 --n-timesteps 1000 --exp-id 0 --folder logs/ --load-best --gym-packages gym_footsteps_planning
