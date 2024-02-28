#!/bin/bash

#ENVS=strategy
ENVS="footsteps-planning-right-withball-multigoal"
# killall -9 python

for env in $ENVS;
do
	nohup python train.py --env $env-v0 -n 100000 --n-trials 3000 --n-jobs 18 --algo td3 -optimize --track --gym-packages gym_footsteps_planning --device cuda > $env_optuna.log &
done
