#!/bin/bash

#ENVS=strategy
ENVS="footsteps-planning-left footsteps-planning-right footsteps-planning-place"
killall -9 python

for env in $ENVS;
do
	nohup python train.py --env $env-v0 --algo td3 --num-threads 4 --track > $env.log &
done

