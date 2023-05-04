#!/bin/bash

#ENVS=strategy
ENVS="approach-left approach-right approach-place"
killall -9 python

for env in $ENVS;
do
	nohup python train.py --env $env-v0 --algo td3 --num-threads 4 --track > $env.log &
done

