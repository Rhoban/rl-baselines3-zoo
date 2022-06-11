#!/bin/bash

for side in left right place;
do
	nohup python train.py --env approach-$side-v0 --algo td3 --num-threads 4 > $side.log &
done

