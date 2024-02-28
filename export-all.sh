#!/bin/bash

ENVS="approach-left approach-right approach-place"

for env in $ENVS
do
    echo "*** Exporting $env"
    python export.py --env $env-v0 --output rl-agents/ --gym-packages gym_footsteps_planning
done

cp -R rl-agents/* ~/workspace/env/common/rl-agents/
