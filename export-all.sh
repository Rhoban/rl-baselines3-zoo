#!/bin/bash

ENVS="approach-left-v0 approach-right-v0 approach-place-v0"

for env in $ENVS
do
    echo "*** Exporting $env"
    python export.py --env $env --output rl-agents/
done
