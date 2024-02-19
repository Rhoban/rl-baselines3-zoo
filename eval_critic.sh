#!/bin/bash
source ~/espaces/travail/rl-env/bin/activate


nohup python eval_critic.py > eval_critic.log &
