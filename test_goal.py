import time
import os
import yaml
import math
from tqdm import tqdm
import gym_footsteps_planning
import gymnasium
import numpy as np
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from stable_baselines3.common.utils import set_random_seed
from rl_zoo3.utils import StoreDict, get_model_path
import matplotlib.pyplot as plt

env_name = "footsteps-planning-any-obstacle-multigoal-v0"
exp_nb = 0
algo = "td3"

obstacle_max_radius = 0.25

foot_length = 0.14
foot_width = 0.08

# set_random_seed(0)
max_episode_len = 90

folder = "logs"

reset_dict = {
    "start_foot_pose": [0.7, 0, 3.14],
    "start_support_foot": "right",
    "target_foot_pose": [0, 0, 0],
    "target_support_foot": "right",
    "obstacle_radius": 0.15,
}

print(f"Env. Name: {env_name}, Exp. Number: {exp_nb}, Algo: {algo}")

env = gymnasium.make(env_name, disable_env_checker=True)

_, model_path, log_path = get_model_path(
    exp_nb,
    folder,
    algo,
    env_name,
    True,  # load-best
    False,  # load-checkpoint
    False,  # load-last-checkpoint
)

parameters = {
    "env": env,
}

model = ALGOS[algo].load(model_path, device="auto", **parameters)

obs, infos = env.reset(options=reset_dict)
episode_reward = 0.0
ep_len = 0
walk_in_ball = 0
truncated_ep = 0
done = False

while (not done) & (ep_len < max_episode_len):
    action, lstm_states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, infos = env.step(action)

    episode_start = done

    episode_reward += reward
    ep_len += 1

    env.render()

    if reward <= -10:
        walk_in_ball += 1

    if (not done) & (ep_len == max_episode_len):
        truncated_ep = 1

    if done | (ep_len == max_episode_len):
        print(
            f"Episode Reward: {episode_reward}, Episode Length: {ep_len}, Walk in Ball: {walk_in_ball}, Truncated Episode: {truncated_ep}"
        )