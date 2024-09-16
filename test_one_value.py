import torch
import torch.nn as nn
import numpy as np
from rl_zoo3 import ALGOS
import gymnasium
import gym_footsteps_planning
from rl_zoo3.utils import get_model_path
from tqdm import tqdm
import math

env_name = "footsteps-planning-any-withball-v0"
exp_nb = 0
algo = "td3"

folder = "logs"

max_episode_len = 90
obstacle_coordinates = [0.3, 0]

def rotation_arround_obstacle(theta: float, center: list = obstacle_coordinates, radius: float = 0.3) -> list:
    x = center[0] + radius * np.cos(np.deg2rad(180 - theta))
    y = center[1] + radius * np.sin(np.deg2rad(180 - theta))
    return [x, y, np.deg2rad(-theta)]

#Catch eye figure situation
reset_dict = {
    "start_foot_pose": [1.0, -0.15, math.pi/4],
    "start_support_foot": "right",
    "target_foot_pose": [0.0, 0.0, 0.0],
    "target_support_foot": "right",
    "obstacle_radius":0.15,
}

reset_dict = {
    "start_foot_pose": [1.0, -0.15, math.pi/4],
    "start_support_foot": "right",
    "target_foot_pose": [0.0, 0.0, 0.0],
    "target_support_foot": "right",
    "obstacle_radius":0.15,
}

reset_dict["target_foot_pose"] = rotation_arround_obstacle(45)

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

model = ALGOS[algo].load(model_path, device="cuda", **parameters)

obs, infos = env.reset(options=reset_dict)
done = False

total_reward = 0
total_step = 0
env.render()

while (not done) & (total_step < max_episode_len):
    action, lstm_states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, infos = env.step(action)
    total_step += 1
    total_reward += reward

print(f"Total reward: {total_reward}, Total steps: {total_step}") 
input()