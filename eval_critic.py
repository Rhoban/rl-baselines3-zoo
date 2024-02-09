import torch
import torch.nn as nn
import numpy as np
from rl_zoo3 import ALGOS
import gymnasium
import gym_footsteps_planning
from rl_zoo3.utils import get_model_path

env_name = "footsteps-planning-any-obstacle-multigoal-v0"
exp_nb = 0
algo = "td3"

folder = "logs"

reset_dict_list = np.array([])


radius_arround_obstacle = 0.3
obstacle_coordinates = [0.3, 0]

for theta in np.arange(0, 360, 22.5):
    for foot in ["left", "right"]:
        x = obstacle_coordinates[0]+radius_arround_obstacle*np.cos(np.deg2rad(180-theta))
        y = obstacle_coordinates[1]+radius_arround_obstacle*np.sin(np.deg2rad(180-theta))

        reset_dict = {
            "start_foot_pose": [-2, 0, 0],
            "start_support_foot": "left",
            "target_foot_pose": [x, y, np.deg2rad(-theta)],
            "target_support_foot": foot,
            "obstacle_radius": 0.15,
        }

        reset_dict_list = np.append(reset_dict_list, reset_dict)

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

for reset_dict in reset_dict_list:
    obs, infos = env.reset(options=reset_dict)
    done = False
    critic = model.critic.eval()
    total_reward = 0
    total_step = 0
    env.render()
    while (not done):
        action, lstm_states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, infos = env.step(action)
        if total_step == 0:
            obs_tensor = model.critic.features_extractor(torch.from_numpy(np.array([obs])))
            action_tensor = model.critic.features_extractor(torch.from_numpy(np.array([action])))
            critic_value = critic(obs_tensor, action_tensor)[0].item()
        total_reward += reward

        if not done:
            total_step += 1

    print(f"Foot: {reset_dict['target_support_foot']}, Angle: {np.round(np.rad2deg(reset_dict['target_foot_pose'][2]),1)}, Critic Value: {critic_value}, Number of steps:  {total_step}, Reward: {total_reward}")
    