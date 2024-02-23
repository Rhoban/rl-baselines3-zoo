import torch
import torch.nn as nn
import numpy as np
from rl_zoo3 import ALGOS
import gymnasium
import gym_footsteps_planning
from rl_zoo3.utils import get_model_path
from tqdm import tqdm
import math

env_name = "footsteps-planning-any-obstacle-multigoal-v0"
exp_nb = 0
algo = "td3"

folder = "logs"

nb_tests = 10
max_episode_len = 90

foot_length = 0.14
foot_width = 0.08
obstacle_coordinates = [0.3, 0]
save_reset_dict = False

def in_obstacle(foot_pose, obstacle_radius):
    in_obstacle = False
    cos_theta = np.cos(foot_pose[2])
    sin_theta = np.sin(foot_pose[2])
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            P_corner_foot = np.array([sx * foot_length / 2, sy * foot_width / 2])
            P_corner_world = (
                foot_pose[:2]
                + P_corner_foot[0] * np.array([cos_theta, sin_theta])
                + P_corner_foot[1] * np.array([-sin_theta, cos_theta])
            )
            if np.linalg.norm(P_corner_world - np.array([0.3, 0]), axis=-1) < obstacle_radius:
                in_obstacle = True
    return in_obstacle


def rotation_arround_obstacle(theta: float, center: list = obstacle_coordinates, radius: float = 0.3) -> list:
    x = center[0] + radius * np.cos(np.deg2rad(180 - theta))
    y = center[1] + radius * np.sin(np.deg2rad(180 - theta))
    return [x, y, np.deg2rad(-theta)]


def get_reset_dict_arr(situation: int, nb_tests: int = 1000, lr: bool = False) -> np.ndarray:

    reset_dict_arr = np.empty(shape=(0, 1))

    for _ in range(nb_tests):
        reset_dict_init = {
            "start_foot_pose": np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
            "start_support_foot": "left" if (np.random.uniform(0, 1) > 0.5) else "right",
            "target_foot_pose": None,
            "target_support_foot": "right",
            "obstacle_radius": None,
        }

        if lr:
            reset_dict_init["target_support_foot"] = "left" if (np.random.uniform(0, 1) > 0.5) else "right",

        if situation == 1:
            reset_dict_init["obstacle_radius"] = 0.0
            reset_dict_init["target_foot_pose"] = [0.0, 0.0, 0.0]

        elif situation == 2:
            radius_arround_obstacle = 0.3

            reset_dict_init["obstacle_radius"] = 0.15
            reset_dict_init["target_foot_pose"] = [0.0, 0.0, 0.0]

        elif situation == 3:
            distx_genzone_obs = 0.7
            genzone_dxy = [0.3, 0.4]
            radius_arround_obstacle = 0.7

            reset_dict_init["obstacle_radius"] = 0.25
            reset_dict_init["start_foot_pose"] = np.random.uniform(
                [-genzone_dxy[0] - distx_genzone_obs, -genzone_dxy[1], -math.pi],
                [genzone_dxy[0] - distx_genzone_obs, genzone_dxy[1], math.pi],
            )
            reset_dict_init["target_foot_pose"] = np.random.uniform(
                [-genzone_dxy[0] + distx_genzone_obs + obstacle_coordinates[0], -genzone_dxy[1], -math.pi],
                [genzone_dxy[0] + distx_genzone_obs + obstacle_coordinates[0], genzone_dxy[1], math.pi],
            )
            # reset_dict_init["target_foot_pose"] = rotation_arround_obstacle(180, [0.3, 0.0], radius_arround_obstacle)

        while in_obstacle(reset_dict_init["start_foot_pose"], reset_dict_init["obstacle_radius"]):
            reset_dict_init["start_foot_pose"] = np.random.uniform([-2, -2, -math.pi], [-2, 2, math.pi])

        reset_dict_arr = np.append(reset_dict_arr, reset_dict_init)

    return reset_dict_arr

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

reset_dict_all_situations_arr = np.empty(shape=(0, nb_tests))
print(reset_dict_all_situations_arr.shape)

if save_reset_dict:
    for nb_situation in range(1, 4):
        reset_dict_arr = np.array([get_reset_dict_arr(nb_situation, nb_tests, False)])
        reset_dict_all_situations_arr = np.append(reset_dict_all_situations_arr, reset_dict_arr, axis=0)
    np.save("reset_dict_all_situations_RL", reset_dict_all_situations_arr, allow_pickle=True, fix_imports=True)
else:
    reset_dict_all_situations_arr = np.load("reset_dict_all_situations_ROS.npy", allow_pickle=True, fix_imports=False, encoding="latin1")
    nb_tests = reset_dict_all_situations_arr.shape[1]

nb_steps_all_situations_arr = np.empty(shape=(0, nb_tests))

for nb_situation in range(0, 3):
    reset_dict_arr = reset_dict_all_situations_arr[nb_situation]
    print(f"\n-----Situation: {nb_situation} with obstacle radius = {reset_dict_arr[0]['obstacle_radius']}-----")
    
    nb_steps_arr = np.empty(shape=(0, 1))

    for reset_dict in tqdm(reset_dict_arr):    
        obs, infos = env.reset(options=reset_dict)
        done = False
        total_reward = 0
        nb_steps = 0
        # env.render()

        while (not done) & (nb_steps < max_episode_len):
            action, lstm_states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, infos = env.step(action)
            nb_steps += 1
            total_reward += reward

        nb_steps_arr = np.append(nb_steps_arr, nb_steps)

    nb_steps_arr = np.array([nb_steps_arr])
    nb_steps_all_situations_arr = np.append(nb_steps_all_situations_arr, nb_steps_arr, axis=0)

np.save("nb_steps_all_situations_RL", nb_steps_all_situations_arr, allow_pickle=False, fix_imports=True)
