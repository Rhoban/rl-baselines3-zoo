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

nb_tests = 1000
max_episode_len = 90

foot_length = 0.14
foot_width = 0.08

obstacle_coordinates = [0.3, 0]

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

def rotation_arround_obstacle(theta: float, center:list = obstacle_coordinates, radius:float = 0.3) -> list:
    x = center[0] + radius * np.cos(np.deg2rad(180 - theta))
    y = center[1] + radius * np.sin(np.deg2rad(180 - theta))
    return [x, y, np.deg2rad(-theta)]

def get_reset_dict_arr(situation: int, nb_tests:int = 1000, lr:bool = False) -> np.ndarray:
    


    for _ in range(nb_tests):
        reset_dict_init = {
            "start_foot_pose": np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
            "start_support_foot": "left" if (np.random.uniform(0, 1) > 0.5) else "right",
            "target_foot_pose": None,
            "target_support_foot": None,
            "obstacle_radius": None,
        }

        if situation == 1:
            dthetas = (-135, -90, -45)
            dxys = [[0.17, 0.17], [0.17, -0.17], [-0.17, 0.17], [-0.17, -0.17]]
            size_list = len(dxys)*len(dthetas)
            reset_dict_arr = np.empty(shape=(0,size_list))

            reset_dict_init["target_foot_pose"] = [0., 0., 0.]
            obstacle_radius = [0]
            radius_arround_obstacle = 0

        elif situation == 2:
            dthetas = (-45, 0, 45)
            dxys = [[0, 0]]
            size_list = len(dxys)*len(dthetas)
            reset_dict_arr = np.empty(shape=(0,size_list))

            reset_dict_init["target_foot_pose"] = [0., 0., 0.]
            obstacle_radius = [0.15,0.25]
            radius_arround_obstacle = 0.5

        elif situation == 3:
            dthetas = (0)
            dxys = [[0, 0]]
            size_list = len(dxys)*len(dthetas)
            reset_dict_arr = np.empty(shape=(0,size_list))

            reset_dict_init["start_foot_pose"] = np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
            reset_dict_init["target_foot_pose"] = [0., 0., 0.]
            obstacle_radius = [0.15,0.25]
            radius_arround_obstacle = 0.5



        if lr:
            foot = ["left", "right"]
        else:
            foot = ["right"]

        xy_situation = reset_dict_init["target_foot_pose"][:2]
        theta_situation = np.rad2deg(reset_dict_init["target_foot_pose"][2])
        reset_dict_dthetadxdy = np.array([])
        for foot in foot:
            for dtheta in dthetas:
                for dxy in dxys:
                    for obs_radius in obstacle_radius:
                        reset_dict_init["obstacle_radius"] = obs_radius
                        reset_dict_init["target_foot_pose"] = rotation_arround_obstacle(theta_situation+dtheta, [sum(x) for x in zip(xy_situation, dxy)],radius_arround_obstacle)
                        reset_dict_init["target_support_foot"] = foot
                        reset_dict_dthetadxdy = np.append(reset_dict_dthetadxdy, reset_dict_init.copy())
                        reset_dict_dthetadxdy = np.array([reset_dict_dthetadxdy])

                        while in_obstacle(reset_dict_init["start_foot_pose"], reset_dict_init["obstacle_radius"]):
                            reset_dict_init["start_foot_pose"] = np.random.uniform([-2, -2, -math.pi], [-2, 2, math.pi])

        reset_dict_arr = np.append(reset_dict_arr, reset_dict_dthetadxdy, axis=0)
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


for nb_situation in range(1, 4):
    reset_dict_arr = get_reset_dict_arr(nb_situation, nb_tests, False)
    
    more_steps_array = np.array([])
    for reset_dict_exp in tqdm(reset_dict_arr):
        critic_value_arr = np.array([])
        nb_steps_arr = np.array([])
        for reset_dict in reset_dict_exp:
            obs, infos = env.reset(options=reset_dict)
            done = False
            critic = model.critic.eval()
            total_reward = 0
            nb_steps = 0
            env.render()
            
            while (not done) & (nb_steps < max_episode_len):
                action, lstm_states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, infos = env.step(action)
                if nb_steps == 0:
                    obs_tensor = model.critic.features_extractor(torch.from_numpy(np.array([obs])).to("cuda"))
                    action_tensor = model.critic.features_extractor(torch.from_numpy(np.array([action])).to("cuda"))
                    critic_value = critic(obs_tensor, action_tensor)[0].item()
                total_reward += reward

                if not done:
                    nb_steps += 1

            critic_value_arr = np.append(critic_value_arr, critic_value)
            nb_steps_arr = np.append(nb_steps_arr, nb_steps)

        index_min_total_step = np.argmin(nb_steps_arr)
        index_min_critic = np.argmax(critic_value_arr)

        if index_min_total_step != index_min_critic:
            more_steps = nb_steps_arr[index_min_critic] - nb_steps_arr[index_min_total_step]
            if more_steps != 0:
                more_steps_array = np.append(
                    more_steps_array, nb_steps_arr[index_min_critic] - nb_steps_arr[index_min_total_step]
                )

    mean_more_steps = np.sum(more_steps_array)/nb_tests
    print(f"Situation: {nb_situation}----------------------")
    print(f"Mean More Steps: {mean_more_steps}, Percentage error: {more_steps_array.shape[0]*100/nb_tests}%")
    print(f"More Steps: {more_steps_array}")
