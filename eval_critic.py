import torch
import torch.nn as nn
import numpy as np
from rl_zoo3 import ALGOS
import gymnasium
import gym_footsteps_planning
from rl_zoo3.utils import get_model_path
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
import math

env_name = "footsteps-planning-any-obstacle-multigoal-v0"
exp_nb = 0
algo = "td3"

folder = "logs"

nb_tests = 100
max_episode_len = 90

set_random_seed(15672578)
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


def rotation_arround_obstacle(theta: float, center: list = obstacle_coordinates, radius: float = 0.3) -> list:
    x = center[0] + radius * np.cos(np.deg2rad(180 - theta))
    y = center[1] + radius * np.sin(np.deg2rad(180 - theta))
    return [x, y, np.deg2rad(-theta)]


def get_reset_dict_arr(situation: int, nb_tests: int = 1000, lr: bool = False, obstacle_radius = 0) -> np.ndarray:
    """_summary_

    :param int situation: number of the situation
    :param int nb_tests: number of tests, defaults to 1000
    :param bool lr: if True, the robot will test the left and right foot for goal, defaults to False
    :return np.ndarray: array of reset_dict for each test
    """
    if situation == 1:
        dthetas = [-135, -90, -45]
        dxys = [[0.17, 0.17], [0.17, -0.17], [-0.17, 0.17], [-0.17, -0.17]]
        size_list = len(dxys) * len(dthetas)
        size_list *= 2 if lr else 1
        reset_dict_arr = np.empty(shape=(0, size_list))

        ref_pose = [0.0, 0.0, 0.0]
        radius_arround_obstacle = 0

    elif situation == 2:
        dthetas = [-45, 0, 45]
        dxys = [[0.0, 0.0]]
        size_list = len(dxys) * len(dthetas)
        size_list *= 2 if lr else 1
        reset_dict_arr = np.empty(shape=(0, size_list))

        ref_pose = [0.3, 0.0, 0.0]
        radius_arround_obstacle = 0.5

    elif situation == 3:
        dthetas = [0]
        dxys = [[0.0, 0.15], [0.0, 0.0], [0.0, -0.15]]
        size_list = len(dxys) * len(dthetas)
        size_list *= 2 if lr else 1
        reset_dict_arr = np.empty(shape=(0, size_list))

        distx_genzone_obs = 0.7
        genzone_dxy = [0.3, 0.4]
        ref_pose = [0.3, 0.0, 180.0]
        radius_arround_obstacle = 0.7

    if lr:
        foots = ["left", "right"]
    else:
        foots = ["left"]

    for _ in range(nb_tests):
        reset_dict_init = {
            "start_foot_pose": np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
            "start_support_foot": "left" if (np.random.uniform(0, 1) > 0.5) else "right",
            "target_foot_pose": None,
            "target_support_foot": None,
            "obstacle_radius": None,
        }

        reset_dict_dthetadxdy = np.array([])
        for foot in foots:
            for dtheta in dthetas:
                for dxy in dxys:
                    reset_dict_init["obstacle_radius"] = obstacle_radius
                    reset_dict_init["target_foot_pose"] = rotation_arround_obstacle(
                        ref_pose[2] + dtheta, [ref_pose[0] + dxy[0], ref_pose[1] + dxy[1]], radius_arround_obstacle
                    )
                    reset_dict_init["target_support_foot"] = foot
                    if situation == 3:
                        reset_dict_init["start_foot_pose"] = np.random.uniform(
                            [-genzone_dxy[0] - distx_genzone_obs, -genzone_dxy[1], -math.pi],
                            [genzone_dxy[0] - distx_genzone_obs, genzone_dxy[1], math.pi],
                        )

                    while in_obstacle(reset_dict_init["start_foot_pose"], reset_dict_init["obstacle_radius"]):
                        reset_dict_init["start_foot_pose"] = np.random.uniform([-2, -2, -math.pi], [-2, 2, math.pi])

                    reset_dict_dthetadxdy = np.append(reset_dict_dthetadxdy, reset_dict_init.copy())
                    reset_dict_dthetadxdy = np.array([reset_dict_dthetadxdy])

        reset_dict_arr = np.append(reset_dict_arr, reset_dict_dthetadxdy, axis=0)

    print(f"\n-----Situation: {situation} with obstacle radius = {obs_radius}-----")
    print(f"Number of tests: {nb_tests}, Number of goal possibilities: {reset_dict_arr.shape[1]} -> theta: {dthetas}, xy: {dxys}")
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

# for situation in [2]:
for situation in range(1, 4):
    if situation == 1:
        obstacle_radius = [0]
    else:
        obstacle_radius = [0.10, 0.15, 0.20, 0.25]

    for obs_radius in obstacle_radius:
        reset_dict_arr = get_reset_dict_arr(situation, nb_tests, False, obs_radius)
        more_steps_array = np.array([])
        total_nb_steps_arr = np.array([])
        total_critic_value_arr = np.array([])
        total_best_nb_steps_arr = np.array([])
        total_worse_nb_steps_arr = np.array([])
        for reset_dict_exp in tqdm(reset_dict_arr, disable=True):
            critic_value_arr = np.array([])
            nb_steps_arr = np.array([])
            for reset_dict in reset_dict_exp:
                obs, infos = env.reset(options=reset_dict)
                done = False
                critic = model.critic.eval()
                total_reward = 0
                nb_steps = 0
                # env.render()

                while (not done) & (nb_steps < max_episode_len):
                    action, lstm_states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, infos = env.step(action)
                    nb_steps += 1

                    if nb_steps == 1:
                        # obs_tensor = model.critic.features_extractor(torch.from_numpy(np.array([obs])))
                        obs_tensor = model.critic.features_extractor(torch.from_numpy(np.array([obs])).to("cuda"))
                        # action_tensor = model.critic.features_extractor(torch.from_numpy(np.array([action])))
                        action_tensor = model.critic.features_extractor(torch.from_numpy(np.array([action])).to("cuda"))
                        critic_value = critic(obs_tensor, action_tensor)[0].item()
                    total_reward += reward

                critic_value_arr = np.append(critic_value_arr, critic_value)
                nb_steps_arr = np.append(nb_steps_arr, nb_steps)
            index_min_total_step = np.argmin(nb_steps_arr)
            index_max_total_step = np.argmax(nb_steps_arr)
            index_min_critic = np.argmax(critic_value_arr)

            if index_min_total_step != index_min_critic:
                more_steps = nb_steps_arr[index_min_critic] - nb_steps_arr[index_min_total_step]
                if more_steps != 0:
                    more_steps_array = np.append(
                        more_steps_array, nb_steps_arr[index_min_critic] - nb_steps_arr[index_min_total_step]
                    )

            total_nb_steps_arr = np.append(total_nb_steps_arr, nb_steps_arr)
            total_critic_value_arr = np.append(total_critic_value_arr, critic_value_arr)
            total_best_nb_steps_arr = np.append(total_best_nb_steps_arr, nb_steps_arr[index_min_total_step])
            total_worse_nb_steps_arr = np.append(total_worse_nb_steps_arr, nb_steps_arr[index_max_total_step])

        
        global_mean_more_steps = np.sum(more_steps_array) / nb_tests
        mean_more_steps = np.mean(more_steps_array)
        mean_diff_critic_steps = np.mean(total_nb_steps_arr + total_critic_value_arr)
        mean_best_nb_steps = np.mean(total_best_nb_steps_arr)
        mean_worse_nb_steps = np.mean(total_worse_nb_steps_arr)
        pourcentage_error = more_steps_array.shape[0]*100/nb_tests
        relative_error = np.abs((-total_critic_value_arr - total_nb_steps_arr)/total_nb_steps_arr)*100

        print(f"Mean More Steps: {np.round(mean_more_steps,2)}, Global Mean More Steps: {global_mean_more_steps}, Percentage error: {np.round(pourcentage_error,2)}%")
        print(f"Mean Steps: {np.round(np.mean(total_nb_steps_arr),2)}, Mean Critic Value: {np.round(np.mean(total_critic_value_arr),2)}, Mean Difference Critic/Steps: {np.round(mean_diff_critic_steps,2)}")
        print(f"More Steps: {more_steps_array}")
        print(f"Mean Best Nb Steps: {np.round(mean_best_nb_steps,2)}, Mean Worse Nb Steps: {np.round(mean_worse_nb_steps,2)}", "Pourcentage Error Best/Worse: ", np.round((mean_best_nb_steps - mean_worse_nb_steps)*100/mean_best_nb_steps,2), "%")
        print(f"Mean Relative Error: {np.round(np.mean(relative_error),2)}%")