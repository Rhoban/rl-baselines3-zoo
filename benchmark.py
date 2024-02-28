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

env_names = ["footsteps-planning-right-obstacle-multigoal-v0", "footsteps-planning-right-obstacle-multigoal-v0"]
exp_nbs = [4, 1]
algos = ["td3", "sac"]

obstacle_max_radius = 0.25

foot_length = 0.14
foot_width = 0.08

# set_random_seed(0)
max_episode_len = 90
nb_tests = 1000
name_of_bench = ["", "", ""]

# Select benchmark :
# 1 : no obstacle
# 2 : with random obstacle from 0 to obstacle_max_radius
# 3 : with ball (obstacle_radius = 0.15)
bench = 2

# Force the target footstep to be right or left
force_target_footstep = "right"

# if multigoal, the target_foot_pose is random else it is [0,0,0]
multigoal = False

goal_type = "multi" if multigoal else "single"

folder = "logs"

saved_filename = f"logs/npy_bench/{env_names[0][19:]}_{exp_nbs[0]}_{algos[0]}_vs_{env_names[1][19:]}_{exp_nbs[1]}_{algos[1]}_{nb_tests}_{goal_type}_{force_target_footstep}.npz"

reset_dict_list = np.array([])

episode_rewards_env1, episode_lengths_env1, walks_in_ball_env1, truncated_eps_env1 = (
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
)
episode_rewards_env2, episode_lengths_env2, walks_in_ball_env2, truncated_eps_env2 = (
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
)


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


if os.path.isfile(saved_filename):
    print("Already done")
    npzfile = np.load(saved_filename, allow_pickle=True)
    episode_lengths_env1 = npzfile["arr_0"]
    episode_lengths_env2 = npzfile["arr_1"]
    reset_dict_list = npzfile["arr_2"]
    name_of_bench = npzfile["arr_3"]
    nb_tests = episode_lengths_env1.shape[0]

else:
    for i in range(nb_tests):
        reset_dict = {
            "start_foot_pose": np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
            "start_support_foot": "left" if (np.random.uniform(0, 1) > 0.5) else "right",
            "target_foot_pose": None,
            "target_support_foot": None,
            "obstacle_radius": 0,
        }

        if (force_target_footstep == "right"):
            name_of_bench[0] = "target_footstep=right "
            max_episode_len = 50
            reset_dict["target_foot_pose"] = np.array([0, 0, 0])
            reset_dict["target_support_foot"] = "right"

        if (force_target_footstep == "left"):
            name_of_bench[0] = "target_footstep=left "
            max_episode_len = 50
            reset_dict["target_foot_pose"] = np.array([0, 0, 0])
            reset_dict["target_support_foot"] = "left"

        if bench == 2:
            name_of_bench[1] = f"with random obstacle from 0 to {obstacle_max_radius} "
            max_episode_len = 90
            reset_dict["obstacle_radius"] = np.random.uniform(0, obstacle_max_radius)

        if bench == 3:
            name_of_bench[1] = "with ball "
            max_episode_len = 90
            reset_dict["obstacle_radius"] = 0.15
            reset_dict["target_foot_pose"] = [0, 0, 0]

        if (bench == 2) | (bench == 3):
            max_episode_len = 90
            start_foot_pose = np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi])

            while in_obstacle(start_foot_pose, reset_dict["obstacle_radius"]):
                start_foot_pose = np.random.uniform([-2, -2, -math.pi], [-2, 2, math.pi])

            reset_dict["start_foot_pose"] = start_foot_pose

        if multigoal:
            max_episode_len = 70
            name_of_bench[2] = "with random target (multigoal)"
            target_foot_pose = np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi])

            if (bench == 2) | (bench == 3):
                max_episode_len = 90
                while in_obstacle(target_foot_pose, reset_dict["obstacle_radius"]):
                    target_foot_pose = np.random.uniform([-2, -2, -math.pi], [-2, 2, math.pi])

            reset_dict["target_foot_pose"] = target_foot_pose

        reset_dict_list = np.append(reset_dict_list, reset_dict)

    for env_name, exp_nb, algo in zip(env_names, exp_nbs, algos):
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

        for reset_dict in tqdm(reset_dict_list):
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

                # env.render()

                if reward <= -10:
                    walk_in_ball = 1

                if (not done) & (ep_len == max_episode_len):
                    truncated_ep = 1

                if done | (ep_len == max_episode_len):
                    if (env_name, exp_nb, algo) == (env_names[0], exp_nbs[0], algos[0]):
                        walks_in_ball_env1 = np.append(walks_in_ball_env1, walk_in_ball)
                        truncated_eps_env1 = np.append(truncated_eps_env1, truncated_ep)
                        episode_rewards_env1 = np.append(episode_rewards_env1, episode_reward)
                        episode_lengths_env1 = np.append(episode_lengths_env1, ep_len)

                    elif (env_name, exp_nb, algo) == (env_names[1], exp_nbs[1], algos[1]):
                        walks_in_ball_env2 = np.append(walks_in_ball_env2, walk_in_ball)
                        truncated_eps_env2 = np.append(truncated_eps_env2, truncated_ep)
                        episode_rewards_env2 = np.append(episode_rewards_env2, episode_reward)
                        episode_lengths_env2 = np.append(episode_lengths_env2, ep_len)

    np.savez(saved_filename, episode_lengths_env1, episode_lengths_env2, reset_dict_list, name_of_bench)

compare_episode_lengths = episode_lengths_env1 - episode_lengths_env2

env2_better_ones = np.zeros(compare_episode_lengths.shape)
env1_better_ones = np.zeros(compare_episode_lengths.shape)

env2_better_ones[compare_episode_lengths > 0] = 1
env1_better_ones[compare_episode_lengths < 0] = 1

env2_better_sum = np.sum(env2_better_ones)
env1_better_sum = np.sum(env1_better_ones)

env2_better_mean = np.sum(compare_episode_lengths[compare_episode_lengths > 0]) / env2_better_sum
env1_better_mean = -np.sum(compare_episode_lengths[compare_episode_lengths < 0]) / env1_better_sum

with open(
    f"logs/bench_txt/{env_names[0][19:]}_{exp_nbs[0]}_{algos[0]}_vs_{env_names[1][19:]}_{exp_nbs[1]}_{algos[1]}_{nb_tests}_{goal_type}_{force_target_footstep}.txt",
    "w",
) as f:
    f.write(f"Name of the benchmark : ")
    f.writelines(name_of_bench)
    f.write("\n")
    print(f"Number of tests : {nb_tests}", file=f)
    print(f"Env. Name: {env_names[0]}, Exp. Number: {exp_nbs[0]}, Algo: {algos[0]}------", file=f)
    print(f"    - Mean nb. of steps : {np.mean(episode_lengths_env1)}", file=f)
    print(
        f"    - Better in {(env1_better_sum*100)/nb_tests}% of the tests with a mean of {env1_better_mean} less steps than the other env.",
        file=f,
    )
    print(f"    - Walks in ball in {(np.sum(walks_in_ball_env1)*100)/nb_tests}% of the tests", file=f)
    print(f"    - Truncated in {(np.sum(truncated_eps_env1)*100)/nb_tests}% of the tests", file=f)
    print(f"Env. Name: {env_names[1]}, Exp. Number: {exp_nbs[1]}, Algo: {algos[1]}------", file=f)
    print(f"    - Mean nb. of steps : {np.mean(episode_lengths_env2)}", file=f)
    print(
        f"    - Better in {(env2_better_sum*100)/nb_tests}% of the tests with a mean of {env2_better_mean} less steps than the other env.",
        file=f,
    )
    print(f"    - Walks in ball in {(np.sum(walks_in_ball_env2)*100)/nb_tests}% of the tests", file=f)
    print(f"    - Truncated in {(np.sum(truncated_eps_env2)*100)/nb_tests}% of the tests", file=f)
    print("-------------", file=f)
    print(
        f"Same nb of steps for both envs in {((nb_tests - (env1_better_sum + env2_better_sum))*100)/nb_tests}% of the tests",
        file=f,
    )
    print("\n", file=f)


# Trace the bargraph of the difference of episode lengths for each test

font = {"size": 8}
plt.rc("font", **font)

plt.figure()
plt.hist(compare_episode_lengths, bins=60)
plt.title(
    f"Episode length difference between : \n {env_names[0]}_{exp_nbs[0]}_{algos[0]} \n {env_names[1]}_{exp_nbs[1]}_{algos[1]}"
)
plt.xlabel("Test number")
plt.ylabel("Episode length difference")
plt.grid()
plt.savefig(
    f"logs/graphs/{env_names[0][19:]}_{exp_nbs[0]}_{algos[0]}_vs_{env_names[1][19:]}_{exp_nbs[1]}_{algos[1]}_{nb_tests}_{goal_type}_{force_target_footstep}.png",
    dpi=1000,
)
plt.show()
