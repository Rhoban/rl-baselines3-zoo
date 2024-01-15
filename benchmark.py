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

env_names = ["footsteps-planning-right-v0", "footsteps-planning-right-v0"]
exp_nbs = [1,1]
algos=["td3","sac"]

obstacle_max_radius = 0.25

foot_length = 0.14
foot_width = 0.08

# set_random_seed(0)
max_episode_len = 70
nb_tests = 500

folder="logs"

saved_filename = f"logs/npy_bench/{env_names[0][19:]}_{exp_nbs[0]}_{algos[0]}_vs_{env_names[1][19:]}_{exp_nbs[1]}_{algos[1]}_{nb_tests}.npz"

reset_dict_list = np.array([])

episode_rewards_env1, episode_lengths_env1, walks_in_ball_env1, truncated_eps_env1 = np.array([]),np.array([]),np.array([]),np.array([])
episode_rewards_env2, episode_lengths_env2, walks_in_ball_env2, truncated_eps_env2 = np.array([]),np.array([]),np.array([]),np.array([])

def in_obstacle(foot_pose, obstacle_radius):
    in_obstacle = False
    cos_theta = np.cos(foot_pose[2])
    sin_theta = np.sin(foot_pose[2])
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            P_corner_foot = np.array([sx * foot_length / 2, sy * foot_width / 2])
            P_corner_world = foot_pose[:2] + P_corner_foot[0] * np.array([cos_theta, sin_theta]) + P_corner_foot[1] * np.array([-sin_theta, cos_theta])
            if np.linalg.norm(P_corner_world - np.array([0.3, 0]), axis=-1) < obstacle_radius:
                in_obstacle = True
    return in_obstacle

if os.path.isfile(saved_filename):
    print("Already done")
    npzfile = np.load(saved_filename,allow_pickle=True)
    episode_lengths_env1 = npzfile['arr_0']
    episode_lengths_env2 = npzfile['arr_1']
    reset_dict_list = npzfile['arr_2']
    nb_tests = episode_lengths_env1.shape[0]

else :
    for i in range(nb_tests):
        
        reset_dict = {
            "start_foot_pose" : np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
            "start_support_foot" : "left" if (np.random.uniform(0, 1) > 0.5) else "right",
            "target_foot_pose" : None,
            "target_support_foot" : None,
            "obstacle_radius" : None
        }

        if ("right" in env_names[0]) | ("right" in env_names[1]): 
            max_episode_len = 50
            reset_dict["target_foot_pose"] = np.array([0, 0, 0])
            reset_dict["target_support_foot"] = "right"
            
        if ("left" in env_names[0]) | ("left" in env_names[1]): 
            max_episode_len = 50
            reset_dict["target_foot_pose"] = np.array([0, 0, 0])
            reset_dict["target_support_foot"] = "left"

        if ("obstacle" in env_names[0]) & ("obstacle" in env_names[1]):
            max_episode_len = 90
            reset_dict["obstacle_radius"] = np.random.uniform(0, obstacle_max_radius)

        if ("withball" in env_names[0]) | ("withball" in env_names[1]):
            max_episode_len = 90
            reset_dict["obstacle_radius"] = 0.15

        if (("withball" in env_names[0])|("obstacle" in env_names[0])) & (("withball" in env_names[1])|("obstacle" in env_names[1])):
            max_episode_len = 90
            start_foot_pose = np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi])

            while in_obstacle(start_foot_pose, reset_dict["obstacle_radius"]):
                start_foot_pose = np.random.uniform([-2, -2, -math.pi], [-2, 2, math.pi])

            reset_dict["start_foot_pose"] = start_foot_pose

        if ("multigoal" in env_names[0]) & ("multigoal" in env_names[1]):
            target_foot_pose = np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi])

            while in_obstacle(start_foot_pose, reset_dict["obstacle_radius"]):
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
            True, #load-best
            False, #load-checkpoint
            False, #load-last-checkpoint
        )

        parameters = {
            'env': env,
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
                action, lstm_states = model.predict(obs,  deterministic=True) 
                
                obs, reward, done, truncated, infos = env.step(action)

                episode_start = done

                episode_reward += reward
                ep_len += 1

                # env.render()

                if reward <= -10:
                    walk_in_ball = 1
                    
                if (not done) & (ep_len==max_episode_len):
                    truncated_ep = 1

                if done | (ep_len==max_episode_len) :
                    
                    if (env_name, exp_nb, algo) == (env_names[0], exp_nbs[0], algos[0]):
                        walks_in_ball_env1 = np.append(walks_in_ball_env1,walk_in_ball)
                        truncated_eps_env1 = np.append(truncated_eps_env1,truncated_ep)
                        episode_rewards_env1 = np.append(episode_rewards_env1,episode_reward)
                        episode_lengths_env1 = np.append(episode_lengths_env1,ep_len)

                    elif (env_name, exp_nb, algo) == (env_names[1], exp_nbs[1], algos[1]):
                        walks_in_ball_env2 = np.append(walks_in_ball_env2,walk_in_ball)
                        truncated_eps_env2 = np.append(truncated_eps_env2,truncated_ep)
                        episode_rewards_env2 = np.append(episode_rewards_env2,episode_reward)
                        episode_lengths_env2 = np.append(episode_lengths_env2,ep_len)

    compare_episode_lengths = episode_lengths_env1 - episode_lengths_env2
    np.savez(saved_filename, episode_lengths_env1, episode_lengths_env2, reset_dict_list)

env2_better_ones = np.zeros(compare_episode_lengths.shape)
env1_better_ones = np.zeros(compare_episode_lengths.shape)

env2_better_ones[compare_episode_lengths > 0] = 1
env1_better_ones[compare_episode_lengths < 0] = 1

env2_better_sum = np.sum(env2_better_ones)
env1_better_sum = np.sum(env1_better_ones)

env2_better_mean = np.sum(compare_episode_lengths[compare_episode_lengths > 0])/env2_better_sum
env1_better_mean = -np.sum(compare_episode_lengths[compare_episode_lengths < 0])/env1_better_sum

print(f"Number of tests : {nb_tests}")
print(f"{env_names[0]}_{exp_nbs[0]}_{algos[0]}------")
print(f"    better in {(env1_better_sum*100)/nb_tests}% of the tests with a mean of {env1_better_mean} less steps than the other environment")
print(f"    walks in ball in {(np.sum(walks_in_ball_env1)*100)/nb_tests}% of the tests")
print(f"    truncated in {(np.sum(truncated_eps_env1)*100)/nb_tests}% of the tests")
print(f"{env_names[1]}_{exp_nbs[1]}_{algos[1]}------")
print(f"    better in {(env2_better_sum*100)/nb_tests}% of the tests with a mean of {env2_better_mean} less steps than the other environment")
print(f"    walks in ball in {(np.sum(walks_in_ball_env2)*100)/nb_tests}% of the tests")
print(f"    truncated in {(np.sum(truncated_eps_env2)*100)/nb_tests}% of the tests")
print("-------------")
print(f"Same number of steps for both envs in {((nb_tests - (env1_better_sum + env2_better_sum))*100)/nb_tests}% of the tests")

print(f"Mean nb of steps {env_names[0]} : {np.mean(episode_lengths_env1)}")
print(f"Mean nb of steps {env_names[1]} : {np.mean(episode_lengths_env2)}")

#Trace the bargraph of the difference of episode lengths for each test
plt.figure()
plt.hist(compare_episode_lengths, bins=60)
plt.title(f"Episode length difference between {env_names[0]}_{exp_nbs[0]}_{algos[0]} and {env_names[1]}_{exp_nbs[1]}_{algos[1]}")
plt.xlabel("Test number")
plt.ylabel("Episode length difference")
plt.grid()
plt.savefig(f"logs/graphs/{env_names[0][19:]}_{exp_nbs[0]}_{algos[0]}_vs_{env_names[1][19:]}_{exp_nbs[1]}_{algos[1]}_{nb_tests}_diff.png", dpi=1000)
plt.show() 