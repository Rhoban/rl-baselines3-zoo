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

env_names = ["footsteps-planning-right-withball-multigoal-her-v0", "footsteps-planning-right-withball-her-v0"]
step = 0

# set_random_seed(0)
max_episode_len = 90
nb_tests = 1000

algo="td3"
folder="logs"

reset_dict_list = np.array([])

episode_rewards_env1, episode_lengths_env1 = np.array([]), np.array([])
episode_rewards_env2, episode_lengths_env2 = np.array([]), np.array([])

for i in range(nb_tests):
    
    reset_dict = {
        "start_foot_pose" : np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
        "start_support_foot" : "left" if (np.random.uniform(0, 1) > 0.5) else "right",
        "target_foot_pose" : None,
        "target_support_foot" : None        
    }
    
    if ("multigoal" in env_names[0]) & ("multigoal" in env_names[1]):
        reset_dict["target_foot_pose"] = np.random.uniform([-2, -2, -1, -1], [2, 2, 1, 1])
        print("multigoal")
        
    if ("right" in env_names[0]) | ("right" in env_names[1]): 
        reset_dict["target_foot_pose"] = np.array([0, 0, 1, 0])
        reset_dict["target_support_foot"] = "right"
        print("right")
        
    if ("left" in env_names[0]) | ("left" in env_names[1]): 
        reset_dict["target_foot_pose"] = np.array([0, 0, 1, 0])
        reset_dict["target_support_foot"] = "left"
        print("left")
        
    reset_dict_list = np.append(reset_dict_list, reset_dict)
    
for env_name in env_names:

    print(f"Environment: {env_name}")

    env = gymnasium.make(env_name, disable_env_checker=True)

    _, model_path, log_path = get_model_path(
        0,
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

    model = ALGOS["td3"].load(model_path, device="auto", **parameters)

    
    for reset_dict in tqdm(reset_dict_list):
        obs, infos = env.reset(options=reset_dict)
        episode_reward = 0.0
        ep_len = 0

        done = False
        
        while (not done) & (ep_len <= max_episode_len):
            step += 1
            action, lstm_states = model.predict(obs,  deterministic=True) 
            
            obs, reward, done, truncated, infos = env.step(action)

            episode_start = done

            episode_reward += reward
            ep_len += 1

            if done :
                # print(f"Episode Reward: {episode_reward:.2f}")
                # print("Episode Length", ep_len)

                if env_name == env_names[0]:
                    episode_rewards_env1 = np.append(episode_rewards_env1,episode_reward)
                    episode_lengths_env1 = np.append(episode_lengths_env1,ep_len)
                elif env_name == env_names[1]:
                    episode_rewards_env2 = np.append(episode_rewards_env2,episode_reward)
                    episode_lengths_env2 = np.append(episode_lengths_env2,ep_len)
                episode_reward = 0.0
                ep_len = 0

compare_episode_lengths = episode_lengths_env1 - episode_lengths_env2

env2_better_ones = np.zeros(compare_episode_lengths.shape)
env1_better_ones = np.zeros(compare_episode_lengths.shape)

env2_better_ones[compare_episode_lengths > 0] = 1
env1_better_ones[compare_episode_lengths < 0] = 1

env2_better_sum = np.sum(env2_better_ones)
env1_better_sum = np.sum(env1_better_ones)

env2_better_mean = -np.sum(compare_episode_lengths[compare_episode_lengths > 0])/env2_better_sum
env1_better_mean = np.sum(compare_episode_lengths[compare_episode_lengths < 0])/env1_better_sum

print(f"Number of tests : {nb_tests}")
print(f"{env_names[0]} better in {(env1_better_sum*100)/nb_tests}% of the tests with a mean of {env1_better_mean} less steps than the other environment")
print(f"{env_names[1]} better in {(env2_better_sum*100)/nb_tests}% of the tests with a mean of {env2_better_mean} less steps than the other environment")
print(f"Same : {((nb_tests - (env1_better_sum + env2_better_sum))*100)/nb_tests}%")