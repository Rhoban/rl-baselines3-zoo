import time
import os
import yaml
import math
import gym_footsteps_planning
import gymnasium
import numpy as np
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from stable_baselines3.common.utils import set_random_seed
from rl_zoo3.utils import StoreDict, get_model_path

env_names = ["footsteps-planning-left-v0", "footsteps-planning-left-her-v0"]
step = 0

# set_random_seed(0)

nb_tests = 1000

algo="td3"
folder="logs"

reset_dict_list = np.array([])

episode_rewards_env1, episode_lengths_env1 = np.array([]), np.array([])
episode_rewards_env2, episode_lengths_env2 = np.array([]), np.array([])

for i in range(nb_tests):

    reset_dict = {
        "start_support_foot" : "left" if (np.random.uniform(0, 1) > 0.5) else "right",
        "target_support_foot" : "left" if (np.random.uniform(0, 1) > 0.5) else "right",
        "foot_pose" : np.random.uniform([-2, -2, -math.pi], [2, 2, math.pi]),
    }
    reset_dict_list = np.append(reset_dict_list, reset_dict)

for env_name in env_names:

    print(f"Environment: {env_name}")

    env = gymnasium.make(env_name)

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

    model = ALGOS["td3"].load(model_path, device="cpu", **parameters)

    
    for reset_dict in reset_dict_list:
        obs, infos = env.reset(seed=0, options=reset_dict)

        episode_reward = 0.0
        ep_len = 0

        done = False

        while not done:
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

env2_better_ones[compare_episode_lengths < 0] = 1
env1_better_ones[compare_episode_lengths > 0] = 1

env2_better_sum = np.sum(env2_better_ones)
env1_better_sum = np.sum(env1_better_ones)

print(f"Number of tests : {nb_tests}")
print(f"{env_names[0]} better : {(env1_better_sum*100)/nb_tests}%") 
print(f"{env_names[1]} better : {(env2_better_sum*100)/nb_tests}%")
print(f"Same : {((nb_tests - (env1_better_sum + env2_better_sum))*100)/nb_tests}%")