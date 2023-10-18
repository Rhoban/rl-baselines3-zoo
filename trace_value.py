from distutils.sysconfig import customize_compiler
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
from torchinfo import summary
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs
import argparse
from rl_zoo3.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Env", type=str, required=True)
parser.add_argument("--env2", help="2nd Env", type=str, required=False)
parser.add_argument("--model", help="TD3 model to trace", type=str, required=False)
args = parser.parse_args()
device = th.device("cpu")

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

print(f"Loading env1 {args.env}")
env = gym.make(args.env)

print(f"Loading env2 {args.env2}")
env2 = gym.make(args.env2)

if args.model is None:
    exp_id = get_latest_run_id("logs/td3/", args.env)
    model_fname = f"logs/td3/{args.env}_{exp_id}/best_model.zip"
else:
    model_fname = args.model

if args.model is None:
    exp_id2 = get_latest_run_id("logs/td3/", args.env2)
    model_fname2 = f"logs/td3/{args.env2}_{exp_id2}/best_model.zip"
else:
    model_fname2 = args.model

print(f"Loading model {model_fname}")
model = ALGOS["td3"].load(model_fname, env=env, custom_objects=custom_objects, device=device)

print(f"Loading model {model_fname2}")
model2 = ALGOS["td3"].load(model_fname2, env=env2, custom_objects=custom_objects, device=device)

policy = model.policy

policy2 = model2.policy

# Creating a dummy observation
obs = th.Tensor(env.reset(), device=device)
obs = preprocess_obs(obs, env.observation_space).unsqueeze(0)

print(f"Generating a dummy observation {obs}")

obs2 = th.Tensor(env2.reset(), device=device)
obs2 = preprocess_obs(obs2, env2.observation_space).unsqueeze(0)

print(f"Generating a dummy observation {obs2}")

actor_model = th.nn.Sequential(policy.actor.features_extractor, policy.actor.mu)
summary(actor_model)

actor_model2 = th.nn.Sequential(policy2.actor.features_extractor, policy2.actor.mu)
summary(actor_model2)

# Value function is a combination of actor and Q
class TD3PolicyValue(th.nn.Module):
    def __init__(self, policy: TD3Policy, actor_model: th.nn.Module):
        super(TD3PolicyValue, self).__init__()

        self.actor = actor_model
        self.critic = policy.critic

    def forward(self, obs):
        action = self.actor(obs)
        critic_features = self.critic.features_extractor(obs)
        # print(self.critic.q_networks[0](th.cat([critic_features, action], dim=1)))
        return (self.critic.q_networks[0](th.cat([critic_features, action], dim=1)) + self.critic.q_networks[1](th.cat([critic_features, action], dim=1)))/2.


# Note(antonin): unused variable action
# action = policy.actor.mu(policy.actor.extract_features(obs))
v_model = TD3PolicyValue(policy, actor_model)

# action2 = policy2.actor.mu(policy2.actor.extract_features(obs2))
v_model2 = TD3PolicyValue(policy2, actor_model2)


a, b = -1, 1
theta = np.pi
N = 256

heatmap = [
    [
        [x, y, np.cos(theta), np.sin(theta), 1., 0., 0., 0., 0.]
        for x in np.linspace(a, b, N)
    ]
    for y in np.linspace(a, b, N)
]

heatmap = th.tensor(heatmap, dtype=th.float32)
heatmap2 = heatmap

with th.no_grad():
    heatmap = v_model(heatmap.view(-1, 9))
    heatmap2 = v_model2(heatmap2.view(-1, 9))

heatmap = heatmap.view(N, N, 1)
heatmap = heatmap.cpu().numpy()

heatmap2 = heatmap2.view(N, N, 1)
heatmap2 = heatmap2.cpu().numpy()

heatmap3 = heatmap2-heatmap

heatmap3 = np.where(heatmap3<0, True, False)
size_map = (a, b, a, b)

plt.subplot(131)
a = plt.imshow(heatmap, vmin=-35, vmax=-10)
a.set_extent(size_map)
plt.colorbar()

plt.subplot(132)
b = plt.imshow(heatmap2, vmin=-35, vmax=-10)
b.set_extent(size_map)
plt.colorbar()

plt.subplot(133)
c = plt.imshow(heatmap3)
c.set_extent(size_map)
plt.colorbar()
plt.show() 
