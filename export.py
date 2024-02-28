from distutils.sysconfig import customize_compiler
import gymnasium as gym
import os
import rl_zoo3.import_envs
import importlib
import torch as th
from torchinfo import summary
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs
import argparse
from rl_zoo3.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
import openvino as ov


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Env", type=str, required=True)
parser.add_argument("--model", help="TD3 model to export", type=str, required=False)
parser.add_argument("--output", help="Target directory", type=str, required=True)
parser.add_argument(
    "--gym-packages",
    type=str,
    nargs="+",
    default=[],
    help="Additional external Gym environment package modules to import",
)
args = parser.parse_args()
device = th.device("cpu")

for env_module in args.gym_packages:
    importlib.import_module(env_module)

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

print(f"Loading env {args.env}")
env = gym.make(args.env)

if args.model is None:
    exp_id = get_latest_run_id("logs/td3/", args.env)
    model_fname = f"logs/td3/{args.env}_{exp_id}/best_model.zip"
else:
    model_fname = args.model

print(f"Loading model {model_fname}")
model = ALGOS["td3"].load(model_fname, env=env, custom_objects=custom_objects, device=device)

policy = model.policy

# Creating a dummy observation
obs = th.Tensor(env.reset()[0], device=device)
obs = preprocess_obs(obs, env.observation_space).unsqueeze(0)

print(f"Generating a dummy observation {obs}")

actor_fname = f"{args.output}{args.env}_actor.onnx"
print(f"Exporting actor model to {actor_fname}")
actor_model = th.nn.Sequential(policy.actor.features_extractor, policy.actor.mu)
th.onnx.export(actor_model, obs, actor_fname, opset_version=13)
summary(actor_model)

# Value function is a combination of actor and Q
class TD3PolicyValue(th.nn.Module):
    def __init__(self, policy: TD3Policy, actor_model: th.nn.Module):
        super(TD3PolicyValue, self).__init__()

        self.actor = actor_model
        self.critic = policy.critic

    def forward(self, obs):
        action = self.actor(obs)
        critic_features = self.critic.features_extractor(obs)
        return self.critic.q_networks[0](th.cat([critic_features, action], dim=1))


# Note(antonin): unused variable action
# action = policy.actor.mu(policy.actor.extract_features(obs))
v_model = TD3PolicyValue(policy, actor_model)
summary(v_model)
value_fname = f"{args.output}{args.env}_value.onnx"
print(f"Exporting value model to {value_fname}")
th.onnx.export(v_model, obs, value_fname, opset_version=13)

print("Exporting models for OpenVino...")
input_shape = obs.shape

input_actor = (input_shape, ov.Type.f32)
input_value = (input_shape, ov.Type.f32)

ov_model_actor = ov.convert_model(input_model=actor_fname, input=input_actor)
ov_model_value = ov.convert_model(input_model=value_fname,input=input_value)
ov.save_model(ov_model_actor, f"{args.output}{args.env}_actor.xml")
ov.save_model(ov_model_value, f"{args.output}{args.env}_value.xml")

# Old way to export model to OpenVino IR using Model Optimizer (mo)
# input_shape = ",".join(map(str, obs.shape))
# os.system(f"mo --input_model {actor_fname} --input_shape [{input_shape}] --data_type FP32 --output_dir {args.output}")
# os.system(f"mo --input_model {value_fname} --input_shape [{input_shape}] --data_type FP32 --output_dir {args.output}")
