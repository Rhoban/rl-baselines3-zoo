import numpy as np
import time
import placo
import gym_footsteps_planning
from gym_footsteps_planning.envs.footsteps_planning_env import FootstepsPlanningEnv
import gymnasium
import kidsize
from gym_footsteps_planning.footsteps_simulator.simulator import Simulator
from rl_zoo3.utils import (
    ALGOS,
    create_test_env,
    get_latest_run_id,
    get_saved_hyperparams,
)


# Creating the env in render mode
env: FootstepsPlanningEnv = gymnasium.make(
    "footsteps-planning-any-obstacle-multigoal-v0",
)
env.visualize = True

model = ALGOS["td3"].load("best_model.zip", env=env)


def find_intermediate_target(result, distance: float = 1.5):
    path = list(result.path)
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        d = np.linalg.norm(p2 - p1)
        if d > distance:
            yaw = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            return (p1 + (p2 - p1) * distance / d), yaw
        distance -= d
    return None


step = 0
support_pose = [-2.25, -1.5, 1.5]
support_foot = "right"
target = [1.5, 1.6, np.pi]

obstacle_avoider = kidsize.ObstacleAvoider()

sim = env.simulator

obstacles = [
    [0.25, -1, 0.3],
    [-1.5, -1, 0.3],
    [-1.5, -1.5, 0.3],
    [-1.5, -1.75, 0.3],
    [-1.5, 0.5, 0.3],
    [-1.75, 1.0, 0.3],
    [-2.25, 1.0, 0.3],
    [-2.75, 1.0, 0.3],
    [-0.75, 0.0, 0.3],
    [0.0, 0.0, 0.3],
    [0.5, -0.5, 0.3],
    [1.0, 1.0, 0.25],
    [1.5, 1.0, 0.25],
    [2, 1.0, 0.25],
    [-1.2, 0.25, 0.25],
    [-0.4, 0, 0.25],
    [0.25, -0.25, 0.2]
]

for x, y, radius in obstacles:
    obstacle_avoider.addObstacle(np.array([x, y]), radius + 0.1)

while True:
    if step % 1 == 0:
        # Finding the A* path
        if step == 0:
            middle_pos = np.array(support_pose[:2])
        else:
            left_pose = sim.foot_pose("left")
            right_pose = sim.foot_pose("right")
            middle_pos = (np.array(left_pose[:2]) + np.array(right_pose[:2])) / 2
        result = obstacle_avoider.findPathClipped(
            np.array(middle_pos), np.array(target[:2]), 0.05, 0.025, -2, 2, -2, 2
        )
        sim.path = list(result.path)

        # Finding the intermediate target
        intermediate_target = find_intermediate_target(result)
        if intermediate_target is not None:
            target_pos, target_yaw = intermediate_target
            desired_goal = target_pos[0], target_pos[1], target_yaw
            sim.extra_footsteps = [["right", target]]
        else:
            desired_goal = target
            sim.extra_footsteps = []

        # Finding the local obstacle
        local_obstacle = None
        candidate = None
        for obstacle_id in result.obstacles:
            if obstacle_id != -1:
                local_obstacle = obstacle_id
                break

        options = {
            "start_foot_pose": support_pose,
            "start_support_foot": support_foot,
            "target_foot_pose": desired_goal,
            "target_support_foot": "right",
        }

        if local_obstacle is not None:
            options["has_obstacle"] = True
            options["obstacle_radius"] = obstacles[local_obstacle][2]+0.05
            env.options["obstacle_position"] = obstacles[local_obstacle][:2]
        else:
            options["has_obstacle"] = False

        footsteps = sim.footsteps.copy()
        env.reset(seed=0, options=options)
        sim.footsteps = footsteps

    # Updating obstacle drawing
    sim.clear_obstacles()
    for k, obstacle in enumerate(obstacles):
        x, y, radius = obstacle
        if k != local_obstacle:
            sim.add_obstacle((x, y), radius, color=(200, 200, 200, 255))
    for k, obstacle in enumerate(obstacles):
        x, y, radius = obstacle
        if k == local_obstacle:
            sim.add_obstacle((x, y), radius, color=(70, 150, 0, 255))

    observation = env.get_observation()
    action, _ = model.predict(observation)
    env.step(action)
    support_pose = sim.support_pose()
    support_foot = sim.support_foot

    sim.render()
    time.sleep(0.02)
    step += 1
    if step == 2:
        input("Press Enter to start the simulation...")
