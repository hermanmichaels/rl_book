import argparse
from collections import defaultdict

import gymnasium as gym

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from policy import solve
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n
import matplotlib.pyplot as plt
import numpy as np

import torch

GAMMA = 0.99
EPS = 0.001
NUM_STEPS = 30


gym_env_train = gym.make(
    "CliffWalking-v0",
)
gym_env_test = gym.make(
    "CliffWalking-v0",
    # render_mode="human"
)

env_train = ParametrizedEnv(gym_env_train, GAMMA, EPS)
observations = defaultdict(int)

for _ in range(15):
    pi = solve(env_train, None, 5000, "reinforce")[1]

    observation, _ = gym_env_test.reset()
    observations[observation] += 1
    for _ in range(NUM_STEPS):
        action = torch.argmax(pi(torch.Tensor([observation]).long())).item()
        observation, _, terminated, truncated, _ = gym_env_test.step(action)
        observations[observation] += 1
        if terminated or truncated:
            break

def plot_visit_heatmap_with_cliff(visit_counts, title="State Visit Frequency with Cliff"):
    grid_height, grid_width = 4, 12
    heatmap = np.zeros((grid_height, grid_width))

    # Fill in visit counts
    for state_idx, count in visit_counts.items():
        row = state_idx // grid_width
        col = state_idx % grid_width
        heatmap[row, col] = count

    # Define start, goal, and cliff
    start = (3, 0)
    goal = (3, 11)
    cliff = [(3, i) for i in range(1, 11)]

    plt.figure(figsize=(10, 4))
    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest')

    # Overlay cliff, start, and goal
    for (r, c) in cliff:
        plt.text(c, r, '☠', ha='center', va='center', fontsize=14, color='cyan')
    plt.text(start[1], start[0], 'S', ha='center', va='center', fontsize=14, color='lime')
    plt.text(goal[1], goal[0], 'G', ha='center', va='center', fontsize=14, color='white')

    plt.colorbar(im, label='Visit Count')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    # plt.gca().invert_yaxis()
    plt.grid(False)
    plt.xticks(np.arange(grid_width))
    plt.yticks(np.arange(grid_height))
    plt.savefig("vis.png")

    # import ipdb
    # ipdb.set_trace()

plot_visit_heatmap_with_cliff(observations)

gym_env_test.close()

