import argparse

import gymnasium as gym

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from policy import solve
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n
import matplotlib.pyplot as plt

import torch

GAMMA = 0.99
EPS = 0.001
NUM_STEPS = 100


gym_env_train = gym.make(
    "CliffWalking-v0",
)

env_train = ParametrizedEnv(gym_env_train, GAMMA, EPS)

_, pi_reinforce, _, reinforce_rewards = solve(env_train, None, 3000, "reinforce")
_, pi_ppo, _, ppo_rewards = solve(env_train, None, 3000, "ppo")

x = [x for x in range(len(reinforce_rewards))]
plt.plot(x, reinforce_rewards, "ro-")
plt.plot(x, ppo_rewards, "bo-")
plt.savefig("plot.png")

gym_env_train.close()


