import argparse

import gymnasium as gym
import numpy as np
from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from policy import ReplayBuffer, reinforce_update, solve
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import torch.optim as optim

import torch

GAMMA = 0.99
EPS = 0.001
NUM_STEPS = 30


gym_env_train = gym.make(
    "CliffWalking-v0",
)

env_train = ParametrizedEnv(gym_env_train, GAMMA, EPS)

gym_env_test = gym.make(
    "CliffWalking-v0",
    render_mode="human"
)

policies = []
rewards = []

HIGH = 0.8
LOW = 0.1

UP = torch.Tensor([HIGH, LOW, LOW, LOW])
DOWN = torch.Tensor([LOW, LOW, HIGH, LOW])
LEFT = torch.Tensor([LOW, LOW, LOW, HIGH])
RIGHT = torch.Tensor([LOW, HIGH, LOW, LOW])


pi = solve(env_train, None, 1, "ppo")[1]

with torch.no_grad():
    pi.policy.weight.copy_(torch.randn_like(pi.policy.weight))
    pi.policy.weight[:, 36] = UP
    offset = 24

    pi.policy.weight[:, 24] = UP
    offset = 12

    for j in range(offset, offset + 11):
        pi.policy.weight[:, j] = RIGHT

    pi.policy.weight[:, 35] = DOWN
    pi.policy.weight[:, 23] = DOWN

observation, _ = gym_env_test.reset()

for _ in range(NUM_STEPS):
    action = torch.argmax(pi(torch.Tensor([observation]).long())).item()
    observation, reward, terminated, truncated, _ = gym_env_test.step(action)
    if terminated or truncated or reward == -100:
        break

optimizer = optim.Adam(pi.policy.parameters(), lr=1)

replay_buffer = ReplayBuffer()

sas = [(36, [1, 0, 0, 0])] + [(24 + i, [0, 1, 0, 0]) for i in range(12)] + [(35, [0, 0, 1, 0])]
for s, a in sas:
    replay_buffer.add(s, a, pi.evaluate(torch.Tensor(torch.eye(48)[s]), torch.Tensor(a))[0][torch.argmax(torch.Tensor(a))], -1, False)

# TODO: wrong
reinforce_update(optimizer, replay_buffer, env_train.gamma)

import ipdb
ipdb.set_trace()

observation, _ = gym_env_test.reset()

for _ in range(NUM_STEPS):
    action = torch.argmax(pi(torch.Tensor([observation]).long())).item()
    observation, reward, terminated, truncated, _ = gym_env_test.step(action)
    if terminated or truncated or reward == -100:
        break


