import argparse

import gymnasium as gym
import numpy as np
from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from policy import solve
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

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
    # render_mode="human"
)

policies = []
rewards = []

HIGH = 0.8
LOW = 0.1

UP = torch.Tensor([HIGH, LOW, LOW, LOW])
DOWN = torch.Tensor([LOW, LOW, HIGH, LOW])
LEFT = torch.Tensor([LOW, LOW, LOW, HIGH])
RIGHT = torch.Tensor([LOW, HIGH, LOW, LOW])

for i in range(50):
    print(i)
    num_steps = 1 if i < 3 else random.randint(0, 2000)

    pi = solve(env_train, None, num_steps, "ppo")[1]

    if i < 3:
        with torch.no_grad():
            pi.policy.weight.copy_(torch.randn_like(pi.policy.weight))
            pi.policy.weight[:, 36] = UP
            offset = 24

            if i >= 1:
                pi.policy.weight[:, 24] = UP
                offset = 12
            if i >= 2:
                pi.policy.weight[:, 12] = UP
                offset = 0

            for j in range(offset, offset + 11):
                pi.policy.weight[:, j] = RIGHT

            pi.policy.weight[:, 35] = DOWN
            if i >= 1:
                pi.policy.weight[:, 23] = DOWN
            if i >= 2:
                pi.policy.weight[:, 11] = DOWN

    observation, _ = gym_env_test.reset()

    rewards_ = []
    for _ in range(NUM_STEPS):
        action = torch.argmax(pi(torch.Tensor([observation]).long())).item()
        observation, reward, terminated, truncated, _ = gym_env_test.step(action)
        rewards_.append(reward)
        if terminated or truncated or reward == -100:
            break

    # TODO: normalize
    policies.append(pi.policy.weight.flatten())
    rewards.append(sum(rewards_))


# import ipdb
# ipdb.set_trace()

X_2d = TSNE(n_components=2).fit_transform(torch.stack(policies).detach())

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=np.stack(rewards), cmap='viridis', s=50)

plt.scatter(X_2d[0, 0], X_2d[0, 1], color='red', marker='*', s=200, label='close')
plt.scatter(X_2d[1, 0], X_2d[1, 1], color='green', marker='*', s=200, label='medium')
plt.scatter(X_2d[2, 0], X_2d[2, 1], color='blue', marker='*', s=200, label='far')

plt.colorbar(sc, label='Reward')
plt.title("t-SNE of Policy Parameters (colored by reward)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend()
plt.grid(True)
plt.savefig("tsne.png")


