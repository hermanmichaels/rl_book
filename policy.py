from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_book.gym_utils import get_observation_action_space


@dataclass
class ReplayBufferItem:
    states: list[int]
    actions: list[int]
    logprobs: list[float]
    rewards: list[float]
    is_terminals: list[bool]


class ReplayBuffer:
    def __init__(self):
        self.items = []

    def add(self, state, action, logprob, reward, is_terminal):
        self.items.append(ReplayBufferItem(state, action, logprob, reward, is_terminal))

    def clear(self):
        self.items = []

    def states(self):
        return [item.states for item in self.items]

    def actions(self):
        return [item.actions for item in self.items]

    def logprobs(self):
        return [item.logprobs for item in self.items]

    def rewards(self):
        return [item.rewards for item in self.items]

    def is_terminals(self):
        return [item.is_terminals for item in self.items]


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.policy = nn.Linear(state_dim, action_dim, bias=False)
        self.value = nn.Linear(state_dim, 1)
        self.state_dim = state_dim

    def forward(self, state_idx):
        state = torch.eye(self.state_dim)[state_idx]
        probs = torch.softmax(self.policy(state), dim=-1)
        return probs

    def evaluate(self, states, actions):
        probs = torch.softmax(self.policy(states), dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.value(states).squeeze()
        return action_logprobs, torch.detach(state_values), dist_entropy

    def select_action(self, state):
        probs = self(state)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs[action]


def compute_returns(rewards, is_terminals, gamma):
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(is_terminals)):
        if done:
            G = 0  # TODO: check for reinforce
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def ppo_update(policy, optimizer, replay_buffer, state_dim, gamma):
    states = torch.eye(state_dim)[replay_buffer.states()]
    actions = torch.tensor(replay_buffer.actions())
    old_logprobs = torch.stack(replay_buffer.logprobs()).detach()
    returns = compute_returns(
        replay_buffer.rewards(), replay_buffer.is_terminals(), gamma
    )
    returns = torch.tensor(returns, dtype=torch.float32)
    values = policy.value(states).squeeze().detach()
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    K_epochs = 4
    eps_clip = 0.2

    for _ in range(K_epochs):
        logprobs, state_values, entropy = policy.evaluate(states, actions)
        ratios = torch.exp(logprobs - old_logprobs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(state_values, returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def reinforce_update(optimizer, replay_buffer, gamma):
    returns = compute_returns(
        replay_buffer.rewards(), [False for _ in replay_buffer.is_terminals()], gamma
    )
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss = -torch.sum(
        torch.stack(replay_buffer.logprobs()) * (returns - returns.mean())
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def solve(
    env, success_cb: Callable[[np.ndarray], bool], max_steps: int, method: str
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    policy = PolicyNetwork(observation_space.n, action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    replay_buffer = ReplayBuffer()

    total_rewards = []

    for episode in range(max_steps):
        state, _ = env.env.reset()

        num_steps_in_episode = 0
        while True:
            action, log_prob = policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action, state)

            replay_buffer.add(state, action, log_prob, reward, terminated or truncated)

            state = next_state

            if terminated or truncated:
                break

            num_steps_in_episode += 1

            if num_steps_in_episode > 1000:
                break

        if method == "reinforce":
            reinforce_update(optimizer, replay_buffer, env.gamma)
        elif method == "ppo":
            ppo_update(policy, optimizer, replay_buffer, observation_space.n, env.gamma)
        else:
            raise ValueError("X")

        total_rewards.append(sum(replay_buffer.rewards()))
        # print(f"Episode {episode + 1}/{max_steps}, Total Reward: {sum(replay_buffer.rewards())}")

        replay_buffer.clear()

    return False, policy, 0, total_rewards
