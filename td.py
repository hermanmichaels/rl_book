import random

import numpy as np

from env import ParametrizedEnv
from gym_utils import get_observation_action_space
from utils import get_eps_greedy_action

ALPHA = 0.1
NUM_STEPS = 1000


def sarsa(env: ParametrizedEnv) -> np.ndarray:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    for _ in range(NUM_STEPS):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation])

        while not terminated and not truncated:
            observation_new, reward, terminated, truncated, _ = env.env.step(action)
            action_new = get_eps_greedy_action(Q[observation_new])
            q_next = Q[observation_new, action_new] if not terminated else 0
            Q[observation, action] = Q[observation, action] + ALPHA * (
                reward + env.gamma * q_next - Q[observation, action]
            )
            observation = observation_new
            action = action_new

    return np.array([np.argmax(Q[s]) for s in range(observation_space.n)])


def q(env) -> np.ndarray:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    for _ in range(100):
        observation, _ = env.env.reset()
        terminated = truncated = False

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q[observation])
            observation_new, reward, terminated, truncated, _ = env.env.step(action)
            Q[observation, action] = Q[observation, action] + ALPHA * (
                reward + env.gamma * np.max(Q[observation_new]) - Q[observation, action]
            )
            observation = observation_new

    return np.array([np.argmax(Q[s]) for s in range(env.env.observation_space.n)])


def expected_sarsa(env: ParametrizedEnv) -> np.ndarray:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    def _get_action_prob(Q: np.ndarray) -> float:
        return (
            Q[observation_new, a] / sum(Q[observation_new, :])
            if sum(Q[observation_new, :])
            else 1
        )

    for _ in range(NUM_STEPS):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation])

        while not terminated and not truncated:
            observation_new, reward, terminated, truncated, _ = env.env.step(action)
            action_new = get_eps_greedy_action(Q[observation_new])
            updated_q_value = Q[observation, action] + ALPHA * (
                reward - Q[observation, action]
            )
            for a in range(action_space.n):
                updated_q_value += ALPHA * _get_action_prob(Q) * Q[observation_new, a]
            Q[observation, action] = updated_q_value
            observation = observation_new
            action = action_new

    return np.array([np.argmax(Q[s]) for s in range(observation_space.n)])


def double_q(env) -> np.ndarray:
    observation_space, action_space = get_observation_action_space(env)
    Q_1 = np.zeros((observation_space.n, action_space.n))
    Q_2 = np.zeros((observation_space.n, action_space.n))

    for _ in range(NUM_STEPS):
        observation, _ = env.env.reset()
        terminated = truncated = False

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q_1[observation] + Q_2[observation])
            observation_new, reward, terminated, truncated, _ = env.env.step(action)
            if random.randint(0, 100) < 50:
                Q_1[observation, action] = Q_1[observation, action] + ALPHA * (
                    reward
                    + env.gamma * Q_2[observation_new, np.argmax(Q_1[observation_new])]
                    - Q_1[observation, action]
                )
            else:
                Q_2[observation, action] = Q_2[observation, action] + ALPHA * (
                    reward
                    + env.gamma * Q_1[observation_new, np.argmax(Q_2[observation_new])]
                    - Q_2[observation, action]
                )
            observation = observation_new

    return np.array([np.argmax(Q_1[s]) for s in range(env.env.observation_space.n)])
