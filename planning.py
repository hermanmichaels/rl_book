import heapq
import random
from collections import defaultdict

import numpy as np

from env import ParametrizedEnv
from gym_utils import get_observation_action_space
from td import ALPHA, get_eps_greedy_action

NUM_STEPS = 1000


class ReplayBuffer:
    def __init__(self, max_length: int = 1000):
        self.replay_buffer: list[tuple[int, int]] = []
        self.max_length = max_length

    def push(self, state: int, action: int) -> None:
        self.replay_buffer.append((state, action))
        self.replay_buffer = self.replay_buffer[-self.max_length :]

    def sample(self) -> tuple[int, int]:
        return self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]


def dyna_q(env: ParametrizedEnv, n=3, plus_mode: bool = False) -> np.ndarray:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))
    model = np.zeros((observation_space.n, action_space.n, 3))

    buffer = ReplayBuffer()
    t = 0
    kappa = 0

    for _ in range(NUM_STEPS):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation])

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q[observation])
            buffer.push(observation, action)
            t += 1

            observation_new, reward, terminated, truncated, _ = env.env.step(action)
            Q[observation, action] = Q[observation, action] + ALPHA * (
                reward + env.gamma * np.max(Q[observation_new]) - Q[observation, action]
            )
            model[observation, action] = observation_new, reward, t

            for _ in range(n):
                observation, action = buffer.sample()
                observation_new_sampled, reward, t_last = model[observation, action]
                bonus_reward = kappa * np.sqrt(t - t_last) if plus_mode else 0.0
                Q[observation, action] = Q[observation, action] + ALPHA * (
                    (reward + bonus_reward)
                    + env.gamma * np.max(Q[int(observation_new_sampled)])
                    - Q[observation, action]
                )

            observation = observation_new

    return np.array([np.argmax(Q[s]) for s in range(observation_space.n)])


def generate_predecessor_states(
    env: ParametrizedEnv,
) -> dict[int, set[tuple[int, int]]]:
    """Generates a dictionary of predecessor states.

    Args:
        env: env to use

    Returns:
        dict[state] containing all (s, a) tuples leading into state
    """
    observation_space, action_space = get_observation_action_space(env)
    predecessors = defaultdict(set)

    for state in range(observation_space.n):
        for action in range(action_space.n):
            _, next_state, _, _ = env.env.P[state][action][0]  # type: ignore
            predecessors[next_state].add((state, action))

    return predecessors


def prioritized_sweeping(env: ParametrizedEnv) -> np.ndarray:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))
    model = np.zeros((observation_space.n, action_space.n, 2))
    p_queue: list[tuple[float, tuple[int, int]]] = []
    theta = 0.1
    predecessors = generate_predecessor_states(env)

    n = 3

    for _ in range(NUM_STEPS):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation])

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q[observation])
            observation_new, reward, terminated, truncated, _ = env.env.step(action)
            model[observation, action] = observation_new, reward
            P = abs(
                reward + env.gamma * max(Q[observation_new]) - Q[observation, action]
            )
            if P > theta:
                heapq.heappush(p_queue, (-P, (observation, action)))

            for _ in range(n):
                if not p_queue:
                    break

                _, (observation, action) = heapq.heappop(p_queue)
                observation_new_sampled, reward = model[observation, action]
                Q[observation, action] = Q[observation, action] + ALPHA * (
                    reward
                    + env.gamma * np.max(Q[int(observation_new_sampled)])
                    - Q[observation, action]
                )

                for s_bar, a_bar in predecessors[observation]:
                    _, reward = model[s_bar, a_bar]
                    P = abs(reward + env.gamma * max(Q[observation]) - Q[s_bar, a_bar])
                    if P > theta:
                        heapq.heappush(p_queue, (-P, (s_bar, a_bar)))

            observation = observation_new

    return np.array([np.argmax(Q[s]) for s in range(observation_space.n)])


class TreeNode:
    def __init__(self, parent = None, action = None):
        self.children = []
        self.action = action
        self.parent = parent
        self.visits = 0
        self.reward = 0
        self.terminal = None
        self.sim_step = parent.sim_step + 1 if parent else 0
        self.visited = False
    
    def update(self, reward, terminal):
        self.visited = True
        self.reward = reward
        self.terminal = terminal

def select_child(node):
    if any([child.terminal is None for child in node.children]):
        return node.children[np.random.choice([i for i in range(len(node.children))])]
    
    ucb_values = []
    for child in node.children:
        ucb_values.append(child.reward / child.visits + 1 * np.sqrt(np.log(node.visits) / child.visits)) # because all visits are 1 sometimes

    ucb_values = np.asarray(ucb_values)
    ucb_values /= np.sum(ucb_values)

    return node.children[np.random.choice([i for i in range(len(node.children))], p=ucb_values)]

def select(env, node):
    reward = 0
    while node.children:
        node = select_child(node)
        _, reward, terminated, truncated, _ = env.env.step(node.action)
        node.update(reward, terminated or truncated)

    return node


def expand(env, node, n):
    node.children = [TreeNode(parent=node, action=i) for i in range(n)]
    expand_idx = random.randint(0, len(node.children) - 1)
    _, reward, terminated, truncated, _ = env.env.step(node.children[expand_idx].action)
    node.children[expand_idx].update(reward, terminated or truncated)
    return node.children[expand_idx]

def backprop(node, reward, gamma):
    reward = reward * gamma
    node.visits += 1
    node.reward += reward

    if node.parent:
        backprop(node.parent, reward, gamma)


def reset_env(env, actions):
    observation, _ = env.env.reset()
    for action in actions:
        observation, _, _, _, _= env.env.step(action)
    return observation

def mcts(env: ParametrizedEnv, pi, state, actions) -> np.ndarray:
    root = TreeNode()
    _, action_space = get_observation_action_space(env)

    for _ in range(1000):
        node = root
        observation = reset_env(env, actions)

        node = select(env, node)
        if not node.terminal:
            node = expand(env, node, action_space.n)

        truncated = False
        terminated = node.terminal

        sim_step = node.sim_step
        total_reward = node.reward

        while not terminated and not truncated:
            action = pi[observation]
            observation, reward, terminated, truncated, _= env.env.step(action)
            print(observation)
            if observation in [-1]:
                reward += 1
            total_reward += reward * env.gamma ** sim_step
            total_reward -= 0.1
            sim_step += 1

        backprop(node, total_reward, env.gamma)

    return np.argmax([child.reward / child.visits for child in root.children])
