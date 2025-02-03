import heapq
import random
from collections import defaultdict
from typing import Optional

import numpy as np

from env import ParametrizedEnv
from gym_utils import get_observation_action_space
from td import ALPHA, get_eps_greedy_action

NUM_STEPS = 1000
NUM_MCTS_ITERATIONS = 1000
UCB_EXPLORATION_CONST = 0.01


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
    def __init__(
        self,
        parent: Optional["TreeNode"] = None,
        action: int | None = None,
    ) -> None:
        """Instantiates a tree node.

        Args:
            parent: parent node
            action: action taken to reach the node
        """
        self.action = action
        self.parent = parent

        self.children: list["TreeNode"] = []

        self.terminal: bool | None = None
        self.reward = 0.0

        self.visits = 0
        self.reward_sum = 0.0

    def update(self, terminal: bool, reward: float) -> None:
        """When we visit a node for the first time,
        update all relevant state stats.
        """
        self.terminal = terminal
        self.reward = reward


def select_child(node: TreeNode) -> TreeNode:
    """Selects a child node."""
    # If any of the child nodes has not been visited yet, first visit these.
    unvisited_children = [child for child in node.children if child.visits == 0]
    if unvisited_children:
        return unvisited_children[
            np.random.choice([i for i in range(len(unvisited_children))])
        ]

    # Otherwise, select child according to UCB rule.
    ucb_values = []
    for child in node.children:
        ucb_values.append(
            child.reward_sum / child.visits
            + UCB_EXPLORATION_CONST * np.sqrt(np.log(node.visits) / child.visits)
        )

    ucb_values_np = np.asarray(ucb_values)
    ucb_values_np /= np.sum(ucb_values_np)

    return node.children[
        np.random.choice([i for i in range(len(node.children))], p=ucb_values_np)
    ]


def select(env, node: TreeNode) -> TreeNode:
    """Select phase of the MCTS algorithm.
    Select nodes until reaching a leaf node.
    """
    while node.children:
        node = select_child(node)
        _, reward, terminated, truncated, _ = env.step(node.action)
        # If node was not visited yet, update state stats.
        if node.visits == 0:
            node.update(terminated or truncated, reward)

    return node


def expand(env, node: TreeNode, n: int) -> TreeNode:
    """Expand phase of the MCTS algorithm.
    Creates all child nodes, and selects a random one.

    Args:
        env: gym environment
        node: node to expand
        n: number of possible actions
    """
    node.children = [TreeNode(parent=node, action=i) for i in range(n)]
    expand_idx = random.randint(0, len(node.children) - 1)
    _, reward, terminated, truncated, _ = env.env.step(
        node.children[expand_idx].action
    )
    node.children[expand_idx].update(terminated or truncated, reward)
    return node.children[expand_idx]


def backprop(node: TreeNode, reward: float, gamma: float) -> None:
    """Backprops result of MCTS run - i.e. travers all visited
    nodes and updates visit count and reward_sum.
    """
    reward = reward * gamma
    node.visits += 1
    node.reward_sum += reward

    if node.parent:
        backprop(node.parent, reward, gamma)


def reset_env(env, actions: list[int]) -> int:
    """'Resets' the env to the state defined / reachable
    by the given action sequence.
    NOTE: this function assumes the environment to be deterministic.

    Args:
        env: gym env
        actions: action sequence

    Returns:
        resulting state
    """
    observation, _ = env.env.reset()
    for action in actions:
        observation, _, _, _, _ = env.env.step(action)
    return observation


def mcts(env: ParametrizedEnv, actions: list[int]) -> int:
    """Runs the MCTS algorithm.

    Args:
        env: environment
        actions: list of actions taken till the current state

    Returns:
        best action to take
    """
    reset_env(env, actions)
    root = TreeNode()

    action_space_n: int = int(get_observation_action_space(env)[1].n)

    for _ in range(NUM_MCTS_ITERATIONS):
        node = root
        reset_env(env, actions)

        # Select nodes until leaf node.
        node = select(env.env, node)

        # Expand leaf node.
        if not node.terminal:
            node = expand(env, node, action_space_n)

        # Simulate step.
        truncated = False
        terminated = node.terminal
        total_reward = node.reward

        while not terminated and not truncated:
            # Use a random rollout policy.
            action = random.randint(0, action_space_n - 1)
            _, reward, terminated, truncated, _ = env.env.step(action)
            total_reward += float(reward)

        # Backprop found reward.
        backprop(node, total_reward, env.gamma)

    return int(
        np.argmax([child.reward_sum / (child.visits + 1) for child in root.children])
    )
