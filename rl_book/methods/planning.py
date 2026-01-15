import copy
import pickle
import random
from collections import defaultdict
from typing import Any, DefaultDict, Optional

import numpy as np
from typing_extensions import override

from rl_book.env import ParametrizedEnv
from rl_book.methods.method import RLMethod
from rl_book.replay_utils import ReplayItem

NUM_STEPS = 1000
NUM_MCTS_ITERATIONS = 1000
UCB_EXPLORATION_CONST = 0.01
ALPHA = 0.1


class ReplayBuffer:
    def __init__(self, max_length: int = 1000):
        self.replay_buffer: list[tuple[int, int]] = []
        self.max_length = max_length

    def push(self, state: int, action: int) -> None:
        self.replay_buffer.append((state, action))
        self.replay_buffer = self.replay_buffer[-self.max_length :]

    def sample(self) -> tuple[int, int]:
        return self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]


def model_factory():
    return 0, 0.0, 0, []


class DynaQ(RLMethod[int]):
    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        n: int = 3,
        plus_mode: bool = False,
        **kwargs: object
    ):
        self.Q: DefaultDict[tuple[int, int], float] = defaultdict(float)
        self.n = n
        self.buffer = ReplayBuffer()
        self.model: DefaultDict[
            tuple[int, int], tuple[int, float, int, np.ndarray | list]
        ] = defaultdict(model_factory)
        self.plus_mode = plus_mode

        super().__init__(env, load_weights, **kwargs)

    @override
    def get_name(self) -> str:
        return "DynaQ"

    @override
    def clone(self):
        cloned = self.__class__(self.env, False, self.n, self.plus_mode)
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

    @override
    def act(self, state: int, step: int | None = None, mask: np.ndarray | list = []):
        allowed_actions = self.get_allowed_actions(mask)
        if (
            self._train
            and step is not None
            and random.uniform(0, 1) < self.env.eps(step)
        ):
            return random.choice(allowed_actions)
        else:
            q_values = [self.Q[state, a] for a in allowed_actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(allowed_actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def _learn(self, step: int) -> None:
        kappa = 0.1

        for _ in range(self.n):
            observation, action = self.buffer.sample()
            observation_new_sampled, reward, t_last, mask = self.model[
                observation, action
            ]
            bonus_reward = kappa * np.sqrt(step - t_last) if self.plus_mode else 0.0

            allowed_actions = self.get_allowed_actions(mask)
            next_q = max(
                [self.Q[observation_new_sampled, a_] for a_ in allowed_actions],
                default=0,
            )

            self.Q[observation, action] = self.Q[observation, action] + ALPHA * (
                (float(reward) + bonus_reward)
                + self.env.gamma * next_q
                - self.Q[observation, action]
            )

    @override
    def update(
        self, episode: list[ReplayItem[int]], step: int
    ) -> None:  # TODO: signtuare
        if len(episode) <= 2:
            return

        self.buffer.push(episode[-2].state, episode[-2].action)

        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]

        allowed_actions = self.get_allowed_actions(next_state.mask)
        next_q = max(
            [self.Q[next_state.state, a_] for a_ in allowed_actions],
            default=0,
        )

        self.Q[cur_state.state, cur_state.action] = self.Q[
            cur_state.state, cur_state.action
        ] + ALPHA * (
            cur_state.reward
            + self.env.gamma * next_q
            - self.Q[cur_state.state, cur_state.action]
        )

        self.model[cur_state.state, cur_state.action] = (
            next_state.state,
            cur_state.reward,
            step,
            next_state.mask,
        )

        self._learn(step)

    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        self.Q[cur_state.state, cur_state.action] += ALPHA * (
            cur_state.reward - self.Q[cur_state.state, cur_state.action]
        )

        self._learn(step)

    def _get_save_data(self) -> Any:
        return self.Q, self.model

    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            self.Q, self.model = pickle.load(f)


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
    _, reward, terminated, truncated, _ = env.env.step(node.children[expand_idx].action)
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

    for _ in range(NUM_MCTS_ITERATIONS):
        node = root
        reset_env(env, actions)

        # Select nodes until leaf node.
        node = select(env.env, node)

        # Expand leaf node.
        if not node.terminal:
            node = expand(env, node, env.get_action_space_len())

        # Simulate step.
        truncated = False
        terminated = node.terminal
        total_reward = node.reward

        while not terminated and not truncated:
            # Use a random rollout policy.
            action = random.randint(0, env.get_action_space_len() - 1)
            _, reward, terminated, truncated, _ = env.env.step(action)
            total_reward += float(reward)

        # Backprop found reward.
        backprop(node, total_reward, env.gamma)

    return int(
        np.argmax([child.reward_sum / (child.visits + 1) for child in root.children])
    )
