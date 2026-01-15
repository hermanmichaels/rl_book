import copy
import pickle
import random
from abc import ABC
from collections import defaultdict
from typing import Any, DefaultDict

import numpy as np
from typing_extensions import override

from rl_book.env import ParametrizedEnv
from rl_book.methods.method import RLMethod
from rl_book.replay_utils import ReplayItem

ALPHA = 0.1


class TDMethod(RLMethod[int], ABC):
    def __init__(
        self, env: ParametrizedEnv, load_weights: bool = False, **kwargs: object
    ) -> None:
        self.Q: DefaultDict[tuple[int, int], float] = defaultdict(float)

        super().__init__(env, load_weights, **kwargs)

    @override
    def clone(self) -> "TDMethod":
        cloned = self.__class__(self.env, False)
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

    @override
    def act(
        self, state: int, step: int | None = None, mask: np.ndarray | list = []
    ) -> int:
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

    @override
    def _get_save_data(self) -> Any:
        return self.Q

    @override
    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            self.Q = pickle.load(f)


class Sarsa(TDMethod):
    @override
    def get_name(self) -> str:
        return "Sarsa"

    @override
    def update(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 2:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        self.Q[prev_state.state, prev_state.action] += ALPHA * (
            float(prev_state.reward)
            + self.env.gamma * self.Q[cur_state.state, cur_state.action]
            - self.Q[prev_state.state, prev_state.action]
        )

    @override
    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        self.Q[cur_state.state, cur_state.action] += ALPHA * (
            float(cur_state.reward) - self.Q[cur_state.state, cur_state.action]
        )


class QLearning(TDMethod):
    @override
    def get_name(self) -> str:
        return "QLearning"

    @override
    def update(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 2:
            return

        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]

        allowed_actions = self.get_allowed_actions(next_state.mask)
        next_q = max(
            [self.Q[next_state.state, a_] for a_ in allowed_actions],
            default=0,
        )

        self.Q[cur_state.state, cur_state.action] += ALPHA * (
            cur_state.reward
            + self.env.gamma * next_q
            - self.Q[cur_state.state, cur_state.action]
        )

    @override
    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        self.Q[cur_state.state, cur_state.action] += ALPHA * (
            cur_state.reward - self.Q[cur_state.state, cur_state.action]
        )


class ExpectedSarsa(TDMethod):
    @override
    def get_name(self) -> str:
        return "ExpectedSarsa"

    def _get_action_prob(self, observation: int, action: int) -> float:
        probs = [self.Q[observation, a] for a in range(self.env.get_action_space_len())]
        probs = np.exp(probs - np.max(probs))
        return probs[action] / sum(probs)

    @override
    def update(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 2:
            return

        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]

        updated_q_value = self.Q[cur_state.state, cur_state.action] + ALPHA * (
            cur_state.reward - self.Q[cur_state.state, cur_state.action]
        )

        actions = self.get_allowed_actions(next_state.mask)
        for a in actions:
            updated_q_value += (
                self.env.gamma
                * ALPHA
                * self._get_action_prob(next_state.state, a)
                * self.Q[next_state.state, a]
            )

        self.Q[cur_state.state, cur_state.action] = updated_q_value

    @override
    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        self.Q[cur_state.state, cur_state.action] += +ALPHA * (
            cur_state.reward - self.Q[cur_state.state, cur_state.action]
        )


class DoubleQ(TDMethod):
    @override
    def get_name(self) -> str:
        return "DoubleQ"

    def __init__(
        self, env: ParametrizedEnv, load_weights: bool = False, **kwargs: object
    ) -> None:
        self.Q_2: DefaultDict[tuple[int, int], float] = defaultdict(float)

        super().__init__(env, load_weights, **kwargs)

    def _update(self, Q1, Q2, cur_state, next_state):
        if next_state:
            allowed_actions = self.get_allowed_actions(next_state.mask)
            max_a = allowed_actions[
                np.argmax(
                    [Q1[next_state.state, a] for a in allowed_actions],
                )
            ]
            next_q = self.env.gamma * Q2[next_state.state, max_a]
        else:
            next_q = 0

        Q1[cur_state.state, cur_state.action] += ALPHA * (
            cur_state.reward + next_q - Q1[cur_state.state, cur_state.action]
        )

    @override
    def update(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 2:
            return

        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]

        if random.random() < 0.5:
            self._update(self.Q, self.Q_2, cur_state, next_state)
        else:
            self._update(self.Q_2, self.Q, cur_state, next_state)

    @override
    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        if random.randint(0, 100) < 50:
            self._update(self.Q, self.Q_2, cur_state, None)
        else:
            self._update(self.Q_2, self.Q, cur_state, None)

    @override
    def _get_save_data(self) -> Any:
        return self.Q, self.Q_2

    @override
    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            self.Q, self.Q_2 = pickle.load(f)
