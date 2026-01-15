from collections import defaultdict
from typing import override

import numpy as np
import torch

from rl_book.env import ParametrizedEnv
from rl_book.methods.td import TDMethod
from rl_book.replay_utils import ReplayItem
from rl_book.utils import ConstantFactory

ALPHA = 0.1


class SarsaN(TDMethod):
    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
        n: int = 3,
    ) -> None:
        self.n = n
        super().__init__(env, load_weights, device)

    @override
    def get_name(self) -> str:
        return "SarsaN"

    def _update(self, episode: list[ReplayItem[int]], tau: int | None = None) -> None:
        if tau is None:
            # tau is set when finalizing the episode - otherwise pick
            # the correct update step here.
            tau = len(episode) - self.n - 1

        if tau >= 0:
            G = sum(
                [
                    episode[i].reward * self.env.gamma ** (i - tau)
                    for i in range(tau, min(tau + self.n, len(episode)))
                ]
            )

            if tau + self.n < len(episode):
                G = (
                    G
                    + self.env.gamma**self.n
                    * self.Q[
                        episode[tau + self.n].state,
                        episode[tau + self.n].action,
                    ]
                )

            self.Q[episode[tau].state, episode[tau].action] += ALPHA * (
                G - self.Q[episode[tau].state, episode[tau].action]
            )

    @override
    def update(self, episode: list[ReplayItem[int]], step: int) -> None:
        self._update(episode, None)

    @override
    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        # Replay has terminated - still finish updating the values
        # by going over the remaining episode.

        for tau in range(len(episode) - self.n, len(episode)):
            self._update(episode, tau)


class TreeN(TDMethod):
    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
        n: int = 3,
    ):
        super().__init__(env, load_weights, device)
        self.Q = defaultdict(ConstantFactory(0.1))
        self.n = n

    @override
    def get_name(self) -> str:
        return "TreeN"

    def _update(self, replay_buffer: list[ReplayItem[int]], tau: int | None = None):
        if tau is None:
            tau = len(replay_buffer) - self.n - 1

        if tau >= 0:
            if tau >= len(replay_buffer) - 1:
                G = replay_buffer[-1].reward
            else:
                allowed_actions = self.get_allowed_actions(replay_buffer[-1].mask)
                G = replay_buffer[-2].reward + self.env.gamma * sum(
                    [
                        self._get_action_prob(replay_buffer[-1].state, a)
                        * self.Q[replay_buffer[-1].state, a]
                        for a in allowed_actions
                    ]
                )

            for k in range(len(replay_buffer) - 2, tau, -1):
                allowed_actions = self.get_allowed_actions(replay_buffer[k].mask)
                G = (
                    replay_buffer[k - 1].reward
                    + self.env.gamma
                    * sum(
                        [
                            self._get_action_prob(replay_buffer[k].state, a)
                            * self.Q[replay_buffer[k].state, a]
                            for a in allowed_actions
                            if a != replay_buffer[k].action
                        ]
                    )
                    + self.env.gamma
                    * self._get_action_prob(
                        replay_buffer[k].state, replay_buffer[k].action
                    )
                    * G
                )

            self.Q[replay_buffer[tau].state, replay_buffer[tau].action] += ALPHA * (
                G - self.Q[replay_buffer[tau].state, replay_buffer[tau].action]
            )

    @override
    def update(
        self, replay_buffer: list[ReplayItem[int]], step: int, tau: int | None = None
    ):
        self._update(replay_buffer, None)

    @override
    def finalize(self, episode: list[ReplayItem[int]], step: int) -> None:
        for tau in range(len(episode) - self.n, len(episode)):
            self._update(episode, tau)

    def _get_action_prob(self, observation: int, action: int) -> float:
        probs = [self.Q[observation, a] for a in range(self.env.get_action_space_len())]
        probs = np.exp(probs - np.max(probs))
        return probs[action] / sum(probs)
