from collections import defaultdict
import math
from dataclasses import dataclass
import random
from typing import Callable

import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.method_wrapper import with_default_values
from rl_book.methods.td import TDMethod
from rl_book.utils import div_with_zero, get_eps_greedy_action, get_policy


@dataclass
class ReplayItem:
    state: int
    action: int
    reward: float


ALPHA = 0.1

class SarsaN(TDMethod):
    def __init__(self, env, n: int = 3):
        super().__init__(env)
        self.n = n

    def finalize(self, replay_buffer, step):
        for tau in range(len(replay_buffer) - self.n - 1, len(replay_buffer)):
            self.update(replay_buffer, step, tau)

    def update(self, replay_buffer, step, tau = None):
        is_final = True
        if tau is None:
            tau = len(replay_buffer) - self.n - 1
            is_final = False

        b = self.Q
        rhos = []

        if tau >= 0:
            rho = math.prod(
                [
                    div_with_zero(
                        self.Q[replay_buffer[i].state, replay_buffer[i].action],
                        b[replay_buffer[i].state, replay_buffer[i].action],
                    )
                    for i in range(tau + 1, min(tau + self.n + 1, len(replay_buffer)))
                ]
            )
            rhos.append(rho)

            G = sum(
                [
                    replay_buffer[i].reward * self.env.eps(step) ** (i - tau)
                    for i in range(tau, min(tau + self.n, len(replay_buffer)))
                ]
            )

            if not is_final:
                G = (
                    G
                    + self.env.gamma**self.n
                    * self.Q[replay_buffer[tau + self.n].state, replay_buffer[tau + self.n].action]
                )

            self.Q[replay_buffer[tau].state, replay_buffer[tau].action] = self.Q[
                replay_buffer[tau].state, replay_buffer[tau].action
            ] + ALPHA * rho / (sum(rhos) + 1) * (
                G - self.Q[replay_buffer[tau].state, replay_buffer[tau].action]
            )

    
class TreeN(TDMethod):
    def __init__(self, env, n: int = 3):
        super().__init__(env)
        self.n = n

    def finalize(self, replay_buffer, step):
        for tau in range(len(replay_buffer) - self.n - 1, len(replay_buffer)):
            self.update(replay_buffer, step, tau)

    def update(self, replay_buffer, step, tau = None):
        is_final = True
        if tau is None:
            tau = len(replay_buffer) - self.n - 1
            is_final = False
        b = self.Q
        rhos = []

        if tau >= 0:
            if is_final:
                G = replay_buffer[-1].reward
            else:
                G = replay_buffer[-1].reward + 0.95 * sum(
                    [
                        self.Q[replay_buffer[-1].state, a]
                        / (sum([self.Q[replay_buffer[-1].state, a] for a in range(4)])  + 0.001)
                        * self.Q[replay_buffer[-1].state, a]
                        for a in range(4)
                    ]
                )

                for k in range(tau, min(tau + self.n, len(replay_buffer))):
                    G = (
                        replay_buffer[k].reward
                        + self.env.gamma
                        * sum(
                            [
                                self.Q[replay_buffer[k].state, a]
                                / (sum([self.Q[replay_buffer[k].state, a] for a in range(self.env.env.action_space.n)]) + 0.001)
                                * self.Q[replay_buffer[k].state, a]
                                for a in range(self.env.env.action_space.n)
                                if a != replay_buffer[k].action
                            ]
                        )
                        + self.env.gamma
                        * self.Q[replay_buffer[k].state, replay_buffer[k].action]
                        / (sum([self.Q[replay_buffer[k].state, a] for a in range(self.env.env.action_space.n)]) + 0.001)
                        * G
                    )

                self.Q[replay_buffer[tau].state, replay_buffer[tau].action] = self.Q[
                    replay_buffer[tau].state, replay_buffer[tau].action
                ] + ALPHA * (G - self.Q[replay_buffer[tau].state, replay_buffer[tau].action])

