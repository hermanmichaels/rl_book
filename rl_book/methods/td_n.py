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
    def __init__(self, n: int = 3):
        self.Q = defaultdict(float)
        self.n = n

    def finalize(self, replay_buffer):
        for tau in range(len(replay_buffer) - self.n - 1, len(replay_buffer)):
            self.update(replay_buffer, tau)

    def update(self, replay_buffer, tau = None):
        is_final = True
        if tau is None:
            tau = len(replay_buffer) - self.n - 1
            is_final = False

        print(f"{tau} -- {len(replay_buffer)}")
        b = self.Q
        rhos = []
        # T = len(replay_buffer) - 1

        if tau >= 0:
            # rho = math.prod(
            #     [
            #         div_with_zero(
            #             self.Q[replay_buffer[i].state, replay_buffer[i].action],
            #             b[replay_buffer[i].state, replay_buffer[i].action],
            #         )
            #         for i in range(tau + 1, tau + self.n + 1)
            #     ]
            # )
            rho = 1
            rhos.append(rho)

            G = sum(
                [
                    replay_buffer[i].reward * 0.95 ** (i - tau)
                    for i in range(tau, min(tau + self.n, len(replay_buffer)))
                ]
            )

            if not is_final:
                print("extend")
                # import ipdb
                # ipdb.set_trace()
                G = (
                    G
                    + 0.95**self.n
                    * self.Q[replay_buffer[tau + self.n].state, replay_buffer[tau + self.n].action]
                )

            # import ipdb
            # ipdb.set_trace()

            # if G != 0:
            #     import ipdb
            #     ipdb.set_trace()

            self.Q[replay_buffer[tau].state, replay_buffer[tau].action] = self.Q[
                replay_buffer[tau].state, replay_buffer[tau].action
            ] + ALPHA * rho / (sum(rhos) + 1) * (
                G - self.Q[replay_buffer[tau].state, replay_buffer[tau].action]
            )

    # TODO: share
    def act(self, state, mask):
        q_values = [self.Q[(state, a)] for a in np.nonzero(mask)[0].tolist()]
        max_q = max(q_values)

        eps_greedy = random.randint(0, 100)

        # TODO: eps greedy?
        max_actions = [a for a, q in zip(np.nonzero(mask)[0].tolist(), q_values) if q == max_q or eps_greedy <= 5]

        # import ipdb
        # ipdb.set_trace()
        
        return random.choice(max_actions)
    
@with_default_values
def sarsa_n(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
    n: int = 3,
    off_policy: bool = False,
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        b = (
            np.random.rand(int(observation_space.n), int(action_space.n))
            if off_policy
            else Q
        )

        observation, _ = env.env.reset()

        terminated = truncated = False
        action = (
            get_eps_greedy_action(Q[observation], env.eps(step))
            if not off_policy
            else get_eps_greedy_action(b[observation], eps=0)
        )

        replay_buffer = [ReplayItem(observation, int(action), 0.0)]

        T = 2**31 - 1  # terminal step
        t = 0  # current step
        tau = 0  # update value estimate for this time step

        rhos = []  # importance sampling weights

        while True:
            if t < T:
                # While not terminal, continue playing episode.
                observation_new, reward, terminated, truncated, _ = env.step(
                    action, observation
                )
                action_new = get_eps_greedy_action(Q[observation_new], env.eps(step))
                replay_buffer.append(
                    ReplayItem(observation_new, action_new, float(reward))
                )
                if terminated or truncated:
                    T = t + 1

                observation = observation_new
                action = action_new

            tau = t - n + 1
            if tau >= 0:
                rho = math.prod(
                    [
                        div_with_zero(
                            Q[replay_buffer[i].state, replay_buffer[i].action],
                            b[replay_buffer[i].state, replay_buffer[i].action],
                        )
                        for i in range(tau + 1, min(tau + n, T - 1) + 1)
                    ]
                )
                rhos.append(rho)

                G = sum(
                    [
                        replay_buffer[i].reward * env.gamma ** (i - tau - 1)
                        for i in range(tau + 1, min(tau + n, T) + 1)
                    ]
                )

                if tau + n < T:
                    G = (
                        G
                        + env.gamma**n
                        * Q[replay_buffer[tau + n].state, replay_buffer[tau + n].action]
                    )

                Q[replay_buffer[tau].state, replay_buffer[tau].action] = Q[
                    replay_buffer[tau].state, replay_buffer[tau].action
                ] + ALPHA * rho / (sum(rhos) + 1) * (
                    G - Q[replay_buffer[tau].state, replay_buffer[tau].action]
                )

            if tau == T - 1:
                break

            t += 1

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q, observation_space), step

class TreeN(TDMethod):
    def __init__(self, n: int = 3):
        self.Q = defaultdict(float)
        self.n = n

    def finalize(self, replay_buffer):
        for tau in range(len(replay_buffer) - self.n - 1, len(replay_buffer)):
            self.update(replay_buffer, tau)

    def update(self, replay_buffer, tau = None):
        is_final = True
        if tau is None:
            tau = len(replay_buffer) - self.n - 1
            is_final = False

        print(f"{tau} -- {len(replay_buffer)}")
        b = self.Q
        rhos = []
        # T = len(replay_buffer) - 1

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
                        + 0.95
                        * sum(
                            [
                                self.Q[replay_buffer[k].state, a]
                                / (sum([self.Q[replay_buffer[k].state, a] for a in range(4)]) + 0.001)
                                * self.Q[replay_buffer[k].state, a]
                                for a in range(4)
                                if a != replay_buffer[k].action
                            ]
                        )
                        + 0.95
                        * self.Q[replay_buffer[k].state, replay_buffer[k].action]
                        / (sum([self.Q[replay_buffer[k].state, a] for a in range(4)]) + 0.001)
                        * G
                    )

                self.Q[replay_buffer[tau].state, replay_buffer[tau].action] = self.Q[
                    replay_buffer[tau].state, replay_buffer[tau].action
                ] + ALPHA * (G - self.Q[replay_buffer[tau].state, replay_buffer[tau].action])
            

    # TODO: share
    def act(self, state, mask):
        q_values = [self.Q[(state, a)] for a in np.nonzero(mask)[0].tolist()]
        max_q = max(q_values)

        eps_greedy = random.randint(0, 100)

        # TODO: eps greedy?
        max_actions = [a for a, q in zip(np.nonzero(mask)[0].tolist(), q_values) if q == max_q or eps_greedy <= 5]

        # import ipdb
        # ipdb.set_trace()
        
        return random.choice(max_actions)

@with_default_values
def tree_n(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
    n: int = 3,
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n)) + 0.1

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation], env.eps(step))

        replay_buffer = [ReplayItem(observation, action, 0.0)]

        T = 2**31 - 1  # terminal step
        t = 0  # current step
        tau = 0  # update value estimate for this time step

        while True:
            if t < T:
                observation_new, reward, terminated, truncated, _ = env.step(
                    action, observation
                )
                action_new = get_eps_greedy_action(Q[observation_new], env.eps(step))
                replay_buffer.append(
                    ReplayItem(observation_new, action_new, float(reward))
                )
                if terminated or truncated:
                    T = t + 1

                observation = observation_new
                action = action_new

            tau = t - n + 1

            if tau >= 0:
                if t + 1 >= T:
                    G = replay_buffer[T].reward
                else:
                    G = replay_buffer[t + 1].reward + env.gamma * sum(
                        [
                            Q[replay_buffer[t + 1].state, a]
                            / sum(Q[replay_buffer[t + 1].state, :])
                            * Q[replay_buffer[t + 1].state, a]
                            for a in range(action_space.n)
                        ]
                    )

                for k in range(min(t, T - 1), tau + 1, -1):
                    G = (
                        replay_buffer[k].reward
                        + env.gamma
                        * sum(
                            [
                                Q[replay_buffer[k].state, a]
                                / sum(Q[replay_buffer[k].state, :])
                                * Q[replay_buffer[k].state, a]
                                for a in range(action_space.n)
                                if a != replay_buffer[k].action
                            ]
                        )
                        + env.gamma
                        * Q[replay_buffer[k].state, replay_buffer[k].action]
                        / sum(Q[replay_buffer[k].state, :])
                        * G
                    )

                Q[replay_buffer[tau].state, replay_buffer[tau].action] = Q[
                    replay_buffer[tau].state, replay_buffer[tau].action
                ] + ALPHA * (G - Q[replay_buffer[tau].state, replay_buffer[tau].action])

            if tau == T - 1:
                break

            t += 1

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q, observation_space), step
