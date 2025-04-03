import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from gymnasium.spaces import Discrete

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.utils import div_with_zero, get_eps_greedy_action


@dataclass
class ReplayItem:
    state: int
    action: int
    reward: float


ALPHA = 0.1


# TODO:  re-use with TD
def get_policy(Q, observation_space: Discrete) -> np.ndarray:
    return np.array([np.argmax(Q[s]) for s in range(observation_space.n)])


def sarsa_n(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray], bool],
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
                observation_new, reward, terminated, truncated, _ = env.step(action, observation)
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


def tree_n(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray], bool],
    max_steps: int,
    n: int = 3,
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n)) + 0.1

    step = 10000
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
                observation_new, reward, terminated, truncated, _ = env.step(action, observation)
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
