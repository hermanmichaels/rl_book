import random
from typing import Callable

import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.method_wrapper import with_default_svalues
from rl_book.utils import get_eps_greedy_action, get_policy

ALPHA = 0.1

@with_default_svalues
def sarsa(
    env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False

        eps = env.eps(step)

        action = get_eps_greedy_action(Q[observation], eps)

        while not terminated and not truncated:
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )
            action_new = get_eps_greedy_action(Q[observation_new], eps)
            q_next = Q[observation_new, action_new] if not terminated else 0
            Q[observation, action] = Q[observation, action] + ALPHA * (
                float(reward) + env.gamma * q_next - Q[observation, action]
            )
            observation = observation_new
            action = action_new

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q, observation_space), step

@with_default_svalues
def q(
    env, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    first_goal = True
    step = 0

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False

        c = 0

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q[observation], env.eps(step))
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            # print(reward)

            if first_goal and reward >= 1:
                # print(step)
                first_goal = False

            Q[observation, action] = Q[observation, action] + ALPHA * (
                reward + env.gamma * np.max(Q[observation_new]) - Q[observation, action]
            )
            observation = observation_new

            c += 1
            if c > 500:
                print("XXXXXXXXXXXXXXXX")
                break

        # import ipdb
        # ipdb.set_trace()

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q, observation_space), step

@with_default_svalues
def expected_sarsa(
    env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    def _get_action_prob(Q: np.ndarray) -> float:
        return (
            Q[observation_new, a] / sum(Q[observation_new, :])
            if sum(Q[observation_new, :])
            else 1
        )

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation])

        c = 0

        while not terminated and not truncated:
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )
            action_new = get_eps_greedy_action(Q[observation_new])
            updated_q_value = Q[observation, action] + ALPHA * (
                reward - Q[observation, action]
            )
            for a in range(action_space.n):
                updated_q_value += ALPHA * _get_action_prob(Q) * Q[observation_new, a]
            Q[observation, action] = updated_q_value
            observation = observation_new
            action = action_new

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

        c += 1
        if c > 100:
            break

    return False, get_policy(Q, observation_space), step

@with_default_svalues
def double_q(
    env, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q_1 = np.zeros((observation_space.n, action_space.n))
    Q_2 = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        observation, _ = env.env.reset()

        terminated = truncated = False

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q_1[observation], env.eps(step))
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

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

        pi = get_policy(Q_1, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q_1, observation_space), step
