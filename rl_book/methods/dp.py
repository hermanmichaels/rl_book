from typing import Callable

import numpy as np
from gymnasium.spaces import Discrete

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.method_wrapper import with_default_svalues


def get_policy(
    V: np.ndarray,
    observation_space: Discrete,
    action_space: Discrete,
    P: dict,  # [dict[list[float, float, float, bool]]], - TODO
    gamma: float,
) -> np.ndarray:
    return np.asarray(
        [
            np.argmax(
                [
                    p * (r + gamma * V[s_next])
                    for a in range(action_space.n)
                    for p, s_next, r, _ in P[s][a]  # type: ignore
                ]
            )
            for s in range(observation_space.n)
        ]
    )

@with_default_svalues
def policy_iteration(
    env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    """Uses 'Policy Iteration' to solve the RL problem
    specified by the passed Gymnasium env.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    observation_space, action_space = get_observation_action_space(env)

    pi = np.zeros(observation_space.n).astype(np.int32)

    def _policy_evaluation() -> np.ndarray:
        """Run's policy evaluation - i.e. evaluates the current
        policy pi, and updates the value estimate V.
        """
        V = np.zeros(observation_space.n)
        while True:
            delta = 0.
            for s in range(observation_space.n):
                v = V[s]
                V[s] = sum(
                    [
                        p * (r + env.gamma * V[s_next])
                        for p, s_next, r, _ in env.env.P[s][pi[s]]  # type: ignore
                    ]
                )
                delta = max(delta, abs(v - V[s]))
            if delta < env.eps:
                break
        return V

    for step in range(max_steps):
        V = _policy_evaluation()

        policy_stable = True
        for s in range(observation_space.n):
            old_a = pi[s]
            pi[s] = np.argmax(
                [
                    p * (r + env.gamma * V[s_next])
                    for a in range(action_space.n)
                    for p, s_next, r, _ in env.env.P[s][a]  # type: ignore
                ]
            )
            if old_a != pi[s]:
                policy_stable = False

        pi = get_policy(V, observation_space, action_space, env.env.P, env.gamma)
        success = success_cb(pi, step)

        if success:
            return success, pi, step

    return False, pi, step

@with_default_svalues
def value_iteration(
    env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    """Uses 'Value Iteration' to solve the RL problem
    specified by the passed Gymnasium env.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    observation_space, action_space = get_observation_action_space(env)

    V = np.zeros(observation_space.n)

    for step in range(max_steps):
        delta = 0
        for s in range(observation_space.n):
            v = V[s]
            V[s] = max(
                [
                    p * (r + env.gamma * V[s_next])
                    for a in range(action_space.n)
                    for p, s_next, r, _ in env.env.P[s][a]  # type: ignore
                ]
            )
            delta = max(delta, abs(v - V[s]))

        pi = get_policy(V, observation_space, action_space, env.env.P, env.gamma)
        success = success_cb(pi, step)
        if success:
            return success, pi, step

    return False, pi, step
