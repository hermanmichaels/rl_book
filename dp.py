import numpy as np

from env import ParametrizedEnv
from gym_utils import get_observation_action_space


def policy_iteration(env: ParametrizedEnv) -> np.ndarray:
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
            delta = 0
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

    while True:
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

        if policy_stable:
            return pi


def value_iteration(env: ParametrizedEnv) -> np.ndarray:
    """Uses 'Value Iteration' to solve the RL problem
    specified by the passed Gymnasium env.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    observation_space, action_space = get_observation_action_space(env)

    V = np.zeros(observation_space.n)

    while True:
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
        if delta < env.eps:
            break

    return np.asarray(
        [
            np.argmax(
                [
                    p * (r + env.gamma * V[s_next])
                    for a in range(action_space.n)
                    for p, s_next, r, _ in env.env.P[s][a]  # type: ignore
                ]
            )
            for s in range(observation_space.n)
        ]
    )
