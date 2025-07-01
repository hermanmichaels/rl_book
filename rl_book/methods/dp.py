import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.methods.method import RLMethod

EPS = 0.05


def extract_policy(
    V: np.ndarray,
    observation_space_len: int,
    action_space_len: int,
    P: dict,
    gamma: float,
) -> np.ndarray:
    """Extracts a policy from the given value function.

    Args:
        V: found value function
        observation_space: observation space
        action_space: action space
        P: transition function
        gamma: discount factor

    Returns:
        policy
    """
    return np.asarray(
        [
            np.argmax(
                [
                    p * (r + gamma * V[s_next])
                    for a in range(action_space_len)
                    for p, s_next, r, _ in P[s][a]  # type: ignore
                ]
            )
            for s in range(observation_space_len)
        ]
    )


class DPMethod(RLMethod):
    def __init__(self, env: ParametrizedEnv, pi: np.ndarray) -> None:
        super().__init__(env)
        self.pi = pi

    def act(
        self, state: int, step: int | None = None, mask: np.ndarray | list = []
    ) -> int:
        return self.pi[state]


def policy_iteration(env: ParametrizedEnv) -> DPMethod:
    """Uses 'Policy Iteration' to solve the RL problem
    specified by the passed Gymnasium env.

    Args:
        env: env containing the problem

    Returns:
        found method
    """
    pi = np.zeros(env.get_observation_space_len()).astype(np.int32)

    def _policy_evaluation() -> np.ndarray:
        """Run's policy evaluation - i.e. evaluates the current
        policy pi, and updates the value estimate V.
        """
        V = np.zeros(env.get_observation_space_len())
        while True:
            delta = 0
            for s in range(env.get_observation_space_len()):
                v = V[s]
                V[s] = sum(
                    [
                        p * (r + env.gamma * V[s_next])
                        for p, s_next, r, _ in env.env.unwrapped.P[s][  # type: ignore
                            pi[s]
                        ]
                    ]
                )
                delta = max(delta, abs(v - V[s]))
            if delta < EPS:
                break
        return V

    while True:
        V = _policy_evaluation()

        policy_stable = True
        for s in range(env.get_observation_space_len()):
            old_a = pi[s]
            pi[s] = np.argmax(
                [
                    p * (r + env.gamma * V[s_next])
                    for a in range(env.get_action_space_len())
                    for p, s_next, r, _ in env.env.unwrapped.P[s][a]  # type: ignore
                ]
            )
            if old_a != pi[s]:
                policy_stable = False
                break

        if policy_stable:
            return DPMethod(env, pi)


def value_iteration(env: ParametrizedEnv) -> DPMethod:
    """Uses 'Value Iteration' to solve the RL problem
    specified by the passed Gymnasium env.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    V = np.zeros(env.get_observation_space_len())

    while True:
        delta = 0
        for s in range(env.get_observation_space_len()):
            v = V[s]
            V[s] = max(
                [
                    p * (r + env.gamma * V[s_next])
                    for a in range(env.get_action_space_len())
                    for p, s_next, r, _ in env.env.unwrapped.P[s][a]  # type: ignore
                ]
            )
            delta = max(delta, abs(v - V[s]))

        if delta < EPS:
            break

    pi = extract_policy(
        V,
        env.get_observation_space_len(),
        env.get_action_space_len(),
        env.env.unwrapped.P,  # type: ignore
        env.gamma,
    )

    return DPMethod(env, pi)
