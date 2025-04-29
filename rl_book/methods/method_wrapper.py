import functools
from typing import Callable

import numpy as np
from gymnasium.core import Env

from rl_book.env import ParametrizedEnv

MAX_INFERENCE_STEPS = 1000
INT_INF = 10000000


def success_callback(pi: np.ndarray, step: int, env: Env) -> bool:
    """Tests whether the given policy can successfully solve the given Gridworld
    environment.

    Args:
        pi: policy
        step: current step
        env: env

    Returns:
        False if current step is not a step to be checked, or policy does not solve env
        - True otherwise.
    """
    observation, _ = env.reset()
    for _ in range(MAX_INFERENCE_STEPS):
        action = pi[observation]
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    return reward == 1


# Decorator to add default values for success_cb and max_steps to all algorithm calls
def with_default_values(func):
    @functools.wraps(func)
    def wrapper(
        env: ParametrizedEnv,
        success_cb: Callable[[np.ndarray, int], bool] | None = None,
        max_steps: int | None = None,
    ):
        if success_cb is None:
            success_cb = functools.partial(success_callback, env=env.env)
        if max_steps is None:
            max_steps = INT_INF
        return func(env, success_cb, max_steps)

    return wrapper
