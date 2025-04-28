import functools
from typing import Callable

import numpy as np

from rl_book.env import ParametrizedEnv

INT_INF = 100000000

# Decorator to add default values for success_cb and max_steps to all algorithm calls
def with_default_values(func):
    @functools.wraps(func)
    def wrapper(
        env: ParametrizedEnv,
        success_cb: Callable[[np.ndarray, int], bool] | None = None,
        max_steps: int | None = None,
    ):
        if success_cb is None:
            success_cb = lambda pi, step: False
        if max_steps is None:
            max_steps = INT_INF
        return func(env, success_cb, max_steps)

    return wrapper
