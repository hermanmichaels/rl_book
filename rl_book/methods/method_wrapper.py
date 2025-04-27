from typing import Callable
from rl_book.env import ParametrizedEnv
import numpy as np

INT_INF = 100000000


def with_default_svalues(func):
    def wrapper(env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool] | None =None, max_steps: int | None = None):
        if success_cb is None:
            success_cb = lambda pi, step: False
        if max_steps is None:
            max_steps = INT_INF
        return func(env, success_cb)
    return wrapper