import random

import numpy as np


def get_eps_greedy_action(q_values: np.ndarray, eps: float = 0.05) -> int:
    if random.uniform(0, 1) < eps or np.all(q_values == q_values[0]):
        return int(np.random.choice([a for a in range(len(q_values))]))
    else:
        return int(np.argmax(q_values))


def div_with_zero(x: float, y: float) -> float:
    if x == 0 and y == 0:
        return 1
    else:
        return x / (y + 0.0001)
