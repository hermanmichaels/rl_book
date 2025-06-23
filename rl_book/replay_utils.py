from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayItem:
    state: int
    action: int
    reward: float
    mask: np.ndarray | None = None
