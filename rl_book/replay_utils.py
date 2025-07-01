from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplayItem:
    state: int
    action: int
    reward: float
    mask: np.ndarray | list = field(default_factory=list)
