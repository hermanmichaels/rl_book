from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

S = TypeVar("S")


@dataclass
class ReplayItem(Generic[S]):
    state: S
    action: int
    reward: float
    mask: np.ndarray | list = field(default_factory=list)
