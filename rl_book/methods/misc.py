import random
from typing import Any

import numpy as np
import torch

from rl_book.env import ParametrizedEnv
from rl_book.methods.method import RLMethod


class Random(RLMethod):
    def __init__(
        self, env: ParametrizedEnv, device: torch.device = torch.device("cpu")
    ):
        super().__init__(env)

    def get_name(self) -> str:
        return "Random"

    def act(self, state: int, step: int | None = None, mask: np.ndarray | list = []):
        allowed_actions = self.get_allowed_actions(mask)
        return random.choice(allowed_actions)

    def _get_save_data(self) -> Any:
        return None
